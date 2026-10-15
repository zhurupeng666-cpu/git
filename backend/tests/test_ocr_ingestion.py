"""OCR门禁、结构切片、页面判别与索引一致性回归；不调用真实OCR和模型接口。"""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.context.token_counter import TokenCounter
from src.ingestion.document_blocks import (
    DocumentBlock,
    parse_table_html,
    table_markdown,
    restore_rotated_bbox,
)
from src.ingestion.ocr_quality_checker import check_block
from src.ingestion.structured_chunker import build_structured_chunks
from src.ingestion.pdf_page_parser import classify_page, parse_pdf_blocks
from src.ingestion.ingestion_service import file_sha256, load_preview, save_preview
from src.ingestion.ingestion_publisher import validate_expansion, vector_text
from src.rag.index_snapshot import RefreshableBM25
from src.ingestion.ocr_review import review_preview


def block(text="压力读数为-0.05 Pa", kind="text", index=0, page=1, score=0.99, details=None):
    return DocumentBlock(
        page, index, kind, text, [0.1, 0.1, 0.9, 0.9], "paddle_structure_v3", score, details or {}
    )


def metadata(source="image.png"):
    return {
        "document_id": "doc_example",
        "source_sha256": "a" * 64,
        "file_name": source,
        "title": "压力测量",
        "file_type": "png",
        "source": source,
        "domain": "equipment",
    }


class OcrIngestionTests(unittest.TestCase):
    def test_reviewed_text_requires_audit(self):
        value = block(score=0.3)
        value.method = "reviewed_transcription"
        self.assertIn("missing_transcription_review_audit", check_block(value)["reasons"])

    def test_explicit_transcription_review_preserves_original_and_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.txt"
            source.write_text("来源文件", encoding="utf-8")
            values = metadata(str(source))
            values["source_sha256"] = file_sha256(source)
            values["ingest_signature"] = "base-signature"
            original = block("引|起电容变化")
            chunks, reviews = build_structured_chunks([original], values)
            preview = save_preview(
                [
                    {
                        "source": values,
                        "pages": {"truncated": False},
                        "chunks": chunks,
                        "review_items": reviews,
                        "blocks": [original.to_dict()],
                    }
                ],
                output_dir=Path(folder) / "original",
            )
            decision = Path(folder) / "decisions.json"
            decision.write_text(
                json.dumps(
                    {
                        "reviewer": "local-source-review",
                        "decisions": [
                            {
                                "document_id": values["document_id"],
                                "source_sha256": values["source_sha256"],
                                "page_number": 1,
                                "block_index": 0,
                                "action": "replace_text",
                                "text": "引起电容变化",
                                "reason": "对照原始图像确认多余分隔符",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result = review_preview(preview, decision, Path(folder) / "reviewed")
            manifest = load_preview(result)
            self.assertEqual(manifest["documents"][0]["chunks"][0].page_content, "引起电容变化")
            self.assertTrue(manifest["documents"][0]["chunks"][0].metadata["transcription_reviews"])
            self.assertNotEqual(
                manifest["documents"][0]["source"]["ingest_signature"], values["ingest_signature"]
            )
            self.assertIn("引|起", (preview / "raw_blocks.json").read_text(encoding="utf-8"))

    def test_short_paragraph_before_table_is_not_lost(self):
        rows = [["项目", "单位"], ["压力", "Pa"]]
        chunks, _ = build_structured_chunks(
            [
                block("下面的值必须与单位一起解释。"),
                block(table_markdown(rows), kind="table", index=1, details={"table_rows": rows}),
            ],
            metadata(),
        )
        self.assertIn(
            "下面的值必须与单位一起解释。", "\n".join(item.page_content for item in chunks)
        )

    def test_suspicious_chinese_separator_is_reviewed_without_rewriting(self):
        value = block("膜片变形引|起电容变化")
        self.assertIn("suspicious_inword_separator", check_block(value)["reasons"])
        self.assertEqual(value.text, "膜片变形引|起电容变化")

    def test_rotated_scan_coordinates_map_back_to_source(self):
        value = restore_rotated_bbox([0.1, 0.2, 0.4, 0.6], 90)
        for actual, expected in zip(value, [0.4, 0.1, 0.8, 0.4]):
            self.assertAlmostEqual(actual, expected)

    def test_nan_confidence_is_not_approved(self):
        self.assertEqual(check_block(block(score=float("nan")))["status"], "needs_review")

    def test_merged_table_cells_and_numeric_characters_are_preserved(self):
        rows = parse_table_html(
            '<table><tr><th rowspan="2">传感器</th><th>值</th></tr><tr><td>-0.05</td></tr><tr><td>PT-008</td><td>00108</td></tr></table>'
        )
        self.assertEqual(rows, [["传感器", "值"], ["传感器", "-0.05"], ["PT-008", "00108"]])
        self.assertIn("00108", table_markdown(rows))

    def test_table_parser_does_not_execute_or_keep_script(self):
        rows = parse_table_html(
            "<table><tr><td>A<script>alert(1)</script></td><td>B</td></tr></table>"
        )
        self.assertEqual(rows, [["A", "B"]])

    def test_low_confidence_numeric_line_is_quarantined(self):
        value = block(details={"lines": [{"text": "PT-008 -0.05 Pa", "confidence": 0.87}]})
        self.assertIn("uncertain_numeric_or_identifier", check_block(value)["reasons"])

    def test_missing_confidence_does_not_look_like_perfect_ocr(self):
        self.assertEqual(check_block(block(score=None))["status"], "needs_review")

    def test_visual_relationship_is_not_invented_from_ocr(self):
        chunks, reviews = build_structured_chunks([block("压力控制图", kind="image")], metadata())
        self.assertFalse(chunks)
        self.assertIn("visual_semantics_require_review", reviews[0]["quality"]["reasons"])

    def test_different_regions_have_distinct_ids_and_repeat_is_stable(self):
        values = [block("压力测量", index=0), block("压力测量", index=1, page=2)]
        first, _ = build_structured_chunks(values, metadata())
        second, _ = build_structured_chunks(values, metadata())
        self.assertEqual(len({item.metadata["chunk_id"] for item in first}), 2)
        self.assertEqual(
            [item.metadata["chunk_id"] for item in first],
            [item.metadata["chunk_id"] for item in second],
        )

    def test_long_text_obeys_budget_without_losing_negative_sign(self):
        text = "压力偏差-0.05 Pa。" * 200
        chunks, _ = build_structured_chunks([block(text)], metadata(), max_tokens=100)
        self.assertEqual("".join(item.page_content for item in chunks), text)
        self.assertTrue(
            all(TokenCounter().count_text_tokens(item.page_content) <= 100 for item in chunks)
        )

    def test_table_split_repeats_headers(self):
        rows = [["项目", "说明"]] + [[f"项目{i}", "测量方法与单位定义" * 4] for i in range(10)]
        chunks, reviews = build_structured_chunks(
            [block(table_markdown(rows), kind="table", details={"table_rows": rows})],
            metadata(),
            max_tokens=150,
        )
        self.assertGreater(len(chunks), 1)
        self.assertFalse(reviews)
        self.assertTrue(
            all(item.page_content.startswith("压力测量\n项目 | 说明") for item in chunks)
        )
        self.assertTrue(
            all(TokenCounter().count_text_tokens(item.page_content) <= 150 for item in chunks)
        )

    def test_incomplete_table_is_not_published_as_plain_text(self):
        chunks, reviews = build_structured_chunks([block("仅有一个表头", kind="table")], metadata())
        self.assertFalse(chunks)
        self.assertTrue(reviews)

    def test_normalized_coordinates_are_validated(self):
        with self.assertRaises(ValueError):
            DocumentBlock(1, 0, "text", "内容", [0, 0, 150, 100], "native")

    def test_native_page_is_preserved_without_ocr(self):
        page = Mock()
        page.extract_text.return_value = "这是足够长度的原生PDF正文。" * 10
        page.get.return_value = {}
        page.mediabox.width = 595
        page.mediabox.height = 842
        self.assertEqual(classify_page(page)["mode"], "native")

    def test_page_with_only_header_and_large_image_is_mixed(self):
        obj = Mock()
        obj.get_object.return_value = {"/Subtype": "/Image"}
        page = Mock()
        page.get.return_value = {"/XObject": {"/Im1": obj}}
        page.mediabox.width = 595
        page.mediabox.height = 842

        def extract(**kwargs):
            if "visitor_operand_before" in kwargs:
                kwargs["visitor_operand_before"](b"Do", ["/Im1"], [500, 0, 0, 700, 0, 0], None)
            return "可以提取的页眉文字" * 8

        page.extract_text.side_effect = extract
        self.assertEqual(classify_page(page)["mode"], "mixed")

    def test_missing_page_result_is_an_error(self):
        reader = Mock(is_encrypted=False, pages=[Mock()])
        with patch("src.ingestion.pdf_page_parser.PdfReader", return_value=reader), patch(
            "src.ingestion.pdf_page_parser.classify_page", return_value={"mode": "ocr", "text": ""}
        ):
            with self.assertRaisesRegex(RuntimeError, "遗漏"):
                parse_pdf_blocks("missing.pdf", engine=Mock(parse=Mock(return_value=[])))

    def test_incomplete_preview_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.txt"
            source.write_text("原始资料", encoding="utf-8")
            values = metadata(str(source))
            values["source_sha256"] = file_sha256(source)
            chunks, _ = build_structured_chunks([block()], values)
            preview = save_preview(
                [
                    {
                        "source": values,
                        "pages": {"truncated": True},
                        "chunks": chunks,
                        "review_items": [],
                        "blocks": [],
                    }
                ],
                output_dir=Path(folder) / "preview",
            )
            with self.assertRaisesRegex(ValueError, "部分页面"):
                load_preview(preview)

    def test_source_modified_after_preview_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.txt"
            source.write_text("原始资料", encoding="utf-8")
            values = metadata(str(source))
            values["source_sha256"] = file_sha256(source)
            chunks, _ = build_structured_chunks([block()], values)
            preview = save_preview(
                [
                    {
                        "source": values,
                        "pages": {"truncated": False},
                        "chunks": chunks,
                        "review_items": [],
                        "blocks": [],
                    }
                ],
                output_dir=Path(folder) / "preview",
            )
            source.write_text("源文件变化", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "源文件发生变化"):
                load_preview(preview)

    def test_non_1024_or_nonfinite_vector_is_rejected(self):
        for vector in ([0.0] * 768, [float("nan")] * 1024):
            with self.assertRaises(ValueError):
                vector_text(vector)

    def test_incomplete_expansion_is_not_silently_published(self):
        with self.assertRaises(ValueError):
            validate_expansion({"summary": "摘要", "synthetic_questions": [], "keywords": []})

    def test_index_refresh_keeps_old_query_snapshot_consistent(self):
        old = [{"content": "压力"}]
        current = [old[0], {"content": "流量"}]
        cache = RefreshableBM25(
            "unused", Mock(return_value=current), list, lambda value: value["content"], old, None
        )
        cache.interval = 0
        cursor = Mock()
        cursor.execute.return_value.fetchone.return_value = (2, "new-generation")
        old_snapshot = cache.snapshot
        with patch("src.rag.index_snapshot.psycopg.connect", return_value=cursor):
            cursor.__enter__ = Mock(return_value=cursor)
            cursor.__exit__ = Mock(return_value=False)
            new_snapshot = cache.get()
        self.assertEqual(len(old_snapshot[0]), 1)
        self.assertEqual(len(new_snapshot[0]), 2)
        self.assertEqual(len(new_snapshot[1].get_scores(list("流量"))), 2)


if __name__ == "__main__":
    unittest.main()
