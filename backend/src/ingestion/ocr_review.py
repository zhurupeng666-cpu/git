"""本地复核入口：必须明确提交逐块决定，保留原始OCR和校正审计，不自动猜测。"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from src.ingestion.document_blocks import DocumentBlock, table_markdown
from src.ingestion.ingestion_service import file_sha256, save_preview
from src.ingestion.structured_chunker import build_structured_chunks


def review_preview(preview_folder, decision_file, output_dir):
    folder = Path(preview_folder)
    decisions = json.loads(Path(decision_file).read_text(encoding="utf-8"))
    reviewer = decisions.get("reviewer", "").strip()
    if not reviewer or not decisions.get("decisions"):
        raise ValueError("必须提供复核者标识和逐块决定，不能批量默认为通过")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    original = json.loads((folder / "raw_blocks.json").read_text(encoding="utf-8"))
    if len(original) != len(manifest["documents"]):
        raise ValueError("原始块文件与预览文档数量不一致")
    digest = hashlib.sha256(
        json.dumps(decisions, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    audit = []
    indexed = {}
    for decision in decisions["decisions"]:
        key = (
            decision["document_id"],
            decision["source_sha256"],
            int(decision["page_number"]),
            int(decision["block_index"]),
        )
        if key in indexed:
            raise ValueError("同一区域不能有多个互相覆盖的复核决定")
        if not str(decision.get("reason", "")).strip():
            raise ValueError("每个决定都需要说明对照依据或排除原因")
        indexed[key] = decision
    used = set()
    results = []
    for entry, info in zip(original, manifest["documents"]):
        source = dict(entry["source"])
        if (
            source["document_id"] != info["source"]["document_id"]
            or source["source_sha256"] != info["source"]["source_sha256"]
        ):
            raise ValueError("原始块与预览来源不一致")
        if file_sha256(source["source"]) != source["source_sha256"]:
            raise ValueError("复核前源文件已变化，不能把旧校正用于新文件")
        source["ingest_signature"] = hashlib.sha256(
            (source["ingest_signature"] + digest).encode()
        ).hexdigest()
        blocks = []
        for value in entry["blocks"]:
            value = dict(value)
            key = (
                source["document_id"],
                source["source_sha256"],
                value["page_number"],
                value["block_index"],
            )
            decision = indexed.get(key)
            if decision:
                used.add(key)
                action = decision["action"]
                review = {
                    "reviewed_by": reviewer,
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                    "original_text_sha256": hashlib.sha256(value["text"].encode()).hexdigest(),
                    "original_method": value["method"],
                    "reason": decision["reason"],
                    "action": action,
                }
                audit.append(
                    {
                        "source_sha256": source["source_sha256"],
                        "page_number": value["page_number"],
                        "block_index": value["block_index"],
                        **review,
                    }
                )
                if action == "exclude":
                    continue
                if action == "replace_table":
                    rows = decision.get("rows")
                    if not isinstance(rows, list) or not all(
                        isinstance(row, list) and all(isinstance(cell, str) for cell in row)
                        for row in rows
                    ):
                        raise ValueError("表格校正需要提供字符串行列数组")
                    value["kind"] = "table"
                    value["text"] = table_markdown(rows)
                    value["details"] = {**value.get("details", {}), "table_rows": rows}
                elif action in ("replace_text", "describe_figure", "confirm_text"):
                    if action != "confirm_text":
                        text = decision.get("text", "").strip()
                        if not text:
                            raise ValueError("文字校正或图示解释不能是空内容")
                        value["text"] = text
                    if action == "describe_figure":
                        value["kind"] = "text"
                else:
                    raise ValueError("不支持的复核动作")
                value["method"] = "reviewed_transcription"
                value["details"] = {**value.get("details", {}), "transcription_review": review}
            blocks.append(DocumentBlock(**value))
        chunks, reviews = build_structured_chunks(blocks, source)
        results.append(
            {
                "source": source,
                "pages": info["pages"],
                "chunks": chunks,
                "review_items": reviews,
                "blocks": [block.to_dict() for block in blocks],
                "block_count": len(blocks),
            }
        )
    if used != set(indexed):
        raise ValueError("复核决定中的来源或区域不存在，未生成新预览")
    destination = save_preview(results, manifest.get("skipped"), output_dir)
    (destination / "review_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination
