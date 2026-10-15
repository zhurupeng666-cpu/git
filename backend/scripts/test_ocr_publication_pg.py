"""真实PG隔离schema验收：发布故障回滚、辅助索引完整和重复发布幂等。"""

import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
import psycopg
from psycopg import sql
from src.ingestion.document_blocks import DocumentBlock
from src.ingestion.structured_chunker import build_structured_chunks
from src.ingestion.ingestion_service import file_sha256, save_preview
from src.ingestion.ingestion_publisher import publish_preview


def main():
    real_connect = psycopg.connect
    url = os.environ["DATABASE_URL"]
    schema = "ocr_publish_test_" + uuid4().hex
    fail = {"enabled": True}
    results = {}
    with real_connect(url) as conn:
        original = conn.execute(
            "SELECT count(*),md5(string_agg(chunk_id||md5(content)||md5(metadata::text)||md5(embedding::text),'' ORDER BY chunk_id)) FROM public.rag_chunks"
        ).fetchone()
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        for table in ("rag_chunks", "rag_chunk_expansions", "rag_chunk_expansion_vectors"):
            conn.execute(
                sql.SQL("CREATE TABLE {}.{} (LIKE public.{} INCLUDING ALL)").format(
                    sql.Identifier(schema), sql.Identifier(table), sql.Identifier(table)
                )
            )
        for table in ("rag_chunk_expansions", "rag_chunk_expansion_vectors"):
            conn.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD FOREIGN KEY(chunk_id) REFERENCES {}.rag_chunks(chunk_id) ON DELETE CASCADE"
                ).format(sql.Identifier(schema), sql.Identifier(table), sql.Identifier(schema))
            )

    class Connection:
        def __init__(self, *args, **kwargs):
            self.conn = real_connect(*args, **kwargs)

        def __enter__(self):
            self.conn.__enter__()
            return self

        def __exit__(self, *args):
            return self.conn.__exit__(*args)

        def execute(self, query, params=None):
            query = query.replace("public.", schema + ".")
            if fail["enabled"] and query.startswith(
                "INSERT INTO " + schema + ".rag_chunk_expansions"
            ):
                raise RuntimeError("injected publication failure")
            return self.conn.execute(query, params)

    class Embedding:
        def embed_documents(self, texts):
            return [[1.0] + [0.0] * 1023 for _ in texts]

    try:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.txt"
            source.write_text("真实PG隔离事务验证资料", encoding="utf-8")
            metadata = {
                "source": str(source),
                "source_relpath": "equipment/isolated.txt",
                "source_sha256": file_sha256(source),
                "document_id": "doc_isolated",
                "ingest_signature": "isolated-v1",
                "title": "事务验证",
                "file_type": "txt",
                "file_name": "isolated.txt",
            }
            blocks = [
                DocumentBlock(
                    1, 0, "text", "故障时必须回滚正文和辅助索引。", [0, 0, 1, 1], "native"
                )
            ]
            chunks, _ = build_structured_chunks(blocks, metadata)
            preview = save_preview(
                [
                    {
                        "source": metadata,
                        "pages": {"truncated": False},
                        "chunks": chunks,
                        "review_items": [],
                        "blocks": [],
                    }
                ],
                output_dir=Path(folder) / "preview",
            )
            expansion = lambda _: {
                "summary": "事务验证",
                "synthetic_questions": [f"事务问题{i}" for i in range(5)],
                "keywords": ["事务", "回滚"],
            }
            with patch("src.ingestion.ingestion_publisher.psycopg.connect", Connection), patch(
                "src.ingestion.ingestion_publisher.backup_knowledge",
                return_value={"isolated_test": True},
            ):
                try:
                    publish_preview(preview, Embedding(), expansion)
                    raise AssertionError("Expected injected failure")
                except RuntimeError as exc:
                    assert "injected" in str(exc)
                with real_connect(url) as conn:
                    assert (
                        conn.execute(
                            sql.SQL("SELECT count(*) FROM {}.rag_chunks").format(
                                sql.Identifier(schema)
                            )
                        ).fetchone()[0]
                        == 0
                    )
                    assert (
                        conn.execute(
                            "SELECT to_regclass(%s)", (schema + ".rag_document_versions",)
                        ).fetchone()[0]
                        is None
                    )
                results["transaction_rollback"] = True
                fail["enabled"] = False
                published = publish_preview(preview, Embedding(), expansion)
                assert published["chunks"] == 1
                assert (
                    publish_preview(preview, Embedding(), expansion)["status"]
                    == "already_published"
                )
                results["repeat_publication_is_noop"] = True
                with real_connect(url) as conn:
                    count = conn.execute(
                        sql.SQL("SELECT count(*) FROM {}.rag_chunk_expansion_vectors").format(
                            sql.Identifier(schema)
                        )
                    ).fetchone()[0]
                    assert count == 6
                results["one_summary_five_question_vectors"] = True
    finally:
        if (
            not schema.startswith("ocr_publish_test_")
            or not schema[len("ocr_publish_test_") :].isalnum()
        ):
            raise RuntimeError("Invalid test schema")
        with real_connect(url) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
            actual = conn.execute(
                "SELECT count(*),md5(string_agg(chunk_id||md5(content)||md5(metadata::text)||md5(embedding::text),'' ORDER BY chunk_id)) FROM public.rag_chunks"
            ).fetchone()
            assert actual == original
            assert (
                conn.execute("SELECT to_regclass(%s)", (schema + ".rag_chunks",)).fetchone()[0]
                is None
            )
        results["original_knowledge_unchanged"] = True
        results["temporary_schema_removed"] = True
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
