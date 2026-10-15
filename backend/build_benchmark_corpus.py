import json
from pathlib import Path

from src.ingestion.knowledge_base_chunker import (
    build_knowledge_base_chunks
)


# ============================================================
# 路径
# ============================================================

output_dir = Path(
    "data/benchmark_v1"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    output_dir / "corpus.jsonl"
)


# ============================================================
# 构建当前知识库 Chunk
# ============================================================

chunks = build_knowledge_base_chunks(
    "data/knowledge_base"
)


# ============================================================
# 导出 Benchmark Corpus
# ============================================================

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    for chunk in chunks:

        metadata = chunk.metadata

        item = {

            "chunk_id":
                metadata["chunk_id"],

            "text":
                chunk.page_content,

            "domain":
                metadata.get(
                    "domain"
                ),

            "file_name":
                metadata.get(
                    "file_name"
                ),

            "file_type":
                metadata.get(
                    "file_type"
                ),

            "chunk_type":
                metadata.get(
                    "chunk_type"
                ),

            "section_title":
                metadata.get(
                    "section_title"
                ),

            "section_index":
                metadata.get(
                    "section_index"
                ),

            "page_number":
                metadata.get(
                    "page_number"
                ),
        }


        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            )
            + "\n"
        )


print("=" * 80)
print("Benchmark V1 Corpus 构建完成")
print("=" * 80)

print(
    "Corpus数量：",
    len(chunks)
)

print(
    "输出文件：",
    output_file
)