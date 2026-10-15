import os

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from src.embedding.bge_embedding import BGEEmbedding
from src.ingestion.knowledge_base_chunker import build_knowledge_base_chunks


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("没有读取到 DATABASE_URL，请检查 .env 文件")


# 构建全部知识库 Chunk
chunks = build_knowledge_base_chunks("data/knowledge_base")
ingest_chunks = chunks

# 使用 Chunk 正文生成 Embedding
texts = [
    chunk.page_content
    for chunk in ingest_chunks
]

print(f"准备写入 Chunk 数量：{len(ingest_chunks)}")


# 生成 BGE-M3 向量
embedding_model = BGEEmbedding()
embeddings = embedding_model.embed_documents(texts)

print(f"Embedding shape：{embeddings.shape}")


# 写入 PostgreSQL
with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:

        for chunk, embedding in zip(ingest_chunks, embeddings):

            vector_string = "[" + ",".join(
                map(str, embedding.tolist())
            ) + "]"

            cur.execute(
                """
                INSERT INTO rag_chunks (
                    chunk_id,
                    content,
                    metadata,
                    embedding
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s::vector
                )
                ON CONFLICT (chunk_id)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                (
                    chunk.metadata["chunk_id"],
                    chunk.page_content,
                    Jsonb(chunk.metadata),
                    vector_string
                )
            )

        conn.commit()

        cur.execute(
            "SELECT COUNT(*) FROM rag_chunks"
        )

        count = cur.fetchone()[0]


print(
    f"数据库 rag_chunks 当前记录数：{count}"
)