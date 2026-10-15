import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg
from dotenv import load_dotenv

from src.embedding.bge_embedding import BGEEmbedding


load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

embedding_model = BGEEmbedding()

CHUNK_BATCH_SIZE = 20


def get_pending_chunks():
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    e.chunk_id,
                    e.summary,
                    e.synthetic_questions
                FROM rag_chunk_expansions e
                LEFT JOIN rag_chunk_expansion_vectors v
                    ON e.chunk_id = v.chunk_id
                GROUP BY
                    e.chunk_id,
                    e.summary,
                    e.synthetic_questions
                HAVING COUNT(v.chunk_id) < 6
                ORDER BY e.chunk_id
            """)

            return cur.fetchall()


def build_views(chunk_id, summary, questions):
    views = [
        {
            "chunk_id": chunk_id,
            "view_type": "summary",
            "view_index": 0,
            "text": summary
        }
    ]

    for index, question in enumerate(
        questions,
        start=1
    ):
        views.append(
            {
                "chunk_id": chunk_id,
                "view_type": "question",
                "view_index": index,
                "text": question
            }
        )

    return views


def save_views(views, embeddings):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:

            for view, embedding in zip(
                views,
                embeddings
            ):
                vector_text = (
                    "["
                    + ",".join(
                        str(float(value))
                        for value in embedding
                    )
                    + "]"
                )

                cur.execute("""
                    INSERT INTO rag_chunk_expansion_vectors (
                        chunk_id,
                        view_type,
                        view_index,
                        embedding
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s::vector
                    )
                    ON CONFLICT (
                        chunk_id,
                        view_type,
                        view_index
                    )
                    DO UPDATE SET
                        embedding = EXCLUDED.embedding
                """, (
                    view["chunk_id"],
                    view["view_type"],
                    view["view_index"],
                    vector_text
                ))

        conn.commit()

#单线程处理
def process_batch(batch_number, batch_chunks):
    views = []

    for chunk_id, summary, questions in batch_chunks:
        views.extend(
            build_views(
                chunk_id,
                summary,
                questions
            )
        )

    texts = [
        view["text"]
        for view in views
    ]

    embeddings = embedding_model.embed_documents(
        texts
    )

    save_views(
        views,
        embeddings
    )

    return {
        "batch_number": batch_number,
        "chunk_count": len(batch_chunks),
        "vector_count": len(views)
    }


def main():
    chunks = get_pending_chunks()

    total = len(chunks)

    print("=" * 100)
    print(f"待处理 Chunk 数量：{total}")
    print(f"每批 Chunk 数量：{CHUNK_BATCH_SIZE}")
    print("=" * 100)

    if total == 0:
        print("所有辅助向量已经生成完成。")
        return

    batches = []

    batch_number = 0

    for start in range(
            0,
            total,
            CHUNK_BATCH_SIZE
    ):
        batch_number += 1

        batch_chunks = chunks[
            start:start + CHUNK_BATCH_SIZE
        ]

        batches.append(
            (
                batch_number,
                batch_chunks
            )
        )

    print(f"总 Batch 数量：{len(batches)}")
    print("并行线程数：3")

    with ThreadPoolExecutor(
            max_workers=3
    ) as executor:

        futures = {}

        for batch_number, batch_chunks in batches:
            future = executor.submit(
                process_batch,
                batch_number,
                batch_chunks
            )

            futures[future] = batch_number

        completed_batches = 0

        for future in as_completed(futures):
            batch_number = futures[future]

            try:
                result = future.result()

                completed_batches += 1

                print(
                    f"Batch {result['batch_number']} 完成 "
                    f"| Chunk：{result['chunk_count']} "
                    f"| 向量：{result['vector_count']} "
                    f"| 总进度："
                    f"{completed_batches}/{len(batches)}"
                )

            except Exception as e:
                print(
                    f"Batch {batch_number} 失败：{e}"
                )


if __name__ == "__main__":
    main()