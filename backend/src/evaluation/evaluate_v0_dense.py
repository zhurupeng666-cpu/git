import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from src.embedding.bge_embedding import (
    BGEEmbedding
)


# ============================================================
# 环境变量
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

if not DATABASE_URL:
    raise ValueError(
        "没有读取到 DATABASE_URL"
    )


# ============================================================
# 文件路径
# ============================================================

query_file = Path(
    "data/benchmark_v1/eval_queries.jsonl"
)

result_dir = Path(
    "data/benchmark_v1/results"
)

result_dir.mkdir(
    parents=True,
    exist_ok=True
)

detail_file = (
    result_dir / "v0_dense_eval_details.jsonl"
)

summary_file = (
    result_dir / "v0_dense_eval_summary.json"
)


# ============================================================
# 读取1000条 Benchmark Query
# ============================================================

queries = []

with open(
    query_file,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        queries.append(
            json.loads(line)
        )


print("=" * 100)
print("V0 Baseline Dense Retrieval Evaluation")
print("=" * 100)

print(
    "Benchmark Query数量：",
    len(queries)
)


# ============================================================
# 一次性批量生成 Query Embedding
# ============================================================

query_texts = [
    item["query"]
    for item in queries
]


embedding_model = BGEEmbedding()


query_embeddings = (
    embedding_model.embed_documents(
        query_texts
    )
)


print(
    "Query Embedding shape：",
    query_embeddings.shape
)


# ============================================================
# 指标统计
# ============================================================

hit_at_1 = 0
hit_at_3 = 0
hit_at_5 = 0

reciprocal_rank_sum = 0.0

details = []


# ============================================================
# PostgreSQL评测
# ============================================================

with psycopg.connect(
    DATABASE_URL
) as conn:

    with conn.cursor() as cur:

        for index, (
            item,
            embedding
        ) in enumerate(
            zip(
                queries,
                query_embeddings
            ),
            start=1
        ):

            # --------------------------------------------
            # 转成 pgvector 可接受格式
            # --------------------------------------------

            vector_string = (
                "["+ ",".join(
                map(
                        str,
                        embedding.tolist()
                    )
                )
                + "]"
            )


            gold_chunk_id = (
                item["gold_chunk_id"]
            )


            # --------------------------------------------
            # 计算 Gold Chunk 在全部877个Chunk中的真实排名
            #
            # rank = 1
            # 代表正确Chunk就是第一名
            # --------------------------------------------

            sql = """
            WITH ranked AS (

                SELECT
                    chunk_id,

                    RANK() OVER (
                        ORDER BY
                            embedding <=> %s::vector
                    ) AS rank

                FROM rag_chunks

            )
            SELECT rank
            FROM ranked
            WHERE chunk_id = %s
            """
            cur.execute(
                sql,
                (
                    vector_string,
                    gold_chunk_id
                )
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(
                    f"数据库中找不到Gold Chunk："
                    f"{gold_chunk_id}"
                )

            gold_rank = int(
                row[0]
            )

            # --------------------------------------------
            # Recall / Hit
            # --------------------------------------------
            if gold_rank <= 1:
                hit_at_1 += 1

            if gold_rank <= 3:
                hit_at_3 += 1

            if gold_rank <= 5:
                hit_at_5 += 1

            # --------------------------------------------
            # Reciprocal Rank
            # --------------------------------------------

            reciprocal_rank = (
                1.0 / gold_rank
            )

            reciprocal_rank_sum += (
                reciprocal_rank
            )

            # --------------------------------------------
            # 保存单条结果
            # --------------------------------------------

            details.append(
                {
                    "query_id":
                        item["query_id"],

                    "query":
                        item["query"],

                    "query_type":
                        item["query_type"],

                    "domain":
                        item["domain"],

                    "gold_chunk_id":
                        gold_chunk_id,

                    "gold_rank":
                        gold_rank,

                    "reciprocal_rank":
                        reciprocal_rank,
                }
            )

            # --------------------------------------------
            # 进度
            # --------------------------------------------

            if index % 100 == 0:

                print(
                    f"已评测："
                    f"{index}/{len(queries)}"
                )

# ============================================================
# 最终指标
# ============================================================

total = len(
    queries
)

recall_at_1 = (
    hit_at_1 / total
)

recall_at_3 = (
    hit_at_3 / total
)

recall_at_5 = (
    hit_at_5 / total
)

mrr = (
    reciprocal_rank_sum / total
)

summary = {

    "version":
        "V0 Dense Retrieval",

    "embedding_model":
        "BAAI/bge-m3",

    "retrieval":
        "pgvector cosine similarity",

    "query_count":
        total,

    "Recall@1":
        recall_at_1,

    "Recall@3":
        recall_at_3,

    "Recall@5":
        recall_at_5,

    "MRR":
        mrr,
}


# ============================================================
# 保存详细结果
# ============================================================

with open(detail_file, "w", encoding="utf-8") as f:
    for item in details:
        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            )
            + "\n"
        )

# ============================================================
# 保存汇总结果
# ============================================================

with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 输出
# ============================================================

print("\n" + "=" * 100)
print("V0 Baseline Evaluation Result")
print("=" * 100)

print(
    f"Recall@1："
    f"{recall_at_1:.4f} "
    f"({recall_at_1 * 100:.2f}%)"
)

print(
    f"Recall@3："
    f"{recall_at_3:.4f} "
    f"({recall_at_3 * 100:.2f}%)"
)

print(
    f"Recall@5："
    f"{recall_at_5:.4f} "
    f"({recall_at_5 * 100:.2f}%)"
)

print(
    f"MRR："
    f"{mrr:.4f}"
)

print(
    "\n详细结果：",
    detail_file
)

print(
    "汇总结果：",
    summary_file
)