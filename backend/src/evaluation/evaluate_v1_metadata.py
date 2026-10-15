import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from src.embedding.bge_embedding import BGEEmbedding


# ============================================================
# 环境变量
# ============================================================

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")


# ============================================================
# 文件路径
# ============================================================

query_file = Path(
    "data/benchmark_v1/eval_queries.jsonl"
)

route_file = Path(
    "data/benchmark_v1/results/v1_domain_routes.jsonl"
)

details_file = Path(
    "data/benchmark_v1/results/v1_metadata_eval_details.jsonl"
)

summary_file = Path(
    "data/benchmark_v1/results/v1_metadata_eval_summary.json"
)

details_file.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 读取 Eval Query
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


# ============================================================
# 读取固定好的 Top-2 Router 结果
# ============================================================

routes = {}

with open(
    route_file,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        item = json.loads(line)

        routes[item["query_id"]] = item


print("=" * 100)
print("V1 Metadata Retrieval Evaluation")
print("=" * 100)

print(
    "Eval Query数量：",
    len(queries)
)

print(
    "Router结果数量：",
    len(routes)
)


# ============================================================
# Embedding Model
# ============================================================

embedding_model = BGEEmbedding()


# ============================================================
# 一次性计算 940 条 Query Embedding
# ============================================================

query_texts = [
    item["query"]
    for item in queries
]

query_embeddings = (
    embedding_model.embed_documents(
        query_texts
    )
)

print(
    "Query Embedding Shape：",
    query_embeddings.shape
)


# ============================================================
# 指标统计变量
# ============================================================

recall_1 = 0
recall_3 = 0
recall_5 = 0

rr_sum = 0.0

details = []


# ============================================================
# PostgreSQL
# ============================================================

with psycopg.connect(
    DB_URL
) as conn:

    with conn.cursor() as cur:

        for index, (
            item,
            query_embedding
        ) in enumerate(
            zip(
                queries,
                query_embeddings
            ),
            start=1
        ):

            query_id = item[
                "query_id"
            ]

            gold_chunk_id = item[
                "gold_chunk_id"
            ]


            # ==================================================
            # 获取该 Query 固定好的 Top-2 Domain
            # ==================================================

            route = routes[
                query_id
            ]

            primary_domain = route[
                "primary_domain"
            ]

            secondary_domain = route[
                "secondary_domain"
            ]


            # ==================================================
            # 向量转 pgvector 格式
            # ==================================================

            vector_string = (
                "["
                + ",".join(
                    map(
                        str,
                        query_embedding.tolist()
                    )
                )
                + "]"
            )


            # ==================================================
            # V1 核心：
            #
            # 只在 Router 预测的两个 Domain 内进行排名
            #
            # 如果 Gold Chunk 不在这两个 Domain 中，
            # SQL自然查不到，rank = None
            # ==================================================

            sql = """
            WITH ranked AS (

                SELECT
                    chunk_id,

                    RANK() OVER (
                        ORDER BY
                            embedding <=> %s::vector
                    ) AS rank

                FROM rag_chunks

                WHERE
                    metadata->>'domain'
                    IN (%s, %s)

            )

            SELECT rank

            FROM ranked

            WHERE chunk_id = %s
            """


            cur.execute(
                sql,
                (
                    vector_string,
                    primary_domain,
                    secondary_domain,
                    gold_chunk_id
                )
            )


            result = cur.fetchone()


            # ==================================================
            # Gold Rank
            # ==================================================

            if result is None:

                gold_rank = None

                reciprocal_rank = 0.0

            else:

                gold_rank = int(
                    result[0]
                )

                reciprocal_rank = (
                    1.0 / gold_rank
                )


            # ==================================================
            # Recall@K
            # ==================================================

            if (
                gold_rank is not None
                and gold_rank <= 1
            ):

                recall_1 += 1


            if (
                gold_rank is not None
                and gold_rank <= 3
            ):

                recall_3 += 1


            if (
                gold_rank is not None
                and gold_rank <= 5
            ):

                recall_5 += 1


            rr_sum += reciprocal_rank


            # ==================================================
            # 保存每条 Query 的详细结果
            # ==================================================

            detail = {

                "query_id":
                    query_id,

                "query":
                    item["query"],

                "query_type":
                    item["query_type"],

                "gold_domain":
                    item["domain"],

                "primary_domain":
                    primary_domain,

                "secondary_domain":
                    secondary_domain,

                "gold_chunk_id":
                    gold_chunk_id,

                "gold_rank":
                    gold_rank,

                "reciprocal_rank":
                    reciprocal_rank
            }


            details.append(
                detail
            )


            if index % 100 == 0:

                print(
                    f"已完成："
                    f"{index}/{len(queries)}"
                )


# ============================================================
# 计算最终指标
# ============================================================

total = len(
    queries
)

summary = {

    "version":
        "V1 Metadata Retrieval",

    "total_queries":
        total,

    "Recall@1":
        recall_1 / total,

    "Recall@3":
        recall_3 / total,

    "Recall@5":
        recall_5 / total,

    "MRR":
        rr_sum / total
}


# ============================================================
# 保存 Details
# ============================================================

with open(
    details_file,
    "w",
    encoding="utf-8"
) as f:

    for item in details:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# 保存 Summary
# ============================================================

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

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
print("V1 Metadata Retrieval Result")
print("=" * 100)

print(
    f"Recall@1："
    f"{summary['Recall@1']:.4f}"
)

print(
    f"Recall@3："
    f"{summary['Recall@3']:.4f}"
)

print(
    f"Recall@5："
    f"{summary['Recall@5']:.4f}"
)

print(
    f"MRR："
    f"{summary['MRR']:.4f}"
)

print(
    "\nDetails：",
    details_file
)

print(
    "Summary：",
    summary_file
)