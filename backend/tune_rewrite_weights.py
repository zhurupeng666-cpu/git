import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.rag.vector_retriever import VectorRetriever
from src.rag.bm25_retriever import BM25Retriever


# ============================================================
# 环境变量
# ============================================================

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")


# ============================================================
# 文件
# ============================================================

dev_file = Path(
    "data/benchmark_v1/dev_queries.jsonl"
)

rewrite_file = Path(
    "data/benchmark_v1/dev_rewrites.jsonl"
)


# ============================================================
# 读取 Dev Query
# ============================================================

dev_queries = {}

with open(
    dev_file,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if not line.strip():
            continue

        item = json.loads(line)

        dev_queries[
            item["query_id"]
        ] = item


# ============================================================
# 读取固定 Rewrite
# ============================================================

rewrites = {}

with open(
    rewrite_file,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if not line.strip():
            continue

        item = json.loads(line)

        rewrites[
            item["query_id"]
        ] = item[
            "rewritten_query"
        ]


print("=" * 100)
print("Query Rewrite Weight Tuning")
print("=" * 100)

print(
    "Dev Query数量：",
    len(dev_queries)
)

print(
    "Rewrite数量：",
    len(rewrites)
)


# ============================================================
# Retriever
# ============================================================

dense_retriever = VectorRetriever(
    DB_URL
)

bm25_retriever = BM25Retriever(
    DB_URL
)


TOTAL_CHUNKS = 877


# ============================================================
# 当前已经调好的 V2 参数
# ============================================================

RRF_K = 10

DENSE_WEIGHT = 0.4
BM25_WEIGHT = 0.6


# ============================================================
# 缓存四路排名
# ============================================================

cached_rankings = []


for index, (
        query_id,
        item
) in enumerate(
    dev_queries.items(),
    start=1
):

    original_query = item[
        "query"
    ]

    rewritten_query = rewrites[
        query_id
    ]

    gold_chunk_id = item[
        "gold_chunk_id"
    ]


    # ========================================================
    # Original Dense
    # ========================================================

    original_dense_results = (
        dense_retriever.search(
            query=original_query,
            top_k=TOTAL_CHUNKS
        )
    )

    original_dense_rank = {
        result[0]: rank
        for rank, result in enumerate(
            original_dense_results,
            start=1
        )
    }


    # ========================================================
    # Original BM25
    # ========================================================

    original_bm25_results = (
        bm25_retriever.search(
            query=original_query,
            top_k=TOTAL_CHUNKS
        )
    )

    original_bm25_rank = {
        result["chunk_id"]: rank
        for rank, result in enumerate(
            original_bm25_results,
            start=1
        )
    }


    # ========================================================
    # Rewrite Dense
    # ========================================================

    rewrite_dense_results = (
        dense_retriever.search(
            query=rewritten_query,
            top_k=TOTAL_CHUNKS
        )
    )

    rewrite_dense_rank = {
        result[0]: rank
        for rank, result in enumerate(
            rewrite_dense_results,
            start=1
        )
    }


    # ========================================================
    # Rewrite BM25
    # ========================================================

    rewrite_bm25_results = (
        bm25_retriever.search(
            query=rewritten_query,
            top_k=TOTAL_CHUNKS
        )
    )

    rewrite_bm25_rank = {
        result["chunk_id"]: rank
        for rank, result in enumerate(
            rewrite_bm25_results,
            start=1
        )
    }


    cached_rankings.append(
        {
            "query_id":
                query_id,

            "gold_chunk_id":
                gold_chunk_id,

            "original_dense_rank":
                original_dense_rank,

            "original_bm25_rank":
                original_bm25_rank,

            "rewrite_dense_rank":
                rewrite_dense_rank,

            "rewrite_bm25_rank":
                rewrite_bm25_rank
        }
    )


    print(
        f"排名缓存完成："
        f"{index}/{len(dev_queries)}"
    )


print(
    "\n四路排名缓存完成"
)


# ============================================================
# Original / Rewrite 权重搜索空间
# ============================================================

original_weights = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]


all_results = []


# ============================================================
# 参数搜索
# ============================================================

for original_weight in original_weights:

    rewrite_weight = (
        1.0 - original_weight
    )

    recall_1 = 0
    recall_3 = 0
    recall_5 = 0

    rr_sum = 0.0


    for cached in cached_rankings:

        gold_chunk_id = cached[
            "gold_chunk_id"
        ]


        chunk_ids = set(
            cached[
                "original_dense_rank"
            ].keys()
        )


        rrf_scores = {}


        # ====================================================
        # 四路 Weighted RRF
        # ====================================================

        for chunk_id in chunk_ids:

            score = 0.0


            # Original Dense
            rank = cached[
                "original_dense_rank"
            ][chunk_id]

            score += (
                original_weight
                * DENSE_WEIGHT
                / (
                    RRF_K
                    + rank
                )
            )


            # Original BM25
            rank = cached[
                "original_bm25_rank"
            ][chunk_id]

            score += (
                original_weight
                * BM25_WEIGHT
                / (
                    RRF_K
                    + rank
                )
            )


            # Rewrite Dense
            rank = cached[
                "rewrite_dense_rank"
            ][chunk_id]

            score += (
                rewrite_weight
                * DENSE_WEIGHT
                / (
                    RRF_K
                    + rank
                )
            )


            # Rewrite BM25
            rank = cached[
                "rewrite_bm25_rank"
            ][chunk_id]

            score += (
                rewrite_weight
                * BM25_WEIGHT
                / (
                    RRF_K
                    + rank
                )
            )


            rrf_scores[
                chunk_id
            ] = score


        # ====================================================
        # 最终排名
        # ====================================================

        ranked_chunk_ids = sorted(
            rrf_scores.keys(),

            key=lambda chunk_id:
                rrf_scores[
                    chunk_id
                ],

            reverse=True
        )


        gold_rank = (
            ranked_chunk_ids.index(
                gold_chunk_id
            )
            + 1
        )


        # ====================================================
        # Metrics
        # ====================================================

        if gold_rank <= 1:
            recall_1 += 1

        if gold_rank <= 3:
            recall_3 += 1

        if gold_rank <= 5:
            recall_5 += 1

        rr_sum += (
            1.0 / gold_rank
        )


    total = len(
        cached_rankings
    )


    result = {

        "original_weight":
            original_weight,

        "rewrite_weight":
            rewrite_weight,

        "Recall@1":
            recall_1 / total,

        "Recall@3":
            recall_3 / total,

        "Recall@5":
            recall_5 / total,

        "MRR":
            rr_sum / total
    }


    all_results.append(
        result
    )


    print(
        f"Original={original_weight:.1f} "
        f"Rewrite={rewrite_weight:.1f} | "
        f"R@1={result['Recall@1']:.4f} "
        f"R@3={result['Recall@3']:.4f} "
        f"R@5={result['Recall@5']:.4f} "
        f"MRR={result['MRR']:.4f}"
    )


# ============================================================
# 排序
# ============================================================

all_results.sort(
    key=lambda x: (
        x["Recall@5"],
        x["MRR"]
    ),
    reverse=True
)


best = all_results[0]


print("\n" + "=" * 100)
print("最佳 Query Rewrite 权重")
print("=" * 100)

print(
    "Original Weight：",
    best["original_weight"]
)

print(
    "Rewrite Weight：",
    best["rewrite_weight"]
)

print(
    f"Recall@1："
    f"{best['Recall@1']:.4f}"
)

print(
    f"Recall@3："
    f"{best['Recall@3']:.4f}"
)

print(
    f"Recall@5："
    f"{best['Recall@5']:.4f}"
)

print(
    f"MRR："
    f"{best['MRR']:.4f}"
)