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
# Dev Query
# ============================================================

dev_file = Path(
    "data/benchmark_v1/dev_queries.jsonl"
)

with open(
    dev_file,
    "r",
    encoding="utf-8"
) as f:

    dev_queries = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


print("=" * 100)
print("V2.1 Weighted RRF Parameter Tuning")
print("=" * 100)

print(
    "Dev Query数量：",
    len(dev_queries)
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
# 缓存 Dense / BM25 全量排名
# ============================================================

cached_rankings = []


for index, item in enumerate(
        dev_queries,
        start=1
):

    query = item[
        "query"
    ]

    gold_chunk_id = item[
        "gold_chunk_id"
    ]


    # ========================================================
    # Dense 全量排名
    # ========================================================

    dense_results = (
        dense_retriever.search(
            query=query,
            top_k=TOTAL_CHUNKS
        )
    )


    dense_rank_map = {}

    for rank, result in enumerate(
            dense_results,
            start=1
    ):

        dense_rank_map[
            result[0]
        ] = rank


    # ========================================================
    # BM25 全量排名
    # ========================================================

    bm25_results = (
        bm25_retriever.search(
            query=query,
            top_k=TOTAL_CHUNKS
        )
    )


    bm25_rank_map = {}

    for rank, result in enumerate(
            bm25_results,
            start=1
    ):

        bm25_rank_map[
            result["chunk_id"]
        ] = rank


    cached_rankings.append(
        {
            "query_id":
                item["query_id"],

            "gold_chunk_id":
                gold_chunk_id,

            "dense_rank_map":
                dense_rank_map,

            "bm25_rank_map":
                bm25_rank_map
        }
    )


    print(
        f"排名缓存完成："
        f"{index}/{len(dev_queries)}"
    )


print(
    "\nDense + BM25 排名缓存完成"
)


# ============================================================
# 参数搜索空间
# ============================================================

dense_weights = [
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


rrf_k_values = [
    10,
    20,
    40,
    60,
    80
]


all_results = []


# ============================================================
# Grid Search
# ============================================================

for dense_weight in dense_weights:

    bm25_weight = (
        1.0 - dense_weight
    )


    for rrf_k in rrf_k_values:

        recall_1 = 0
        recall_3 = 0
        recall_5 = 0

        rr_sum = 0.0


        for cached in cached_rankings:

            gold_chunk_id = cached[
                "gold_chunk_id"
            ]

            dense_rank_map = cached[
                "dense_rank_map"
            ]

            bm25_rank_map = cached[
                "bm25_rank_map"
            ]


            chunk_ids = set(
                dense_rank_map.keys()
            ) | set(
                bm25_rank_map.keys()
            )


            rrf_scores = {}


            # =================================================
            # Weighted RRF
            # =================================================

            for chunk_id in chunk_ids:

                score = 0.0


                dense_rank = (
                    dense_rank_map.get(
                        chunk_id
                    )
                )

                if dense_rank is not None:

                    score += (
                        dense_weight
                        / (
                            rrf_k
                            + dense_rank
                        )
                    )


                bm25_rank = (
                    bm25_rank_map.get(
                        chunk_id
                    )
                )

                if bm25_rank is not None:

                    score += (
                        bm25_weight
                        / (
                            rrf_k
                            + bm25_rank
                        )
                    )


                rrf_scores[
                    chunk_id
                ] = score


            # =================================================
            # 最终排名
            # =================================================

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


            # =================================================
            # Metrics
            # =================================================

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

            "dense_weight":
                dense_weight,

            "bm25_weight":
                bm25_weight,

            "rrf_k":
                rrf_k,

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
            f"Dense={dense_weight:.1f} "
            f"BM25={bm25_weight:.1f} "
            f"k={rrf_k:<3} | "
            f"R@5={result['Recall@5']:.4f} | "
            f"MRR={result['MRR']:.4f}"
        )


# ============================================================
# 排序
# Recall@5 优先，MRR 第二
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
print("最佳 Weighted RRF 参数")
print("=" * 100)

print(
    "Dense Weight：",
    best["dense_weight"]
)

print(
    "BM25 Weight：",
    best["bm25_weight"]
)

print(
    "RRF k：",
    best["rrf_k"]
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


print("\n" + "=" * 100)
print("Top 10 参数组合")
print("=" * 100)


for index, result in enumerate(
        all_results[:10],
        start=1
):

    print(
        f"{index:>2}. "
        f"Dense={result['dense_weight']:.1f} "
        f"BM25={result['bm25_weight']:.1f} "
        f"k={result['rrf_k']:<3} | "
        f"R@1={result['Recall@1']:.4f} "
        f"R@3={result['Recall@3']:.4f} "
        f"R@5={result['Recall@5']:.4f} "
        f"MRR={result['MRR']:.4f}"
    )