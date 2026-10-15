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
# 文件路径
# ============================================================

query_file = Path(
    "data/benchmark_v1/eval_queries.jsonl"
)

rewrite_file = Path(
    "data/benchmark_v1/eval_rewrites.jsonl"
)

details_file = Path(
    "data/benchmark_v1/results/"
    "v2_hybrid_eval_details.jsonl"
)

summary_file = Path(
    "data/benchmark_v1/results/"
    "v2_hybrid_eval_summary.json"
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

        if line.strip():

            queries.append(
                json.loads(line)
            )


# ============================================================
# 读取固定好的 Rewrite
# ============================================================

rewrites = {}

with open(
    rewrite_file,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if line.strip():

            item = json.loads(line)

            rewrites[
                item["query_id"]
            ] = item[
                "rewritten_query"
            ]


print("=" * 100)
print("V2 Hybrid + Query Rewrite Evaluation")
print("=" * 100)

print(
    "Eval Query数量：",
    len(queries)
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


# ============================================================
# 固定参数
# ============================================================

TOTAL_CHUNKS = 877

RRF_K = 10

DENSE_WEIGHT = 0.5
BM25_WEIGHT = 0.5

ORIGINAL_WEIGHT = 0.9
REWRITE_WEIGHT = 0.1


# ============================================================
# Metrics
# ============================================================

recall_1 = 0
recall_3 = 0
recall_5 = 0

rr_sum = 0.0

details = []


# ============================================================
# 940 条正式评测
# ============================================================

for index, item in enumerate(
        queries,
        start=1
):

    query_id = item[
        "query_id"
    ]

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
    # 1. Original Dense
    # ========================================================

    original_dense = (
        dense_retriever.search(
            query=original_query,
            top_k=TOTAL_CHUNKS
        )
    )


    # ========================================================
    # 2. Original BM25
    # ========================================================

    original_bm25 = (
        bm25_retriever.search(
            query=original_query,
            top_k=TOTAL_CHUNKS
        )
    )


    # ========================================================
    # 3. Rewrite Dense
    # ========================================================

    rewrite_dense = (
        dense_retriever.search(
            query=rewritten_query,
            top_k=TOTAL_CHUNKS
        )
    )


    # ========================================================
    # 4. Rewrite BM25
    # ========================================================

    rewrite_bm25 = (
        bm25_retriever.search(
            query=rewritten_query,
            top_k=TOTAL_CHUNKS
        )
    )


    # ========================================================
    # 四路 Weighted RRF
    # ========================================================

    rrf_scores = {}


    # --------------------------------------------------------
    # Original Dense
    # --------------------------------------------------------

    for rank, result in enumerate(
            original_dense,
            start=1
    ):

        chunk_id = result[0]

        if chunk_id not in rrf_scores:

            rrf_scores[
                chunk_id
            ] = 0.0


        rrf_scores[
            chunk_id
        ] += (
            ORIGINAL_WEIGHT
            * DENSE_WEIGHT
            / (
                RRF_K
                + rank
            )
        )


    # --------------------------------------------------------
    # Original BM25
    # --------------------------------------------------------

    for rank, result in enumerate(
            original_bm25,
            start=1
    ):

        chunk_id = result[
            "chunk_id"
        ]

        if chunk_id not in rrf_scores:

            rrf_scores[
                chunk_id
            ] = 0.0


        rrf_scores[
            chunk_id
        ] += (
            ORIGINAL_WEIGHT
            * BM25_WEIGHT
            / (
                RRF_K
                + rank
            )
        )


    # --------------------------------------------------------
    # Rewrite Dense
    # --------------------------------------------------------

    for rank, result in enumerate(
            rewrite_dense,
            start=1
    ):

        chunk_id = result[0]

        if chunk_id not in rrf_scores:

            rrf_scores[
                chunk_id
            ] = 0.0


        rrf_scores[
            chunk_id
        ] += (
            REWRITE_WEIGHT
            * DENSE_WEIGHT
            / (
                RRF_K
                + rank
            )
        )


    # --------------------------------------------------------
    # Rewrite BM25
    # --------------------------------------------------------

    for rank, result in enumerate(
            rewrite_bm25,
            start=1
    ):

        chunk_id = result[
            "chunk_id"
        ]

        if chunk_id not in rrf_scores:

            rrf_scores[
                chunk_id
            ] = 0.0


        rrf_scores[
            chunk_id
        ] += (
            REWRITE_WEIGHT
            * BM25_WEIGHT
            / (
                RRF_K
                + rank
            )
        )


    # ========================================================
    # 最终融合排名
    # ========================================================

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


    # ========================================================
    # Reciprocal Rank
    # ========================================================

    reciprocal_rank = (
        1.0 / gold_rank
    )

    rr_sum += reciprocal_rank


    # ========================================================
    # Recall
    # ========================================================

    if gold_rank <= 1:

        recall_1 += 1


    if gold_rank <= 3:

        recall_3 += 1


    if gold_rank <= 5:

        recall_5 += 1


    # ========================================================
    # Details
    # ========================================================

    details.append(
        {
            "query_id":
                query_id,

            "query":
                original_query,

            "rewritten_query":
                rewritten_query,

            "query_type":
                item["query_type"],

            "gold_domain":
                item["domain"],

            "gold_chunk_id":
                gold_chunk_id,

            "gold_rank":
                gold_rank,

            "reciprocal_rank":
                reciprocal_rank
        }
    )


    if index % 50 == 0:

        print(
            f"已完成："
            f"{index}/{len(queries)}"
        )


# ============================================================
# 最终指标
# ============================================================

total = len(
    queries
)


summary = {

    "version": "V2 Hybrid Retrieval",
    "retrieval": "Dense + BM25 + Weighted RRF + Query Rewrite",
    "rrf_k": 10,
    "dense_weight": 0.4,
    "bm25_weight": 0.6,
    "original_weight": 0.5,
    "rewrite_weight": 0.5,

    "total_queries":
        total,

    "rrf_k":
        RRF_K,

    "dense_weight":
        DENSE_WEIGHT,

    "bm25_weight":
        BM25_WEIGHT,

    "original_weight":
        ORIGINAL_WEIGHT,

    "rewrite_weight":
        REWRITE_WEIGHT,

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
print("V2 Final Retrieval Result")
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