import json
import os
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from src.rag.vector_retriever import VectorRetriever
from src.rag.bm25_retriever import BM25Retriever
from src.rag.reranker import BGEReranker


load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

DEV_FILE = Path("data/benchmark_v1/dev_queries.jsonl")
REWRITE_FILE = Path("data/benchmark_v1/dev_rewrites.jsonl")

RESULT_DIR = Path("data/benchmark_v1/results")
DETAIL_FILE = RESULT_DIR / "v3_reranker_dev_details.jsonl"
SUMMARY_FILE = RESULT_DIR / "v3_reranker_dev_summary.json"

TOTAL_CHUNKS = 877
RERANK_K = 50

RRF_K = 10
DENSE_WEIGHT = 0.4
BM25_WEIGHT = 0.6
ORIGINAL_WEIGHT = 0.5
REWRITE_WEIGHT = 0.5


def load_jsonl(path):
    items = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                items[item["query_id"]] = item

    return items


dev_queries = load_jsonl(DEV_FILE)
rewrite_items = load_jsonl(REWRITE_FILE)

rewrites = {
    query_id: item["rewritten_query"]
    for query_id, item in rewrite_items.items()
}


dense_retriever = VectorRetriever(DB_URL)
bm25_retriever = BM25Retriever(DB_URL)
reranker = BGEReranker()

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


v2_recall_1 = 0
v2_recall_3 = 0
v2_recall_5 = 0
v2_rr_sum = 0.0

v3_recall_1 = 0
v3_recall_3 = 0
v3_recall_5 = 0
v3_rr_sum = 0.0

candidate_hit_50 = 0

details = []


for index, (query_id, item) in enumerate(
    dev_queries.items(),
    start=1
):
    query = item["query"]
    rewritten_query = rewrites[query_id]
    gold_chunk_id = item["gold_chunk_id"]

    # 四路检索都取完整排名，保持和 V2 Dev 评测方式一致
    original_dense = dense_retriever.search(
        query=query,
        top_k=TOTAL_CHUNKS
    )

    original_bm25 = bm25_retriever.search(
        query=query,
        top_k=TOTAL_CHUNKS
    )

    rewrite_dense = dense_retriever.search(
        query=rewritten_query,
        top_k=TOTAL_CHUNKS
    )

    rewrite_bm25 = bm25_retriever.search(
        query=rewritten_query,
        top_k=TOTAL_CHUNKS
    )

    rrf_scores = defaultdict(float)
    document_map = {}

    for rank, result in enumerate(
        original_dense,
        start=1
    ):
        chunk_id = result[0]

        rrf_scores[chunk_id] += (
            ORIGINAL_WEIGHT
            * DENSE_WEIGHT
            / (RRF_K + rank)
        )

        document_map[chunk_id] = {
            "chunk_id": result[0],
            "content": result[1],
            "metadata": result[2]
        }

    for rank, result in enumerate(
        original_bm25,
        start=1
    ):
        chunk_id = result["chunk_id"]

        rrf_scores[chunk_id] += (
            ORIGINAL_WEIGHT
            * BM25_WEIGHT
            / (RRF_K + rank)
        )

    for rank, result in enumerate(
        rewrite_dense,
        start=1
    ):
        chunk_id = result[0]

        rrf_scores[chunk_id] += (
            REWRITE_WEIGHT
            * DENSE_WEIGHT
            / (RRF_K + rank)
        )

    for rank, result in enumerate(
        rewrite_bm25,
        start=1
    ):
        chunk_id = result["chunk_id"]

        rrf_scores[chunk_id] += (
            REWRITE_WEIGHT
            * BM25_WEIGHT
            / (RRF_K + rank)
        )

    # V2 完整 RRF 排名
    v2_ranked_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    v2_gold_rank = (
        v2_ranked_ids.index(gold_chunk_id)
        + 1
    )

    # 只把 V2 Top50 交给 Reranker
    candidate_ids = v2_ranked_ids[:RERANK_K]

    candidates = []

    for chunk_id in candidate_ids:
        candidate = document_map[chunk_id].copy()
        candidate["rrf_score"] = rrf_scores[chunk_id]

        candidates.append(candidate)

    reranked = reranker.rerank(
        query=query,
        documents=candidates,
        top_k=RERANK_K
    )

    reranked_ids = [
        result["chunk_id"]
        for result in reranked
    ]

    # Top50 被重新排序，50名之后继续保持原 V2 顺序
    v3_ranked_ids = (
        reranked_ids
        + v2_ranked_ids[RERANK_K:]
    )

    v3_gold_rank = (
        v3_ranked_ids.index(gold_chunk_id)
        + 1
    )

    if gold_chunk_id in candidate_ids:
        candidate_hit_50 += 1

    if v2_gold_rank <= 1:
        v2_recall_1 += 1

    if v2_gold_rank <= 3:
        v2_recall_3 += 1

    if v2_gold_rank <= 5:
        v2_recall_5 += 1

    v2_rr_sum += 1.0 / v2_gold_rank


    if v3_gold_rank <= 1:
        v3_recall_1 += 1

    if v3_gold_rank <= 3:
        v3_recall_3 += 1

    if v3_gold_rank <= 5:
        v3_recall_5 += 1

    v3_rr_sum += 1.0 / v3_gold_rank


    details.append(
        {
            "query_id": query_id,
            "query_type": item.get("query_type"),
            "domain": item.get("domain"),
            "query": query,
            "rewritten_query": rewritten_query,
            "gold_chunk_id": gold_chunk_id,
            "v2_gold_rank": v2_gold_rank,
            "v3_gold_rank": v3_gold_rank,
            "in_rerank_top50": gold_chunk_id in candidate_ids
        }
    )

    print(
        f"完成：{index}/{len(dev_queries)} | "
        f"V2 Rank={v2_gold_rank} | "
        f"V3 Rank={v3_gold_rank}"
    )


total = len(dev_queries)

summary = {
    "total_queries": total,
    "rerank_candidate_k": RERANK_K,

    "V2": {
        "Recall@1": v2_recall_1 / total,
        "Recall@3": v2_recall_3 / total,
        "Recall@5": v2_recall_5 / total,
        "MRR": v2_rr_sum / total
    },

    "V3": {
        "Recall@1": v3_recall_1 / total,
        "Recall@3": v3_recall_3 / total,
        "Recall@5": v3_recall_5 / total,
        "MRR": v3_rr_sum / total
    },

    "candidate_recall@50":
        candidate_hit_50 / total
}


with open(
    DETAIL_FILE,
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


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\n" + "=" * 100)
print("V2 vs V3 Dev Evaluation")
print("=" * 100)

print("\nV2：")
print(f"Recall@1：{summary['V2']['Recall@1']:.4f}")
print(f"Recall@3：{summary['V2']['Recall@3']:.4f}")
print(f"Recall@5：{summary['V2']['Recall@5']:.4f}")
print(f"MRR：     {summary['V2']['MRR']:.4f}")

print("\nV3：")
print(f"Recall@1：{summary['V3']['Recall@1']:.4f}")
print(f"Recall@3：{summary['V3']['Recall@3']:.4f}")
print(f"Recall@5：{summary['V3']['Recall@5']:.4f}")
print(f"MRR：     {summary['V3']['MRR']:.4f}")

print(
    f"\nCandidate Recall@50："
    f"{summary['candidate_recall@50']:.4f}"
)

print("\n结果已保存：")
print(DETAIL_FILE)
print(SUMMARY_FILE)