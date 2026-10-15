import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from src.rag.vector_retriever import VectorRetriever
from src.rag.bm25_retriever import BM25Retriever
from src.rag.expansion_vector_retriever import ExpansionVectorRetriever
from src.rag.keyword_bm25_retriever import KeywordBM25Retriever
from src.rag.reranker import BGEReranker


load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

DEV_FILE = Path("data/benchmark_v1/dev_queries.jsonl")
REWRITE_FILE = Path("data/benchmark_v1/dev_rewrites.jsonl")
DETAIL_FILE = Path("data/benchmark_v1/results/v4_dev_details.jsonl")
SUMMARY_FILE = Path("data/benchmark_v1/results/v4_dev_summary.json")

TOTAL_CHUNKS = 877
V2_K = 50
QUESTION_K = 25
KEYWORD_K = 25

RRF_K = 10
DENSE_WEIGHT = 0.4
BM25_WEIGHT = 0.6
ORIGINAL_WEIGHT = 0.5
REWRITE_WEIGHT = 0.5

MAX_WORKERS = 3


def load_jsonl(path):
    items = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                items[item["query_id"]] = item
    return items


def get_rank(ranked_ids, gold_chunk_id):
    try:
        return ranked_ids.index(gold_chunk_id) + 1
    except ValueError:
        return None


def calculate_metrics(ranks):
    total = len(ranks)

    recall_1 = sum(1 for rank in ranks if rank is not None and rank <= 1) / total
    recall_3 = sum(1 for rank in ranks if rank is not None and rank <= 3) / total
    recall_5 = sum(1 for rank in ranks if rank is not None and rank <= 5) / total
    mrr = sum(1 / rank for rank in ranks if rank is not None) / total

    return {
        "Recall@1": recall_1,
        "Recall@3": recall_3,
        "Recall@5": recall_5,
        "MRR": mrr
    }


def build_v2_top50(query, rewritten_query, dense_retriever, bm25_retriever):
    original_dense = dense_retriever.search(query=query, top_k=TOTAL_CHUNKS)
    original_bm25 = bm25_retriever.search(query=query, top_k=TOTAL_CHUNKS)
    rewrite_dense = dense_retriever.search(query=rewritten_query, top_k=TOTAL_CHUNKS)
    rewrite_bm25 = bm25_retriever.search(query=rewritten_query, top_k=TOTAL_CHUNKS)

    rrf_scores = defaultdict(float)
    document_map = {}

    for rank, result in enumerate(original_dense, start=1):
        chunk_id = result[0]
        rrf_scores[chunk_id] += ORIGINAL_WEIGHT * DENSE_WEIGHT / (RRF_K + rank)
        document_map[chunk_id] = {
            "chunk_id": result[0],
            "content": result[1],
            "metadata": result[2]
        }

    for rank, result in enumerate(original_bm25, start=1):
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] += ORIGINAL_WEIGHT * BM25_WEIGHT / (RRF_K + rank)

    for rank, result in enumerate(rewrite_dense, start=1):
        chunk_id = result[0]
        rrf_scores[chunk_id] += REWRITE_WEIGHT * DENSE_WEIGHT / (RRF_K + rank)

    for rank, result in enumerate(rewrite_bm25, start=1):
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] += REWRITE_WEIGHT * BM25_WEIGHT / (RRF_K + rank)

    ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return ranked_ids[:V2_K], document_map


def add_candidates(candidate_map, results, source):
    for rank, result in enumerate(results, start=1):
        chunk_id = result["chunk_id"]

        if chunk_id not in candidate_map:
            candidate_map[chunk_id] = {
                "chunk_id": chunk_id,
                "content": result["content"],
                "metadata": result["metadata"],
                "sources": []
            }

        candidate_map[chunk_id]["sources"].append({
            "source": source,
            "rank": rank
        })


def evaluate_one_query(index, total, query_id, item, rewrite_items,
                       dense_retriever, bm25_retriever,
                       question_retriever, keyword_retriever, reranker):

    query = item["query"]
    rewritten_query = rewrite_items[query_id]["rewritten_query"]
    gold_chunk_id = item["gold_chunk_id"]

    # 1. V2 Hybrid Top50
    v2_ids, document_map = build_v2_top50(
        query,
        rewritten_query,
        dense_retriever,
        bm25_retriever
    )

    candidate_map = {}

    for rank, chunk_id in enumerate(v2_ids, start=1):
        document = document_map[chunk_id]

        candidate_map[chunk_id] = {
            "chunk_id": chunk_id,
            "content": document["content"],
            "metadata": document["metadata"],
            "sources": [{"source": "v2", "rank": rank}]
        }

    # 2. Synthetic Question Dense Top25
    question_results = question_retriever.search(
        query=query,
        top_k=QUESTION_K,
        view_type="question"
    )

    add_candidates(candidate_map, question_results, "question_dense")

    # 3. Keyword BM25 Top25
    keyword_results = keyword_retriever.search(
        query=query,
        top_k=KEYWORD_K
    )

    add_candidates(candidate_map, keyword_results, "keyword_bm25")

    # 4. 三路直接聚合去重
    candidates = list(candidate_map.values())
    candidate_count = len(candidates)
    gold_in_candidates = gold_chunk_id in candidate_map

    # 5. BGE Reranker
    reranked = reranker.rerank(
        query=query,
        documents=candidates,
        top_k=len(candidates)
    )

    reranked_ids = [result["chunk_id"] for result in reranked]
    gold_rank = get_rank(reranked_ids, gold_chunk_id)

    detail = {
        "query_id": query_id,
        "query_type": item["query_type"],
        "domain": item["domain"],
        "query": query,
        "rewritten_query": rewritten_query,
        "gold_chunk_id": gold_chunk_id,
        "candidate_count": candidate_count,
        "gold_in_candidates": gold_in_candidates,
        "v4_gold_rank": gold_rank,
        "candidate_sources": candidate_map[gold_chunk_id]["sources"] if gold_in_candidates else []
    }

    return {
        "index": index,
        "total": total,
        "query_id": query_id,
        "candidate_count": candidate_count,
        "gold_in_candidates": gold_in_candidates,
        "gold_rank": gold_rank,
        "detail": detail
    }


def main():
    dev_items = load_jsonl(DEV_FILE)
    rewrite_items = load_jsonl(REWRITE_FILE)

    print("正在加载检索模型...")
    dense_retriever = VectorRetriever(DB_URL)
    bm25_retriever = BM25Retriever(DB_URL)
    question_retriever = ExpansionVectorRetriever(DB_URL)
    keyword_retriever = KeywordBM25Retriever(DB_URL)
    reranker = BGEReranker()

    print(f"\n开始 V4 Dev 评测，共 {len(dev_items)} 条 Query，线程数：{MAX_WORKERS}\n")

    v4_ranks = []
    candidate_hits = 0
    candidate_counts = []
    details = []

    total = len(dev_items)
    futures = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for index, (query_id, item) in enumerate(dev_items.items(), start=1):
            future = executor.submit(
                evaluate_one_query,
                index,
                total,
                query_id,
                item,
                rewrite_items,
                dense_retriever,
                bm25_retriever,
                question_retriever,
                keyword_retriever,
                reranker
            )
            futures[future] = query_id

        finished = 0

        for future in as_completed(futures):
            query_id = futures[future]

            try:
                result = future.result()
            except Exception as e:
                print(f"[ERROR] {query_id}: {e}")
                raise

            finished += 1

            v4_ranks.append(result["gold_rank"])
            candidate_counts.append(result["candidate_count"])
            details.append(result["detail"])

            if result["gold_in_candidates"]:
                candidate_hits += 1

            print(
                f"[完成 {finished}/{total}] {result['query_id']} "
                f"| Candidates={result['candidate_count']} "
                f"| GoldInPool={result['gold_in_candidates']} "
                f"| Rank={result['gold_rank']}"
            )

    metrics = calculate_metrics(v4_ranks)
    candidate_recall = candidate_hits / total
    average_candidate_count = sum(candidate_counts) / len(candidate_counts)

    details.sort(key=lambda x: x["query_id"])

    summary = {
        "total_queries": total,
        "workers": MAX_WORKERS,
        "v2_k": V2_K,
        "question_k": QUESTION_K,
        "keyword_k": KEYWORD_K,
        "candidate_recall": candidate_recall,
        "average_candidate_count": average_candidate_count,
        **metrics
    }

    DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(DETAIL_FILE, "w", encoding="utf-8") as f:
        for item in details:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("V4 Aggregation + Reranker Dev Evaluation")
    print("=" * 90)
    print(f"Candidate Recall：{candidate_recall:.4f}")
    print(f"平均候选数量：{average_candidate_count:.2f}")
    print(f"Recall@1：{metrics['Recall@1']:.4f}")
    print(f"Recall@3：{metrics['Recall@3']:.4f}")
    print(f"Recall@5：{metrics['Recall@5']:.4f}")
    print(f"MRR：{metrics['MRR']:.4f}")
    print(f"\nDetails：{DETAIL_FILE}")
    print(f"Summary：{SUMMARY_FILE}")


if __name__ == "__main__":
    main()