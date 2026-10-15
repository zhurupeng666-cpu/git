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
DETAIL_FILE = Path("data/benchmark_v1/results/v4_consensus_dev_details.jsonl")
SUMMARY_FILE = Path("data/benchmark_v1/results/v4_consensus_dev_summary.json")

TOTAL_CHUNKS = 877
V2_K = 50
QUESTION_K = 25
KEYWORD_K = 25

RRF_K = 10
DENSE_WEIGHT = 0.4
BM25_WEIGHT = 0.6
ORIGINAL_WEIGHT = 0.5
REWRITE_WEIGHT = 0.5

CONSENSUS_K = 10
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

MAX_WORKERS = 3


def load_jsonl(path):
    data = {}
    if not path.exists():
        return data

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data[item["query_id"]] = item

    return data


def build_v2_top50(query, rewritten_query, dense_retriever, bm25_retriever):
    original_dense = dense_retriever.search(query=query, top_k=TOTAL_CHUNKS)
    original_bm25 = bm25_retriever.search(query=query, top_k=TOTAL_CHUNKS)
    rewrite_dense = dense_retriever.search(query=rewritten_query, top_k=TOTAL_CHUNKS)
    rewrite_bm25 = bm25_retriever.search(query=rewritten_query, top_k=TOTAL_CHUNKS)

    scores = defaultdict(float)
    document_map = {}

    for rank, result in enumerate(original_dense, start=1):
        chunk_id = result[0]
        scores[chunk_id] += ORIGINAL_WEIGHT * DENSE_WEIGHT / (RRF_K + rank)
        document_map[chunk_id] = {
            "chunk_id": result[0],
            "content": result[1],
            "metadata": result[2]
        }

    for rank, result in enumerate(original_bm25, start=1):
        scores[result["chunk_id"]] += ORIGINAL_WEIGHT * BM25_WEIGHT / (RRF_K + rank)

    for rank, result in enumerate(rewrite_dense, start=1):
        scores[result[0]] += REWRITE_WEIGHT * DENSE_WEIGHT / (RRF_K + rank)

    for rank, result in enumerate(rewrite_bm25, start=1):
        scores[result["chunk_id"]] += REWRITE_WEIGHT * BM25_WEIGHT / (RRF_K + rank)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
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


def get_consensus_score(candidate):
    rank_map = {item["source"]: item["rank"] for item in candidate["sources"]}

    score = 0.0

    for source in ["v2", "question_dense", "keyword_bm25"]:
        rank = rank_map.get(source)

        if rank is not None:
            score += 1 / (CONSENSUS_K + rank)

    return score / 3


def get_gold_rank(ranked_ids, gold_chunk_id):
    try:
        return ranked_ids.index(gold_chunk_id) + 1
    except ValueError:
        return None


def evaluate_one_query(query_id, item, rewrite_items,
                       dense_retriever, bm25_retriever,
                       question_retriever, keyword_retriever, reranker):

    query = item["query"]
    rewritten_query = rewrite_items[query_id]["rewritten_query"]
    gold_chunk_id = item["gold_chunk_id"]

    v2_ids, document_map = build_v2_top50(
        query, rewritten_query, dense_retriever, bm25_retriever
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

    question_results = question_retriever.search(
        query=query,
        top_k=QUESTION_K,
        view_type="question"
    )
    add_candidates(candidate_map, question_results, "question_dense")

    keyword_results = keyword_retriever.search(
        query=query,
        top_k=KEYWORD_K
    )
    add_candidates(candidate_map, keyword_results, "keyword_bm25")

    candidates = list(candidate_map.values())
    gold_in_candidates = gold_chunk_id in candidate_map

    reranked = reranker.rerank(
        query=query,
        documents=candidates,
        top_k=len(candidates)
    )

    rerank_rank = {
        result["chunk_id"]: rank
        for rank, result in enumerate(reranked, start=1)
    }

    gold_ranks = {}

    for alpha in ALPHAS:
        final_scores = {}

        for candidate in candidates:
            chunk_id = candidate["chunk_id"]

            rank = rerank_rank[chunk_id]
            reranker_score = 1 / (CONSENSUS_K + rank)
            consensus_score = get_consensus_score(candidate)

            final_score = (
                (1 - alpha) * reranker_score
                + alpha * consensus_score
            )

            final_scores[chunk_id] = final_score

        final_ids = sorted(
            final_scores,
            key=final_scores.get,
            reverse=True
        )

        gold_ranks[str(alpha)] = get_gold_rank(
            final_ids,
            gold_chunk_id
        )

    return {
        "query_id": query_id,
        "query_type": item["query_type"],
        "domain": item["domain"],
        "gold_chunk_id": gold_chunk_id,
        "candidate_count": len(candidates),
        "gold_in_candidates": gold_in_candidates,
        "gold_sources": candidate_map[gold_chunk_id]["sources"] if gold_in_candidates else [],
        "gold_rank_by_alpha": gold_ranks
    }


def calculate_metrics(ranks):
    total = len(ranks)

    return {
        "Recall@1": sum(1 for r in ranks if r is not None and r <= 1) / total,
        "Recall@3": sum(1 for r in ranks if r is not None and r <= 3) / total,
        "Recall@5": sum(1 for r in ranks if r is not None and r <= 5) / total,
        "MRR": sum(1 / r for r in ranks if r is not None) / total
    }


def main():
    dev_items = load_jsonl(DEV_FILE)
    rewrite_items = load_jsonl(REWRITE_FILE)
    completed = load_jsonl(DETAIL_FILE)

    pending = {
        query_id: item
        for query_id, item in dev_items.items()
        if query_id not in completed
    }

    print(f"总 Query：{len(dev_items)}")
    print(f"已完成：{len(completed)}")
    print(f"待处理：{len(pending)}")

    if pending:
        print("\n加载模型...")

        dense_retriever = VectorRetriever(DB_URL)
        bm25_retriever = BM25Retriever(DB_URL)
        question_retriever = ExpansionVectorRetriever(DB_URL)
        keyword_retriever = KeywordBM25Retriever(DB_URL)
        reranker = BGEReranker()

        futures = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for query_id, item in pending.items():
                future = executor.submit(
                    evaluate_one_query,
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

            finished = len(completed)

            for future in as_completed(futures):
                query_id = futures[future]

                try:
                    result = future.result()
                except Exception as e:
                    print(f"[ERROR] {query_id}: {e}")
                    raise

                with open(DETAIL_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                finished += 1

                print(
                    f"[完成 {finished}/{len(dev_items)}] {query_id} "
                    f"| Candidates={result['candidate_count']} "
                    f"| GoldInPool={result['gold_in_candidates']} "
                    f"| alpha0={result['gold_rank_by_alpha']['0.0']}"
                )

    results = load_jsonl(DETAIL_FILE)

    summary = {}

    print("\n" + "=" * 90)
    print("V4 Retrieval Consensus Dev Evaluation")
    print("=" * 90)

    for alpha in ALPHAS:
        key = str(alpha)

        ranks = [
            results[query_id]["gold_rank_by_alpha"][key]
            for query_id in dev_items
        ]

        metrics = calculate_metrics(ranks)
        summary[key] = metrics

        print(f"\nAlpha = {alpha}")
        print(f"Recall@1：{metrics['Recall@1']:.4f}")
        print(f"Recall@3：{metrics['Recall@3']:.4f}")
        print(f"Recall@5：{metrics['Recall@5']:.4f}")
        print(f"MRR：{metrics['MRR']:.4f}")

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存：{SUMMARY_FILE}")


if __name__ == "__main__":
    main()