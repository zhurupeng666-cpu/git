
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psycopg
import torch

from src.rag.vector_retriever import VectorRetriever
from src.rag.bm25_retriever import BM25Retriever
from src.rag.expansion_vector_retriever import ExpansionVectorRetriever
from src.rag.keyword_bm25_retriever import KeywordBM25Retriever
from src.rag.reranker import BGEReranker
from src.evaluation.evaluate_v4_consensus_dev import (
    DB_URL, DEV_FILE, REWRITE_FILE, load_jsonl,
    build_v2_top50, add_candidates, get_consensus_score
)

SCORE_FILE = Path("data/benchmark_v1/results/v4_multiview_scores_dev.jsonl")
RESULT_FILE = Path("data/benchmark_v1/results/v4_multiview_summary.json")

MAX_WORKERS = 3
ALPHA = 0.3
RRF_K = 10
INFER_SEMAPHORE = threading.Semaphore(1 if torch.cuda.is_available() else 3)

WEIGHTS = [(1.0, 0.0, 0.0)]
WEIGHTS += [
    (c / 10, s / 10, (10 - c - s) / 10)
    for c in range(3, 9)
    for s in range(1, 4)
    if 1 <= 10 - c - s <= 6
]


def evaluate_query(query_id, item, rewrites, dense, bm25, question, keyword, reranker):
    query = item["query"]
    rewritten = rewrites[query_id]["rewritten_query"]

    v2_ids, documents = build_v2_top50(query, rewritten, dense, bm25)
    candidate_map = {}

    for rank, chunk_id in enumerate(v2_ids, start=1):
        candidate_map[chunk_id] = {
            "chunk_id": chunk_id,
            "content": documents[chunk_id]["content"],
            "sources": [{"source": "v2", "rank": rank}]
        }

    add_candidates(candidate_map, question.search(
        query=query, top_k=25, view_type="question"
    ), "question_dense")

    add_candidates(candidate_map, keyword.search(
        query=query, top_k=25
    ), "keyword_bm25")

    candidates = list(candidate_map.values())
    chunk_ids = list(candidate_map)

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, summary, synthetic_questions
                FROM rag_chunk_expansions
                WHERE chunk_id = ANY(%s)
            """, (chunk_ids,))
            expansions = {
                chunk_id: (summary, questions)
                for chunk_id, summary, questions in cur.fetchall()
            }

    pairs = [[query, c["content"]] for c in candidates]

    for c in candidates:
        chunk_id = c["chunk_id"]
        if chunk_id not in expansions:
            raise ValueError(f"缺少扩展数据：{chunk_id}")
        pairs.append([query, expansions[chunk_id][0]])

    for c in candidates:
        questions = expansions[c["chunk_id"]][1]
        if isinstance(questions, str):
            questions = json.loads(questions)
        if len(questions) != 5:
            raise ValueError(f"合成问题数量异常：{c['chunk_id']}")
        pairs.extend([[query, q] for q in questions])

    with INFER_SEMAPHORE:
        raw_scores = reranker.model.compute_score(pairs, batch_size=16)

    scores = [float(x) for x in raw_scores]
    n = len(candidates)
    assert len(scores) == n * 7

    compact = []
    for i, c in enumerate(candidates):
        compact.append({
            "chunk_id": c["chunk_id"],
            "sources": c["sources"],
            "score_content": scores[i],
            "score_summary": scores[n + i],
            "score_question": max(scores[2*n + 5*i:2*n + 5*i + 5])
        })

    return {
        "query_id": query_id,
        "gold_chunk_id": item["gold_chunk_id"],
        "candidates": compact
    }


def rank_map(candidates, score_key):
    ordered = sorted(candidates, key=lambda c: c[score_key], reverse=True)
    return {c["chunk_id"]: rank for rank, c in enumerate(ordered, start=1)}


def get_final_rank(item, weights):
    candidates = item["candidates"]
    wc, ws, wq = weights

    content_ranks = rank_map(candidates, "score_content")
    summary_ranks = rank_map(candidates, "score_summary")
    question_ranks = rank_map(candidates, "score_question")

    final_scores = {}

    for c in candidates:
        chunk_id = c["chunk_id"]

        sc = 1 / (RRF_K + content_ranks[chunk_id])
        ss = 1 / (RRF_K + summary_ranks[chunk_id])
        sq = 1 / (RRF_K + question_ranks[chunk_id])

        multi_view = wc * sc + ws * ss + wq * sq
        consensus = get_consensus_score(c)
        final_scores[chunk_id] = (1 - ALPHA) * multi_view + ALPHA * consensus

    ordered_ids = sorted(final_scores, key=final_scores.get, reverse=True)

    try:
        return ordered_ids.index(item["gold_chunk_id"]) + 1
    except ValueError:
        return None


def calculate_metrics(ranks):
    total = len(ranks)
    return {
        "Recall@1": sum(r is not None and r <= 1 for r in ranks) / total,
        "Recall@3": sum(r is not None and r <= 3 for r in ranks) / total,
        "Recall@5": sum(r is not None and r <= 5 for r in ranks) / total,
        "MRR": sum(1 / r for r in ranks if r is not None) / total
    }


def objective(metrics):
    return (
        metrics["Recall@5"],
        metrics["Recall@1"],
        metrics["Recall@3"],
        metrics["MRR"]
    )


def main():
    dev = load_jsonl(DEV_FILE)
    rewrites = load_jsonl(REWRITE_FILE)
    completed = load_jsonl(SCORE_FILE)

    if len(dev) != 60:
        raise ValueError(f"Dev 应为60条，当前为{len(dev)}条")

    pending = {qid: item for qid, item in dev.items() if qid not in completed}

    print(f"总数：{len(dev)} | 已缓存：{len(completed)} | 待计算：{len(pending)}")
    SCORE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if pending:
        dense = VectorRetriever(DB_URL)
        bm25 = BM25Retriever(DB_URL)
        question = ExpansionVectorRetriever(DB_URL)
        keyword = KeywordBM25Retriever(DB_URL)
        reranker = BGEReranker()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    evaluate_query, qid, item, rewrites,
                    dense, bm25, question, keyword, reranker
                ): qid
                for qid, item in pending.items()
            }

            with open(SCORE_FILE, "a", encoding="utf-8") as f:
                for future in as_completed(futures):
                    result = future.result()
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    completed[result["query_id"]] = result
                    print(f"[{len(completed)}/60] {result['query_id']} 缓存完成")

    experiments = []

    for weights in WEIGHTS:
        ranks = [get_final_rank(completed[qid], weights) for qid in dev]
        metrics = calculate_metrics(ranks)
        experiments.append({"weights": weights, **metrics})

    baseline = experiments[0]
    full_views = experiments[1:]
    best = max(full_views, key=objective)
    selected = best if objective(best) > objective(baseline) else baseline

    experiments.sort(key=objective, reverse=True)

    print("\n" + "=" * 85)
    print("V4 Multi-view + Consensus (alpha=0.3)")
    print("=" * 85)

    print("\n现有 V4 基线：")
    print(baseline)

    print("\n完整 Multi-view 最优：")
    print(best)

    print("\n最终选用：")
    print(selected)

    print("\n表现最好的前10组：")
    for row in experiments[:10]:
        print(row)

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "alpha": ALPHA,
            "baseline": baseline,
            "best_multiview": best,
            "selected": selected,
            "experiments": experiments
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果保存：{RESULT_FILE}")


if __name__ == "__main__":
    main()
