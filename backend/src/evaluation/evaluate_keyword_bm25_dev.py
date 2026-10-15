import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.rag.bm25_retriever import BM25Retriever
from src.rag.keyword_bm25_retriever import KeywordBM25Retriever


load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

DEV_FILE = Path(
    "data/benchmark_v1/dev_queries.jsonl"
)

RESULT_FILE = Path(
    "data/benchmark_v1/results/"
    "v4_keyword_bm25_dev_details.jsonl"
)

TOP_K = 50


def load_queries():
    queries = []

    with open(
        DEV_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                queries.append(
                    json.loads(line)
                )

    return queries


def get_rank(results, gold_chunk_id):
    for rank, result in enumerate(
        results,
        start=1
    ):
        if result["chunk_id"] == gold_chunk_id:
            return rank

    return None


def calculate_recall(ranks, k):
    hit_count = sum(
        1
        for rank in ranks
        if rank is not None and rank <= k
    )

    return hit_count / len(ranks)


def main():
    queries = load_queries()

    old_bm25 = BM25Retriever(DB_URL)

    keyword_bm25 = KeywordBM25Retriever(
        DB_URL
    )

    old_ranks = []
    keyword_ranks = []

    details = []

    total = len(queries)

    for index, item in enumerate(
        queries,
        start=1
    ):
        query = item["query"]
        query_id = item["query_id"]
        gold_chunk_id = item["gold_chunk_id"]

        print(
            f"[{index}/{total}] "
            f"{query_id}"
        )

        old_results = old_bm25.search(
            query=query,
            top_k=TOP_K
        )

        keyword_results = keyword_bm25.search(
            query=query,
            top_k=TOP_K
        )

        old_rank = get_rank(
            old_results,
            gold_chunk_id
        )

        keyword_rank = get_rank(
            keyword_results,
            gold_chunk_id
        )

        old_ranks.append(old_rank)
        keyword_ranks.append(keyword_rank)

        details.append(
            {
                "query_id": query_id,
                "query": query,
                "gold_chunk_id": gold_chunk_id,
                "old_bm25_rank": old_rank,
                "keyword_bm25_rank": keyword_rank
            }
        )

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULT_FILE,
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

    print("\n" + "=" * 90)
    print("BM25 vs Keyword BM25 Dev Evaluation")
    print("=" * 90)

    for name, ranks in [
        ("旧 BM25", old_ranks),
        ("Keyword BM25", keyword_ranks)
    ]:
        print(f"\n{name}")

        print(
            f"Recall@1： "
            f"{calculate_recall(ranks, 1):.4f}"
        )

        print(
            f"Recall@3： "
            f"{calculate_recall(ranks, 3):.4f}"
        )

        print(
            f"Recall@5： "
            f"{calculate_recall(ranks, 5):.4f}"
        )

        print(
            f"Recall@50："
            f"{calculate_recall(ranks, 50):.4f}"
        )

    print("\n结果已保存：")
    print(RESULT_FILE)


if __name__ == "__main__":
    main()