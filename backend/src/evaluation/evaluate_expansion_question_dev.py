import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.rag.expansion_vector_retriever import ExpansionVectorRetriever


load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

DEV_FILE = Path(
    "data/benchmark_v1/dev_queries.jsonl"
)

DETAIL_FILE = Path(
    "data/benchmark_v1/results/"
    "v4_question_dense_dev_details.jsonl"
)

SUMMARY_FILE = Path(
    "data/benchmark_v1/results/"
    "v4_question_dense_dev_summary.json"
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


def calculate_recall(ranks, k):
    hits = sum(
        1
        for rank in ranks
        if rank is not None and rank <= k
    )

    return hits / len(ranks)


def main():
    queries = load_queries()

    retriever = ExpansionVectorRetriever(
        DB_URL
    )

    ranks = []
    details = []

    total = len(queries)

    for index, item in enumerate(
        queries,
        start=1
    ):
        query_id = item["query_id"]
        query = item["query"]
        gold_chunk_id = item["gold_chunk_id"]

        print(
            f"[{index}/{total}] "
            f"{query_id}"
        )

        results = retriever.search(
            query=query,
            top_k=TOP_K,
            view_type="question"
        )

        ranked_ids = [
            result["chunk_id"]
            for result in results
        ]

        if gold_chunk_id in ranked_ids:
            gold_rank = (
                ranked_ids.index(
                    gold_chunk_id
                )
                + 1
            )
        else:
            gold_rank = None

        ranks.append(gold_rank)

        details.append(
            {
                "query_id": query_id,
                "query_type": item["query_type"],
                "domain": item["domain"],
                "query": query,
                "gold_chunk_id": gold_chunk_id,
                "gold_rank": gold_rank,
                "top50_chunk_ids": ranked_ids
            }
        )

    recall_1 = calculate_recall(
        ranks,
        1
    )

    recall_3 = calculate_recall(
        ranks,
        3
    )

    recall_5 = calculate_recall(
        ranks,
        5
    )

    recall_50 = calculate_recall(
        ranks,
        50
    )

    summary = {
        "total_queries": total,
        "retriever": "synthetic_question_dense",
        "top_k": TOP_K,
        "Recall@1": recall_1,
        "Recall@3": recall_3,
        "Recall@5": recall_5,
        "Recall@50": recall_50
    }

    DETAIL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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

    print("\n" + "=" * 90)
    print("Synthetic Question Dense Dev Evaluation")
    print("=" * 90)

    print(f"Recall@1： {recall_1:.4f}")
    print(f"Recall@3： {recall_3:.4f}")
    print(f"Recall@5： {recall_5:.4f}")
    print(f"Recall@50：{recall_50:.4f}")

    print("\n结果已保存：")
    print(DETAIL_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()