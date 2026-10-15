
import os
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from src.rag.v4_rerank import V4Rerank


# ==================== 1. 路径配置 ====================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

EVAL_PATH = (
    PROJECT_ROOT / "data/benchmark_v1/eval_queries.jsonl"
)

RESULT_DIR = (
    PROJECT_ROOT / "data/benchmark_v1/results/retrieval_eval"
)

RESULT_PATH = RESULT_DIR / "eval_v4_results.jsonl"
ERROR_PATH = RESULT_DIR / "eval_v4_errors.jsonl"


# ==================== 2. 读取JSONL ====================

def read_jsonl(path):
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


# ==================== 3. 计算单题检索指标 ====================

def calculate_metrics(gold_chunk_id, retrieved_ids):
    """
    每道题只有一个Gold Chunk。
    判断正确Chunk是否进入Top1、Top3、Top5。
    """

    gold_rank = None

    for index, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id == gold_chunk_id:
            gold_rank = index
            break

    return {
        "gold_rank": gold_rank,
        "hit_at_1": int(gold_rank == 1),
        "hit_at_3": int(
            gold_rank is not None and gold_rank <= 3
        ),
        "hit_at_5": int(
            gold_rank is not None and gold_rank <= 5
        ),
        "rr_at_5": (
            1.0 / gold_rank if gold_rank is not None else 0.0
        ),
    }


# ==================== 4. 汇总全部结果 ====================

def print_summary():
    results = read_jsonl(RESULT_PATH)

    if not results:
        print("暂无有效评测结果")
        return

    count = len(results)

    recall_1 = sum(x["hit_at_1"] for x in results) / count
    recall_3 = sum(x["hit_at_3"] for x in results) / count
    recall_5 = sum(x["hit_at_5"] for x in results) / count
    mrr_5 = sum(x["rr_at_5"] for x in results) / count

    print("\n========== V4 Eval 检索评测 ==========")
    print(f"成功评测数量：{count}")
    print(f"Recall@1：{recall_1:.4f}")
    print(f"Recall@3：{recall_3:.4f}")
    print(f"Recall@5：{recall_5:.4f}")
    print(f"MRR@5：{mrr_5:.4f}")


# ==================== 5. 执行940条Eval ====================

def evaluate_retrieval():

    if not EVAL_PATH.exists():
        raise FileNotFoundError(EVAL_PATH)

    # 读取全部Eval问题
    queries = read_jsonl(EVAL_PATH)

    # 创建评测结果目录
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取已完成ID，支持断点续跑
    completed_ids = {
        item["query_id"]
        for item in read_jsonl(RESULT_PATH)
    }

    pending = [
        item
        for item in queries
        if item["query_id"] not in completed_ids
    ]

    print(f"Eval问题总数：{len(queries)}")
    print(f"已完成：{len(queries) - len(pending)}")
    print(f"待处理：{len(pending)}")

    if not pending:
        print_summary()
        return

    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise ValueError("未找到DATABASE_URL")

    # 初始化V4检索器
    retriever = V4Rerank(db_url)

    success_count = 0
    failed_count = 0

    # 追加写入，避免覆盖已有数据
    with (
        open(RESULT_PATH, "a", encoding="utf-8") as result_file,
        open(ERROR_PATH, "a", encoding="utf-8") as error_file,
    ):

        for index, item in enumerate(pending, start=1):

            query_id = item["query_id"]
            query = item["query"]
            gold_chunk_id = item["gold_chunk_id"]

            print(f"\n[{index}/{len(pending)}] {query_id}")

            try:
                start_time = time.perf_counter()

                # V4检索及重排序
                retrieval = retriever.search(
                    query=query,
                    top_k=5
                )

                elapsed = time.perf_counter() - start_time

                documents = retrieval["top_documents"]

                retrieved_ids = [
                    doc["chunk_id"]
                    for doc in documents
                ]

                # 计算Recall与MRR
                metrics = calculate_metrics(
                    gold_chunk_id,
                    retrieved_ids
                )

                # 组织结果
                result = {
                    "query_id": query_id,
                    "query": query,
                    "domain": item.get("domain"),
                    "query_type": item.get("query_type"),
                    "difficulty": item.get("difficulty"),
                    "gold_chunk_id": gold_chunk_id,
                    "retrieved_chunk_ids": retrieved_ids,
                    "candidate_count": retrieval.get(
                        "candidate_count"
                    ),
                    "retrieval_seconds": round(elapsed, 3),
                    **metrics
                }

                # 保存成功结果
                result_file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                result_file.flush()
                os.fsync(result_file.fileno())

                success_count += 1

                print(
                    f"完成 | Gold排名：{metrics['gold_rank']} "
                    f"| Hit@5：{metrics['hit_at_5']} "
                    f"| 耗时：{elapsed:.2f}s"
                )

            except Exception as exc:
                failed_count += 1

                error = {
                    "query_id": query_id,
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    )
                }

                error_file.write(
                    json.dumps(error, ensure_ascii=False) + "\n"
                )
                error_file.flush()

                print(f"失败：{query_id} | {exc}")

    print("\n========== 执行完成 ==========")
    print("本次成功：", success_count)
    print("本次失败：", failed_count)

    print_summary()

    print("\n结果文件：", RESULT_PATH)


if __name__ == "__main__":
    evaluate_retrieval()
