
import os
import json
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from src.rag.v4_rerank import V4Rerank
from src.rag.answer_generator import AnswerGenerator

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEV_PATH = PROJECT_ROOT / "data/benchmark_v1/dev_queries.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data/benchmark_v1/results/ragas_results/ragas_dev_samples.jsonl"


def load_completed_ids():
    """读取已经保存的样本，实现断点续跑。"""

    completed_ids = set()

    if not OUTPUT_PATH.exists():
        return completed_ids

    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                sample = json.loads(line)
                completed_ids.add(sample["query_id"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(
                    f"已有结果文件第{line_number}行无效，请先检查文件"
                ) from exc

    return completed_ids


def process_query(item, retriever, generator, retrieval_lock):
    """处理单个Dev问题：检索Top5并生成原始答案。"""

    query = item["query"]

    # V4检索器包含共享状态，先加锁避免线程安全问题
    with retrieval_lock:
        retrieval = retriever.search(query=query, top_k=5)

    documents = retrieval["top_documents"]

    # 不在检索锁内生成答案，允许多个LLM请求并发执行
    generation = generator.generate(
        query=query,
        documents=documents
    )

    return {
        "query_id": item["query_id"],
        "query": query,
        "query_type": item["query_type"],
        "domain": item["domain"],
        "difficulty": item.get("difficulty"),
        "gold_chunk_id": item["gold_chunk_id"],
        "answer": generation["answer"],
        "retrieved_contexts": [
            doc["content"] for doc in documents
        ],
        "retrieved_chunk_ids": [
            doc["chunk_id"] for doc in documents
        ],
        "retrieved_file_names": [
            (doc.get("metadata") or {}).get("file_name")
            for doc in documents
        ]
    }


def build_dev_samples(limit=None, max_workers=3):
    """并发生成Dev样本，支持断点续跑与失败重试。"""

    # 1. 读取Dev数据
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    if limit is not None and limit < 1:
        raise ValueError("limit必须大于等于1，或设置为None")

    if max_workers < 1:
        raise ValueError("max_workers必须大于等于1")

    selected = queries if limit is None else queries[:limit]

    # 2. 检查已完成任务
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = load_completed_ids()

    pending = [
        item for item in selected
        if item["query_id"] not in completed_ids
    ]

    print(f"Dev问题总数：{len(queries)}")
    print(f"本次选择：{len(selected)}")
    print(f"已完成：{len(selected) - len(pending)}")
    print(f"待处理：{len(pending)}")

    if not pending:
        print("所有选定问题均已完成，无须重复执行。")
        return

    # 3. 初始化模型
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise ValueError("未找到DATABASE_URL环境变量")

    retriever = V4Rerank(db_url)
    generator = AnswerGenerator()

    # 保护共享检索器
    retrieval_lock = threading.Lock()

    success_count = 0
    failed_count = 0

    # 4. 创建线程池
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {
            executor.submit(
                process_query,
                item,
                retriever,
                generator,
                retrieval_lock
            ): item
            for item in pending
        }

        # 5. 以追加模式保存，不覆盖已有结果
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:

            for future in as_completed(futures):
                item = futures[future]
                query_id = item["query_id"]

                try:
                    sample = future.result()

                    # 只由主线程写入，避免并发写文件冲突
                    f.write(
                        json.dumps(sample, ensure_ascii=False) + "\n"
                    )
                    f.flush()
                    os.fsync(f.fileno())

                    success_count += 1
                    print(
                        f"[完成 {success_count}/{len(pending)}] "
                        f"{query_id}"
                    )

                except Exception as exc:
                    failed_count += 1
                    print(
                        f"[失败] {query_id}："
                        f"{type(exc).__name__}: {exc}"
                    )

    print("\n========== 执行结果 ==========")
    print("本次成功：", success_count)
    print("本次失败：", failed_count)
    print("结果文件：", OUTPUT_PATH)

    if failed_count:
        print("失败任务未写入结果，下次运行时会自动重试。")


if __name__ == "__main__":
    build_dev_samples(limit=None, max_workers=3)
