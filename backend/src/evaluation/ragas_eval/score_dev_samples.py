
import os
import json
import math
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
)

from src.embedding.bge_embedding import BGEEmbedding
from src.evaluation.ragas_eval.ragas_bge_adapter import RagasBGEAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data/benchmark_v1/results/ragas_results"

SAMPLE_PATH = DATA_DIR / "ragas_dev_samples.jsonl"
SCORE_PATH = DATA_DIR / "ragas_dev_scores.jsonl"
ERROR_PATH = DATA_DIR / "ragas_dev_errors.jsonl"


def read_jsonl(path):
    """读取JSONL文件。"""
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


async def score_one(sample, metrics, semaphore):
    """对单条问题计算三个RAGAS指标。"""

    async with semaphore:
        query_id = sample["query_id"]

        try:
            query = sample["query"]
            answer = sample["answer"]
            contexts = sample["retrieved_contexts"]

            if not answer or not contexts:
                raise ValueError("答案或检索证据为空")

            # 1. 忠实度
            faithfulness_result = await metrics["faithfulness"].ascore(
                user_input=query,
                response=answer,
                retrieved_contexts=contexts,
            )

            # 2. 答案相关性
            relevancy_result = await metrics["answer_relevancy"].ascore(
                user_input=query,
                response=answer,
            )

            # 3. 无参考答案的上下文精确度
            precision_result = await metrics["context_precision"].ascore(
                user_input=query,
                response=answer,
                retrieved_contexts=contexts,
            )

            scores = {
                "faithfulness": float(faithfulness_result.value),
                "answer_relevancy": float(relevancy_result.value),
                "context_precision": float(precision_result.value),
            }

            # 拒绝NaN或Infinity等无效分数
            if not all(math.isfinite(v) for v in scores.values()):
                raise ValueError(f"评分存在无效值：{scores}")

            return {
                "query_id": query_id,
                "query": query,
                "domain": sample["domain"],
                "query_type": sample["query_type"],
                "difficulty": sample.get("difficulty"),
                "gold_chunk_id": sample["gold_chunk_id"],
                "scores": scores,
            }

        except Exception as exc:
            return {
                "query_id": query_id,
                "error": f"{type(exc).__name__}: {exc}",
            }


async def evaluate_dev(max_concurrent=2):
    """批量评分，逐条保存，支持断点续跑。"""

    if max_concurrent < 1:
        raise ValueError("max_concurrent必须大于等于1")

    # 1. 读取此前生成的评测样本
    samples = read_jsonl(SAMPLE_PATH)

    if not samples:
        raise FileNotFoundError(
            f"没有找到有效的评测样本：{SAMPLE_PATH}"
        )

    # 2. 加载已经成功评分的题号
    completed = {
        item["query_id"]
        for item in read_jsonl(SCORE_PATH)
        if "scores" in item
    }

    pending = [
        item for item in samples
        if item["query_id"] not in completed
    ]

    print(f"样本总数：{len(samples)}")
    print(f"已经评分：{len(samples) - len(pending)}")
    print(f"待评分：{len(pending)}")

    if not pending:
        print("所有样本均已评分完成。")
        return

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未找到DEEPSEEK_API_KEY")

    # 3. 加载已有BGE-M3及RAGAS适配器
    bge = BGEEmbedding()
    adapter = RagasBGEAdapter(bge)

    # 4. 创建异步DeepSeek客户端
    async with AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    ) as client:

        ragas_llm = llm_factory(
            model="deepseek-v4-flash",
            provider="openai",
            client=client,
            max_tokens=4096,
            extra_body={
                "thinking": {"type": "disabled"}
            },
        )

        # 5. 创建三个评分器
        metrics = {
            "faithfulness": Faithfulness(llm=ragas_llm),
            "answer_relevancy": AnswerRelevancy(
                llm=ragas_llm,
                embeddings=adapter,
                strictness=3,
            ),
            "context_precision": ContextPrecisionWithoutReference(
                llm=ragas_llm
            ),
        }

        semaphore = asyncio.Semaphore(max_concurrent)

        # 6. 创建全部待执行任务
        tasks = [
            asyncio.create_task(
                score_one(sample, metrics, semaphore)
            )
            for sample in pending
        ]

        success_count = 0
        failed_count = 0

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        with (
            open(SCORE_PATH, "a", encoding="utf-8") as score_file,
            open(ERROR_PATH, "a", encoding="utf-8") as error_file,
        ):
            # 7. 哪条先评分完成，就先保存哪条
            for task in asyncio.as_completed(tasks):
                result = await task

                if "error" in result:
                    failed_count += 1
                    error_file.write(
                        json.dumps(result, ensure_ascii=False) + "\n"
                    )
                    error_file.flush()

                    print(
                        f"[失败] {result['query_id']} "
                        f"{result['error']}"
                    )
                    continue

                score_file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                score_file.flush()
                os.fsync(score_file.fileno())

                success_count += 1

                print(
                    f"[完成 {success_count}/{len(pending)}] "
                    f"{result['query_id']} "
                    f"{result['scores']}"
                )

    print("\n========== RAGAS评测结束 ==========")
    print("本次成功：", success_count)
    print("本次失败：", failed_count)
    print("评分文件：", SCORE_PATH)
    print("失败记录：", ERROR_PATH)


if __name__ == "__main__":
    asyncio.run(evaluate_dev(max_concurrent=2))
