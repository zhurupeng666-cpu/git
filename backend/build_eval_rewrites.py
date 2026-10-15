import json
import threading
import time

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed
)
from pathlib import Path

from src.rag.query_rewriter import QueryRewriter


# ============================================================
# 并发数
# ============================================================

MAX_WORKERS = 10


# ============================================================
# 文件路径
# ============================================================

input_file = Path(
    "data/benchmark_v1/eval_queries.jsonl"
)

output_file = Path(
    "data/benchmark_v1/eval_rewrites.jsonl"
)


# ============================================================
# 读取 940 条 Eval Query
# ============================================================

queries = []

with open(
    input_file,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if not line.strip():
            continue

        queries.append(
            json.loads(line)
        )


# ============================================================
# 断点续跑
# ============================================================

finished_ids = set()


if output_file.exists():

    with open(
        output_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            item = json.loads(line)

            finished_ids.add(
                item["query_id"]
            )


pending_queries = [

    item

    for item in queries

    if item["query_id"]
    not in finished_ids
]


print("=" * 100)
print("Eval Query Rewrite - 10并发")
print("=" * 100)

print(
    "Eval Query数量：",
    len(queries)
)

print(
    "已完成数量：",
    len(finished_ids)
)

print(
    "待处理数量：",
    len(pending_queries)
)

print(
    "并发数量：",
    MAX_WORKERS
)


# ============================================================
# 每个线程使用自己的 QueryRewriter
# ============================================================

thread_local = threading.local()


def get_rewriter():

    if not hasattr(
        thread_local,
        "rewriter"
    ):

        thread_local.rewriter = (
            QueryRewriter()
        )

    return thread_local.rewriter


# ============================================================
# 单条 Query Rewrite
# ============================================================

def process_query(
        item
):

    rewriter = get_rewriter()

    query_id = item[
        "query_id"
    ]

    original_query = item[
        "query"
    ]


    # ========================================================
    # 最多重试 3 次
    # ========================================================

    for attempt in range(
        1,
        4
    ):

        try:

            rewritten_query = (
                rewriter.rewrite(
                    original_query
                )
            )


            return {

                "query_id":
                    query_id,

                "original_query":
                    original_query,

                "rewritten_query":
                    rewritten_query
            }


        except Exception as e:

            print(
                f"\n{query_id} "
                f"第{attempt}次失败："
                f"{repr(e)}"
            )


            if attempt < 3:

                time.sleep(
                    attempt * 2
                )

            else:

                raise


# ============================================================
# 10 并发执行
# ============================================================

completed = len(
    finished_ids
)

failed = []


with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    future_map = {

        executor.submit(
            process_query,
            item
        ): item

        for item in pending_queries
    }


    for future in as_completed(
        future_map
    ):

        original_item = (
            future_map[
                future
            ]
        )


        try:

            record = future.result()


        except Exception as e:

            failed.append(
                original_item[
                    "query_id"
                ]
            )

            print(
                "\n最终失败：",
                original_item[
                    "query_id"
                ],
                repr(e)
            )

            continue


        # ====================================================
        # 只由主线程写文件
        # ====================================================

        with open(
            output_file,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


        finished_ids.add(
            record[
                "query_id"
            ]
        )

        completed += 1


        if completed % 20 == 0:

            print(
                f"已完成："
                f"{completed}/{len(queries)}"
            )


# ============================================================
# 最终结果
# ============================================================

print("\n" + "=" * 100)
print("Eval Query Rewrite完成")
print("=" * 100)

print(
    "总数量：",
    len(finished_ids)
)

print(
    "失败数量：",
    len(failed)
)

print(
    "输出文件：",
    output_file
)


if failed:

    print(
        "失败 Query ID：",
        failed
    )