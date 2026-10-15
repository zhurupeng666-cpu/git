import json
import threading
import time

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed
)
from pathlib import Path

from src.rag.domain_router import (
    DomainRouter
)


# ============================================================
# 并发配置
# ============================================================

MAX_WORKERS = 10


# ============================================================
# 文件路径
# ============================================================

query_file = Path(
    "data/benchmark_v1/eval_queries.jsonl"
)

output_file = Path(
    "data/benchmark_v1/results/"
    "v1_domain_routes.jsonl"
)

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 读取 Query
# ============================================================

queries = []

with open(
    query_file,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        queries.append(
            json.loads(line)
        )


# ============================================================
# 读取已经完成的 Query
# 支持断点续跑
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


# ============================================================
# 本次真正需要处理的数据
# ============================================================

pending_queries = [
    item
    for item in queries
    if item["query_id"]
    not in finished_ids
]


print("=" * 100)
print("V1 Top-2 Domain Routing - 10并发")
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
# 每个线程拥有自己的 Router
#
# 避免多个线程同时操作同一个模型客户端实例
# ============================================================

thread_local = threading.local()


def get_router():

    if not hasattr(
        thread_local,
        "router"
    ):

        thread_local.router = (
            DomainRouter()
        )

    return thread_local.router


# ============================================================
# 单条 Query 路由
# ============================================================

def process_query(
        item
):

    router = get_router()

    query_id = item[
        "query_id"
    ]

    query = item[
        "query"
    ]


    # ================================================
    # 最多重试3次
    # ================================================

    for attempt in range(
        1,
        4
    ):

        try:

            domains = router.route(
                query
            )

            return {

                "query_id":
                    query_id,

                "query":
                    query,

                # 只用于后续评测
                # 没有传给 Router
                "gold_domain":
                    item["domain"],

                "primary_domain":
                    domains[0],

                "secondary_domain":
                    domains[1],
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
# 10线程并发执行
# ============================================================

completed = len(
    finished_ids
)

failed = []


with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    # --------------------------------------------
    # 提交所有未完成任务
    # --------------------------------------------

    future_map = {

        executor.submit(
            process_query,
            item
        ): item

        for item in pending_queries
    }


    # --------------------------------------------
    # 哪条先完成就先处理哪条
    # --------------------------------------------

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


        # ========================================
        # 注意：
        # 文件写入放在主线程
        # 不让10个线程同时写JSONL
        # ========================================

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


        # ========================================
        # 进度
        # ========================================

        if completed % 20 == 0:

            print(
                f"已完成："
                f"{completed}/{len(queries)}"
            )


# ============================================================
# 最终结果
# ============================================================

print("\n" + "=" * 100)
print("V1 Domain Routing完成")
print("=" * 100)

print(
    "总路由数量：",
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