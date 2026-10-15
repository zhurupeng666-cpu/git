import json
import re
from pathlib import Path

from src.evaluation.benchmark_query_generator import (
    generate_queries
)

# ============================================================
# 路径
# ============================================================

gold_file = Path(
    "data/benchmark_v1/gold_chunks.jsonl"
)

output_file = Path(
    "data/benchmark_v1/queries.jsonl"
)

failed_file = Path(
    "data/benchmark_v1/failed_queries.jsonl"
)


# ============================================================
# Query 类型
# ============================================================

QUERY_TYPES = [
    "direct",
    "colloquial",
    "scenario",
    "indirect",
    "reasoning",
]


# ============================================================
# 文本标准化
# 用于 Evidence 校验
# ============================================================

def normalize_text(text: str) -> str:

    return re.sub(
        r"\s+",
        "",
        text
    )


# ============================================================
# 读取 Gold Chunk
# ============================================================

gold_chunks = []

with open(
    gold_file,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        gold_chunks.append(
            json.loads(line)
        )


# ============================================================
# 读取已经生成过的 Query
# 支持中断后继续运行
# ============================================================

finished_keys = set()


if output_file.exists():

    with open(
        output_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            item = json.loads(line)

            finished_keys.add(
                (
                    item["gold_chunk_id"],
                    item["query_type"]
                )
            )


print("=" * 100)
print("Benchmark V1 Query 批量生成")
print("=" * 100)

print(
    "Gold Chunk数量：",
    len(gold_chunks)
)

print(
    "已完成Query数量：",
    len(finished_keys)
)


# ============================================================
# 批量生成
# ============================================================

success_count = len(finished_keys)

failed_count = 0


for gold_index, gold_chunk in enumerate(
    gold_chunks
):

    chunk_id = gold_chunk["chunk_id"]

    print(
        f"\n[{gold_index + 1}/{len(gold_chunks)}]"
        f" Gold Chunk：{chunk_id}"
    )


    # ========================================================
    # 如果5种问题全部生成过，则跳过
    # ========================================================

    all_finished = all(
        (
            chunk_id,
            query_type
        )
        in finished_keys

        for query_type
        in QUERY_TYPES
    )


    if all_finished:

        print(
            "5种 Query 已存在，跳过。"
        )

        continue


    # ========================================================
    # 最多尝试3次
    # ========================================================

    valid_queries = None


    for attempt in range(
        1,
        4
    ):

        print(
            f"生成尝试：{attempt}/3"
        )


        try:

            generated = generate_queries(
                gold_chunk
            )


            current_valid = []


            for item in generated:

                query_type = item[
                    "query_type"
                ]

                query = item[
                    "query"
                ]

                evidence = item[
                    "evidence"
                ]


                # ============================================
                # Evidence必须真实存在于Gold Chunk
                # ============================================

                evidence_valid = (
                    normalize_text(
                        evidence
                    )
                    in
                    normalize_text(
                        gold_chunk["text"]
                    )
                )


                if not evidence_valid:

                    print(
                        f"Evidence校验失败："
                        f"{query_type}"
                    )

                    break


                current_valid.append(
                    {
                        "query_type":
                            query_type,

                        "query":
                            query,

                        "evidence":
                            evidence,
                    }
                )


            # 必须5条全部通过
            if len(current_valid) == 5:

                valid_queries = (
                    current_valid
                )

                break


        except Exception as e:

            print(
                "生成异常：",
                repr(e)
            )


    # ========================================================
    # 连续3次失败
    # ========================================================

    if valid_queries is None:

        failed_count += 1

        failed_record = {

            "gold_chunk_id":
                chunk_id,

            "domain":
                gold_chunk["domain"],

            "file_name":
                gold_chunk["file_name"],
        }


        with open(
            failed_file,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(
                    failed_record,
                    ensure_ascii=False
                )
                + "\n"
            )


        print(
            "该 Gold Chunk 生成失败，已记录。"
        )

        continue


    # ========================================================
    # 写入5种Query
    # ========================================================

    with open(
        output_file,
        "a",
        encoding="utf-8"
    ) as f:


        for type_index, item in enumerate(
            valid_queries
        ):

            query_type = item[
                "query_type"
            ]


            # 如果已经存在，则不重复写
            key = (
                chunk_id,
                query_type
            )


            if key in finished_keys:

                continue


            # ================================================
            # 固定 Query ID
            #
            # 200 Chunk × 5种Query = 1000
            # ================================================

            query_number = (
                gold_index * 5
                +
                QUERY_TYPES.index(
                    query_type
                )
                +
                1
            )


            record = {

                "query_id":
                    f"Q-{query_number:05d}",

                "query":
                    item["query"],

                "query_type":
                    query_type,

                "domain":
                    gold_chunk["domain"],

                "gold_chunk_id":
                    chunk_id,

                "evidence":
                    item["evidence"],

                "file_name":
                    gold_chunk["file_name"],

                "difficulty":
                    (
                        "hard"
                        if query_type
                        in [
                            "indirect",
                            "reasoning"
                        ]
                        else
                        "medium"
                    ),
            }


            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


            finished_keys.add(
                key
            )

            success_count += 1


    print(
        "当前累计Query数量：",
        success_count
    )


# ============================================================
# 最终统计
# ============================================================

print("\n" + "=" * 100)
print("Benchmark Query 生成完成")
print("=" * 100)

print(
    "成功Query数量：",
    success_count
)

print(
    "失败Gold Chunk数量：",
    failed_count
)

print(
    "输出文件：",
    output_file
)