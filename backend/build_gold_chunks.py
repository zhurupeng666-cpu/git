import json
import random
from collections import defaultdict
from pathlib import Path


# ============================================================
# 配置
# ============================================================

corpus_file = Path(
    "data/benchmark_v1/corpus.jsonl"
)

output_file = Path(
    "data/benchmark_v1/gold_chunks.jsonl"
)


# 固定随机种子
# 保证以后重新运行得到完全相同的200个Gold Chunk
random.seed(42)


sample_plan = {
    "equipment": 33,
    "general": 11,
    "process": 44,
    "product": 34,
    "quality": 45,
    "standard": 33,
}


# ============================================================
# 读取 corpus
# ============================================================

domain_chunks = defaultdict(list)


with open(
    corpus_file,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        item = json.loads(line)

        domain = item["domain"]

        domain_chunks[
            domain
        ].append(item)


# ============================================================
# 分层抽样
# ============================================================

gold_chunks = []


for domain, sample_size in sample_plan.items():

    candidates = domain_chunks[
        domain
    ]

    selected = random.sample(
        candidates,
        sample_size
    )

    gold_chunks.extend(
        selected
    )


# ============================================================
# 为避免业务域顺序影响后续处理，再打乱一次
# ============================================================

random.shuffle(
    gold_chunks
)


# ============================================================
# 保存
# ============================================================

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    for item in gold_chunks:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# 统计输出
# ============================================================

print("=" * 80)
print("Benchmark V1 Gold Chunk 分层抽样完成")
print("=" * 80)

print(
    "Gold Chunk总数：",
    len(gold_chunks)
)


for domain, sample_size in sample_plan.items():

    actual_count = sum(
        1
        for item in gold_chunks
        if item["domain"] == domain
    )

    print(
        f"{domain}: {actual_count}"
    )


print(
    "输出文件：",
    output_file
)