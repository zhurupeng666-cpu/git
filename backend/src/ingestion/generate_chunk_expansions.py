import json
import os
import time

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("DEEPSEEK_API_KEY")


llm = ChatOpenAI(
    model="deepseek-v4-pro",
    base_url="https://api.deepseek.com/v1",
    api_key=API_KEY,
    temperature=0.2
)


SYSTEM_PROMPT = """
你是一名半导体企业知识库检索数据构建助手。

你的任务不是回答用户问题，而是根据给定的真实知识库 Chunk，
生成用于提升 RAG 检索召回率的辅助索引数据。

所有内容必须严格依据原始 Chunk。
不得补充原文没有提供的具体参数、阈值、设备型号、流程或结论。

请生成：

1. summary
用 2~3 句话概括 Chunk 中最重要的知识。
尽量覆盖多个主要知识点，而不是只总结标题主题。

2. synthetic_questions
生成 5 个用户可能提出的问题。

5 个问题必须尽量对应 Chunk 中不同的知识点，
禁止只是对同一个问题进行同义改写。

尽量分别覆盖：
- 核心流程或操作要求
- 原因或判断依据
- 证据、记录或数据要求
- 边界条件、异常情况或禁止事项
- 实际工作中的场景化或口语化问题

问题应像真实工程师会提出的问题，
可以使用专业表达，也可以使用自然口语表达。

3. keywords
提取 8~15 个真正有助于检索的关键词。

优先包含：
- 中文专业术语
- 英文术语
- 常用缩写
- 工艺或设备名称
- 参数名称
- 关键动作或判断概念
- 同义表达

不要把纯教学编号、页码、章节号、检索锚点作为关键词，
除非该编号本身具有真实业务检索价值。

只返回合法 JSON，不要输出解释。

格式：

{
  "summary": "...",
  "synthetic_questions": [
    "...",
    "...",
    "...",
    "...",
    "..."
  ],
  "keywords": [
    "...",
    "...",
    "..."
  ]
}
"""


def clean_json(text: str):
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def generate_expansion(content: str):
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"""
原始知识库 Chunk：

{content}
"""
        )
    ])

    result = json.loads(
        clean_json(response.content)
    )

    if not isinstance(result.get("summary"), str):
        raise ValueError("summary 格式错误")

    questions = result.get("synthetic_questions")
    keywords = result.get("keywords")

    if not isinstance(questions, list) or len(questions) != 5:
        raise ValueError("synthetic_questions 必须包含 5 个问题")

    if not isinstance(keywords, list):
        raise ValueError("keywords 格式错误")

    return result

def process_chunk(chunk_id: str, content: str):
    # 单条最多尝试 3 次
    for attempt in range(1, 4):
        try:
            result = generate_expansion(content)

            # 每个线程使用自己的数据库连接
            with psycopg.connect(DB_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO rag_chunk_expansions (
                            chunk_id,
                            summary,
                            synthetic_questions,
                            keywords
                        )
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (chunk_id)
                        DO UPDATE SET
                            summary = EXCLUDED.summary,
                            synthetic_questions =
                                EXCLUDED.synthetic_questions,
                            keywords = EXCLUDED.keywords
                    """, (
                        chunk_id,
                        result["summary"],
                        Jsonb(result["synthetic_questions"]),
                        Jsonb(result["keywords"])
                    ))

                conn.commit()

            return {
                "chunk_id": chunk_id,
                "success": True,
                "error": None
            }

        except Exception as e:
            if attempt < 3:
                time.sleep(2)

            else:
                return {
                    "chunk_id": chunk_id,
                    "success": False,
                    "error": str(e)
                }

def main():
    # 先找出还没有生成扩展数据的 Chunk
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    r.chunk_id,
                    r.content
                FROM rag_chunks r
                LEFT JOIN rag_chunk_expansions e
                    ON r.chunk_id = e.chunk_id
                WHERE e.chunk_id IS NULL
                ORDER BY r.chunk_id
            """)

            chunks = cur.fetchall()

    total = len(chunks)

    print("=" * 100)
    print(f"待生成 Chunk 数量：{total}")
    print("并发数：10")
    print("=" * 100)

    if total == 0:
        print("所有 Chunk 已经生成完成。")
        return

    completed = 0
    success_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = {
            executor.submit(
                process_chunk,
                chunk_id,
                content
            ): chunk_id
            for chunk_id, content in chunks
        }

        for future in as_completed(futures):

            completed += 1
            chunk_id = futures[future]

            try:
                result = future.result()

                if result["success"]:
                    success_count += 1

                    print(
                        f"[{completed}/{total}] "
                        f"生成成功：{chunk_id}"
                    )

                else:
                    failed_count += 1

                    print(
                        f"[{completed}/{total}] "
                        f"生成失败：{chunk_id}"
                    )

                    print(
                        f"错误：{result['error']}"
                    )

            except Exception as e:
                failed_count += 1

                print(
                    f"[{completed}/{total}] "
                    f"线程异常：{chunk_id}"
                )
                print(f"错误：{e}")

    print("\n" + "=" * 100)
    print("本轮处理结束")
    print(f"成功：{success_count}")
    print(f"失败：{failed_count}")
    print("=" * 100)


if __name__ == "__main__":
    main()