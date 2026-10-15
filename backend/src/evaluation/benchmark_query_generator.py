import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


llm = ChatOpenAI(
    model="deepseek-v4-pro",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
)


QUERY_TYPES = [
    "direct",
    "colloquial",
    "scenario",
    "indirect",
    "reasoning",
]


def generate_queries(
        chunk: dict
) -> list[dict]:

    prompt = f"""
你正在构建半导体企业 RAG 检索评测集。

下面给出一个知识库 Chunk。

【业务域】
{chunk.get("domain")}

【正文】
{chunk.get("text")}

请基于这一个 Chunk 生成 5 个用户可能真实提出的问题。

必须分别属于以下五种类型：

1. direct
   直接的专业技术问题。

2. colloquial
   更像工程师日常交流的口语化问题。

3. scenario
   描述一个具体设备、工艺、质量或产品场景后提问。

4. indirect
   不直接复述章节标题或核心关键词，
   但答案仍然必须来自该 Chunk。

5. reasoning
   需要根据现象、原因、证据或判断边界进行分析的问题。


重要要求：

- 只能根据给出的 Chunk 出题。
- 不允许引入 Chunk 中不存在的知识。
- 不要直接复制原文句子。
- 不要出现文件名。
- 不要出现 section_title。
- 不要出现 chunk_id。
- 不要在问题中泄露答案。
- 每个问题必须能够由当前 Chunk 独立回答。
- 五个问题不能只是同一句话的简单改写。
- 保持半导体企业工程场景风格。
- 输出中文。
- 只输出 JSON，不要输出其他解释。
- evidence 必须直接复制自当前 Chunk 原文，不允许改写。
- evidence 必须确实能够支撑对应 query。
- 如果当前 Chunk 无法生成某一类型的问题，也不能引入外部知识硬凑。
- reasoning 只能基于当前 Chunk 中明确存在的现象、证据、判断边界或处置逻辑进行推理。
- 禁止引入当前 Chunk 中没有出现的试验、设备、参数、故障、样本或处置步骤。


输出格式必须严格为：

{{
    "direct": { {
        "query": "问题",
        "evidence": "能够支持这个问题的原文片段"
    } },
  "colloquial": { {
        "query": "问题",
        "evidence": "能够支持这个问题的原文片段"
    } },
  "scenario": { {
        "query": "问题",
        "evidence": "能够支持这个问题的原文片段"
    } },
  "indirect": { {
        "query": "问题",
        "evidence": "能够支持这个问题的原文片段"
    } },
  "reasoning": { {
        "query": "问题",
        "evidence": "能够支持这个问题的原文片段"
    } }
  }}
"""

    response = llm.invoke(
        prompt
    )

    content = response.content.strip()

    # 防止模型偶尔返回 ```json
    if content.startswith("```"):
        content = content.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()

    data = json.loads(
        content
    )

    results = []

    for query_type in QUERY_TYPES:
        results.append(
            {
                "query_type": query_type,
                "query": data[query_type]["query"],
                "evidence": data[query_type]["evidence"],
            }
        )

    return results
