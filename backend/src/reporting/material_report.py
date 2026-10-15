
import json

from datetime import date

from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

from src.rag.answer_generator import AnswerGenerator
from src.tools.material_scope import MATERIAL_CATEGORY_LABELS, validate_report_scope

from src.tools.material_tools import (
    generate_material_statistics
)


def generate_material_report(
    start_date: str,
    end_date: str,
    report_type: str = "monthly",
    on_token=None,
    scope: str = "all",
    category: str | None = None,
    material_code: str | None = None
) -> str:
    """
    根据PostgreSQL物料统计数据自动生成材料周报或月报。
    :param start_date: 报告开始日期，格式YYYY-MM-DD
    :param end_date: 报告结束日期，格式YYYY-MM-DD
    :param report_type: 报告类型，weekly或monthly
    :return: DeepSeek生成的Markdown格式材料管理报告
    """

    # 1. 参数校验
    if report_type not in ("weekly", "monthly"):
        raise ValueError(
            "report_type只能是weekly或monthly"
        )

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if start > end:
        raise ValueError("开始日期不能晚于结束日期")

    category, material_code = validate_report_scope(scope, category, material_code)
    scope_name = MATERIAL_CATEGORY_LABELS[category] if category else (material_code or "全部半导体材料")

    # 2. 调用已经完成的库存统计Tool
    statistics = generate_material_statistics.invoke({
        "start_date": start_date,
        "end_date": end_date,
        "category": category,
        "material_code": material_code
    })
    if not statistics.get("materials"):
        message = f"{scope_name}在{start_date}至{end_date}期间没有可用的库存统计数据，请核实材料范围与日期。"
        if on_token is not None:
            on_token(message)
        return message

    # 3. 设置报告名称
    report_name = scope_name + ("管理周报" if report_type == "weekly" else "管理月报")

    # 4. 复用现有DeepSeek模型
    llm = AnswerGenerator().llm

    # 5. 设置系统提示词
    system_prompt = """
你是一名半导体制造企业的物料管理分析助手。

请根据提供的PostgreSQL统计数据，
撰写专业、准确、简洁的材料出入库管理报告。

必须遵循：

1. 只能使用输入数据，不得编造库存数量。
2. 不同材料的计量单位不能直接相加。
3. 期末库存 = 期初库存 + 入库 - 出库。
4. 库存增加或减少不一定代表异常。
5. 没有历史对比和预警阈值时，
   不得直接认定存在异常消耗。
6. 不得将账面库存等同于可用于生产的库存。
7. 所有结论必须能够追溯到提供的数据。
8. 不输出JSON，直接输出Markdown报告。

报告结构：

# 报告标题

## 一、报告概况
说明报告周期、统计物料数量。

## 二、材料出入库情况
按物料类别概括主要库存变化。
必要时列出重点物料的库存数据。

## 三、重点材料分析
仅分析本次材料范围内的物料。
不得增加输入数据之外的材料分类、物料或总库存。
未查到的材料不生成分析章节。

## 四、需要关注的问题
指出值得复核的库存变化。
如没有充分依据，应明确说明不能判定为异常。

## 五、管理建议
提出合理的库存管理和数据核实建议。

## 六、报告结论
简洁总结本周期材料流转情况。

要求：
语言专业、内容简洁。
避免无意义的重复描述。
"""

    # 6. 将真实统计数据发送给DeepSeek
    user_prompt = f"""
请生成以下材料管理报告。

报告名称：{report_name}
报告材料范围：{scope_name}
分类代码：{category or '未限定分类'}
物料编码：{material_code or '未限定单一物料'}
只分析上述范围，不扩展到全部材料。

统计周期：
{start_date} 至 {end_date}

数据库统计结果：

{json.dumps(
    statistics,
    ensure_ascii=False,
    indent=2
)}

请严格根据以上数据撰写报告。
"""

    # 7. 调用DeepSeek
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    if on_token is not None:
        parts = []
        for chunk in llm.stream(messages):
            if isinstance(chunk.content, str) and chunk.content:
                parts.append(chunk.content)
                on_token(chunk.content)
        return "".join(parts)

    response = llm.invoke(messages)

    # 8. 返回Markdown报告
    return response.content
