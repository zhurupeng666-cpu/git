
import os
import re
import httpx2
from src.config.request_limits import seconds_timeout
from src.context.request_budget import validate_request_budget

from pathlib import Path
from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient, Choice, RetryPolicy
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage
)

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 加载环境变量
load_dotenv(PROJECT_ROOT / ".env")


# 支持的业务意图
VALID_INTENTS = {
    "knowledge_qa",
    "data_query",
    "report_generation",
    "conversation_memory",
    "unclear"
}


def classify_intent(
    query: str,
    context: list[BaseMessage] | None = None
) -> dict:
    """
    使用Jev结合历史对话识别用户业务意图。
    :param query: 当前用户输入
    :param context: 最近的历史对话消息
    :return: 意图、置信度、概率分布和澄清标记
    """

    # 1. 检查问题
    if not isinstance(query, str) or not query.strip():
        raise ValueError("用户问题不能为空")

    # 2. 读取OpenRouter API Key
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("未配置OPENROUTER_API_KEY")


    # 整理最近历史对话
    history_lines = []

    for message in (context or [])[-8:]:

        if isinstance(message, HumanMessage):
            role = "用户"

        elif isinstance(message, AIMessage):
            role = "助手"

        else:
            continue

        if not isinstance(message.content, str):
            continue

        # 长回答只传编号小标题，保留“第二个部件”等指代，不发送完整数据库结果。
        content = message.content[:200]
        if role == "助手" and len(message.content) > 300:
            headings = re.findall(r"(?m)^\s*(?:#{1,6}\s*)?\d+[.、．)）]\s*[^\n]+", message.content)
            if not headings:
                continue
            content = "\n".join(headings[:12])[:600]
        history_lines.append(f"{role}：{content}")

    # 限制发送给Jev的上下文长度
    history_text = "\n".join(history_lines)[-1500:]


    # 3. 初始化Jev客户端；SDK的None表示默认值，显式Timeout(None)才是关闭超时。
    request_timeout = seconds_timeout("INTENT_REQUEST_TIMEOUT_SECONDS", "15")
    with TypeSafeClient(
        api_key=api_key,
        base_url="https://openrouter.ai/api",
        timeout=request_timeout if request_timeout is not None else httpx2.Timeout(None),
        retry=RetryPolicy(max_retries=1, timeout=20.0 if request_timeout else None,
                          http_statuses=set(), api_timeout_error=False)
    ) as client:

        # 4. 调用Jev进行意图分类
        payload = dict(
            model="jev-1.13",

            state={
                "user_query": query.strip(),
                "recent_dialogue": history_text
            },

            questions={
                "intent": Choice(
                    instructions=(
                        "你需要结合user_query和recent_dialogue判断业务意图。"
                        "user_query是当前用户输入，recent_dialogue是历史对话。"
                        "优先根据当前问题判断。"
                        "只有当前问题存在省略或上下文指代时，才参考历史对话。"
                        "例如用户说'它的预防措施呢'，需要结合历史判断并继续knowledge_qa。"
                        "记录或回忆本会话中的用户信息、编号、型号、先前说过的内容选conversation_memory。"
                        "例如'我之前告诉你的内部测试编号是什么'。"
                        "询问前面提到的部件如何排查属于knowledge_qa，不属于conversation_memory。"
                        "用户明确提出新问题时，不得让历史话题覆盖新意图。"
                        "生成报告优先选report_generation；"
                        "查询库存或出入库数据选data_query；"
                        "询问半导体专业知识选knowledge_qa；"
                        "无法明确判断时选unclear。"
                        "判断用户问题的主要业务意图。"
                        "生成报告优先选report_generation；"
                        "查询库存或出入库数据选data_query；"
                        "询问半导体专业知识选knowledge_qa；"
                        "无法明确判断时选unclear。"
                    ),

                    criteria={
                        "knowledge_qa": (
                            "半导体专业知识问答，"
                            "包括CVD、CMP、工艺规范、"
                            "设备维护和故障排查"
                        ),

                        "data_query": (
                            "查询企业数据库中的材料库存、"
                            "出入库记录、材料消耗量或"
                            "指定周期的库存统计数据"
                        ),

                        "report_generation": (
                            "要求撰写或生成材料管理周报、"
                            "月报、出入库分析报告"
                        ),

                        "conversation_memory": (
                            "要求记住用户提供的信息，或仅依据当前对话回忆用户或助手先前的内容。"
                            "回忆编号、型号、个人说明不需要检索知识库；"
                            "利用历史指代继续提出技术排查或数据库查询问题仍属于对应业务意图。"
                        ),

                        "unclear": (
                            "无法判断业务意图，"
                            "或不属于以上功能"
                        )
                    }
                )
            }
        )

        # 完整分类请求（含问题说明）也进入应用预算校验。
        validate_request_budget(payload, output_tokens=512)
        response = client.system_one(**payload)

    # 5. 解析Jev结构化结果
    decision = response.answers["intent"]

    intent = decision.choice
    confidence = float(decision.confidence)

    # 6. 检查意图是否合法
    if intent not in VALID_INTENTS:
        raise ValueError(f"未知意图类型：{intent}")

    # 7. 低置信度时不自动路由
    # 阈值0.70为初始设置，后续根据测试结果调整
    needs_clarification = (
        intent == "unclear"
        or confidence < 0.70
    )

    # 8. 返回识别结果
    return {
        "intent": intent,
        "confidence": round(confidence, 4),
        "probabilities": dict(decision.probabilities),
        "needs_clarification": needs_clarification
    }
