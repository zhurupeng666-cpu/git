
"""
文件作用：
    管理半导体Agent的短期对话上下文。

主要功能：
    1. 读取LangGraph保存的历史消息。
    2. 选择最近若干轮有效对话。
    3. 限制历史消息长度，避免直接传入全部历史。
    4. 为后续Token预算和摘要压缩提供基础。
"""

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage
)


def get_recent_context(
    messages: list[BaseMessage],
    max_turns: int = 6,
    max_chars: int = 6000
) -> list[BaseMessage]:
    """
    获取最近几轮历史对话，供大模型理解上下文。
    :param messages: LangGraph保存的历史消息
    :param max_turns: 最多保留的用户对话轮数
    :param max_chars: 历史消息的字符预算
    :return: 筛选后的历史消息列表
    """

    if max_turns <= 0 or max_chars <= 0:
        raise ValueError("上下文限制必须大于0")

    history = list(messages)

    # 当前最新用户消息由调用方单独传入
    # 避免重复加入本轮问题
    if history and isinstance(history[-1], HumanMessage):
        history = history[:-1]

    selected = []
    used_chars = 0
    user_turns = 0

    # 从最近的历史消息开始筛选
    for message in reversed(history):

        if not isinstance(message, (HumanMessage, AIMessage)):
            continue

        if not isinstance(message.content, str):
            continue

        if isinstance(message, HumanMessage):
            if user_turns >= max_turns:
                break
            user_turns += 1

        remaining = max_chars - used_chars

        if remaining <= 0:
            break

        # 避免把很长的数据库JSON结果全部送入模型
        content = message.content[:min(1000, remaining)]

        if not content.strip():
            continue

        selected.append(
            type(message)(content=content)
        )

        used_chars += len(content)

    # 恢复为正常的时间顺序
    return list(reversed(selected))
