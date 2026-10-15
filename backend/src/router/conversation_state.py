
from typing import TypedDict, Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ConversationState(TypedDict, total=False):
    """
    LangGraph多轮对话状态。
    """

    # 1. 历史对话消息
    messages: Annotated[list[BaseMessage], add_messages]

    # 2. Jev识别的当前业务意图
    intent: str | None

    # 3. 尚未完成的业务意图
    pending_intent: str | None

    # 4. 等待执行的工具名称
    pending_tool: str | None

    # 5. 已经收集到的工具参数
    collected_args: dict[str, Any]

    # 6. 仍然缺少的参数
    missing_args: list[str]

    # 7. 当前对话执行状态
    # running / needs_clarification / success / error
    status: str

    # 8. 当前需要回复用户的内容
    reply: str | dict[str, Any]
    user_id: str

    # 9. 当前需要回复用户的内容
    answer_chunk: str

    # 增量摘要与原始消息一起由PG检查点持久化；不删除原始历史。
    context_summary: str
    summary_message_count: int
    summary_prefix_hash: str
