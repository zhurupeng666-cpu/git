"""
文件作用：
    FastAPI请求和响应数据模型。

核心功能：
    1. 定义聊天请求
    2. 定义聊天响应
"""


from pydantic import BaseModel


class ChatRequest(BaseModel):
    """
    用户聊天请求。
    """

    query: str
    thread_id: str = "default"
    user_id: str = "engineer_001"



class ChatResponse(BaseModel):
    """
    用户聊天响应。
    """

    status: str
    intent: str | None = None
    answer: str | None = None
    sources: list = []
    verified: bool = False
    coverage_status: str | None = None
    handoff: dict | None = None