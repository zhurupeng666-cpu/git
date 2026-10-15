"""
文件作用：
    统一管理大模型实例。

核心功能：
    1. 创建ChatOpenAI
    2. 提供全局LLM调用
"""

import os
import httpx
from src.api.stream_control import track_model_response

from src.context.request_budget import BudgetedChatOpenAI
from src.config.settings import settings
from src.config.request_limits import seconds_timeout

request_timeout = seconds_timeout("LLM_REQUEST_TIMEOUT_SECONDS", "60")


llm = BudgetedChatOpenAI(
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
    temperature=settings.temperature,
    timeout=request_timeout,
    max_retries=0,
    # 跟踪每次模型响应，退出登录时关闭正在生成答案的上游HTTP连接
    http_client=httpx.Client(timeout=request_timeout, event_hooks={"response": [track_model_response]})
)