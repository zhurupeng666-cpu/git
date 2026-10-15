"""
文件作用：
    FastAPI请求中间件。

核心功能：
    1. 生成请求ID
    2. 记录请求耗时
"""


import time
import uuid
from fastapi import Request
from src.api.backend_logging import current_request_id

async def request_middleware(request: Request, call_next):
    """
    请求日志中间件。
    1. 生成请求ID
    2. 记录请求耗时、
    3. 添加请求ID到响应头
    4. 打印请求日志
    """
    request_id = str(
        uuid.uuid4()
    )

    start_time = time.time()

    context_token = current_request_id.set(request_id)
    try:
        response = await call_next(request)
    finally:
        current_request_id.reset(context_token)

    cost_time = time.time() - start_time

    response.headers["X-Request-ID"] = request_id

    print(
        f"request_id={request_id}, "
        f"path={request.url.path}, "
        f"time={cost_time:.2f}s"
    )

    return response