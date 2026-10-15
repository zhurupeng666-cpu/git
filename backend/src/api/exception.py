"""
文件作用：
    FastAPI异常处理。

核心功能：
    1. 捕获接口异常
    2. 返回统一错误格式
"""


from fastapi import Request
from fastapi.responses import JSONResponse



async def global_exception_handler(
        request: Request,
        exc: Exception
):
    """
    全局异常处理。
    """

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "服务内部异常，请稍后重试。",
            "error_type": type(exc).__name__
        }
    )