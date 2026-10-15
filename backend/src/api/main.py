"""
文件作用：
    FastAPI服务入口。

核心功能：
    1. 创建FastAPI应用
    2. 注册接口路由
"""

from src.api.backend_logging import configure_logging
configure_logging()

from fastapi import FastAPI
from src.api.routes import router
from src.api.exception import global_exception_handler
from src.api.middleware import request_middleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from src.api.auth import router as auth_router
from src.api.sessions import router as sessions_router
from src.api.engineer import router as engineer_router

app = FastAPI(
    title="Semiconductor RAG Agent",
    description="半导体企业知识库智能问答系统",
    version="1.0.0"
)

# 注册JWT登录认证接口，统一使用/api前缀
app.include_router(auth_router, prefix="/api")

# 注册会话管理接口，统一使用/api前缀
app.include_router(sessions_router, prefix="/api")

# 注册员工转人工、工程师接单与回复接口
app.include_router(engineer_router, prefix="/api")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=[
        "*"
    ],
    allow_headers=[
        "*"
    ]
)

app.add_exception_handler(
    Exception,
    global_exception_handler
)

app.add_middleware(
    BaseHTTPMiddleware,
    dispatch=request_middleware
)

app.include_router(
    router,
    prefix="/api"
)

@app.get("/")
def root():
    """
    服务健康检查。
    """
    return {
        "status": "running",
        "service": "semiconductor-rag"
    }