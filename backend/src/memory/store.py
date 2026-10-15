"""
文件作用：
    PostgreSQL记忆存储初始化。

核心功能：
    1. 创建PostgresStore
    2. 管理长期记忆数据库连接
"""

from langgraph.store.postgres import PostgresStore
from src.config.settings import settings


def create_memory_store():

    """
    创建长期记忆存储
    :return:PostgresStore上下文管理器
    """

    return PostgresStore.from_conn_string(
        settings.postgres_url
    )