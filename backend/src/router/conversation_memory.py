
import os

from pathlib import Path
from contextlib import contextmanager

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from src.api.stream_control import get_current_run


class GuardedPostgresSaver(PostgresSaver):
    """在真实检查点写入事务中锁定租约，防止过期任务覆盖新任务。"""
    @contextmanager
    def _cursor(self, *, pipeline=False):
        if pipeline:
            # put和put_writes都经此入口；不用pipeline以保证校验先完成。
            with self.lock, self.conn.transaction():
                run = get_current_run()
                if run:
                    run.guard_write(self.conn)
                with self.conn.cursor(binary=True, row_factory=dict_row) as cursor:
                    yield cursor
        else:
            with super()._cursor(pipeline=False) as cursor:
                yield cursor


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 加载.env配置
load_dotenv(PROJECT_ROOT / ".env")


@contextmanager
def open_conversation_memory():
    """
    创建PostgreSQL多轮对话持久化连接。
    :return: PostgresSaver检查点管理器
    """

    # 1. 获取数据库连接地址
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise ValueError("未配置DATABASE_URL")

    # 2. 创建PostgreSQL Checkpointer
    with GuardedPostgresSaver.from_conn_string(db_url) as checkpointer:

        # 3. 初始化检查点数据库表
        # 已经初始化过时，会根据迁移记录跳过已完成的步骤
        checkpointer.setup()

        # 4. 返回检查点管理器
        yield checkpointer


def create_thread_config(thread_id: str) -> dict:
    """
    根据会话ID生成LangGraph运行配置。
    :param thread_id: 当前用户的会话ID
    :return: LangGraph运行配置
    """

    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("thread_id不能为空")

    return {
        "configurable": {
            "thread_id": f"semi:{thread_id.strip()}",
            "checkpoint_ns": "semiconductor_chat"
        }
    }
