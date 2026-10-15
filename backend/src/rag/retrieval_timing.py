"""记录检索子阶段耗时；不记录用户问题、文档内容或凭证。"""
import logging
from contextlib import contextmanager
from time import monotonic
from langgraph.config import get_stream_writer
from src.api.stream_control import check_cancelled

logger = logging.getLogger(__name__)


@contextmanager
def retrieval_stage(stage: str):
    check_cancelled()
    # 在LangGraph内直接发送子阶段；命令行检索没有运行上下文时只记录日志。
    try:
        writer = get_stream_writer()
    except RuntimeError:
        writer = None
    if writer:
        writer({"type": "status", "data": {
            "stage": stage, "message": "正在处理知识检索..."
        }})
    started = monotonic()
    logger.info("Retrieval stage started: %s", stage)
    try:
        yield
        check_cancelled()
    finally:
        logger.info("Retrieval stage finished: %s elapsed=%.3fs",
                    stage, monotonic() - started)
