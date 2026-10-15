"""按Worker保存异常日志，并把请求ID传递到后台问答线程。"""
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

current_request_id = ContextVar("backend_request_id", default="-")


class RequestContextFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = current_request_id.get()
        return True


def configure_logging():
    logger = logging.getLogger()
    if any(getattr(handler, "semiconductor_backend", False) for handler in logger.handlers):
        return
    folder = Path(__file__).resolve().parents[2] / "logs"
    folder.mkdir(exist_ok=True)
    handler = RotatingFileHandler(folder / f"backend_{os.getpid()}.log",
                                  maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
    handler.semiconductor_backend = True
    handler.addFilter(RequestContextFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s pid=%(process)d request_id=%(request_id)s %(name)s %(message)s"))
    logger.addHandler(handler)
    if logger.level > logging.INFO:
        logger.setLevel(logging.INFO)


@contextmanager
def request_log_context(request_id):
    token = current_request_id.set(request_id)
    try:
        yield
    finally:
        current_request_id.reset(token)
