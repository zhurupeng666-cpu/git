
import json
import logging
import time
import uuid

from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "rag_events.jsonl"


# 初始化日志记录器
logger = logging.getLogger("rag_monitor")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )

    handler.setFormatter(
        logging.Formatter("%(message)s")
    )

    logger.addHandler(handler)


def create_request_id() -> str:
    """为每次RAG问答生成唯一请求编号。"""
    return uuid.uuid4().hex


def log_event(
    request_id: str,
    stage: str,
    status: str = "success",
    **details
):
    """记录结构化监控事件。"""

    event = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "request_id": request_id,
        "stage": stage,
        "status": status,
        **details
    }

    logger.info(
        json.dumps(
            event,
            ensure_ascii=False,
            default=str
        )
    )


@contextmanager
def track_stage(
    request_id: str,
    stage: str
):
    """自动统计阶段耗时，并记录执行状态。"""

    start_time = time.perf_counter()

    try:
        yield

    except Exception as exc:
        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_event(
            request_id=request_id,
            stage=stage,
            status="error",
            elapsed_ms=round(elapsed_ms, 2),
            error_type=type(exc).__name__
        )

        raise

    else:
        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_event(
            request_id=request_id,
            stage=stage,
            status="success",
            elapsed_ms=round(elapsed_ms, 2)
        )


def record_request_completion(
    request_id: str,
    start_time: float,
    result: dict
) -> dict:
    """记录一次完整RAG问答的执行结果。"""

    # 计算完整问答耗时
    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    # 获取最终审核状态
    verified = result.get("verified", False)
    coverage_status = result.get("coverage_status")

    # 获取工程师工单状态
    handoff = result.get("handoff")

    if isinstance(handoff, dict):
        handoff_status = handoff.get("status", "unknown")
        handoff_id = handoff.get("question_id")
    else:
        handoff_status = "not_required"
        handoff_id = None

    # 写入请求汇总日志
    log_event(
        request_id=request_id,
        stage="rag_request",
        status="success",
        elapsed_ms=round(elapsed_ms, 2),
        verified=verified,
        coverage_status=coverage_status,
        handoff_status=handoff_status,
        handoff_id=handoff_id,
        candidate_count=result.get("candidate_count", 0)
    )

    # 保持RAGPipeline原有返回结构不变
    return result
