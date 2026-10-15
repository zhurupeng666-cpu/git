"""限制进程内本地模型资源占用，并在等待或批次之间检查取消信号。"""
import os
from contextlib import contextmanager
from threading import BoundedSemaphore, Lock

from src.api.stream_control import check_cancelled


INFERENCE_THREADS = int(os.getenv("LOCAL_INFERENCE_THREADS", "2"))
INFERENCE_CONCURRENCY = int(os.getenv("LOCAL_INFERENCE_CONCURRENCY", "1"))
if INFERENCE_THREADS < 1 or INFERENCE_CONCURRENCY < 1:
    raise ValueError("本地推理线程数和并发数必须大于0")
inference_gate = BoundedSemaphore(INFERENCE_CONCURRENCY)
runtime_lock = Lock()
runtime_ready = False


def configure_runtime():
    """每个Worker只设置一次PyTorch线程预算，为认证和任务监控保留CPU。"""
    global runtime_ready
    with runtime_lock:
        if not runtime_ready:
            import torch
            torch.set_num_threads(INFERENCE_THREADS)
            runtime_ready = True


@contextmanager
def inference_slot():
    # 不用无法取消的长时间acquire，退出或断连后排队任务及时返回。
    while True:
        check_cancelled()
        if inference_gate.acquire(timeout=0.1):
            break
    try:
        check_cancelled()
        yield
        check_cancelled()
    finally:
        inference_gate.release()
