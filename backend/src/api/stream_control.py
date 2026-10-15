"""按登录凭证管理活动回答，并把取消信号传递到模型HTTP连接。"""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event, Lock
import logging
from starlette.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from anyio import CancelScope
from src.api.backend_logging import current_request_id


class StreamCancelled(RuntimeError):
    """回答因退出登录、断开连接或超时而取消。"""


class StreamCancellation:
    def __init__(self):
        self.request_id = current_request_id.get()
        self.event = Event()
        self.reason = "回答已取消"
        self.lock = Lock()
        self.responses = []
        self.run = None

    def track(self, response):
        # HTTP响应到达后登记连接，取消时关闭正在消费的上游流
        with self.lock:
            cancelled = self.event.is_set()
            if not cancelled:
                self.responses.append(response)
        if cancelled:
            response.close()

    def cancel(self, reason="回答已取消"):
        with self.lock:
            if self.event.is_set():
                return
            self.reason = reason
            self.event.set()
            responses, self.responses = self.responses, []
        for response in responses:
            try:
                response.close()
            except Exception:
                # 取消时连接可能已经被模型生成器关闭，不影响其它连接清理
                pass

    def check(self):
        if self.event.is_set():
            raise StreamCancelled(self.reason)


current_cancellation = ContextVar("current_stream_cancellation", default=None)
registry_lock = Lock()
active_streams = {}


@contextmanager
def cancellation_scope(cancellation):
    token = current_cancellation.set(cancellation)
    try:
        cancellation.check()
        yield
    finally:
        current_cancellation.reset(token)


def get_current_run():
    """LangGraph线程继承取消上下文，检查点管理器据此取得任务所有权。"""
    cancellation = current_cancellation.get()
    return cancellation.run if cancellation else None


def check_cancelled():
    cancellation = current_cancellation.get()
    if cancellation:
        cancellation.check()


def track_model_response(response):
    cancellation = current_cancellation.get()
    if cancellation:
        cancellation.track(response)


def register_stream(token_hash):
    cancellation = StreamCancellation()
    with registry_lock:
        active_streams.setdefault(token_hash, set()).add(cancellation)
    return cancellation


def finish_stream(token_hash, cancellation):
    cancellation.cancel("回答连接已关闭")
    with registry_lock:
        streams = active_streams.get(token_hash)
        if streams:
            streams.discard(cancellation)
            if not streams:
                active_streams.pop(token_hash, None)


def cancel_token_streams(token_hash):
    # 仅取消同一登录凭证的回答，不影响其他用户或其他登录会话
    with registry_lock:
        streams = list(active_streams.get(token_hash, ()))
    for cancellation in streams:
        cancellation.cancel("登录已失效，请重新登录")


# 某些ASGI版本在客户端断开时抛出异常，普通BackgroundTask不会执行。


class CancellableStreamingResponse(StreamingResponse):
    def __init__(self, content, token_hash, cancellation, **kwargs):
        super().__init__(content, **kwargs)
        self.token_hash = token_hash
        self.cancellation = cancellation

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # 无论发送成功、客户端断开或任务被取消，都释放上游和注册记录
            # 数据库释放和HTTP close都是同步I/O，不能阻塞ASGI事件循环导致其他用户超时。
            # 客户端取消时仍执行清理，但只屏蔽清理阶段，不继续屏蔽回答任务的取消。
            with CancelScope(shield=True):
                await run_in_threadpool(finish_stream, self.token_hash, self.cancellation)
                if self.cancellation.run:
                    try:
                        await run_in_threadpool(self.cancellation.run.release)
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "释放回答任务租约失败", extra={"request_id": self.cancellation.request_id}
                        )
