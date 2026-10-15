"""
文件作用：
    FastAPI接口路由。

核心功能：
    1. 接收用户问题
    2. 调用LangGraph对话流程
    3. 返回Agent结果
"""

from fastapi import APIRouter

from src.api.schemas import (
    ChatRequest,
    ChatResponse
)

from src.router.conversation_graph import chat, stream_query
from fastapi.responses import StreamingResponse
import json
import logging
import os
import time
from queue import Queue, Empty, Full
from threading import Event, Thread
from fastapi import Depends, HTTPException
from src.api.auth import get_current_user, is_stream_session_active
from src.api.stream_control import StreamCancellation, StreamCancelled, cancellation_scope, register_stream, finish_stream, CancellableStreamingResponse
from uuid import UUID
from src.api.auth import get_connection
from src.api.run_control import ActiveRun
from src.api.backend_logging import request_log_context
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)


router = APIRouter()
SSE_HEARTBEAT_SECONDS = 10.0
from src.context.token_counter import ContextBudgetExceeded
from src.config.request_limits import seconds_timeout
SSE_TIMEOUT_SECONDS = seconds_timeout("SSE_TIMEOUT_SECONDS", "300")


def stream_with_heartbeat(cancellation=None, validate_session=None, **kwargs):
    """在同一工作线程中执行整个同步 graph.stream，保留 LangGraph 上下文。

    SSE 消费者独立等待队列，检索或模型等待期间仍能发心跳和超时。
    """
    queue = Queue(maxsize=128)
    cancellation = cancellation or StreamCancellation()
    stopped = cancellation.event
    finished = object()
    completed = False

    def enqueue(item):
        while not stopped.is_set():
            try:
                queue.put(item, timeout=0.1)
                return True
            except Full:
                pass
        return False

    def produce():
        iterator = None
        try:
            with request_log_context(cancellation.request_id), cancellation_scope(cancellation):
                iterator = stream_query(**kwargs)
                for event in iterator:
                    cancellation.check()
                    if not enqueue(event):
                        break
        except Exception as exc:
            enqueue(exc)
        finally:
            close = getattr(iterator, "close", None)
            try:
                if close:
                    close()
            finally:
                enqueue(finished)

    # 0配置对应None，关闭整轮业务截止时间，心跳和取消仍继续工作。
    deadline = time.monotonic() + SSE_TIMEOUT_SECONDS if SSE_TIMEOUT_SECONDS else None
    next_validation = time.monotonic()
    next_heartbeat = time.monotonic() + SSE_HEARTBEAT_SECONDS
    Thread(target=produce, name="chat-sse", daemon=True).start()
    try:
        while True:
            cancellation.check()
            now = time.monotonic()
            if validate_session and cancellation.run is None and now >= next_validation:
                if not validate_session():
                    cancellation.cancel("登录已失效，请重新登录")
                    cancellation.check()
                next_validation = now + 1.0
            remaining = deadline - now if deadline is not None else None
            if remaining is not None and remaining <= 0:
                raise TimeoutError("SSE对话处理超时")
            try:
                item = queue.get(timeout=min(0.25, remaining) if remaining is not None else 0.25)
            except Empty:
                if time.monotonic() >= next_heartbeat:
                    yield None
                    next_heartbeat = time.monotonic() + SSE_HEARTBEAT_SECONDS
                continue
            cancellation.check()
            if item is finished:
                completed = True
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        # 客户端断开或超时后不再积压事件，生产者在当前阻塞调用返回后退出。
        if not completed:
            cancellation.cancel("回答连接已关闭")

@router.post(
    "/chat",
    response_model=ChatResponse
)
def chat_api(request: ChatRequest, user: dict = Depends(get_current_user)):
    """
    对话接口。

    接收用户问题，调用 LangGraph 对话流程进行处理，
    并返回结构化的 Agent 结果。

    :param request: 对话请求体，包含 query、thread_id、user_id
    :return: ChatResponse，包含状态、意图、答案、来源、校验等字段
    """

    # 兼容接口也使用认证身份和账号租约，避免绕过SSE并发保护。
    session_id = validate_chat_session(request, user)
    cancellation = register_stream(user["_token_hash"])
    run = ActiveRun(user, session_id, cancellation)
    try:
        run.acquire()
        with cancellation_scope(cancellation):
            result = chat(query=request.query, thread_id=f"{user['id']}:{session_id}",
                          user_id=str(user["id"]))
        with get_connection() as conn:
            run.guard_write(conn)
    except ContextBudgetExceeded as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except StreamCancelled as exc:
        raise HTTPException(status_code=401 if "登录" in str(exc) else 409, detail=str(exc))
    finally:
        finish_stream(user["_token_hash"], cancellation)
        run.release()

    reply = result.get(
        "reply",
        {}
    )
    if not isinstance(reply, dict):
        reply = {"answer": str(reply)}

    return ChatResponse(
        status=result.get(
            "status",
            "unknown"
        ),

        intent=result.get(
            "intent"
        ),

        answer=reply.get(
            "answer"
        ),

        sources=reply.get(
            "sources",
            []
        ),

        verified=reply.get(
            "verified",
            False
        ),

        coverage_status=reply.get(
            "coverage_status"
        ),

        handoff=reply.get(
            "handoff"
        )
    )


def validate_chat_session(request, user):
    """同步和流式接口统一验证会话归属。"""
    try:
        session_id = UUID(request.thread_id.strip())
    except (ValueError, AttributeError):
        raise HTTPException(status_code=422, detail="thread_id必须为有效的会话UUID")
    with get_connection() as conn:
        session = conn.execute(
            "SELECT id FROM app_chat_sessions WHERE id=%s AND user_id=%s AND deleted_at IS NULL",
            (session_id, user["id"])
        ).fetchone()
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")
    return session_id


@router.get("/chat/run")
def get_active_run(user: dict = Depends(get_current_user)):
    """数据库是任务状态的唯一依据，标签页通知只用于加快页面同步。"""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT run_id,thread_id,lease_expires_at FROM app_active_runs WHERE user_id=%s AND lease_expires_at>clock_timestamp()",
            (user["id"],)
        ).fetchone()
    return {"running": bool(row), "run_id": str(row["run_id"]) if row else None}


@router.post("/chat/stream/sse")
def chat_stream_sse(request: ChatRequest, user: dict = Depends(get_current_user)):
    """通过JWT和CSRF验证用户身份，并按用户隔离LangGraph会话。"""
    # 用户身份必须来自数据库认证结果，禁止使用前端传入的user_id
    # 获取后端认证用户ID，禁止客户端伪造用户身份
    user_id = str(user["id"])
    # thread_id必须是数据库创建的会话UUID
    try:
        session_id = UUID(request.thread_id.strip())
    except (ValueError, AttributeError):
        raise HTTPException(status_code=422, detail="thread_id必须为有效的会话UUID")
    # 验证会话是否存在、是否属于当前用户以及是否已删除
    with get_connection() as conn:
        session = conn.execute(
            "SELECT id FROM app_chat_sessions WHERE id=%s AND user_id=%s AND deleted_at IS NULL",
            (session_id, user["id"])
        ).fetchone()
        if session:
            # 第一条提问即更新默认标题，生成失败时也能识别该会话；后续提问不覆盖。
            first_question_title = " ".join(request.query.split())[:200]
            if first_question_title:
                conn.execute(
                    """UPDATE app_chat_sessions SET title=%s
                    WHERE id=%s AND user_id=%s AND deleted_at IS NULL AND title='新会话'""",
                    (first_question_title, session_id, user["id"])
                )
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")
    # 使用经过验证的会话ID，隔离不同用户的LangGraph检查点
    client_thread_id = str(session_id)
    thread_id = f"{user_id}:{client_thread_id}"

    cancellation = register_stream(user["_token_hash"])
    run = ActiveRun(user, session_id, cancellation)
    try:
        run.acquire()
    except StreamCancelled as exc:
        finish_stream(user["_token_hash"], cancellation)
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception:
        finish_stream(user["_token_hash"], cancellation)
        raise

    def event_generator():
        """执行SSE流式问答，并在成功完成后持久化聊天记录。"""
        answer_parts = []
        sources = []
        stream_failed = False
        # 通知前端当前会话ID
        yield "event: start\n" + f"data: {json.dumps({'thread_id': client_thread_id}, ensure_ascii=False)}\n\n"
        try:
            # 接收LangGraph事件，保留原有心跳功能
            for event in stream_with_heartbeat(
                query=request.query, thread_id=thread_id, user_id=user_id,
                cancellation=cancellation, validate_session=lambda: is_stream_session_active(user)
            ):
                if event is None:
                    yield ": heartbeat\n\n"
                    continue
                # 兼容旧版LangGraph事件格式
                if "type" not in event:
                    if "sources" in event:
                        event = {"type": "source", "data": event["sources"]}
                    elif event.get("answer_chunk"):
                        event = {"type": "token", "data": {"content": event["answer_chunk"]}}
                    else:
                        continue
                event_type = event["type"]
                data = event.get("data")
                if event_type == "done":
                    continue
                # 收集AI生成的Token，避免逐Token写入数据库
                if event_type == "token" and isinstance(data, dict):
                    content = data.get("content", "")
                    if isinstance(content, str):
                        answer_parts.append(content)
                # 收集RAG引用来源
                elif event_type == "source":
                    if isinstance(data, list):
                        sources = data
                    elif isinstance(data, dict) and isinstance(data.get("sources"), list):
                        sources = data["sources"]
                elif event_type == "error":
                    stream_failed = True
                # 保持实时SSE输出
                yield "event: message\n" + f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if not is_stream_session_active(user):
                raise StreamCancelled("登录已失效，请重新登录")
            # 流式生成成功后，一次性保存员工提问和AI回答
            answer = "".join(answer_parts).strip()
            if not stream_failed and answer:
                with get_connection() as conn:
                    # 保存消息也必须与租约校验处于同一事务。
                    run.guard_write(conn)
                    # 再次校验会话归属并锁定，防止保存期间会话被删除
                    session = conn.execute(
                        "SELECT id FROM app_chat_sessions WHERE id=%s AND user_id=%s AND deleted_at IS NULL FOR UPDATE",
                        (session_id, user["id"])
                    ).fetchone()
                    if not session:
                        raise RuntimeError("会话不存在或已删除，无法保存聊天记录")
                    # 保存员工提问
                    conn.execute(
                        "INSERT INTO app_chat_messages(session_id,sender_role,sender_user_id,content) VALUES (%s,%s,%s,%s)",
                        (session_id, "user", user["id"], request.query)
                    )
                    # 保存完整AI回答及引用来源
                    conn.execute(
                        "INSERT INTO app_chat_messages(session_id,sender_role,content,sources) VALUES (%s,%s,%s,%s)",
                        (session_id, "assistant", answer, Jsonb(sources))
                    )
                    # 更新会话最后活跃时间
                    conn.execute(
                        "UPDATE app_chat_sessions SET updated_at=NOW() WHERE id=%s",
                        (session_id,)
                    )
        except StreamCancelled as exc:
            logger.info("回答任务已取消：%s", exc, extra={"request_id": cancellation.request_id})
            yield "event: message\n" + f"data: {json.dumps({'type': 'error', 'data': {'code': 'session_expired' if '登录' in str(exc) else 'run_cancelled', 'message': str(exc)}}, ensure_ascii=False)}\n\n"
        except ContextBudgetExceeded as exc:
            yield "event: message\n" + f"data: {json.dumps({'type': 'error', 'data': {'code': 'context_budget_exceeded', 'message': str(exc)}}, ensure_ascii=False)}\n\n"
        except TimeoutError:
            logger.exception("SSE对话处理超时", extra={"request_id": cancellation.request_id})
            yield "event: message\n" + f"data: {json.dumps({'type': 'error', 'data': {'message': '对话处理超时，请稍后重试。'}}, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("SSE对话执行或消息保存失败", extra={"request_id": cancellation.request_id})
            yield "event: message\n" + f"data: {json.dumps({'type': 'error', 'data': {'message': '对话执行或记录保存失败，请稍后重试。'}}, ensure_ascii=False)}\n\n"
        finally:
            finish_stream(user["_token_hash"], cancellation)
        # 通知前端本轮流式处理结束
        yield "event: message\n" + f"data: {json.dumps({'type': 'done', 'data': None}, ensure_ascii=False)}\n\n"

    # 必须返回响应对象，否则FastAPI会将函数的None返回值序列化为JSON null。
    return CancellableStreamingResponse(
        event_generator(),
        token_hash=user["_token_hash"], cancellation=cancellation,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
