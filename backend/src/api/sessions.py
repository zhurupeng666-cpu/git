from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from src.api.auth import get_connection, get_current_user

# 会话管理路由，所有接口均需要登录身份验证
router = APIRouter(prefix="/sessions", tags=["会话管理"])

# 新建会话请求模型
class SessionCreate(BaseModel):
    title: str = Field(default="新会话", min_length=1, max_length=200)

@router.post("", status_code=201)
def create_session(data: SessionCreate, user: dict = Depends(get_current_user)):
    """创建会话，用户ID由后端JWT认证结果确定。"""
    title = data.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="会话标题不能为空")
    # 只能为当前登录用户创建会话，禁止前端指定user_id
    with get_connection() as conn:
        session = conn.execute(
            "INSERT INTO app_chat_sessions(user_id,title) VALUES (%s,%s) RETURNING id,user_id,title,created_at,updated_at",
            (user["id"], title)
        ).fetchone()
    return session

@router.get("")
def list_sessions(user: dict = Depends(get_current_user), limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
    """分页查询当前用户的会话列表，不返回其他用户的数据。"""
    with get_connection() as conn:
        sessions = conn.execute(
            """SELECT s.id,
                COALESCE(NULLIF(LEFT(REGEXP_REPLACE(first_question.content, '[[:space:]]+', ' ', 'g'), 200), ''), s.title) AS title,
                s.created_at,s.updated_at
            FROM app_chat_sessions s
            LEFT JOIN LATERAL (
                SELECT content FROM app_chat_messages
                WHERE session_id=s.id AND sender_role='user'
                ORDER BY created_at ASC,id ASC LIMIT 1
            ) first_question ON TRUE
            WHERE s.user_id=%s AND s.deleted_at IS NULL
            ORDER BY s.updated_at DESC,s.id DESC LIMIT %s OFFSET %s""",
            (user["id"], limit, offset)
        ).fetchall()
    return {"sessions": sessions, "limit": limit, "offset": offset}

@router.get("/{session_id}/messages")
def list_messages(session_id: UUID, user: dict = Depends(get_current_user), limit: int = Query(100, ge=1, le=100), offset: int = Query(0, ge=0)):
    """校验会话归属后，分页读取历史聊天消息。"""
    with get_connection() as conn:
        # 未拥有该会话或会话已删除，统一返回404，避免泄露其他用户的会话信息
        session = conn.execute(
            "SELECT id FROM app_chat_sessions WHERE id=%s AND user_id=%s AND deleted_at IS NULL",
            (session_id, user["id"])
        ).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        messages = conn.execute(
            """SELECT id,sender_role,sender_user_id,content,sources,created_at
            FROM app_chat_messages WHERE session_id=%s
            ORDER BY created_at ASC,id ASC LIMIT %s OFFSET %s""",
            (session_id, limit, offset)
        ).fetchall()
    return {"session_id": str(session_id), "messages": messages, "limit": limit, "offset": offset}

@router.delete("/{session_id}")
def delete_session(session_id: UUID, user: dict = Depends(get_current_user)):
    """软删除当前用户的会话，不允许删除其他用户的会话。"""
    with get_connection() as conn:
        deleted = conn.execute(
            """UPDATE app_chat_sessions SET deleted_at=NOW(),updated_at=NOW()
            WHERE id=%s AND user_id=%s AND deleted_at IS NULL RETURNING id""",
            (session_id, user["id"])
        ).fetchone()
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"message": "会话删除成功", "session_id": str(session_id)}