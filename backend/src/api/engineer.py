from typing import Literal
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from src.api.auth import get_connection, get_current_user, require_engineer

# 员工提交入口和工程师处理入口，沿用Cookie、角色及CSRF认证
router = APIRouter(tags=["工程师人工处理"])


class QuestionCreate(BaseModel):
    question: str = Field(min_length=1, max_length=8000)


class EngineerReply(BaseModel):
    answer: str = Field(min_length=1, max_length=20000)


# 统一返回问题详情，兼容已有未关联用户的RAG待答问题
QUESTION_SELECT = """SELECT q.*,u.display_name AS requester_name,
    engineer.display_name AS assigned_name,s.title AS session_title
    FROM engineer_questions q
    LEFT JOIN app_users u ON u.id=q.requester_id
    LEFT JOIN app_users engineer ON engineer.id=q.assigned_engineer_id
    LEFT JOIN app_chat_sessions s ON s.id=q.session_id"""


@router.post("/sessions/{session_id}/engineer-questions", status_code=201)
def submit_question(session_id: UUID, data: QuestionCreate, user: dict = Depends(get_current_user)):
    """员工将当前会话问题转人工，重复提交未完成的问题返回原工单。"""
    question = data.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空")
    with get_connection() as conn:
        # 锁定当前用户自己的会话，防止并发重复建单和提交到已删除会话
        session = conn.execute(
            """SELECT id FROM app_chat_sessions
            WHERE id=%s AND user_id=%s AND deleted_at IS NULL FOR UPDATE""",
            (session_id, user["id"])
        ).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在或无权访问")
        existing = conn.execute(
            """SELECT * FROM engineer_questions
            WHERE session_id=%s AND requester_id=%s AND question=%s
            AND status IN ('pending','processing') ORDER BY id DESC LIMIT 1""",
            (session_id, user["id"], question)
        ).fetchone()
        if existing:
            return existing
        # 手动转人工只保存员工提交的问题，不调用外部模型
        ticket = conn.execute(
            """INSERT INTO engineer_questions(question,session_id,requester_id)
            VALUES (%s,%s,%s) RETURNING *""",
            (question, session_id, user["id"])
        ).fetchone()
    return ticket


@router.get("/sessions/{session_id}/engineer-questions")
def session_questions(session_id: UUID, user: dict = Depends(get_current_user)):
    """员工查看自己当前会话的工单状态和人工回复。"""
    with get_connection() as conn:
        session = conn.execute(
            "SELECT id FROM app_chat_sessions WHERE id=%s AND user_id=%s AND deleted_at IS NULL",
            (session_id, user["id"])
        ).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在或无权访问")
        questions = conn.execute(
            QUESTION_SELECT + " WHERE q.session_id=%s AND q.requester_id=%s ORDER BY q.created_at DESC,q.id DESC",
            (session_id, user["id"])
        ).fetchall()
    return {"questions": questions}


@router.get("/engineer/questions")
def list_questions(
    status: Literal["pending", "processing", "answered", "closed"] = "pending",
    mine: bool = False,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: dict = Depends(require_engineer)
):
    """工程师分页查看待接单、处理中和已回复问题，可仅查看本人接单。"""
    with get_connection() as conn:
        questions = conn.execute(
            QUESTION_SELECT + """ WHERE q.status=%s
            AND (%s=FALSE OR q.assigned_engineer_id=%s)
            ORDER BY q.created_at ASC,q.id ASC LIMIT %s OFFSET %s""",
            (status, mine, user["id"], limit, offset)
        ).fetchall()
    return {"questions": questions, "limit": limit, "offset": offset}


@router.post("/engineer/questions/{question_id}/claim")
def claim_question(question_id: int, user: dict = Depends(require_engineer)):
    """原子领取待处理问题，同一问题只能被一个工程师接单。"""
    with get_connection() as conn:
        question = conn.execute(
            """UPDATE engineer_questions
            SET status='processing',assigned_engineer_id=%s,claimed_at=NOW(),updated_at=NOW()
            WHERE id=%s AND status='pending' AND assigned_engineer_id IS NULL RETURNING *""",
            (user["id"], question_id)
        ).fetchone()
        if not question:
            exists = conn.execute("SELECT id FROM engineer_questions WHERE id=%s", (question_id,)).fetchone()
            if not exists:
                raise HTTPException(status_code=404, detail="问题不存在")
            raise HTTPException(status_code=409, detail="该问题已被接单或已完成，请刷新队列")
    return question


@router.post("/engineer/questions/{question_id}/reply")
def reply_question(question_id: int, data: EngineerReply, user: dict = Depends(require_engineer)):
    """接单工程师回复问题，并在同一事务中将回复写回员工会话。"""
    answer = data.answer.strip()
    if not answer:
        raise HTTPException(status_code=422, detail="回复内容不能为空")
    with get_connection() as conn:
        # 锁定问题，防止重复回复或同时修改处理状态
        question = conn.execute("SELECT * FROM engineer_questions WHERE id=%s FOR UPDATE", (question_id,)).fetchone()
        if not question:
            raise HTTPException(status_code=404, detail="问题不存在")
        if question["assigned_engineer_id"] != user["id"]:
            raise HTTPException(status_code=403, detail="只有接单工程师可以回复该问题")
        if question["status"] != "processing":
            raise HTTPException(status_code=409, detail="该问题已经完成，请勿重复回复")
        question = conn.execute(
            """UPDATE engineer_questions SET status='answered',engineer_answer=%s,
            answered_by=%s,answered_at=NOW(),updated_at=NOW() WHERE id=%s RETURNING *""",
            (answer, str(user["id"]), question_id)
        ).fetchone()
        # 旧版RAG工单可能没有会话关联，仍可接单并保留回复
        if question["session_id"]:
            session = conn.execute(
                "SELECT id FROM app_chat_sessions WHERE id=%s AND user_id=%s AND deleted_at IS NULL FOR UPDATE",
                (question["session_id"], question["requester_id"])
            ).fetchone()
            if session:
                content = f"工单 #{question_id} 人工回复\n\n原问题：{question['question']}\n\n{answer}"
                conn.execute(
                    """INSERT INTO app_chat_messages(session_id,sender_role,sender_user_id,content)
                    VALUES (%s,'engineer',%s,%s)""",
                    (question["session_id"], user["id"], content)
                )
                conn.execute("UPDATE app_chat_sessions SET updated_at=NOW() WHERE id=%s", (question["session_id"],))
    return question
