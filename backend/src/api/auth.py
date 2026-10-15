import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4
import jwt
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request, Response, Header
from pydantic import BaseModel, Field
from pwdlib import PasswordHash
import hashlib
from src.api.stream_control import cancel_token_streams

load_dotenv()
router = APIRouter(prefix="/auth", tags=["用户认证"])
password_hash = PasswordHash.recommended()
ALGORITHM = "HS256"
COOKIE_NAME = "semiconductor_access"
# 绝对有效期不随操作延长；空闲超时只由有效业务操作或活动请求刷新。
TOKEN_EXPIRE_SECONDS = int(os.getenv("AUTH_SESSION_MAX_SECONDS", "7200"))
SESSION_IDLE_SECONDS = int(os.getenv("AUTH_SESSION_IDLE_SECONDS", "1800"))
if TOKEN_EXPIRE_SECONDS <= 0 or SESSION_IDLE_SECONDS <= 0:
    raise RuntimeError("登录有效期和空闲超时必须大于0")
IS_PRODUCTION = os.getenv("APP_ENV", "development") == "production"

# 登录请求模型
class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1)

# 获取数据库连接
def get_connection():
    """连接PostgreSQL用户数据库。"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("请配置DATABASE_URL")
    return psycopg.connect(database_url, row_factory=dict_row)

# 获取JWT签名密钥
def get_jwt_secret():
    """读取服务端JWT签名密钥。"""
    secret = os.getenv("JWT_SECRET_KEY")
    if not secret or len(secret) < 32:
        raise RuntimeError("JWT_SECRET_KEY长度不能少于32位")
    return secret

# 验证请求来源
def check_origin(request: Request):
    """拒绝来源不可信的跨站请求。"""
    origin = request.headers.get("origin")
    if not origin:
        return
    default_origins = "http://localhost:5173,http://127.0.0.1:5173,http://127.0.0.1:8000"
    allowed = os.getenv("APP_ALLOWED_ORIGINS", "" if IS_PRODUCTION else default_origins)
    if origin not in {item.strip() for item in allowed.split(",") if item.strip()}:
        raise HTTPException(status_code=403, detail="请求来源不可信")

# 签发JWT
def create_access_token(user_id: str, session_id: str):
    """生成包含用户ID、登录会话ID、有效期和CSRF随机值的JWT。"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "sid": session_id,
        "iat": now,
        "exp": now + timedelta(seconds=TOKEN_EXPIRE_SECONDS),
        "csrf": secrets.token_urlsafe(32)
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM), payload["csrf"]

def get_session_cookie_name(session_id):
    """每个登录会话使用独立HttpOnly Cookie，避免同浏览器不同账号相互覆盖。"""
    return f"{COOKIE_NAME}_{UUID(str(session_id)).hex}"


def get_request_token(request):
    """页面会话ID只用于选择Cookie，真实身份仍由签名JWT和数据库验证。"""
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        try:
            cookie_name = get_session_cookie_name(session_id)
        except (ValueError, TypeError, AttributeError):
            raise HTTPException(status_code=401, detail="登录会话无效，请重新登录")
        token = request.cookies.get(cookie_name)
        if token:
            return token
    # 禁止只凭浏览器共享Cookie恢复登录；每个请求都必须指定自己的会话。
    return None


# 用户登录
@router.post("/login")
def login(data: LoginRequest, request: Request, response: Response):
    """验证账户并将JWT写入HttpOnly Cookie。"""
    check_origin(request)
    with get_connection() as conn:
        user = conn.execute(
            "SELECT id,username,password_hash,display_name,role FROM app_users WHERE username=%s AND is_active=TRUE",
            (data.username,)
        ).fetchone()
    if not user or not password_hash.verify(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # 每次登录生成新的会话ID，更新成功后旧设备的JWT立即失效。
    session_id = uuid4()
    token, csrf_token = create_access_token(str(user["id"]), str(session_id))
    tab_token = secrets.token_urlsafe(32)
    tab_token_hash = hashlib.sha256(tab_token.encode("utf-8")).hexdigest()
    with get_connection() as conn:
        updated = conn.execute(
            "UPDATE app_users SET active_session_id=%s,updated_at=NOW() WHERE id=%s AND is_active=TRUE RETURNING id",
            (session_id, user["id"])
        ).fetchone()
        if not updated:
            raise HTTPException(status_code=401, detail="用户不存在或已禁用")
        # 与切换active_session_id同一事务，旧会话不能在新登录后继续使用。
        conn.execute("UPDATE app_auth_sessions SET revoked_at=clock_timestamp() WHERE user_id=%s AND revoked_at IS NULL",
                     (user["id"],))
        conn.execute(
            """INSERT INTO app_auth_sessions(id,user_id,tab_token_hash,expires_at)
            VALUES (%s,%s,%s,clock_timestamp()+%s*INTERVAL '1 second')""",
            (session_id, user["id"], tab_token_hash, TOKEN_EXPIRE_SECONDS)
        )
        # 只清理已失效的凭证记录，保留用户、聊天和业务数据。
        conn.execute("""DELETE FROM app_auth_sessions WHERE id IN (
            SELECT id FROM app_auth_sessions WHERE expires_at<=clock_timestamp()
                OR last_active_at<=clock_timestamp()-%s*INTERVAL '1 second' OR revoked_at IS NOT NULL
            LIMIT 500 FOR UPDATE SKIP LOCKED)""",
                     (SESSION_IDLE_SECONDS,))

    # JWT仅通过HttpOnly Cookie传递，不放入响应JSON
    # 清理旧版默认Cookie，不再允许新标签页自动继承它。
    response.delete_cookie(key=COOKIE_NAME, path="/api", secure=IS_PRODUCTION,
                           httponly=True, samesite="lax")
    # JWT仍只保存在当前登录会话对应的HttpOnly Cookie中。
    response.set_cookie(
        key=get_session_cookie_name(session_id), value=token, max_age=TOKEN_EXPIRE_SECONDS,
        httponly=True, secure=IS_PRODUCTION, samesite="lax", path="/api"
    )
    return {
        "user_id": str(user["id"]),
        "username": user["username"],
        "display_name": user["display_name"],
        "role": user["role"],
        "csrf_token": csrf_token,
        "session_id": str(session_id),
        "tab_token": tab_token,
        "idle_timeout_seconds": SESSION_IDLE_SECONDS
    }

# 验证当前登录用户
def get_current_user(request: Request, csrf_token: str | None = Header(default=None, alias="X-CSRF-Token")):
    """验证JWT、撤销状态、CSRF及数据库用户身份。"""
    # Tab Token是标签页持有的第二份凭证，不能从共享Cookie或公开sid推导。
    tab_token = request.headers.get("X-Tab-Token", "")
    if not tab_token or len(tab_token) > 256:
        raise HTTPException(status_code=401, detail="标签页凭证缺失或无效，请重新登录")
    tab_token_hash = hashlib.sha256(tab_token.encode("utf-8")).hexdigest()
    # 从HttpOnly Cookie获取JWT
    token = get_request_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="请先登录")
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[ALGORITHM])
        user_id = UUID(payload["sub"])
        session_id = UUID(payload["sid"])
    except (jwt.InvalidTokenError, ValueError, KeyError, TypeError):
        raise HTTPException(status_code=401, detail="登录凭证无效或已过期")

    # 页面绑定原登录会话，避免Cookie被另一个标签页更新后冒用新身份。
    page_session = request.headers.get("X-Session-ID")
    if page_session != str(session_id):
        raise HTTPException(status_code=401, detail="登录会话已更换，请重新登录")

    # 检查JWT是否已被服务端撤销
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with get_connection() as conn:
        revoked = conn.execute("SELECT 1 FROM auth_revoked_tokens WHERE token_hash=%s", (token_hash,)).fetchone()
        if revoked:
            raise HTTPException(status_code=401, detail="登录凭证已注销")

        # 联合校验账户、sid、凭证哈希、绝对有效期及空闲有效期。
        user = conn.execute(
            """SELECT u.id,u.username,u.display_name,u.role,s.tab_token_hash
            FROM app_users u JOIN app_auth_sessions s ON s.user_id=u.id AND s.id=u.active_session_id
            WHERE u.id=%s AND s.id=%s AND u.is_active=TRUE AND s.revoked_at IS NULL
                AND s.expires_at>clock_timestamp()
                AND s.last_active_at>clock_timestamp()-%s*INTERVAL '1 second'""",
            (user_id, session_id, SESSION_IDLE_SECONDS)
        ).fetchone()
        if not user or not hmac.compare_digest(user["tab_token_hash"], tab_token_hash):
            raise HTTPException(status_code=401, detail="登录会话已失效或空闲超时，请重新登录")
        # 未通过CSRF的请求不能延长会话；后台轮询和/me也不算用户活动。
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            check_origin(request)
            if not csrf_token or not hmac.compare_digest(csrf_token, payload.get("csrf", "")):
                raise HTTPException(status_code=403, detail="CSRF验证失败")
            refreshed = conn.execute(
                """UPDATE app_auth_sessions SET last_active_at=clock_timestamp()
                WHERE id=%s AND tab_token_hash=%s AND revoked_at IS NULL
                    AND expires_at>clock_timestamp()
                    AND last_active_at>clock_timestamp()-%s*INTERVAL '1 second'
                RETURNING id""", (session_id, tab_token_hash, SESSION_IDLE_SECONDS)
            ).fetchone()
            if not refreshed:
                raise HTTPException(status_code=401, detail="登录会话已失效，请重新登录")
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在或已禁用")

    # POST等修改类请求必须通过Origin和CSRF校验
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        check_origin(request)
        if not csrf_token or not hmac.compare_digest(csrf_token, payload.get("csrf", "")):
            raise HTTPException(status_code=403, detail="CSRF验证失败")
    # 内部流式校验信息不包含JWT原文，也不出现在/me响应中
    user.pop("tab_token_hash", None)
    user["_tab_token_hash"] = tab_token_hash
    user["_session_id"] = session_id
    user["_token_hash"] = token_hash
    user["_token_exp"] = payload["exp"]
    user["csrf_token"] = payload.get("csrf", "")
    return user

# 工程师角色权限
def require_engineer(user: dict = Depends(get_current_user)):
    """限制接口仅允许工程师访问。"""
    if user["role"] != "engineer":
        raise HTTPException(status_code=403, detail="仅工程师有权执行此操作")
    return user

# 获取当前登录用户
@router.get("/me")
def get_profile(user: dict = Depends(get_current_user)):
    """返回当前用户及CSRF Token，不暴露JWT。"""
    return {
        "user_id": str(user["id"]),
        "username": user["username"],
        "display_name": user["display_name"],
        "role": user["role"],
        "csrf_token": user["csrf_token"],
        "session_id": str(user["_session_id"]),
        "idle_timeout_seconds": SESSION_IDLE_SECONDS
    }


@router.post("/activity")
def record_activity(user: dict = Depends(get_current_user)):
    """仅前台用户操作调用；依赖已经校验凭证并更新最后活动时间。"""
    return {"message": "活动状态已更新"}


# 用户登出
@router.post("/logout")
def logout(request: Request, response: Response, user: dict = Depends(get_current_user)):
    """将当前JWT加入撤销记录，并清除浏览器认证Cookie。"""
    # 获取当前JWT并计算SHA-256哈希
    token = get_request_token(request)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    payload = jwt.decode(token, get_jwt_secret(), algorithms=[ALGORITHM])
    expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    # 将JWT撤销记录持久化到PostgreSQL
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO auth_revoked_tokens(token_hash,user_id,expires_at) VALUES (%s,%s,%s) ON CONFLICT(token_hash) DO NOTHING",
            (token_hash, user["id"], expires_at)
        )
        conn.execute("UPDATE app_auth_sessions SET revoked_at=clock_timestamp() WHERE id=%s AND user_id=%s",
                     (user["_session_id"], user["id"]))
        # 仅清除当前会话，旧设备退出不能注销后来登录的新设备。
        conn.execute("UPDATE app_users SET active_session_id=NULL WHERE id=%s AND active_session_id=%s",
                     (user["id"], user["_session_id"]))
    # 撤销事务提交后立即取消当前进程中的活动SSE，上游连接也同时关闭
    cancel_token_streams(token_hash)
    # 数据库写入成功后再清除浏览器Cookie
    response.delete_cookie(key=get_session_cookie_name(user["_session_id"]), path="/api",
                           secure=IS_PRODUCTION, httponly=True, samesite="lax")
    # 默认Cookie可能属于另一个账号，退出当前账号时不能删除它。
    if request.cookies.get(COOKIE_NAME) == token:
        response.delete_cookie(key=COOKIE_NAME, path="/api", secure=IS_PRODUCTION,
                               httponly=True, samesite="lax")
    return {"message": "退出登录成功"}

def is_stream_session_active(user: dict) -> bool:
    """活动SSE每秒复查撤销记录，兼容多进程和账户禁用。"""
    if datetime.now(timezone.utc).timestamp() >= user["_token_exp"]:
        return False
    with psycopg.connect(os.getenv("DATABASE_URL"), row_factory=dict_row,
                         connect_timeout=5, options="-c statement_timeout=3000") as conn:
        revoked = conn.execute("SELECT 1 FROM auth_revoked_tokens WHERE token_hash=%s", (user["_token_hash"],)).fetchone()
        active = conn.execute(
            """SELECT u.id FROM app_users u JOIN app_auth_sessions s ON s.id=u.active_session_id AND s.user_id=u.id
            WHERE u.id=%s AND s.id=%s AND s.tab_token_hash=%s AND u.is_active=TRUE
                AND s.revoked_at IS NULL AND s.expires_at>clock_timestamp()
                AND s.last_active_at>clock_timestamp()-%s*INTERVAL '1 second'""",
            (user["id"], user["_session_id"], user["_tab_token_hash"], SESSION_IDLE_SECONDS)
        ).fetchone()
    return not revoked and bool(active)
