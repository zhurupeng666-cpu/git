"""使用PostgreSQL租约限制同一账号的并发任务，并校验实际写入权限。"""
import os
import time
import logging
from threading import Event, Thread
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from fastapi import HTTPException

from src.api.stream_control import StreamCancelled
from src.api.auth import SESSION_IDLE_SECONDS


LEASE_SECONDS = 120
RENEW_SECONDS = 10


def get_run_connection():
    # 独立连接避免长模型调用占用数据库事务，数据库故障时有明确超时。
    return psycopg.connect(os.getenv("DATABASE_URL"), row_factory=dict_row,
                           connect_timeout=5, options="-c statement_timeout=5000")


class ActiveRun:
    def __init__(self, user, thread_id, cancellation):
        self.user = user
        self.thread_id = thread_id
        self.run_id = uuid4()
        self.cancellation = cancellation
        self.stopped = Event()
        self.monitor = None
        cancellation.run = self

    def acquire(self):
        """原子抢占租约；有效任务不能被另一标签页或进程覆盖。"""
        with get_run_connection() as conn:
            self.guard_session(conn)
            row = conn.execute(
                """INSERT INTO app_active_runs(user_id,run_id,thread_id,lease_expires_at)
                VALUES (%s,%s,%s,clock_timestamp()+%s*INTERVAL '1 second')
                ON CONFLICT(user_id) DO UPDATE SET run_id=EXCLUDED.run_id,
                    thread_id=EXCLUDED.thread_id,lease_expires_at=EXCLUDED.lease_expires_at,
                    created_at=clock_timestamp()
                WHERE app_active_runs.lease_expires_at <= clock_timestamp()
                RETURNING run_id""",
                (self.user["id"], self.run_id, self.thread_id, LEASE_SECONDS)
            ).fetchone()
            if not row:
                raise HTTPException(status_code=409, detail="当前账号正在处理另一个问题，请等待回答完成后再发送")
        # 租约监控独立于SSE消费，即使检索阻塞仍能检查退出和异地登录。
        self.monitor = Thread(target=self.watch, name="chat-run-lease", daemon=True)
        self.monitor.start()
        return self

    def guard_session(self, conn):
        """先锁用户行再锁任务行，统一锁顺序避免与退出登录形成死锁。"""
        row = conn.execute(
            "SELECT active_session_id,is_active FROM app_users WHERE id=%s FOR SHARE",
            (self.user["id"],)
        ).fetchone()
        if (not row or not row["is_active"] or
                str(row["active_session_id"]) != str(self.user["_session_id"])):
            raise StreamCancelled("登录已失效，请重新登录")
        if conn.execute("SELECT clock_timestamp() >= to_timestamp(%s) AS expired",
                        (self.user["_token_exp"],)).fetchone()["expired"]:
            raise StreamCancelled("登录已失效，请重新登录")

        # 与检查点写入同一事务锁定登录凭证，过期或撤销后不能继续保存。
        session = conn.execute(
            """SELECT id FROM app_auth_sessions WHERE id=%s AND user_id=%s AND tab_token_hash=%s
                AND revoked_at IS NULL AND expires_at>clock_timestamp()
                AND last_active_at>clock_timestamp()-%s*INTERVAL '1 second' FOR SHARE""",
            (self.user["_session_id"], self.user["id"], self.user["_tab_token_hash"], SESSION_IDLE_SECONDS)
        ).fetchone()
        if not session:
            raise StreamCancelled("登录会话已失效或空闲超时，请重新登录")

    def guard_write(self, conn):
        """与检查点/消息写入处于同一事务，锁定归属直到提交。

        过期接管会等待此事务；旧任务在接管后无法再次写入，避免只在节点入口校验。
        """
        self.cancellation.check()
        self.guard_session(conn)
        row = conn.execute(
            "SELECT run_id,lease_expires_at FROM app_active_runs WHERE user_id=%s FOR SHARE",
            (self.user["id"],)
        ).fetchone()
        valid = row and str(row["run_id"]) == str(self.run_id)
        if valid:
            valid = conn.execute("SELECT %s > clock_timestamp() AS valid",
                                 (row["lease_expires_at"],)).fetchone()["valid"]
        if not valid:
            raise StreamCancelled("当前任务已失去执行权限，请重新发送问题")

    def _renew(self, conn):
        self.guard_session(conn)
        row = conn.execute(
            """UPDATE app_active_runs SET lease_expires_at=clock_timestamp()+%s*INTERVAL '1 second'
            WHERE user_id=%s AND run_id=%s AND lease_expires_at>clock_timestamp()
            RETURNING run_id""", (LEASE_SECONDS, self.user["id"], self.run_id)
        ).fetchone()
        if not row:
            raise StreamCancelled("当前任务已失去执行权限，请重新发送问题")

    def renew(self):
        with get_run_connection() as conn:
            self._renew(conn)

    def watch(self):
        try:
            # 监控线程复用独立连接，避免20用户每秒重复建连和多条锁查询。
            # autocommit确保普通监控读取不保持数据库事务或行锁。
            with get_run_connection() as conn:
                conn.autocommit = True
                next_renewal = time.monotonic() + RENEW_SECONDS
                while not self.stopped.wait(1):
                    if self.cancellation.event.is_set():
                        return
                    if time.monotonic() >= next_renewal:
                        # 按真实经过时间续租，不能按循环次数计算；慢查询不会推迟到第10轮。
                        with conn.transaction():
                            self._renew(conn)
                        next_renewal = time.monotonic() + RENEW_SECONDS
                    else:
                        row = conn.execute(
                            """SELECT 1 FROM app_active_runs r
                            JOIN app_users u ON u.id=r.user_id
                            JOIN app_auth_sessions s ON s.id=u.active_session_id AND s.user_id=u.id
                            WHERE r.user_id=%s AND r.run_id=%s AND r.lease_expires_at>clock_timestamp()
                                AND u.is_active=TRUE AND s.id=%s AND s.tab_token_hash=%s
                                AND s.revoked_at IS NULL AND s.expires_at>clock_timestamp()
                                AND s.last_active_at>clock_timestamp()-%s*INTERVAL '1 second'
                                AND clock_timestamp()<to_timestamp(%s)
                                AND NOT EXISTS(SELECT 1 FROM auth_revoked_tokens WHERE token_hash=%s)""",
                            (self.user["id"], self.run_id, self.user["_session_id"], self.user["_tab_token_hash"],
                             SESSION_IDLE_SECONDS, self.user["_token_exp"], self.user["_token_hash"])
                        ).fetchone()
                        if not row:
                            self.guard_session(conn)
                            raise StreamCancelled("当前任务已失去执行权限，请重新发送问题")
        except StreamCancelled as exc:
            self.cancellation.cancel(str(exc))
        except Exception:
            logging.getLogger(__name__).exception(
                "任务监控失败 run_id=%s", self.run_id,
                extra={"request_id": self.cancellation.request_id}
            )
            # 数据库校验失败仍然停止任务，不能为了压测成功忽略鉴权或所有权。
            self.cancellation.cancel("任务状态校验失败，请稍后重试")

    def release(self):
        self.stopped.set()
        with get_run_connection() as conn:
            conn.execute("DELETE FROM app_active_runs WHERE user_id=%s AND run_id=%s",
                         (self.user["id"], self.run_id))
