"""使用PostgresStore保存长期资料，并保持活动任务的写入权限校验。"""
from contextlib import contextmanager
from langgraph.store.postgres import PostgresStore
from src.api.stream_control import get_current_run, check_cancelled


class LongMemory:
    def __init__(self, store: PostgresStore):
        self.store = store

    @contextmanager
    def write_guard(self):
        check_cancelled()
        run = get_current_run()
        if run:
            from src.api.run_control import get_run_connection
            # 持有账号/会话/租约共享锁直至Store写入完成，撤销操作不能穿过此窗口。
            with get_run_connection() as conn:
                run.guard_write(conn)
                yield
        else:
            yield

    def save_memory(self, user_id: str, key: str, value: dict):
        with self.write_guard():
            self.store.put(("semiconductor", user_id), key, value)

    def get_memory(self, user_id: str, key: str):
        result = self.store.get(("semiconductor", user_id), key)
        return result.value if result else None

    def update_memory(self, user_id: str, key: str, updates: dict):
        # 正式聊天已有PG每账号单活动任务租约；更新前再校验，防止失效任务覆盖新资料。
        with self.write_guard():
            value = dict(self.get_memory(user_id, key) or {})
            for field, item in updates.items():
                if item is None:
                    value.pop(field, None)
                elif field == "facts":
                    value[field] = list(dict.fromkeys(value.get(field, []) + item))[-12:]
                else:
                    value[field] = item
            self.store.put(("semiconductor", user_id), key, value)
