"""管理用户明确授权保存的长期资料，不把普通会话事实自动转为长期记忆。"""
import re
from src.memory.long_memory import LongMemory
from src.context.token_counter import TokenCounter, ContextBudgetExceeded


def requests_long_term(query):
    profile_question = re.search(r"我的(?:回答|回复|输出)?偏好|我的设备型号", query) and re.search(r"是什么|什么|哪个|[？?]", query)
    return bool(profile_question or re.search(r"长期记住|长期保存|跨会话|以后.{0,8}新会话|我的长期|长期.{0,8}(?:偏好|资料|信息|记忆)", query))


class MemoryManager:
    def __init__(self, long_memory: LongMemory):
        self.long_memory = long_memory

    def get_user_memory(self, user_id: str):
        return self.long_memory.get_memory(user_id=user_id, key="profile") or {}

    def save_user_memory(self, user_id: str, profile: dict):
        self.long_memory.save_memory(user_id=user_id, key="profile", value=profile)

    def save_requested_memory(self, user_id: str, query: str):
        """只有明确长期保存指令才写库，事实保留原文，常见偏好另存结构化字段。"""
        if not user_id or not requests_long_term(query):
            return False
        if re.search(r"不要|不用|不必|无需", query.split("记住", 1)[0]):
            return False
        match = re.search(r"(?:长期记住|长期保存)(?:以下(?:信息|内容))?\s*[:：，,]?\s*(.+)", query)
        if not match:
            match = re.search(r"记住\s*[:：，,]?\s*(.+)", query)
        if not match or re.search(r"[？?]|是什么|哪个|多少", query):
            return False
        statement = match[1].strip()
        if TokenCounter().count_text_tokens(statement) > 512:
            raise ContextBudgetExceeded("需要长期保存的信息过长，请精简为明确的偏好、型号或事实。")
        updates = {}
        if "中文" in statement:
            updates["language"] = "中文"
        elif "英文" in statement:
            updates["language"] = "英文"
        formats = list(re.finditer(r"表格|条目|列表|段落", statement))
        if formats:
            last = formats[-1]
            updates["format"] = None if re.search(r"不要|不用|不使用", statement[max(0,last.start()-4):last.start()]) else last[0]
        equipment = re.search(r"设备型号\s*(?:是|为|[:：])\s*([A-Za-z0-9][A-Za-z0-9._/-]{0,127})", statement)
        if equipment:
            updates["equipment_model"] = equipment[1]
        updates["facts"] = [statement]
        self.long_memory.update_memory(user_id=user_id, key="profile", updates=updates)
        return True
