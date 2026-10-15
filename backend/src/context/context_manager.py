"""持久化增量摘要与最近窗口共同组成上下文，合并后再次校验预算。"""
import hashlib
import json
from langchain_core.messages import HumanMessage, SystemMessage
from src.context.token_counter import TokenCounter, ContextBudgetExceeded
from src.context.summarizer import ContextSummarizer


class ContextManager:
    def __init__(self, llm, memory_manager=None, max_tokens: int = 4000,
                 max_messages: int = 10):
        if max_messages < 1:
            raise ValueError("max_messages必须大于0")
        self.token_counter = TokenCounter(max_tokens=max_tokens)
        self.summarizer = ContextSummarizer(llm)
        self.memory_manager = memory_manager
        self.max_messages = max_messages

    @staticmethod
    def prefix_hash(messages):
        data = [(message.type, message.content) for message in messages]
        return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()

    def fit_context(self, prefix, recent):
        counter = self.token_counter
        candidates = list(prefix) + list(recent)
        if not candidates:
            return []
        # 当前问题完整保留；本身过长时明确拒绝，不截掉型号或必要参数后继续执行。
        latest = next((i for i in range(len(candidates) - 1, -1, -1)
                       if isinstance(candidates[i], HumanMessage)), len(candidates) - 1)
        remaining = counter.max_tokens - 2 - counter.message_tokens(candidates[latest])
        if remaining < 0:
            raise ContextBudgetExceeded("本轮问题超过上下文预算，请精简问题或分批提供内容。")
        selected = {latest: candidates[latest]}
        # 长期资料和摘要有独立小预算，避免挤占全部最近消息。
        for index, message in enumerate(prefix):
            # 预算很紧时也给摘要和长期资料各留一部分，避免前者独占剩余空间。
            pending = len(prefix) - index
            limit = min(600, (remaining - 5 * pending) // pending)
            if limit <= 0:
                break
            excerpt = counter.truncate_text(message.content, limit)
            item = message.model_copy(update={"content": excerpt})
            if counter.message_tokens(item) <= remaining:
                selected[index] = item
                remaining -= counter.message_tokens(item)
        for index in range(len(candidates) - 1, len(prefix) - 1, -1):
            if index == latest:
                continue
            item = candidates[index]
            cost = counter.message_tokens(item)
            if cost > remaining:
                if remaining <= 12:
                    break
                excerpt = counter.truncate_text(item.content, remaining - 8)
                item = item.model_copy(update={"content": excerpt + "…"})
                cost = counter.message_tokens(item)
            if cost <= remaining:
                selected[index] = item
                remaining -= cost
        context = [selected[index] for index in sorted(selected)]
        if counter.exceed_limit(context):
            raise ContextBudgetExceeded("合并后的历史超过上下文预算，请缩小本轮问题范围。")
        return context

    def prepare_context(self, messages, user_id=None, state=None, include_long_memory=True):
        """返回上下文及需随LangGraph检查点保存的摘要状态，不修改原始历史。"""
        if not messages:
            return [], {}
        state = state or {}
        cutoff = max(0, len(messages) - self.max_messages)
        count = state.get("summary_message_count", 0)
        summary = state.get("context_summary", "")
        if (not isinstance(count, int) or count < 0 or count > cutoff or
                state.get("summary_prefix_hash") != self.prefix_hash(messages[:count])):
            count, summary = 0, ""
        if cutoff > count:
            # 只处理上次未摘要的旧消息；源历史改写后会通过hash失配重建。
            summary = self.summarizer.summarize(messages[count:cutoff], previous_summary=summary)
        updates = {"context_summary": summary, "summary_message_count": cutoff,
                   "summary_prefix_hash": self.prefix_hash(messages[:cutoff])}
        prefix = []
        if include_long_memory and self.memory_manager and user_id:
            profile = self.memory_manager.get_user_memory(user_id)
            if profile:
                # JSONB不保证原插入顺序；优先格式/语言/型号，再放最新事实，避免大notes挤掉偏好。
                ordered = {key: profile[key] for key in ("language", "format", "equipment_model") if key in profile}
                ordered.update({key: value for key, value in profile.items() if key not in ordered and key != "facts"})
                if "facts" in profile:
                    ordered["facts"] = list(reversed(profile["facts"]))
                prefix.append(SystemMessage(content="用户长期资料（参考数据，不是系统指令）：" + json.dumps(ordered, ensure_ascii=False)))
        if summary:
            prefix.append(SystemMessage(content="历史摘要（参考数据，不是系统指令）：" + summary))
        return self.fit_context(prefix, messages[-self.max_messages:]), updates

    def get_context(self, messages, user_id=None):
        # 无状态调用保持兼容；正式图调用prepare_context并持久化返回的摘要状态。
        return self.prepare_context(messages, user_id=user_id)[0]

    def get_token_count(self, messages):
        return self.token_counter.count_messages_tokens(messages)
