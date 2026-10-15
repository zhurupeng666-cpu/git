"""使用离线参考Tokenizer计数；不是字符数，也不冒充服务商的精确用量。"""
import hashlib
import json
import os
from pathlib import Path
import tiktoken


class ContextBudgetExceeded(ValueError):
    """当前请求无法在应用预算内完整处理，需要缩小输入范围。"""


class TokenCounter:
    def __init__(self, max_tokens: int = 8000):
        if max_tokens < 1:
            raise ValueError("max_tokens必须大于0")
        self.max_tokens = max_tokens
        # 使用安装阶段缓存的官方分词表，请求过程中不联网下载Tokenizer。
        cache = Path(os.getenv("TIKTOKEN_CACHE_DIR", str(Path(__file__).resolve().parents[2] / "data" / "tokenizers")))
        url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        if not (cache / hashlib.sha1(url.encode()).hexdigest()).is_file():
            raise RuntimeError("缺少本地cl100k_base分词表，请先准备data/tokenizers缓存")
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache))
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.counting_method = "cl100k_base_reference_estimate"

    def count_text_tokens(self, text: str) -> int:
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False, default=str)
        return len(self.encoding.encode(text or "", disallowed_special=()))

    def message_tokens(self, message) -> int:
        # 消息边界与角色也占预算，不能只计算content。
        return 4 + self.count_text_tokens(getattr(message, "type", "message")) + self.count_text_tokens(getattr(message, "content", ""))

    def count_messages_tokens(self, messages) -> int:
        return 2 + sum(self.message_tokens(message) for message in messages)

    def exceed_limit(self, messages) -> bool:
        return self.count_messages_tokens(messages) > self.max_tokens

    def truncate_text(self, text: str, limit: int) -> str:
        """按Token边界节选，并避免在UTF-8字符中间产生替代字符。"""
        if limit <= 0:
            return ""
        tokens = self.encoding.encode(text or "", disallowed_special=())
        return self.encoding.decode_bytes(tokens[:limit]).decode("utf-8", errors="ignore")

    def split_text(self, text: str, limit: int):
        # 按字符位置切分、按Token验证，避免跨中文字符时丢字。
        while text:
            if self.count_text_tokens(text) <= limit:
                yield text
                return
            low, high = 1, len(text)
            while low < high:
                middle = (low + high + 1) // 2
                if self.count_text_tokens(text[:middle]) <= limit:
                    low = middle
                else:
                    high = middle - 1
            yield text[:low]
            text = text[low:]
