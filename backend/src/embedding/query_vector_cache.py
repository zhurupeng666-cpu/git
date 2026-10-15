"""有容量上限的查询向量LRU缓存，同一查询并发只计算一次。"""
from collections import OrderedDict
from threading import Event, Lock
from src.api.stream_control import check_cancelled


class QueryVectorCache:
    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError("QUERY_VECTOR_CACHE_SIZE不能小于0")
        self.capacity = capacity
        self.values = OrderedDict()
        self.pending = {}
        self.lock = Lock()
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key, compute):
        check_cancelled()
        if not self.capacity:
            return compute()
        while True:
            check_cancelled()
            with self.lock:
                if key in self.values:
                    self.hits += 1
                    self.values.move_to_end(key)
                    # 返回副本，避免调用方修改向量影响其他请求。
                    return self.values[key].copy()
                ready = self.pending.get(key)
                if ready is None:
                    ready = Event()
                    self.pending[key] = ready
                    self.misses += 1
                    break
            # 等待不占推理名额，退出登录或断连后可独立取消。
            ready.wait(0.1)
        try:
            value = compute()
            check_cancelled()
            with self.lock:
                self.values[key] = value.copy()
                self.values.move_to_end(key)
                while len(self.values) > self.capacity:
                    self.values.popitem(last=False)
            return value.copy()
        finally:
            # 异常和取消不缓存；唤醒其他请求，让仍有效的请求重新计算。
            with self.lock:
                self.pending.pop(key, None)
                ready.set()
