"""验证缓存容量、副本隔离、相同查询合并、异常重试和等待取消。"""
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.embedding.query_vector_cache import QueryVectorCache
from src.api.stream_control import StreamCancellation, StreamCancelled, cancellation_scope


class VectorCacheTests(unittest.TestCase):
    def test_twenty_concurrent_queries_compute_once(self):
        cache = QueryVectorCache(2)
        calls = []
        def compute():
            calls.append(1)
            time.sleep(0.05)
            return [0.1, 0.2]
        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(lambda _: cache.get_or_compute('same-model-query', compute), range(20)))
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.hits, 19)
        results[0][0] = 9
        self.assertEqual(cache.get_or_compute('same-model-query', compute), [0.1, 0.2])

    def test_lru_bounded_and_model_keys_isolated(self):
        cache = QueryVectorCache(2)
        cache.get_or_compute(('model-a', 'q'), lambda: [1])
        cache.get_or_compute(('model-b', 'q'), lambda: [2])
        self.assertEqual(cache.get_or_compute(('model-a', 'q'), lambda: [9]), [1])
        cache.get_or_compute(('model-c', 'q'), lambda: [3])
        self.assertEqual(len(cache.values), 2)
        self.assertNotIn(('model-b', 'q'), cache.values)

    def test_failure_does_not_poison_cache(self):
        cache = QueryVectorCache(2)
        def broken():
            raise ValueError('synthetic failure')
        with self.assertRaises(ValueError):
            cache.get_or_compute('q', broken)
        self.assertFalse(cache.pending)
        self.assertEqual(cache.get_or_compute('q', lambda: [3]), [3])

    def test_cancelled_waiter_does_not_cancel_owner(self):
        cache = QueryVectorCache(2)
        entered, release = threading.Event(), threading.Event()
        token = StreamCancellation()
        def owner():
            entered.set()
            release.wait(3)
            return [4]
        def waiter():
            with cancellation_scope(token):
                return cache.get_or_compute('q', lambda: self.fail('duplicate compute'))
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(cache.get_or_compute, 'q', owner)
            self.assertTrue(entered.wait(1))
            second = pool.submit(waiter)
            token.cancel('synthetic disconnect')
            try:
                with self.assertRaises(StreamCancelled):
                    second.result(timeout=1)
            finally:
                release.set()
            self.assertEqual(first.result(timeout=1), [4])

    def test_disabled_cache(self):
        cache = QueryVectorCache(0)
        calls = []
        for _ in range(2):
            cache.get_or_compute('q', lambda: calls.append(1) or [1])
        self.assertEqual(len(calls), 2)
        self.assertFalse(cache.values)


if __name__ == '__main__':
    unittest.main()
