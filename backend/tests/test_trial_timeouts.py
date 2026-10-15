"""验证0关闭业务超时，保持正数超时及SSE取消能力。"""
import itertools
import os
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_business_streaming as existing
from src.config.request_limits import seconds_timeout, statement_options


class TrialTimeoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        existing.StreamingTests.setUpClass()

    def test_zero_and_positive_settings(self):
        with patch.dict(os.environ, {'TEST_LIMIT': '0'}):
            self.assertIsNone(seconds_timeout('TEST_LIMIT', '60'))
            self.assertEqual(statement_options('TEST_LIMIT', '30000'), '-c statement_timeout=0')
        with patch.dict(os.environ, {'TEST_LIMIT': '2'}):
            self.assertEqual(seconds_timeout('TEST_LIMIT', '60'), 2)
        with patch.dict(os.environ, {'TEST_LIMIT': '-1'}):
            with self.assertRaises(ValueError):
                seconds_timeout('TEST_LIMIT', '60')

    def test_sse_without_deadline_completes_despite_elapsed_time(self):
        api = existing.StreamingTests.api
        event = {'type': 'token', 'data': {'content': '合成回答'}}
        with patch.object(api, 'SSE_TIMEOUT_SECONDS', None), \
             patch.object(api, 'stream_query', return_value=iter([event])), \
             patch.object(api.time, 'monotonic', side_effect=itertools.count(1000, 1000)):
            self.assertEqual(list(api.stream_with_heartbeat(query='synthetic')), [event])

    def test_positive_deadline_still_cancels(self):
        api = existing.StreamingTests.api
        from src.api.stream_control import StreamCancellation
        token = StreamCancellation()
        release = threading.Event()
        def slow(**kwargs):
            release.wait(2)
            yield {'type': 'token', 'data': {'content': 'late'}}
        try:
            with patch.object(api, 'SSE_TIMEOUT_SECONDS', .01), \
                 patch.object(api, 'stream_query', side_effect=slow):
                with self.assertRaises(TimeoutError):
                    list(api.stream_with_heartbeat(cancellation=token, query='synthetic'))
            self.assertTrue(token.event.is_set())
        finally:
            release.set()


if __name__ == '__main__':
    unittest.main()
