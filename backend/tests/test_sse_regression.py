"""Offline streaming regression checks for the actual project files."""
import importlib.util
import json
import sys
import threading
import time
import unittest
from contextlib import nullcontext, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tests'))
import test_business_streaming as existing
from langgraph.checkpoint.memory import InMemorySaver


class SSERegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        existing.StreamingTests.setUpClass()
        cls.graph = existing.StreamingTests.graph
        cls.api = existing.StreamingTests.api

    def test_singleton_initialization_is_thread_safe(self):
        pipeline = Mock()
        results = []
        def construct():
            time.sleep(0.02)
            return pipeline
        with patch.object(self.graph, 'rag_pipeline', None), patch.object(
                self.graph, 'RAGPipeline', side_effect=construct) as factory:
            threads = [threading.Thread(target=lambda: results.append(self.graph.get_rag_pipeline()))
                       for _ in range(4)]
            for thread in threads: thread.start()
            for thread in threads: thread.join(1)
            self.assertEqual(results, [pipeline] * 4)
            factory.assert_called_once()

    @contextmanager
    def authenticated_route(self):
        # 与现有业务流式测试一样隔离数据库和租约，保留真实图及SSE协议。
        session_id = uuid4()
        connection = MagicMock()
        connection.__enter__.return_value.execute.return_value.fetchone.return_value = {"id": session_id}
        user = {"id": uuid4(), "_token_hash": "offline-regression", "_token_exp": 4102444800}
        with patch.object(self.api, 'CancellableStreamingResponse', side_effect=lambda iterator, **kwargs: iterator), \
             patch.object(self.api, 'ActiveRun', return_value=Mock()), \
             patch.object(self.api, 'get_connection', return_value=connection), \
             patch.object(self.api, 'is_stream_session_active', return_value=True):
            yield session_id, user

    def test_actual_langgraph_and_sse_deliver_before_generation_finishes(self):
        allow_finish = threading.Event()
        finished = threading.Event()
        def chunks(**kwargs):
            yield {'type': 'chunk', 'content': '首个'}
            allow_finish.wait(2)
            yield {'type': 'chunk', 'content': '\n尾段'}
            finished.set()
        pipeline = SimpleNamespace(ask_stream=chunks)
        manager = SimpleNamespace(get_context=lambda messages, user_id: messages, prepare_context=lambda messages, user_id, state: (messages, {}))
        with self.authenticated_route() as (session_id, user), patch.object(self.graph, 'get_rag_pipeline', return_value=pipeline), patch.object(
                self.graph, 'get_context_manager', return_value=manager), patch.object(
                self.graph, 'classify_intent', return_value={'intent': 'knowledge_qa'}), patch.object(
                self.graph, 'open_conversation_memory', return_value=nullcontext(InMemorySaver())), patch.object(
                self.api, 'StreamingResponse', side_effect=lambda iterator, **kwargs: iterator):
            stream = self.api.chat_stream_sse(self.api.ChatRequest(query='CVD压力异常', thread_id=str(session_id)), user=user)
            try:
                next(stream)  # start
                frames = []
                for frame in stream:
                    frames.append(frame)
                    if '首个' in frame:
                        self.assertFalse(finished.is_set())
                        allow_finish.set()
                        break
                else:
                    self.fail('No first token arrived')
                frames.extend(stream)
                self.assertTrue(any('尾段' in frame for frame in frames))
                self.assertEqual(sum('"type": "done"' in f for f in frames), 1)
            finally:
                allow_finish.set()
                stream.close()

    def test_waiting_stream_heartbeats_then_times_out_and_closes(self):
        release = threading.Event()
        closed = threading.Event()
        def blocked(**kwargs):
            try:
                release.wait(2)
                yield {'type': 'token', 'data': {'content': 'late'}}
            finally:
                closed.set()
        with self.authenticated_route() as (session_id, user), patch.object(self.api, 'stream_query', side_effect=blocked), patch.object(
                self.api, 'SSE_HEARTBEAT_SECONDS', 0.01), patch.object(
                self.api, 'SSE_TIMEOUT_SECONDS', 0.05), patch.object(
                self.api, 'StreamingResponse', side_effect=lambda iterator, **kwargs: iterator):
            try:
                frames = list(self.api.chat_stream_sse(self.api.ChatRequest(query='test', thread_id=str(session_id)), user=user))
                self.assertIn(': heartbeat\n\n', frames)
                self.assertEqual(sum('"type": "error"' in f for f in frames), 1)
                self.assertEqual(sum('"type": "done"' in f for f in frames), 1)
                self.assertFalse(any('late' in f for f in frames))
            finally:
                release.set()
                self.assertTrue(closed.wait(1))

    def test_immediate_producer_failure_reaches_client(self):
        with patch.object(self.api, 'stream_query', side_effect=RuntimeError('test')):
            with self.assertRaisesRegex(RuntimeError, 'test'):
                list(self.api.stream_with_heartbeat(query='test'))

    def test_disconnect_closes_producer_after_pending_call(self):
        release = threading.Event()
        closed = threading.Event()
        def producer(**kwargs):
            try:
                yield {'type': 'token'}
                release.wait(2)
                yield {'type': 'token'}
            finally:
                closed.set()
        with patch.object(self.api, 'stream_query', side_effect=producer):
            stream = self.api.stream_with_heartbeat(query='test')
            self.assertEqual(next(stream), {'type': 'token'})
            stream.close()
            release.set()
            self.assertTrue(closed.wait(1))

    def test_real_rag_stream_preserves_whitespace_and_reports_sources_before_tokens(self):
        modules = {
            'src.rag.v4_rerank': existing.stub('rerank', V4Rerank=Mock()),
            'src.rag.citation_verifier': existing.stub('verifier', CitationVerifier=Mock()),
            'src.rag.answer_coverage_checker': existing.stub('coverage', AnswerCoverageChecker=Mock()),
            'src.rag.engineer_queue': existing.stub('queue', EngineerQueue=Mock()),
        }
        spec = importlib.util.spec_from_file_location('offline_rag', PROJECT / 'src/rag/rag_pipeline.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules): spec.loader.exec_module(module)
        pipeline = object.__new__(module.RAGPipeline)
        pipeline.retriever = Mock()
        pipeline.retriever.search.return_value = {'top_documents': [
            {'chunk_id': 123, 'metadata': {'file_name': '工艺.pdf'}}]}
        pipeline.generator = SimpleNamespace(stream_generate=lambda **kwargs: iter(['甲', '\n', ' ', '乙']))
        # 使用真实生成器协议，包含 close。
        def chunks(**kwargs): yield from ['甲', '\n', ' ', '乙']
        pipeline.generator.stream_generate = chunks
        events = list(pipeline.ask_stream('CVD'))
        self.assertEqual([e['type'] for e in events[:2]], ['status', 'source'])
        self.assertEqual(events[1]['sources'][0]['chunk_id'], 123)
        self.assertEqual(''.join(e['content'] for e in events if e['type'] == 'chunk'), '甲\n 乙')
        pipeline.generator.stream_generate = lambda **kwargs: iter(())
        # 空流也要产生明确失败，而不能静默完成。
        def empty(**kwargs): yield from []
        pipeline.generator.stream_generate = empty
        with self.assertRaisesRegex(RuntimeError, '没有返回答案'):
            list(pipeline.ask_stream('CVD'))

    def test_clarification_saves_assistant_and_resumes_pending_tool(self):
        from langchain_core.messages import AIMessage, HumanMessage
        router = existing.StreamingTests.router
        llm = Mock()
        llm.bind_tools.return_value.invoke.side_effect = [
            SimpleNamespace(tool_calls=[{"name": "query_material_transactions", "args": {"material_code": "MAT-WAF-001"}}]),
            SimpleNamespace(tool_calls=[{"name": "query_material_transactions", "args": {"start_date": "2026-09-01", "end_date": "2026-09-30"}}])]
        llm.stream.return_value = iter([AIMessage(content="查询结果：1条记录")])
        manager = SimpleNamespace(get_context=lambda messages, user_id: messages, prepare_context=lambda messages, user_id, state: (messages, {}))
        with patch.object(router, 'AnswerGenerator', return_value=SimpleNamespace(llm=llm)), patch.object(
                self.graph, 'get_context_manager', return_value=manager), patch.object(
                self.graph, 'classify_intent', side_effect=[{'intent': 'data_query'}, {'intent': 'unclear', 'needs_clarification': True}]), patch.object(
                type(router.DATA_TOOLS['query_material_transactions']), 'invoke', return_value={'total_records': 1}) as invoke, patch.object(
                self.graph, 'get_rag_pipeline') as rag:
            graph = self.graph.build_conversation_graph(InMemorySaver())
            config = {'configurable': {'thread_id': 'clarification-regression'}}
            first = graph.invoke({'messages': [HumanMessage(content='查询MAT-WAF-001流水')], 'user_id': 'test'}, config)
            self.assertEqual(first['status'], 'needs_clarification')
            self.assertEqual(first['pending_tool'], 'query_material_transactions')
            self.assertEqual(first['collected_args'], {'material_code': 'MAT-WAF-001'})
            self.assertIsInstance(first['messages'][-1], AIMessage)
            self.assertIn('开始日期', first['messages'][-1].content)
            invoke.assert_not_called()
            second = graph.invoke({'messages': [HumanMessage(content='2026-09-01至2026-09-30')]}, config)
            self.assertEqual(second['status'], 'success')
            self.assertEqual(len(second['messages']), 4)
            self.assertIsNone(second['pending_intent'])
            self.assertIsNone(second['pending_tool'])
            invoke.assert_called_once_with({'material_code': 'MAT-WAF-001', 'start_date': '2026-09-01', 'end_date': '2026-09-30'})
            rag.assert_not_called()
            history = llm.bind_tools.return_value.invoke.call_args.args[0]
            self.assertTrue(any(isinstance(message, AIMessage) and '开始日期' in message.content for message in history))

    def test_invalid_dates_and_empty_material_ask_before_database(self):
        router = existing.StreamingTests.router
        llm = Mock()
        with patch.object(router, 'AnswerGenerator', return_value=SimpleNamespace(llm=llm)), patch.object(
                type(router.DATA_TOOLS['query_material_transactions']), 'invoke') as invoke:
            for args in [
                {'material_code': 'MAT-WAF-001', 'start_date': 'bad', 'end_date': '2026-09-30'},
                {'material_code': 'MAT-WAF-001', 'start_date': '2026-09-30', 'end_date': '2026-09-01'},
                {'material_code': ' ', 'start_date': '2026-09-01', 'end_date': '2026-09-30'},
            ]:
                llm.bind_tools.return_value.invoke.return_value = SimpleNamespace(tool_calls=[{'name':'query_material_transactions','args':args}])
                self.assertEqual(router.execute_tools('查询流水', 'data_query')['status'], 'needs_clarification')
            invoke.assert_not_called()

if __name__ == '__main__': unittest.main(verbosity=2)
