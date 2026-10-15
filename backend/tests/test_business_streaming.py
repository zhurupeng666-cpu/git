"""Offline regression tests; no LLM or PostgreSQL requests."""
import asyncio
import importlib
import json
import sys
import types
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from src.memory.memory_manager import requests_long_term


def stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only replace infrastructure; router, tools, report and API code stay real.
        modules = {
            "src.rag.answer_generator": stub("answer", AnswerGenerator=Mock()),
            "src.rag.rag_pipeline": stub("rag", RAGPipeline=Mock()),
            "src.context.context_manager": stub("context", ContextManager=Mock()),
            "src.memory.store": stub("store", create_memory_store=Mock()),
            "src.memory.long_memory": stub("long", LongMemory=Mock()),
            "src.memory.memory_manager": stub("manager", MemoryManager=Mock(), requests_long_term=requests_long_term),
            "src.config.llm": stub("llm", llm=Mock()),
            "src.router.intent_classifier": stub("classifier", classify_intent=Mock()),
        }
        with patch.dict(sys.modules, modules):
            cls.router = importlib.import_module("src.router.intent_router")
            cls.report = importlib.import_module("src.reporting.material_report")
            cls.graph = importlib.import_module("src.router.conversation_graph")
            cls.api = importlib.import_module("src.api.routes")

    def setUp(self):
        self.events = []
        self.llm = Mock()
        self.router_llm = patch.object(self.router, "AnswerGenerator",
                                       return_value=SimpleNamespace(llm=self.llm))
        self.router_llm.start()
        self.addCleanup(self.router_llm.stop)

    def select(self, name, args):
        self.llm.bind_tools.return_value.invoke.return_value = SimpleNamespace(
            tool_calls=[{"name": name, "args": args}])

    def test_memory_store_context_is_entered_and_kept_open(self):
        store = Mock()
        context = Mock()
        context.__enter__ = Mock(return_value=store)
        context.__exit__ = Mock(return_value=False)
        manager = Mock()
        with patch.object(self.graph, "memory_manager", None), patch.object(
                self.graph, "memory_store", None), patch.object(
                self.graph, "create_memory_store", return_value=context), patch.object(
                self.graph, "LongMemory") as long_memory, patch.object(
                self.graph, "MemoryManager", return_value=manager), patch.object(
                self.graph.atexit, "register") as register:
            self.assertIs(self.graph.get_memory_manager(), manager)
            self.assertIs(self.graph.get_memory_manager(), manager)
            context.__enter__.assert_called_once()
            store.setup.assert_called_once()
            long_memory.assert_called_once_with(store)
            context.__exit__.assert_not_called()
            register.assert_called_once()
            register.call_args.args[0]()
            context.__exit__.assert_called_once()

    def test_memory_store_initialization_failure_closes_connection(self):
        store = Mock()
        store.setup.side_effect = RuntimeError("offline setup failure")
        context = Mock()
        context.__enter__ = Mock(return_value=store)
        context.__exit__ = Mock(return_value=False)
        with patch.object(self.graph, "memory_manager", None), patch.object(
                self.graph, "memory_store", None), patch.object(
                self.graph, "create_memory_store", return_value=context):
            with self.assertRaisesRegex(RuntimeError, "offline setup failure"):
                self.graph.get_memory_manager()
            context.__exit__.assert_called_once()
            self.assertIsNone(self.graph.memory_manager)
            self.assertIsNone(self.graph.memory_store)

    def test_data_tokens_arrive_before_generation_finishes(self):
        self.select("query_material_inventory", {"material_code": "MAT-WAF-001"})

        def chunks(messages):
            yield AIMessage(content="库存：")
            self.assertEqual(self.events[-1]["type"], "token")
            self.assertEqual(self.events[-1]["data"]["content"], "库存：")
            yield AIMessage(content="10片")

        self.llm.stream.side_effect = chunks
        with patch.object(type(self.router.DATA_TOOLS["query_material_inventory"]),
                          "invoke", return_value={"total_stock": "10", "unit": "片"}) as invoke:
            result = self.router.execute_tools("查询库存", "data_query", writer=self.events.append)
        invoke.assert_called_once_with({"material_code": "MAT-WAF-001"})
        self.assertEqual(result["answer"], "库存：10片")
        self.assertEqual(self.events[-1], {"type": "source", "data": []})

    def test_missing_parameters_do_not_execute(self):
        self.select("query_material_inventory", {})
        with patch.object(type(self.router.DATA_TOOLS["query_material_inventory"]), "invoke") as invoke:
            result = self.router.execute_tools("查库存", "data_query", writer=self.events.append)
        self.assertEqual(result["status"], "needs_clarification")
        invoke.assert_not_called()

    def test_disallowed_tool_does_not_execute(self):
        self.select("material_report_tool", {})
        result = self.router.execute_tools("查询", "data_query", writer=self.events.append)
        self.assertEqual(result["status"], "invalid_tool")

    def test_ambiguous_selection(self):
        self.llm.bind_tools.return_value.invoke.return_value = SimpleNamespace(tool_calls=[])
        self.assertEqual(self.router.execute_tools("查询", "data_query")["status"],
                         "needs_clarification")

    def test_sync_router_preserves_raw_result(self):
        self.select("query_material_inventory", {"material_code": "MAT-WAF-001"})
        with patch.object(type(self.router.DATA_TOOLS["query_material_inventory"]),
                          "invoke", return_value={"total_stock": "10"}):
            result = self.router.execute_tools("库存", "data_query")
        self.assertEqual(result["result"], {"total_stock": "10"})
        self.llm.stream.assert_not_called()

    def test_report_streams_once_and_accumulates(self):
        self.select("material_report_tool", {
            "start_date": "2026-09-01", "end_date": "2026-09-30", "report_type": "monthly", "scope": "all"
        })

        def chunks(messages):
            yield AIMessage(content="# 月报")
            self.assertEqual(self.events[-1]["data"]["content"], "# 月报")
            yield AIMessage(content="\n统计结果")

        self.llm.stream.side_effect = chunks
        with patch.object(self.report, "AnswerGenerator",
                          return_value=SimpleNamespace(llm=self.llm)), patch.object(
                              type(self.report.generate_material_statistics), "invoke",
                              return_value={"material_count": 1, "materials": [{"material_code": "MAT-WAF-001"}]}) as statistics:
            result = self.router.execute_tools("生成月报", "report_generation",
                                               writer=self.events.append)
        statistics.assert_called_once()
        self.assertEqual(result["answer"], "# 月报\n统计结果")
        self.llm.stream.assert_called_once()
        self.llm.invoke.assert_not_called()

    def test_real_langgraph_stream_and_sync_api(self):
        self.select("query_material_inventory", {"material_code": "MAT-WAF-001"})
        self.llm.stream.return_value = iter([AIMessage(content="库存10片")])
        manager = SimpleNamespace(get_context=lambda messages, user_id: messages, prepare_context=lambda messages, user_id, state: (messages, {}))
        with patch.object(self.graph, "get_context_manager", return_value=manager), patch.object(
                self.graph, "classify_intent", return_value={"intent": "data_query"}), patch.object(
                type(self.router.DATA_TOOLS["query_material_inventory"]), "invoke",
                return_value={"total_stock": "10"}):
            graph = self.graph.build_conversation_graph(InMemorySaver())
            config = {"configurable": {"thread_id": "offline"}}
            state = {"messages": [HumanMessage(content="查询MAT-WAF-001库存")],
                     "user_id": "test-user"}
            events = list(graph.stream(state, config=config, stream_mode="custom"))
            self.assertIn({"type": "token", "data": {"content": "库存10片"}}, events)
            saved = graph.get_state(config).values
            self.assertEqual(saved["reply"]["answer"], "库存10片")
            self.assertEqual(saved["user_id"], "test-user")
            self.llm.stream.return_value = iter([AIMessage(content="库存10片")])
            result = graph.invoke(state, config=config)
            self.assertEqual(result["status"], "success")
            # 接口已要求认证和账号租约，此处仅隔离基础设施，保留实际响应转换。
            user = {"id": uuid4(), "_token_hash": "offline-user"}
            with patch.object(self.api, "chat", return_value=result), patch.object(
                    self.api, "validate_chat_session", return_value=uuid4()), patch.object(
                    self.api, "ActiveRun", return_value=Mock()), patch.object(
                    self.api, "get_connection", return_value=MagicMock()):
                response = self.api.chat_api(self.api.ChatRequest(query="库存"), user=user)
                self.assertEqual(response.answer, "库存10片")

    def frames(self, side_effect):
        # 真实凭证、租约和数据库写入在scripts独立PG测试覆盖；本组只测SSE协议。
        session_id = uuid4()
        connection = MagicMock()
        connection.__enter__.return_value.execute.return_value.fetchone.return_value = {"id": session_id}
        user = {"id": uuid4(), "_token_hash": "offline-sse", "_token_exp": 4102444800}
        with patch.object(self.api, "stream_query", side_effect=side_effect), patch.object(
                self.api, "CancellableStreamingResponse", side_effect=lambda iterator, **kwargs: iterator), patch.object(
                self.api, "ActiveRun", return_value=Mock()), patch.object(
                self.api, "get_connection", return_value=connection), patch.object(
                self.api, "is_stream_session_active", return_value=True):
            frames = list(self.api.chat_stream_sse(self.api.ChatRequest(query="test", thread_id=str(session_id)), user=user))
        return [json.loads(line[6:]) for frame in frames for line in frame.splitlines()
                if line.startswith("data: ")]

    def test_sse_forwards_all_events_including_empty_sources(self):
        events = [{"type": "status", "data": {"stage": "generation"}},
                  {"type": "token", "data": {"content": "甲"}},
                  {"type": "token", "data": {"content": "乙"}},
                  {"type": "source", "data": []}]
        frames = self.frames(lambda **kwargs: iter(events))
        self.assertEqual(frames[1:-1], events)
        self.assertEqual(frames[-1]["type"], "done")

    def test_sse_error_has_one_done(self):
        def broken(**kwargs):
            yield {"type": "token", "data": {"content": "部分"}}
            raise RuntimeError("offline failure")
        frames = self.frames(broken)
        self.assertEqual([f.get("type") for f in frames[1:]], ["token", "error", "done"])

    def test_rag_sources_use_same_protocol(self):
        manager = SimpleNamespace(get_context=lambda messages, user_id: messages, prepare_context=lambda messages, user_id, state: (messages, {}))
        pipeline = Mock()
        pipeline.ask_stream.return_value = iter([
            {"type": "chunk", "content": "答案"},
            {"type": "source", "sources": [{"file_name": "工艺.pdf"}]}
        ])
        with patch.object(self.graph, "get_context_manager", return_value=manager), patch.object(
                self.graph, "RAGPipeline", return_value=pipeline), patch.object(
                self.graph, "get_stream_writer", return_value=self.events.append):
            result = self.graph.handle_query({"intent": "knowledge_qa",
                                             "messages": [HumanMessage(content="工艺")]})
        self.assertEqual(result["reply"]["sources"], [{"file_name": "工艺.pdf"}])
        self.assertEqual(self.events[-1]["type"], "source")


if __name__ == "__main__":
    unittest.main()
