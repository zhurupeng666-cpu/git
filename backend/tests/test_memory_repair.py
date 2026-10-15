"""记忆修复的独立回归：预算在HTTP前执行、摘要增量、显式授权和写入取消。"""
import json
import unittest
from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch
import httpx
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.context.token_counter import TokenCounter, ContextBudgetExceeded
from src.context.context_manager import ContextManager
from src.context.summarizer import ContextSummarizer
from src.context.request_budget import BudgetedChatOpenAI
from src.memory.memory_manager import MemoryManager
from src.memory.long_memory import LongMemory
from src.api.stream_control import StreamCancellation, StreamCancelled, cancellation_scope


class MemoryRepairTests(unittest.TestCase):
    def manager(self):
        llm = Mock()
        llm.invoke.return_value = AIMessage(content='合成摘要：编号保留，原因未确认。')
        return ContextManager(llm)

    def test_tokenizer_counts_tokens_and_preserves_unicode_chunks(self):
        counter = TokenCounter()
        self.assertEqual(counter.count_text_tokens('hello world'), 2)
        text = '半导体🧪CVD-123' * 200
        pieces = list(counter.split_text(text, 100))
        self.assertEqual(''.join(pieces), text)
        self.assertTrue(all(counter.count_text_tokens(piece) <= 100 for piece in pieces))

    def test_current_question_not_truncated_under_budget(self):
        manager = self.manager()
        current = HumanMessage(content='当前设备型号是CVD-CURRENT-987，如何检查？')
        context = manager.get_context([AIMessage(content='长回答' * 5000), current])
        self.assertEqual(context[-1].content, current.content)
        self.assertLessEqual(manager.get_token_count(context), 4000)

    def test_oversized_current_question_rejected(self):
        with self.assertRaises(ContextBudgetExceeded):
            self.manager().get_context([HumanMessage(content='测试' * 10000)])

    def test_incremental_summary_input_is_bounded(self):
        manager = self.manager()
        messages = [HumanMessage(content=f'独立事实-{index}：' + '测试' * 600) for index in range(15)]
        context, state = manager.prepare_context(messages)
        for call in manager.summarizer.llm.invoke.call_args_list:
            self.assertLess(manager.token_counter.count_messages_tokens(call.args[0]), 3500)
            self.assertEqual(call.kwargs['max_tokens'], 600)
        calls = manager.summarizer.llm.invoke.call_count
        manager.prepare_context(messages, state=state)
        self.assertEqual(manager.summarizer.llm.invoke.call_count, calls)
        self.assertLessEqual(manager.get_token_count(context), 4000)

    def test_edited_history_invalidates_summary(self):
        manager = self.manager()
        messages = [HumanMessage(content=f'原始事实-{index}') for index in range(15)]
        _, state = manager.prepare_context(messages)
        count = manager.summarizer.llm.invoke.call_count
        messages[0] = HumanMessage(content='更正后的旧事实CORRECTED')
        manager.prepare_context(messages, state=state)
        added = manager.summarizer.llm.invoke.call_args_list[count:]
        self.assertTrue(any('CORRECTED' in call.args[0][1].content for call in added))

    def test_empty_summary_does_not_advance_state(self):
        manager = self.manager()
        manager.summarizer.llm.invoke.return_value = AIMessage(content='')
        with self.assertRaisesRegex(RuntimeError, '摘要'):
            manager.prepare_context([HumanMessage(content=f'事实-{i}') for i in range(15)])

    def test_ordinary_remember_does_not_persist(self):
        store = Mock()
        manager = MemoryManager(store)
        self.assertFalse(manager.save_requested_memory('user-a', '请记住本次测试编号是CVD-TEST-1。'))
        self.assertFalse(manager.save_requested_memory('user-a', '不要长期记住：我的回答偏好是中文表格。'))
        store.update_memory.assert_not_called()

    def test_explicit_long_term_save_preserves_scope_and_fields(self):
        store = Mock()
        manager = MemoryManager(store)
        self.assertTrue(manager.save_requested_memory('user-a', '请长期记住：我的回答偏好是中文表格，设备型号是CVD-2026。'))
        values = store.update_memory.call_args.kwargs
        self.assertEqual(values['user_id'], 'user-a')
        self.assertEqual(values['updates']['format'], '表格')
        self.assertEqual(values['updates']['equipment_model'], 'CVD-2026')

    def test_profile_updates_preserve_other_fields_and_bound_facts(self):
        store = Mock()
        store.get.return_value = SimpleNamespace(value={'language': '中文', 'facts': [str(i) for i in range(12)]})
        LongMemory(store).update_memory('user-a', 'profile', {'format': '表格', 'facts': ['new-fact']})
        namespace, key, value = store.put.call_args.args
        self.assertEqual(namespace, ('semiconductor', 'user-a'))
        self.assertEqual(value['language'], '中文')
        self.assertEqual(len(value['facts']), 12)
        self.assertEqual(value['facts'][-1], 'new-fact')

    def test_revoked_run_cannot_write_long_term(self):
        store = Mock()
        token = StreamCancellation()
        token.run = Mock()
        token.run.guard_write.side_effect = StreamCancelled('synthetic revoked lease')
        with patch('src.api.run_control.get_run_connection', return_value=MagicMock()), cancellation_scope(token):
            with self.assertRaises(StreamCancelled):
                LongMemory(store).save_memory('user-a', 'profile', {'format': '表格'})
        store.put.assert_not_called()

    def test_complete_budget_checked_before_http(self):
        sent = []
        def respond(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={'id': 'synthetic', 'object': 'chat.completion', 'created': 1,
                'model': 'synthetic', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '合成回答'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8}})
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            model = BudgetedChatOpenAI(model='synthetic', api_key='synthetic', base_url='https://example.invalid/v1', http_client=client, max_retries=0)
            self.assertEqual(model.invoke('合成短问题').content, '合成回答')
            self.assertEqual(sent[0]['max_tokens'], 2048)
            with self.assertRaises(ContextBudgetExceeded):
                model.invoke('短问题', tools=[{'type': 'function', 'function': {'name': 'test', 'description': '巨大工具说明' * 20000}}])
            self.assertEqual(len(sent), 1)

    def test_output_reservation_cannot_exceed_configured_cap(self):
        model = BudgetedChatOpenAI(model='synthetic', api_key='synthetic')
        self.assertEqual(model._get_request_payload('合成问题', max_tokens=99999)['max_tokens'], 2048)
        with self.assertRaises(ValueError):
            model._get_request_payload('合成问题', max_tokens=0)

    def test_sync_budget_error_is_413_not_500(self):
        import test_business_streaming as existing
        from fastapi import HTTPException
        existing.StreamingTests.setUpClass()
        api = existing.StreamingTests.api
        user = {'id': uuid4(), '_token_hash': 'synthetic-budget-user'}
        with patch.object(api, 'validate_chat_session', return_value=uuid4()), \
             patch.object(api, 'ActiveRun', return_value=Mock()), \
             patch.object(api, 'chat', side_effect=ContextBudgetExceeded('synthetic budget rejection')):
            with self.assertRaises(HTTPException) as error:
                api.chat_api(api.ChatRequest(query='synthetic'), user=user)
        self.assertEqual(error.exception.status_code, 413)

    def test_sse_budget_error_has_explicit_code_and_done(self):
        import test_business_streaming as existing
        existing.StreamingTests.setUpClass()
        helper = existing.StreamingTests()
        frames = helper.frames(ContextBudgetExceeded('synthetic budget rejection'))
        errors = [frame for frame in frames if frame.get('type') == 'error']
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['data']['code'], 'context_budget_exceeded')
        self.assertEqual(sum(frame.get('type') == 'done' for frame in frames), 1)

    def test_classifier_budget_keeps_structured_choice_contract(self):
        import importlib
        import os
        module = importlib.import_module('src.router.intent_classifier')
        client = MagicMock()
        client.__enter__.return_value = client
        client.system_one.return_value = SimpleNamespace(answers={'intent': SimpleNamespace(
            choice='knowledge_qa', confidence=.99, probabilities={'knowledge_qa': .99})})
        with patch.object(module, 'TypeSafeClient', return_value=client), patch.dict(os.environ, {'OPENROUTER_API_KEY': 'synthetic'}):
            result = module.classify_intent('合成测试：CVD设备如何检查？', context=[])
        self.assertEqual(result['intent'], 'knowledge_qa')
        self.assertIn('intent', client.system_one.call_args.kwargs['questions'])

    def test_large_fact_list_does_not_hide_structured_preferences(self):
        manager = self.manager()
        manager.memory_manager = SimpleNamespace(get_user_memory=lambda user_id: {
            'facts': ['合成长事实' * 1000 for _ in range(12)], 'format': '表格', 'language': '中文'})
        context = manager.get_context([HumanMessage(content='合成技术问题')], user_id='synthetic')
        self.assertIn('中文', context[0].content)
        self.assertIn('表格', context[0].content)
        self.assertLessEqual(manager.get_token_count(context), 4000)

    def test_tight_budget_preserves_both_prefix_kinds(self):
        manager = ContextManager(Mock(), max_tokens=200)
        context = manager.fit_context([SystemMessage(content='长期资料：' + '资料' * 100),
                                       SystemMessage(content='历史摘要：' + '事实' * 100)],
                                      [HumanMessage(content='hello ' * 100)])
        self.assertEqual(sum(isinstance(message, SystemMessage) for message in context), 2)
        self.assertLessEqual(manager.get_token_count(context), 200)


if __name__ == '__main__':
    unittest.main(verbosity=2)
