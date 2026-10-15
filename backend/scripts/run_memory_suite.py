"""一次执行四类记忆测试：只新增测试，不修改业务实现；PG使用独立临时schema。"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from src.context.context_manager import ContextManager
from src.context.token_counter import TokenCounter, ContextBudgetExceeded
from src.context.summarizer import ContextSummarizer
from src.memory.memory_manager import MemoryManager
from src.memory.long_memory import LongMemory
from src.rag.answer_generator import AnswerGenerator
from src.router.conversation_recall import get_identifier_answer, is_conversation_request

RESULTS = []


def record(test_id, group, title, function, layer='deterministic_code_test'):
    started = time.monotonic()
    try:
        evidence = function() or {}
        status = 'PASS'
        if isinstance(evidence, dict) and '_status' in evidence:
            status = evidence.pop('_status')
    except AssertionError as exc:
        status, evidence = 'FAIL', {'reason': str(exc)}
    except Exception as exc:
        status, evidence = 'BLOCKED', {'reason': f'{type(exc).__name__}: {exc}'}
    RESULTS.append({'id': test_id, 'group': group, 'title': title, 'layer': layer,
                    'status': status, 'seconds': round(time.monotonic() - started, 3), 'evidence': evidence})
    print(f'{test_id} {status} {title}', flush=True)


def require(condition, reason, **evidence):
    assert condition, reason + (' | ' + json.dumps(evidence, ensure_ascii=False) if evidence else '')
    return evidence


def manager(max_tokens=4000, profile=None, summary='摘要：旧事实保留。'):
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=summary)
    memory = SimpleNamespace(get_user_memory=lambda user_id: profile or {})
    return ContextManager(llm, memory_manager=memory, max_tokens=max_tokens)


def chars(messages):
    return sum(len(message.content) for message in messages if isinstance(message.content, str))


def offline_tests():
    query = '我之前告诉你的CVD故障排查内部测试编号是什么？请只回答编号。'
    history = [HumanMessage(content='本次CVD故障排查内部测试编号是CVD-MEM-A81X，请记住。')]
    record('ST01', '短期记忆', '精确回忆用户编号', lambda: require(get_identifier_answer(query, history) == 'CVD-MEM-A81X', '编号不一致'))
    record('ST02', '短期记忆', '同字段更正后取最新值', lambda: require(get_identifier_answer(query, history + [HumanMessage(content='CVD故障排查内部测试编号更正为CVD-MEM-B72Y，请记住。')]) == 'CVD-MEM-B72Y', '未取更正后的编号'))
    record('ST03', '短期记忆', '不把助手编造编号当作用户事实', lambda: require(get_identifier_answer(query, [AIMessage(content='CVD故障排查内部测试编号是CVD-FAKE-000。')]) == '不知道', '误用助手内容'))
    record('ST04', '短期记忆', '没有历史时不猜测编号', lambda: require(get_identifier_answer(query, []) == '不知道', '无历史仍给出编号'))
    record('ST05', '短期记忆', '技术追问保持知识问答路由', lambda: require(not is_conversation_request('针对你刚才提到的第二个部件，应该怎么进一步排查？'), '技术追问误入纯回忆'))
    assistant_history = [HumanMessage(content='请列出部件。'), AIMessage(content='背景说明。' * 120 + '\n1. 压力规\n2. 节流阀\n3. 真空泵')]
    record('ST06', '短期记忆', '助手长回答中的编号部件进入答案历史', lambda: require('2. 节流阀' in AnswerGenerator._build_history(None, assistant_history), '助手历史被截断'))

    record('LT01', '长期记忆', '长期资料进入上下文管理器', lambda: require('中文表格' in manager(profile={'preference': '中文表格'}).get_context([HumanMessage(content='你好')], user_id='synthetic')[0].content, '长期信息未注入'))
    def long_in_final():
        context = manager(profile={'preference': '长期偏好标记LONG-82A'}).get_context([HumanMessage(content=f'第{i}条消息') for i in range(10)], user_id='synthetic')
        payload = AnswerGenerator._build_history(None, context)
        return require('LONG-82A' in payload, '长期记忆在最终答案历史构建时丢失', context_messages=len(context), final_contains_long_memory='LONG-82A' in payload)
    record('LT02', '长期记忆', '长期信息抵达RAG答案生成输入', long_in_final)

    counter = TokenCounter(max_tokens=4000)
    record('TK01', 'Token控制', '计数器区分Token和字符', lambda: require(counter.count_text_tokens('hello world') != len('hello world'), '当前返回len(text)，不是模型Token计数', counted=counter.count_text_tokens('hello world'), characters=11))
    for number, delta in enumerate((-1, 0, 1), start=2):
        def boundary(offset=delta):
            messages = [HumanMessage(content="半导体工艺设备维护" * 10)]
            tokens = counter.count_messages_tokens(messages)
            limited = TokenCounter(max_tokens=tokens + offset)
            return require(limited.exceed_limit(messages) == (offset < 0), "Token边界判断不符", counted_tokens=tokens, configured_budget=tokens+offset)
        record(f'TK{number:02}', 'Token控制', f'Token预算上下边界{delta:+}', boundary)
    def few_long():
        try:
            context = manager().get_context([HumanMessage(content='测' * 3000), AIMessage(content='试' * 3000)])
        except ContextBudgetExceeded as exc:
            return {'controlled_rejection': str(exc)}
        tokens = counter.count_messages_tokens(context)
        return require(tokens <= 4000, '少于10条时仍超出Token预算', counted_tokens=tokens, configured_limit=4000)
    record('TK05', 'Token控制', '少量超长消息仍受预算约束', few_long)
    def long_summary():
        context = manager(summary='摘要' * 2000).get_context([HumanMessage(content='测' * 500) for _ in range(14)])
        return require(counter.count_messages_tokens(context) <= 4000, '摘要加最近消息没有再次校验预算', counted_tokens=counter.count_messages_tokens(context), configured_limit=4000)
    record('TK06', 'Token控制', '摘要与最近消息合并后受预算约束', long_summary)
    def long_profile():
        context = manager(profile={'note': '测' * 4500}).get_context([HumanMessage(content='普通问题')], user_id='synthetic')
        return require(counter.count_messages_tokens(context) <= 4000, '长期资料不计入总预算', counted_tokens=counter.count_messages_tokens(context), configured_limit=4000)
    record('TK07', 'Token控制', '长期资料计入上下文预算', long_profile)
    def recall_budget():
        from src.router import conversation_graph as graph
        captured = []
        fake = Mock()
        def stream(prompt):
            captured.extend(prompt)
            yield AIMessageChunk(content='合成回顾')
        fake.stream.side_effect = stream
        fake.invoke.return_value = AIMessage(content='合成历史摘要')
        messages = [HumanMessage(content='合成历史' * 200) for _ in range(20)] + [HumanMessage(content='回顾我们刚才的对话。')]
        with patch.object(graph, 'llm', fake), patch.object(graph, 'get_stream_writer', return_value=lambda event: None):
            graph.handle_conversation_memory(messages)
        return require(counter.count_messages_tokens(captured[1:]) <= 4000, '回顾历史超出历史预算', sent_history_tokens=counter.count_messages_tokens(captured[1:]), sent_messages=len(captured))
    record('TK08', 'Token控制', '通用回忆分支遵守上下文预算', recall_budget)
    def request_budget():
        from src.context.request_budget import validate_request_budget
        small = validate_request_budget({'messages': [{'role': 'user', 'content': '合成问题'}]})
        try:
            validate_request_budget({'messages': [{'role': 'user', 'content': '短问题'}], 'tools': [{'description': '超长工具Schema' * 20000}]})
        except ContextBudgetExceeded:
            return {'small_payload': small, 'oversized_tool_schema_rejected': True}
        raise AssertionError('完整请求未计入工具Schema')
    record('TK09', 'Token控制', '完整请求计入工具Schema与输出预留', request_budget)


    messages = [HumanMessage(content=f'窗口消息-{i}') for i in range(14)]
    record('SW01', '滑动窗口', '不足10条时保持消息顺序', lambda: require(manager().get_context(messages[:6]) == messages[:6], '短历史顺序或数量变化'))
    record('SW02', '滑动窗口', '超过10条时保留最近10条原文', lambda: require([message for message in manager().get_context(messages) if isinstance(message, HumanMessage)] == messages[-10:], '最近窗口不正确'))
    record('SW03', '滑动窗口', '窗口操作不修改原始消息列表', lambda: require((manager().get_context(messages) is not messages) and len(messages) == 14, '修改了原始历史'))
    heavy = [HumanMessage(content=f'旧事实-{i}：' + '测' * 400) for i in range(14)]
    def summary_scope():
        context_manager = manager(summary='摘要标记SUM-71B')
        context = context_manager.get_context(heavy)
        sent = '\n'.join(call.args[0][1].content for call in context_manager.summarizer.llm.invoke.call_args_list)
        return require('旧事实-0：' in sent and '旧事实-3：' in sent and '旧事实-4：' not in sent and any('SUM-71B' in message.content for message in context), '摘要范围不等于窗口外历史')
    record('SW04', '滑动窗口', '仅摘要窗口外历史', summary_scope)
    record('SW05', '滑动窗口', '摘要出现在有效上下文', lambda: require(any('SUM-71B' in message.content for message in manager(summary='SUM-71B').get_context(heavy)), '上下文缺少摘要'))
    def summary_final():
        context = manager(summary='旧事实SUM-71B：已排除压力规故障。').get_context(heavy)
        payload = AnswerGenerator._build_history(None, context)
        return require('SUM-71B' in payload, '答案生成仅取最后6条，摘要未进入最终历史', context_messages=len(context), final_contains_summary='SUM-71B' in payload)
    record('SW06', '滑动窗口', '摘要抵达最终答案输入', summary_final)
    def effective_window():
        context = manager().get_context(messages)
        payload = AnswerGenerator._build_history(None, context)
        return require(all(message.content in payload for message in messages[-10:]), '管理器10条窗口在答案生成处变成6条', configured_window=10, raw_recent_messages_in_final=sum(message.content in payload for message in messages[-10:] ))
    record('SW07', '滑动窗口', '配置的10条窗口贯穿答案生成', effective_window)
    def repeated_summary():
        context_manager = manager()
        all_messages = [HumanMessage(content=f'增量事实-{number:03}：' + '测' * 100) for number in range(24)]
        _, state = context_manager.prepare_context(all_messages[:20])
        first_calls = context_manager.summarizer.llm.invoke.call_count
        _, next_state = context_manager.prepare_context(all_messages, state=state)
        added_calls = context_manager.summarizer.llm.invoke.call_args_list[first_calls:]
        new_inputs = '\n'.join(call.args[0][1].content for call in added_calls)
        require('增量事实-000：' not in new_inputs and '增量事实-010：' in new_inputs, '重复发送了已摘要的原文')
        count = context_manager.summarizer.llm.invoke.call_count
        context_manager.prepare_context(all_messages, state=next_state)
        return require(context_manager.summarizer.llm.invoke.call_count == count and next_state['summary_message_count'] == 14,
                       '相同历史重复触发摘要', first_summary_count=state['summary_message_count'], next_summary_count=next_state['summary_message_count'])
    record('SW08', '滑动窗口', '新增旧消息增量摘要且相同历史不重复摘要', repeated_summary)


def pg_worker(mode, schema, spec, target):
    import psycopg
    from psycopg.rows import dict_row
    from langgraph.store.postgres import PostgresStore
    from src.router import conversation_graph as graph
    from src.router.conversation_memory import GuardedPostgresSaver, create_thread_config
    assert re.fullmatch(r'memory_suite_[a-f0-9]{32}', schema)
    url = os.environ['DATABASE_URL']
    options = f'-c search_path={schema} -c statement_timeout=15000'
    @contextmanager
    def checkpoint():
        with psycopg.connect(url, autocommit=True, row_factory=dict_row, connect_timeout=5, options=options) as conn:
            saver = GuardedPostgresSaver(conn)
            saver.setup()
            yield saver
    def ask(text, thread):
        events = list(graph.stream_query(text, thread_id=thread, user_id=thread.split(':')[0]))
        return ''.join(event['data']['content'] for event in events if event.get('type') == 'token')
    def forbid(*args, **kwargs):
        raise AssertionError('精确记忆不应调用外部模型或知识库')
    query = '我之前告诉你的CVD故障排查内部测试编号是什么？请只回答编号。'
    result = {'pid': os.getpid(), 'stage': mode}
    with psycopg.connect(url, autocommit=True, row_factory=dict_row, connect_timeout=5, options=options) as conn:
        store = PostgresStore(conn)
        store.setup()
        memory = MemoryManager(LongMemory(store))
        with patch.object(graph, 'open_conversation_memory', checkpoint), patch.object(graph, 'get_rag_pipeline', side_effect=forbid), \
             patch.object(graph, 'classify_intent', side_effect=forbid), patch.object(graph, 'get_context_manager', side_effect=forbid), \
             patch.object(graph, 'get_memory_manager', return_value=memory):
            if mode == 'write':
                result['remember_ack'] = ask(f'本次CVD故障排查内部测试编号是{spec["code"]}，请记住。', spec['thread'])
                result['exact_before_restart'] = ask(query, spec['thread'])
                result['correction_ack'] = ask(f'CVD故障排查内部测试编号更正为{spec["new_code"]}，请记住。', spec['thread'])
                for number in range(12):
                    ask(f'请记住：这是第{number}轮合成临时备注。', spec['thread'])
                result['exact_after_window'] = ask(query, spec['thread'])
                result['natural_long_term_ack'] = ask('请长期记住：我的回答偏好是中文表格，后续新会话也要使用。', spec['thread'])
                result['profile_after_natural_request'] = memory.get_user_memory(spec['user'])
                memory.save_user_memory(spec['user'], {'language': '中文', 'format': '表格', 'marker': spec['marker']})
                result['profile_saved'] = memory.get_user_memory(spec['user'])
                memory.save_user_memory(spec['user'], {'language': '中文', 'format': '条目', 'marker': spec['marker']})
                result['profile_updated'] = memory.get_user_memory(spec['user'])
                context_manager = ContextManager(llm=Mock(), memory_manager=memory)
                context_manager.summarizer.llm.invoke.return_value = AIMessage(content='持久化增量摘要标记SUMMARY-PG')
                pipeline = SimpleNamespace(ask_stream=lambda **kwargs: iter([{'type':'chunk','content':'合成技术回顾'}]))
                with patch.object(graph, 'get_context_manager', return_value=context_manager), patch.object(graph, 'classify_intent', return_value={'intent':'knowledge_qa'}), patch.object(graph, 'get_rag_pipeline', return_value=pipeline):
                    ask('请结合当前对话说明检查情况。', spec['thread'])
            else:
                result['exact_after_restart'] = ask(query, spec['thread'])
                result['new_session_answer'] = ask(query, f'{spec["user"]}:{uuid4()}')
                result['other_user_answer'] = ask(query, f'{spec["other_user"]}:{spec["session"]}')
                result['profile_after_restart'] = memory.get_user_memory(spec['user'])
                result['other_user_profile'] = memory.get_user_memory(spec['other_user'])
                result['new_session_long_preference'] = ask('我的长期回答偏好是什么？', f'{spec["user"]}:{uuid4()}')
                result['other_user_long_preference'] = ask('我的长期回答偏好是什么？', f'{spec["other_user"]}:{uuid4()}')
                result['implicit_preference'] = ask('我的回答偏好是什么？', f'{spec["user"]}:{uuid4()}')
                result['implicit_other_user'] = ask('我的回答偏好是什么？', f'{spec["other_user"]}:{uuid4()}')
                with checkpoint() as saver:
                    compiled = graph.build_conversation_graph(saver)
                    root_config = create_thread_config(spec['thread'])
                    # get_state的非空checkpoint_ns表示子图定位；本测试检查根图状态。
                    root_config['configurable'].pop('checkpoint_ns', None)
                    state = compiled.get_state(root_config).values
                    result['persisted_summary'] = state.get('context_summary')
                    result['persisted_summary_count'] = state.get('summary_message_count', 0)
                    history = state.get('messages', [])
                    before = len(history)
                    manager(max_tokens=100000).get_context(history)
                    after = compiled.get_state(root_config).values.get('messages', [])
                    result['raw_messages_before_window'] = before
                    result['raw_messages_after_window'] = len(after)
                    result['original_fact_in_checkpoint'] = any(spec['code'] in message.content for message in after)
    Path(target).write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')


def subprocess_worker(arguments, target, timeout):
    log = Path(target).with_suffix('.log')
    with log.open('w', encoding='utf-8') as output:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *arguments], stdout=output, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # 只结束本函数创建的测试进程树，不触碰现有后端。
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
            raise RuntimeError(f'测试子进程达到{timeout}秒诊断预算；见{log.name}')
    if code != 0:
        raise RuntimeError(f'测试子进程退出{code}；见{log.name}')
    return json.loads(Path(target).read_text(encoding='utf-8'))


def database_tests(folder):
    import psycopg
    from psycopg import sql
    schema = 'memory_suite_' + uuid4().hex
    spec = {'user': str(uuid4()), 'other_user': str(uuid4()), 'session': str(uuid4()),
            'code': 'CVD-MEM-' + uuid4().hex[:8].upper(), 'new_code': 'CVD-NEW-' + uuid4().hex[:8].upper(),
            'marker': 'PROFILE-' + uuid4().hex[:8]}
    spec['thread'] = spec['user'] + ':' + spec['session']
    spec_file = folder / 'synthetic_test_inputs.json'
    spec_file.write_text(json.dumps(spec, ensure_ascii=False), encoding='utf-8')
    with psycopg.connect(os.environ['DATABASE_URL'], autocommit=True, connect_timeout=5) as admin:
        admin.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        try:
            written = subprocess_worker(['--worker', 'write', '--schema', schema, '--spec', str(spec_file), '--result', str(folder/'pg_write.json')], folder/'pg_write.json', 180)
            loaded = subprocess_worker(['--worker', 'read', '--schema', schema, '--spec', str(spec_file), '--result', str(folder/'pg_read.json')], folder/'pg_read.json', 180)
            checks = [
                ('PG01', '短期记忆', '真实PG同会话精确回忆', written['exact_before_restart'] == spec['code'], written['exact_before_restart']),
                ('PG02', '短期记忆', '跨窗口后仍精确回忆最新编号', written['exact_after_window'] == spec['new_code'], written['exact_after_window']),
                ('PG03', '短期记忆', '独立新进程重建图后从PG回忆', loaded['pid'] != written['pid'] and loaded['exact_after_restart'] == spec['new_code'], loaded['exact_after_restart']),
                ('PG04', '短期记忆', '同账号新会话不读取旧会话临时编号', loaded['new_session_answer'] == '不知道', loaded['new_session_answer']),
                ('PG05', '短期记忆', '其他账号使用相同会话尾号仍隔离', loaded['other_user_answer'] == '不知道', loaded['other_user_answer']),
                ('PG06', '滑动窗口', '裁剪上下文不删除PG原始历史', loaded['raw_messages_before_window'] == loaded['raw_messages_after_window'] and loaded['raw_messages_before_window'] > 10 and loaded['original_fact_in_checkpoint'], loaded['raw_messages_after_window']),
                ('PG07', '长期记忆', '真实PostgresStore写入并读取', written['profile_saved'].get('marker') == spec['marker'], written['profile_saved']),
                ('PG08', '长期记忆', '长期资料更新', written['profile_updated'].get('format') == '条目', written['profile_updated']),
                ('PG09', '长期记忆', '新进程读取长期资料', loaded['profile_after_restart'].get('marker') == spec['marker'] and loaded['profile_after_restart'].get('format') == '条目', loaded['profile_after_restart']),
                ('PG10', '长期记忆', '长期资料账号隔离', loaded['other_user_profile'] == {}, loaded['other_user_profile']),
                ('PG11', '长期记忆', '聊天请求长期记住后实际写入profile', written['profile_after_natural_request'].get('format') == '表格', {'reply': written['natural_long_term_ack'], 'saved_profile': written['profile_after_natural_request']}),
                ('PG12', '滑动窗口', '增量摘要状态跨进程持久化', 'SUMMARY-PG' in (loaded['persisted_summary'] or '') and loaded['persisted_summary_count'] > 0, loaded['persisted_summary_count']),
                ('PG13', '长期记忆', '新会话通过聊天读取长期偏好', '条目' in loaded['new_session_long_preference'] and '中文' in loaded['new_session_long_preference'], loaded['new_session_long_preference']),
                ('PG14', '长期记忆', '其他账号聊天不能读取长期偏好', loaded['other_user_long_preference'] == '不知道', loaded['other_user_long_preference']),
                ('PG15', '长期记忆', '新会话自然问法读取已保存偏好', '条目' in loaded['implicit_preference'], loaded['implicit_preference']),
                ('PG16', '长期记忆', '自然问法仍保持账号隔离', loaded['implicit_other_user'] == '不知道', loaded['implicit_other_user'])
            ]
            for test_id, group, title, condition, evidence in checks:
                record(test_id, group, title, lambda c=condition,e=evidence: require(c, '实际结果不符合通过标准', actual=e), 'real_postgresql_isolated_schema')
        finally:
            # schema为本轮刚创建的固定前缀随机名，绝不清理public或其他测试空间。
            assert re.fullmatch(r'memory_suite_[a-f0-9]{32}', schema)
            admin.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
            exists = admin.execute('SELECT 1 FROM pg_namespace WHERE nspname=%s', (schema,)).fetchone()
            require(exists is None, '本轮临时schema清理失败')


def live_worker(target):
    from src.router import conversation_graph as graph
    from src.config.llm import llm
    bounded_llm = llm.bind(timeout=60)
    history = [HumanMessage(content='请记住这个合成列表：第一个部件是压力规，第二个部件是节流阀，第三个部件是真空泵。'),
               AIMessage(content='已记录三个部件及顺序。'),
               HumanMessage(content='回顾我们刚才的对话，第二个部件是什么？请只回答部件名称。')]
    with patch.object(graph, 'llm', bounded_llm), patch.object(graph, 'get_stream_writer', return_value=lambda event: None):
        answer = graph.handle_conversation_memory(history)['reply']['answer']
    marker = 'CVD-SYN-' + uuid4().hex[:8].upper()
    summary = ContextSummarizer(bounded_llm).summarize([
        HumanMessage(content=f'合成测试编号是{marker}。已经排除压力规故障；尚未确认节流阀故障；已完成过滤器清理。'),
        AIMessage(content='继续检查节流阀，故障原因尚未确认。')])
    Path(target).write_text(json.dumps({'answer': answer, 'summary': summary, 'marker': marker}, ensure_ascii=False), encoding='utf-8')


def main(folder):
    folder.mkdir(parents=True, exist_ok=True)
    offline_tests()
    try:
        database_tests(folder)
    except Exception as exc:
        record('PG-BLOCK', '数据库测试', '真实PG测试执行环境', lambda: {'_status': 'BLOCKED', 'reason': f'{type(exc).__name__}: {exc}'})
    live_tests(folder)

    write_report(folder)


def live_tests(folder):
    try:
        live = subprocess_worker(['--live-worker', '--result', str(folder/'live_model.json')], folder/'live_model.json', 180)
        record('REAL01', '短期记忆', '真实配置模型理解助手/用户历史中的顺序', lambda: require('节流阀' in live['answer'], '模型未正确回忆第二个部件', answer=live['answer']), 'real_external_model_synthetic_facts')
        record('REAL02', '滑动窗口', '真实摘要保留精确测试编号', lambda: require(live['marker'] in live['summary'], '真实摘要未保留精确编号', summary=live['summary']), 'real_external_model_synthetic_facts')
        record('REAL03', '滑动窗口', '真实摘要不把待确认原因改成已确认', lambda: {'_status': 'REVIEW', 'reason': '语义正确性需人工核对，不以关键词出现代替准确率。', 'source_facts': '压力规已排除；节流阀故障未确认；过滤器已清理。', 'summary': live['summary']}, 'real_external_model_synthetic_facts')
    except Exception as exc:
        record('REAL-BLOCK', '真实模型', '真实模型测试执行环境', lambda: {'_status': 'BLOCKED', 'reason': f'{type(exc).__name__}: {exc}'})


def write_report(folder):
    counts = {status: sum(row['status'] == status for row in RESULTS) for status in ['PASS','FAIL','GAP','BLOCKED','REVIEW']}
    report = {'scope': '后端记忆、上下文、真实隔离PG、独立进程重启、合成事实的真实模型调用',
              'not_tested': ['浏览器刷新按钮和线上真实登录操作', '生产账号权限绕过攻击', '模型完整上下文容量极限'],
              'business_code_modified': False, 'counts': counts, 'results': RESULTS}
    (folder/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 记忆与上下文一次性测试报告\n', '\n本报告使用真实业务代码；不修复业务实现。PG数据仅在本轮临时schema写入，结束后清理。\n',
             '\nPASS不等于整个功能通过，特别是字符边界检查不代表Token计数准确。\n',
             '\n统计：' + json.dumps(counts, ensure_ascii=False) + '\n',
             '\n| 编号 | 分类 | 状态 | 检查 | 证据 |\n|---|---|---|---|---|\n']
    for row in RESULTS:
        evidence = json.dumps(row['evidence'], ensure_ascii=False).replace('|', '\\|').replace('\n',' ')[:1600]
        lines.append(f'| {row["id"]} | {row["group"]} | {row["status"]} | {row["title"]} | {evidence} |\n')
    lines.append('\n未覆盖浏览器实际刷新与线上登录操作；进程重启通过两个独立Python测试进程验证，不重启现有后端。\n')
    (folder/'report.md').write_text(''.join(lines), encoding='utf-8')
    print(json.dumps({'counts': counts, 'report': str(folder/'report.md')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=['write','read'])
    parser.add_argument('--live-worker', action='store_true')
    parser.add_argument('--resume-pg', action='store_true')
    parser.add_argument('--resume-live', action='store_true')
    parser.add_argument('--schema')
    parser.add_argument('--spec')
    parser.add_argument('--result')
    parser.add_argument('--report-dir', default=str(ROOT/'logs'/'memory_suite_latest'))
    args = parser.parse_args()
    if args.worker:
        pg_worker(args.worker, args.schema, json.loads(Path(args.spec).read_text(encoding='utf-8')), args.result)
    elif args.live_worker:
        live_worker(args.result)
    elif args.resume_live:
        folder = Path(args.report_dir)
        previous = json.loads((folder/'report.json').read_text(encoding='utf-8'))
        RESULTS.extend(row for row in previous['results'] if not row['id'].startswith('REAL'))
        live_tests(folder)
        write_report(folder)
    elif args.resume_pg:
        folder = Path(args.report_dir)
        previous = json.loads((folder/'report.json').read_text(encoding='utf-8'))
        RESULTS.extend(row for row in previous['results'] if not row['id'].startswith('PG'))
        database_tests(folder)
        write_report(folder)
    else:
        main(Path(args.report_dir))
