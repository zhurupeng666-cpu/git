"""Report scope regression tests without database or model connections."""
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch
import test_business_streaming as existing


class ReportScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        existing.StreamingTests.setUpClass()
        cls.router = existing.StreamingTests.router
        cls.report = existing.StreamingTests.report
        from src.tools import material_tools
        cls.tools = material_tools

    def test_statistics_filters_category_in_parameterized_sql(self):
        connection = MagicMock()
        cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [{
            'material_code':'MAT-GAS-001', 'material_name':'氮气', 'category':'specialty_gas',
            'unit':'瓶', 'opening_stock':Decimal('10'), 'total_inbound':Decimal('3'),
            'total_outbound':Decimal('2'), 'closing_stock':Decimal('11')}]
        with patch.object(self.tools.psycopg, 'connect', return_value=connection):
            result = self.tools.generate_material_statistics.invoke({
                'start_date':'2026-08-01', 'end_date':'2026-08-31', 'category':'半导体工艺气体'})
        sql, params = cursor.execute.call_args.args
        self.assertIn('m.category = %s', sql)
        self.assertEqual(params[-4:], ('specialty_gas','specialty_gas',None,None))
        self.assertEqual(result['category'], 'specialty_gas')
        self.assertEqual(result['material_count'], 1)

    def test_statistics_filters_specific_material_in_sql(self):
        connection = MagicMock()
        cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []
        with patch.object(self.tools.psycopg, 'connect', return_value=connection):
            self.tools.generate_material_statistics.invoke({
                'start_date':'2026-08-01', 'end_date':'2026-08-31', 'material_code':'MAT-GAS-001'})
        self.assertEqual(cursor.execute.call_args.args[1][-4:], (None,None,'MAT-GAS-001','MAT-GAS-001'))

    def test_router_transmits_category_to_streamed_report(self):
        llm = Mock()
        llm.bind_tools.return_value.invoke.return_value = SimpleNamespace(tool_calls=[{
            'name':'material_report_tool', 'args':{'start_date':'2026-08-01','end_date':'2026-08-31',
            'report_type':'monthly','scope':'category','category':'半导体工艺气体'}}])
        with patch.object(self.router, 'AnswerGenerator', return_value=SimpleNamespace(llm=llm)), patch.object(
                self.router, 'generate_material_report', return_value='气体月报') as generate:
            result = self.router.execute_tools('工艺气体月报','report_generation',writer=lambda event:None)
        self.assertEqual(result['tool_args']['category'], 'specialty_gas')
        self.assertEqual(generate.call_args.kwargs['scope'], 'category')
        self.assertEqual(generate.call_args.kwargs['category'], 'specialty_gas')

    def test_missing_unknown_or_conflicting_scope_never_generates_report(self):
        llm = Mock()
        base = {'start_date':'2026-08-01','end_date':'2026-08-31','report_type':'monthly'}
        with patch.object(self.router, 'AnswerGenerator', return_value=SimpleNamespace(llm=llm)), patch.object(
                self.router, 'generate_material_report') as generate:
            for more in [{}, {'scope':'category'}, {'scope':'category','category':'不存在的分类'},
                         {'scope':'all','category':'specialty_gas'}, {'scope':'material'}]:
                llm.bind_tools.return_value.invoke.return_value = SimpleNamespace(tool_calls=[{
                    'name':'material_report_tool','args':{**base,**more}}])
                result = self.router.execute_tools('做月报','report_generation',writer=lambda event:None)
                self.assertEqual(result['status'], 'needs_clarification')
            generate.assert_not_called()

    def test_report_prompt_and_statistics_keep_category_scope(self):
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content='半导体工艺气体管理月报')
        with patch.object(self.report, 'AnswerGenerator', return_value=SimpleNamespace(llm=llm)), patch.object(
                type(self.report.generate_material_statistics), 'invoke', return_value={
                    'material_count':1, 'materials':[{'category':'specialty_gas','material_code':'MAT-GAS-001'}]}) as statistics:
            answer = self.report.generate_material_report('2026-08-01','2026-08-31',scope='category',category='specialty_gas')
        statistics.assert_called_once_with({'start_date':'2026-08-01','end_date':'2026-08-31',
                                            'category':'specialty_gas','material_code':None})
        messages = llm.invoke.call_args.args[0]
        self.assertIn('报告材料范围：半导体工艺气体', messages[1].content)
        self.assertNotIn('晶圆材料', messages[0].content)
        self.assertIn('工艺气体', answer)

    def test_empty_category_data_does_not_fall_back_to_all_materials(self):
        with patch.object(type(self.report.generate_material_statistics), 'invoke', return_value={
                'material_count':0,'materials':[]}) as statistics, patch.object(self.report,'AnswerGenerator') as generator:
            events = []
            answer = self.report.generate_material_report('2026-08-01','2026-08-31',on_token=events.append,
                                                         scope='category',category='specialty_gas')
        self.assertIn('没有可用', answer)
        self.assertEqual(events,[answer])
        statistics.assert_called_once()
        generator.assert_not_called()

if __name__ == '__main__': unittest.main()
