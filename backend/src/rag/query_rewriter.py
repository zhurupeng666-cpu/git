import os

from dotenv import load_dotenv

from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

from src.config.llm import llm


load_dotenv()



class QueryRewriter:


    def __init__(self):

        self.llm = llm



    def rewrite(
            self,
            query: str,
            context=None
    ):
        """
        根据当前问题和历史上下文生成RAG检索问题。

        :param query:
            当前用户问题

        :param context:
            LangGraph短期历史消息

        :return:
            改写后的检索问题
        """


        system_prompt = """
你是一个半导体企业知识库的检索查询改写器。

你的任务不是回答用户问题，
而是结合历史对话，
把用户问题改写成更适合 RAG 检索的查询语句。


知识库可能包含以下领域：

- 半导体基础知识
- 晶圆制造工艺
- 光刻
- 刻蚀
- CVD
- CMP
- 半导体设备维护
- Equipment Alarm
- FMEA
- SPC
- CAPA
- 8D
- 产品规格
- MOSFET
- MCU
- 企业测试标准
- 可靠性测试
- EHS


改写要求：

1. 保留用户当前问题的核心意图。

2. 如果当前问题包含：
   - 它
   - 这个
   - 该问题
   - 上述设备
   - 还有哪些
   - 怎么处理
   - 如何解决

   等指代表达，
   必须结合历史对话补全真实对象。


3. 改写后的查询必须是完整的问题，
   不允许出现：
   “它”
   “这个”
   “该设备”
   等无法独立理解的词。


4. 保留所有设备名称、工艺名称、型号、
   Alarm Code、数字和专业缩写。


5. 将口语化、模糊化表达转换成企业技术文档常用术语。


6. 可以补充与原问题直接对应的同义词和专业表达。


7. 不要回答问题。


8. 不要虚构用户没有提供的具体设备型号、
参数、故障代码或实验结果。


9. 不要写解释。


10. 不要写“改写后的查询：”。


11. 最终只输出一条适合检索的查询语句。


12. 查询应简洁，不生成长篇文本。


示例：

历史对话：

用户：
CVD腔体压力异常如何排查？

当前问题：

它的预防措施有哪些？

输出：

CVD腔体压力异常预防措施、CVD设备压力控制异常预防性维护、腔体压力异常风险控制


---

历史对话：

用户：
CVD机台最近经常报alarm，应该先查哪里？

当前问题：

这个问题怎么避免？

输出：

CVD设备Alarm报警预防措施、设备状态检查与维护、异常复发预防


---

用户：

刻完以后图形尺寸总是不太对，该看什么？

输出：

刻蚀后关键尺寸CD异常、刻蚀尺寸偏差、工艺参数异常排查
"""


        history_text = ""


        if context:


            history_lines = []


            for message in context:


                content = getattr(
                    message,
                    "content",
                    ""
                )


                if isinstance(
                        content,
                        str
                ):

                    # 检索改写与答案生成都需要助手历史，不能只保留回答前500字。
                    role = getattr(message, "type", "message")
                    excerpt = content if len(content) <= 4000 else content[:3000] + "\n[中间内容省略]\n" + content[-1000:]
                    history_lines.append(f"{role}：{excerpt}")


            history_text = "\n".join(
                history_lines
            )



        messages = [

            SystemMessage(
                content=system_prompt
            ),


            HumanMessage(
                content=f"""
历史对话：

{history_text}


当前用户问题：

{query}


请输出完整的RAG检索查询：
"""
            )

        ]



        response = self.llm.invoke(
            messages
        )


        rewritten_query = (
            response.content.strip()
        )


        return rewritten_query