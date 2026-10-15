import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json
load_dotenv()


class DomainRouter:

    DOMAINS = [
        "equipment",
        "general",
        "process",
        "product",
        "quality",
        "standard",
    ]


    def __init__(self):

        self.llm = ChatOpenAI(
            model="deepseek-v4-pro",
            base_url="https://api.deepseek.com/v1",
            api_key=os.getenv(
                "DEEPSEEK_API_KEY"
            ),
            temperature=0,
        )


    def route(
            self,
            query: str
    ) -> str:
        prompt = f"""
        你是半导体企业知识库的领域路由器。

        你的任务是根据用户问题的“核心知识对象”，
        判断这个问题最应该从哪个业务知识域中检索答案。

        只能输出以下六个标签之一：

        equipment
        general
        process
        product
        quality
        standard


        ==============================
        一、领域定义
        ==============================

        equipment：
        以半导体生产设备本身为核心对象。

        包括：
        - 设备故障
        - Alarm / 报警
        - Robot / Chamber / Module
        - 设备维护
        - PM
        - 硬件状态
        - 设备恢复
        - 设备部件
        - 设备异常
        - 设备维修与诊断


        process：
        以晶圆制造工艺、工艺步骤、工艺条件为核心对象。

        包括：
        - 光刻
        - 刻蚀
        - CVD / PVD 等沉积工艺
        - CMP
        - 清洗
        - 氧化
        - 工艺参数
        - 工艺窗口
        - 工艺异常
        - Wafer加工过程
        - 工艺量测
        - 工艺能力


        product：
        以半导体器件、产品规格和应用为核心对象。

        包括：
        - MOSFET
        - MCU
        - 产品规格参数
        - Datasheet
        - Application Note
        - 器件电气特性
        - 板级应用
        - 驱动
        - 波形
        - 器件测试与应用条件


        quality：
        以质量管理方法和质量体系活动为核心对象。

        包括：
        - FMEA
        - 8D
        - CAPA
        - SPC
        - Root Cause质量管理
        - 质量改善
        - 不符合项
        - 质量闭环
        - 质量风险分析

        注意：
        仅仅出现“异常、证据、记录、报告、风险、调查、结论”
        并不能判定为 quality。


        standard：
        以企业规范、安全、环境、可靠性和标准要求为核心对象。

        包括：
        - EHS
        - 安全管理
        - 环境管理
        - 废物管理
        - 企业测试标准
        - 可靠性测试规范
        - 合规
        - 制度要求
        - 标准化要求


        general：
        以半导体基础知识和通用原理为核心对象。

        包括：
        - 半导体基本原理
        - 材料基础
        - PN结
        - 晶体管基础
        - 基础物理概念
        - 通用机理
        - 不明显属于其他五类的基础知识


        ==============================
        二、判断原则
        ==============================

        必须优先判断“问题在讨论什么对象”，
        而不是根据“问题用了什么表达方式”分类。


        例如：

        设备报警的证据应该如何保存？

        核心对象是设备报警，
        因此属于：

        equipment

        而不是 quality。


        CMP工艺参数改变后如何判断影响？

        核心对象是CMP工艺，
        因此属于：

        process。


        MOSFET开关波形为什么出现尖峰？

        核心对象是MOSFET应用，
        因此属于：

        product。


        FMEA中如何评价严重度？

        核心对象是FMEA质量工具，
        因此属于：

        quality。


        危险废物应该如何分类？

        核心对象是EHS与环境管理，
        因此属于：

        standard。


        硅为什么是常用半导体材料？

        核心对象是半导体基础知识，
        因此属于：

        general。


        ==============================
        三、容易混淆的情况
        ==============================

        equipment 与 process：

        如果核心问题是：
        设备硬件、报警、维护、Module、Robot、Chamber
        → equipment

        如果核心问题是：
        工艺参数、工艺条件、Wafer结果、工艺窗口
        → process


        process 与 quality：

        如果是在讨论某个具体制造工艺本身
        → process

        只有明确以FMEA、8D、CAPA、SPC、
        质量管理方法或质量体系为中心
        → quality


        product 与 quality：

        如果核心对象是器件规格、应用条件、
        电气特性、板级波形
        → product

        如果核心对象是质量分析方法
        → quality


        standard 与 quality：

        EHS、安全、环境、可靠性规范、
        企业标准和制度要求
        → standard

        FMEA、8D、CAPA、SPC等质量管理活动
        → quality


        ==============================
        用户问题
        ==============================

        {query}


请输出最相关的两个知识领域。

第一个必须是你认为最相关的领域，
第二个是可能相关的备选领域。

只能从以下标签中选择：

equipment
general
process
product
quality
standard

必须严格输出 JSON：

{{
    "primary": "equipment",
    "secondary": "process"
}}

禁止输出任何解释。
        """

        response = self.llm.invoke(
            prompt
        )

        content = response.content.strip()

        data = json.loads(
            content
        )

        primary = data["primary"].lower()
        secondary = data["secondary"].lower()

        if primary not in self.DOMAINS:
            raise ValueError(
                f"未知Primary Domain：{primary}"
            )

        if secondary not in self.DOMAINS:
            raise ValueError(
                f"未知Secondary Domain：{secondary}"
            )

        return [
            primary,
            secondary
        ]


        if domain not in self.DOMAINS:

            raise ValueError(
                f"未知Domain：{domain}"
            )


        return domain