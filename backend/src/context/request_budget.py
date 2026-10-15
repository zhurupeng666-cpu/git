"""在模型HTTP请求前检查完整输入和输出预留，覆盖工具Schema和检索正文。"""
import dataclasses
import json
import math
import os
from langchain_openai import ChatOpenAI
from src.context.token_counter import TokenCounter, ContextBudgetExceeded


def json_value(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def validate_request_budget(payload, output_tokens=None):
    total_limit = int(os.getenv("LLM_TOTAL_TOKEN_BUDGET", "16384"))
    default_output = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))
    margin = float(os.getenv("TOKEN_SAFETY_MARGIN", "1.15"))
    if total_limit < 1 or default_output < 1 or not math.isfinite(margin) or margin < 1:
        raise ValueError("模型预算和输出预留必须为正，Token安全系数至少为1")
    reserve = min(int(default_output if output_tokens is None else output_tokens), default_output)
    if reserve < 1:
        raise ValueError("输出Token预算必须为正")
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=json_value)
    # 参考Tokenizer不能保证与服务商完全一致，增加协议开销和可配置安全余量。
    estimated = math.ceil((TokenCounter().count_text_tokens(text) + 32) * margin)
    if estimated + reserve > total_limit:
        raise ContextBudgetExceeded("当前问题、历史或查询结果超过模型请求预算，请缩小查询范围或分批提问。")
    return {"estimated_input_tokens": estimated, "reserved_output_tokens": reserve,
            "application_total_budget": total_limit, "counting_method": "cl100k_base_reference_with_margin"}


class BudgetedChatOpenAI(ChatOpenAI):
    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        output = payload.pop("max_completion_tokens", payload.pop("max_tokens", None))
        report = validate_request_budget(payload, output_tokens=output)
        # 本项目使用DeepSeek兼容接口；如更换服务商，可显式选择输出参数名。
        parameter = os.getenv("LLM_OUTPUT_TOKEN_PARAMETER", "max_tokens")
        if parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("LLM_OUTPUT_TOKEN_PARAMETER配置无效")
        payload[parameter] = report["reserved_output_tokens"]
        return payload
