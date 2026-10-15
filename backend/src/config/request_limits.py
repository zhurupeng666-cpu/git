"""业务执行超时配置：0表示关闭，不改变登录与任务租约校验。"""
import math
import os


def seconds_timeout(name: str, default: str):
    value = float(os.getenv(name, default))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}必须是非负有限数，0表示不限制")
    return value if value > 0 else None


def statement_options(name: str, default: str) -> str:
    value = int(os.getenv(name, default))
    if value < 0:
        raise ValueError(f"{name}不能小于0")
    # PostgreSQL明确以0关闭SQL执行超时；连接建立仍使用原来的超时。
    return f"-c statement_timeout={value}"
