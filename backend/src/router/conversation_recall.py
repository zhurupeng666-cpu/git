"""识别当前对话的信息记录和回忆请求，不查询知识库或跨会话长期记忆。"""
import re

from langchain_core.messages import HumanMessage


def is_conversation_request(query):
    """只截获明确要求记忆或回顾个人对话的请求，保留专业知识问答路由。"""
    # 引用自己此前提供的设备信息再询问排查方法，仍是专业问答，不是纯回忆。
    technical_followup = re.search(
        r"(?:如何|怎么|怎样|应当|应该).{0,15}(?:排查|检查|处理|解决|预防|修复)|进一步(?:排查|检查)", query
    )
    identifier_only = re.search(r"(?:只|仅)(?:要)?(?:回答|返回|输出)(?:编号|型号|编码|代号)", query)
    if technical_followup and not identifier_only:
        return False
    if re.search(r"我的(?:回答|回复|输出)?偏好|我的设备型号", query) and re.search(r"是什么|什么|哪个|[？?]", query):
        return True
    if re.search(r"我的长期|长期保存的|长期记住|长期保存|跨会话.{0,15}(?:资料|信息|偏好|记忆)", query):
        return True
    return bool(re.search(
        r"(?:请)?(?:记住|记下|记一下)|(?:刚才|刚刚|之前|前面|此前|上次|先前)"
        r"(?:我|本人)(?:向你|给你)?(?:提供|告诉|说过|提到|说|给).{0,25}|"
        r"(?:我|本人)(?:刚才|刚刚|之前|前面|此前|上次|先前)(?:曾经|曾)?(?:向你|给你)?"
        r"(?:提供|告诉|说过|提到|说|给)|(?:我|本人)(?:向你|给你)?(?:提供过|告诉过|说过|提到过)|"
        r"(?:你)?(?:还)?记得.{0,20}(?:我|编号|型号|编码)|(?:当前|本次|这段)(?:对话|会话).{0,20}(?:记录|信息)|"
        r"(?:根据|回顾|总结|回忆|复述).{0,12}(?:当前|我们|之前|刚才|本次|这段).{0,8}"
        r"(?:对话|会话|聊天|说过|提到)", query
    ))


def get_identifier_answer(query, messages):
    """精确回忆用户提供的编号或型号，避免模型改写标识符。

    仅在用户要求只回答标识符时使用，复杂的回顾请求交给对话模型。
    只读取用户历史，AI回复或用户本轮举例不作为已提供的事实。
    """
    if not re.search(r"(?:只|仅)(?:要)?(?:回答|返回|输出)(?:那个|这个|该)?(?:编号|型号|编码|代号)", query):
        return None
    kinds = re.findall(r"编号|型号|编码|代号", query)
    if not kinds:
        return None
    kind = kinds[0]
    normalized_query = re.sub(r"[\s的，。？?:：]", "", query)
    candidates = []
    for message in messages:
        if not isinstance(message, HumanMessage) or not isinstance(message.content, str):
            continue
        for match in re.finditer(
            r"(?P<label>[\u4e00-\u9fffA-Za-z0-9 _/-]{0,60}?(?:编号|型号|编码|代号))"
            r"\s*(?:改为|更正为|更新为|是|为|[:：])\s*(?P<value>[A-Za-z0-9][A-Za-z0-9._/-]{0,127})",
            message.content
        ):
            label = re.sub(r"[\s的]", "", match["label"])
            label = re.sub(r"^(?:本次|此次|这次)", "", label)
            if not label.endswith(kind):
                continue
            # 优先完整字段名；若只有一个标识符，允许“刚才的编号”这类省略。
            candidates.append((label, match["value"]))
    matches = [value for label, value in candidates if label in normalized_query]
    if matches:
        return matches[-1]
    values = list(dict.fromkeys(value for _, value in candidates))
    generic_reference = re.search(
        r"(?:刚才|之前|前面|此前|先前)(?:我)?(?:提供|告诉你|说过|提到)?的?(?:那个|这个)?" + kind,
        query
    )
    if len(values) == 1 and generic_reference:
        return values[0]
    if not values:
        return "不知道"
    # 多个不同编号且无法确认字段时不猜测，让模型根据完整用户要求判断。
    return None
