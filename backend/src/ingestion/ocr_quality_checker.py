"""OCR质量门禁：置信度只是识别指标，不能代替工程事实审核。"""

import os
import re
import math


def check_block(block):
    reasons = []
    text = block.text.strip()
    transcription = block.details.get("transcription_review", {})
    reviewed = block.method == "reviewed_transcription"
    if reviewed and (
        not transcription.get("reviewed_by")
        or not transcription.get("reviewed_at")
        or len(transcription.get("original_text_sha256", "")) != 64
    ):
        reasons.append("missing_transcription_review_audit")
    if not text:
        reasons.append("empty_text")
    if text and (text.count("\ufffd") / len(text) > 0.01 or "\x00" in text):
        reasons.append("invalid_characters")
    if (
        not reviewed
        and block.kind != "table"
        and re.search(r"[\u4e00-\u9fff]\|[\u4e00-\u9fff]", text)
    ):
        reasons.append("suspicious_inword_separator")
    if block.method not in ("native", "reviewed_transcription"):
        threshold = float(os.getenv("OCR_MIN_CONFIDENCE", "0.85"))
        if not 0 < threshold <= 1:
            raise ValueError("OCR_MIN_CONFIDENCE必须在0到1之间")
        if (
            block.confidence is None
            or not math.isfinite(block.confidence)
            or not threshold <= block.confidence <= 1
        ):
            reasons.append("low_or_missing_confidence")
        # 数字、负号和型号在低置信度行中出现时不能自动改写、猜测或放行。
        for line in block.details.get("lines", []):
            if (
                re.search(r"\d|Pa\b|Torr\b|sccm\b|°C", line.get("text", ""))
                and line.get("confidence", 0) < 0.90
            ):
                reasons.append("uncertain_numeric_or_identifier")
                break
    if block.kind == "table":
        rows = block.details.get("table_rows", [])
        if len(rows) < 2 or len(rows[0]) < 2 or len({len(row) for row in rows}) != 1:
            reasons.append("incomplete_table_structure")
    if block.kind in ("image", "figure", "chart", "formula"):
        reasons.append("visual_semantics_require_review")
    return {
        "status": "approved" if not reasons else "needs_review",
        "reasons": sorted(set(reasons)),
    }
