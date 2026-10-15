"""按标题、段落及表格切片；原始OCR结果保持不变，便于复核。"""

import hashlib
import re
from langchain_core.documents import Document
from src.context.token_counter import TokenCounter
from src.ingestion.document_blocks import table_markdown
from src.ingestion.ocr_quality_checker import check_block

PARSER_VERSION = "ocr-ingest-v1"


def build_structured_chunks(blocks, base_metadata, max_tokens=600):
    counter = TokenCounter(max_tokens=max_tokens)
    chunks, reviews = [], []
    title = base_metadata.get("title") or base_metadata.get("file_name", "")
    pending, pending_blocks, pending_page = [], [], None

    def emit(text, selected):
        if not text.strip():
            return
        metadata = dict(base_metadata)
        first = selected[0]
        metadata.update(
            {
                "page_number": first.page_number,
                "page_index": first.page_number - 1,
                "block_indices": [block.block_index for block in selected],
                "regions": [block.bbox for block in selected],
                "coordinate_system": "normalized_top_left",
                "section_title": title,
                "section_titles": [
                    block.text
                    for block in selected
                    if block.kind in ("paragraph_title", "section_title")
                ],
                "chunk_type": "ocr_table" if first.kind == "table" else "ocr_text",
                "parser_version": PARSER_VERSION,
                "quality_status": "approved",
                "chunk_index": len(chunks),
                "parse_method": first.method,
                "parse_methods": sorted({block.method for block in selected}),
                "transcription_reviews": [
                    block.details["transcription_review"]
                    for block in selected
                    if "transcription_review" in block.details
                ],
                "ocr_confidence": min(
                    (block.confidence for block in selected if block.confidence is not None),
                    default=None,
                ),
            }
        )
        if metadata.get("logical_page_kind") == "docx_section":
            metadata["section_index"] = metadata.pop("page_number")
            metadata.pop("page_index", None)
        fingerprint = hashlib.sha256(text.encode()).hexdigest()
        raw = f"{metadata['document_id']}|{metadata['source_sha256']}|{first.page_number}|{metadata['block_indices']}|{metadata['chunk_index']}|{fingerprint}|{PARSER_VERSION}"
        metadata["content_sha256"] = fingerprint
        metadata["chunk_id"] = "ocr_" + hashlib.sha256(raw.encode()).hexdigest()[:32]
        chunks.append(Document(page_content=text, metadata=metadata))

    def flush():
        nonlocal pending, pending_blocks
        if pending:
            emit("\n\n".join(pending), pending_blocks)
        pending, pending_blocks = [], []

    for block in blocks:
        if block.kind == "number" and (block.bbox[1] > 0.94 or block.bbox[3] < 0.05):
            continue
        quality = check_block(block)
        if quality["status"] != "approved":
            reviews.append({**block.to_dict(), "quality": quality})
            continue
        if block.page_number != pending_page:
            flush()
            pending_page = block.page_number
        if block.details.get("section_title"):
            title = block.details["section_title"]
        if block.kind in ("doc_title", "title"):
            flush()
            title = block.text.strip()
        if block.kind == "table":
            # 表格前仅有标题时不制造一个缺乏内容的碎片；表格正文重复所属专题标题。
            if counter.count_text_tokens("\n".join(pending)) > 120:
                flush()
            prefix = "\n".join(pending) if pending else title
            context_blocks = list(pending_blocks)
            pending, pending_blocks = [], []
            rows = block.details["table_rows"]
            batch = [rows[0]]
            table_text = lambda values: prefix + "\n" + table_markdown(values)
            for row in rows[1:]:
                value = table_text(batch + [row])
                if counter.count_text_tokens(value) > max_tokens:
                    if len(batch) > 1:
                        emit(table_text(batch), [block] + context_blocks)
                        batch = [rows[0]]
                    if counter.count_text_tokens(table_text(batch + [row])) > max_tokens:
                        reviews.append(
                            {
                                **block.to_dict(),
                                "quality": {
                                    "status": "needs_review",
                                    "reasons": ["single_table_row_exceeds_budget"],
                                },
                            }
                        )
                        continue
                batch.append(row)
            if len(batch) > 1:
                emit(table_text(batch), [block] + context_blocks)
            continue
        # 非表格长段按Token裁剪，但不更改数字、单位和原始文字。
        for part in counter.split_text(block.text, max_tokens):
            proposed = "\n\n".join(pending + [part])
            if pending and counter.count_text_tokens(proposed) > max_tokens:
                flush()
            pending.append(part)
            pending_blocks.append(block)
    flush()
    # 重复文字可能位于不同位置，不能按内容哈希丢弃合法的重复段落。
    return chunks, reviews
