"""逐页判断原生文字、扫描及混合页面；较大图像区域不能仅凭页眉文字跳过OCR。"""

from pathlib import Path
from pypdf import PdfReader
from src.ingestion.document_blocks import DocumentBlock
from src.ingestion.ocr_engine import OcrEngine


def classify_page(page):
    text = (page.extract_text() or "").strip()
    image_area = 0.0
    page_area = float(page.mediabox.width) * float(page.mediabox.height)
    resources = page.get("/Resources", {})
    if hasattr(resources, "get_object"):
        resources = resources.get_object()
    objects = resources.get("/XObject", {})
    if hasattr(objects, "get_object"):
        objects = objects.get_object()

    def visit(operation, arguments, matrix, text_matrix):
        nonlocal image_area
        if operation == b"Do" and arguments:
            obj = objects.get(arguments[0])
            if obj is not None and obj.get_object().get("/Subtype") in ("/Image", "/Form"):
                image_area += abs(matrix[0] * matrix[3] - matrix[1] * matrix[2])

    # Form也保守视为需要版面检查，避免嵌套图像被漏掉。
    page.extract_text(visitor_operand_before=visit)
    coverage = min(1, image_area / max(1, page_area))
    bad_text = "\ufffd" in text or (
        len(text) and sum(ch.isprintable() for ch in text) / len(text) < 0.90
    )
    if len(text) >= 40 and not bad_text and coverage < 0.15:
        mode = "native"
    elif text and coverage >= 0.15:
        mode = "mixed"
    elif not text and not objects:
        mode = "blank"
    else:
        mode = "ocr"
    return {"mode": mode, "text": text, "image_coverage": coverage}


def inspect_pdf(source):
    reader = PdfReader(source)
    if len(reader.pages) > 1000:
        raise ValueError("单文档超过1000页，请先按章节拆分")
    if reader.is_encrypted:
        raise ValueError("加密PDF需要先提供可读取的授权副本")
    return [classify_page(page) for page in reader.pages]


def parse_pdf_blocks(source, engine=None, max_pages=None):
    if max_pages is not None and max_pages < 1:
        raise ValueError("预览页数必须大于0")
    reader = PdfReader(source)
    if len(reader.pages) > 1000:
        raise ValueError("单文档超过1000页，请先按章节拆分")
    if reader.is_encrypted:
        raise ValueError("加密PDF不能静默跳过")
    limit = min(len(reader.pages), max_pages) if max_pages else len(reader.pages)
    pages = [classify_page(reader.pages[index]) for index in range(limit)]
    needed = [index + 1 for index, item in enumerate(pages) if item["mode"] in ("ocr", "mixed")]
    results = (engine or OcrEngine()).parse(source, needed) if needed else []
    parsed = {item["page_number"]: item for item in results}
    blocks = []
    for index, item in enumerate(pages, 1):
        if item["mode"] == "blank":
            continue
        if item["mode"] == "native":
            blocks.append(DocumentBlock(index, 0, "text", item["text"], [0, 0, 1, 1], "native"))
        else:
            if index not in parsed:
                raise RuntimeError(f"OCR遗漏第{index}页，不能按成功处理")
            # 混合页由同一版面引擎读取整页，不再叠加原生层，避免隐藏OCR层重复。
            blocks.extend(DocumentBlock(**value) for value in parsed[index]["blocks"])
    return blocks, {
        "total_pages": len(reader.pages),
        "processed_pages": limit,
        "page_modes": [item["mode"] for item in pages],
        "truncated": limit < len(reader.pages),
    }
