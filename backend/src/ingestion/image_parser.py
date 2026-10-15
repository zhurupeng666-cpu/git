"""图片使用与扫描PDF相同的块结构、质量门禁和引用坐标。"""

from src.ingestion.document_blocks import DocumentBlock
from src.ingestion.ocr_engine import OcrEngine


def parse_image_blocks(source, engine=None):
    pages = (engine or OcrEngine()).parse(source, [1])
    if len(pages) != 1 or pages[0]["page_number"] != 1:
        raise RuntimeError("图片OCR结果不完整")
    return [DocumentBlock(**value) for value in pages[0]["blocks"]], {
        "total_pages": 1,
        "processed_pages": 1,
        "page_modes": ["image"],
        "truncated": False,
    }
