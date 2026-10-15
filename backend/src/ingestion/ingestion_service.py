"""OCR预览与受控发布：先保留原始结果和待复核项，再发布质量通过的切片。"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4
from langchain_core.documents import Document
from src.ingestion.pdf_page_parser import parse_pdf_blocks
from src.ingestion.image_parser import parse_image_blocks
from src.ingestion.structured_chunker import PARSER_VERSION, build_structured_chunks

ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for batch in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(batch)
    return digest.hexdigest()


def library_links(root):
    manifest = Path(root) / "公开知识来源清单.json"
    if not manifest.exists():
        return {}, {}
    value = json.loads(manifest.read_text(encoding="utf-8"))
    images = {item["image"]: item for item in value.get("documents", [])}
    return images, value.get("sources", {})


def parse_document(source, root=None, engine=None, max_pages=None):
    root = Path(root or ROOT / "data/knowledge_base").resolve()
    source = Path(source).resolve()
    relative = source.relative_to(root).as_posix()
    if source.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("单文件超过100MB，请先按文档拆分")
    digest = file_sha256(source)
    configuration = {
        "parser": PARSER_VERSION,
        "dpi": os.getenv("OCR_RENDER_DPI", "180"),
        "confidence": os.getenv("OCR_MIN_CONFIDENCE", "0.85"),
        "orientation": os.getenv("OCR_CORRECT_ORIENTATION", "1"),
        "chunk_tokens": 600,
    }
    metadata = {
        "source": str(source),
        "source_relpath": relative,
        "file_name": source.name,
        "file_type": source.suffix.lower().lstrip("."),
        "domain": relative.split("/")[0] if "/" in relative else "general",
        "document_id": "doc_" + hashlib.sha256(relative.encode()).hexdigest()[:24],
        "source_sha256": digest,
        "ingest_signature": hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode()
        ).hexdigest(),
        "title": source.stem,
    }
    linked, references = library_links(root)
    # 清单只补充来源和文件关联，不以原文或答案清单代替实际OCR。
    matched = [
        item for item in linked.values() if item["pdf"] == relative or item["image"] == relative
    ]
    if matched:
        metadata["document_origin"] = "public_source_synthesis"
        metadata["authoring"] = "AI辅助编写的公开知识综述，非企业受控SOP"
        keys = sorted({key for item in matched for key in item.get("sources", [])})
        metadata["reference_sources"] = [references[key] for key in keys if key in references]
    if relative in linked:
        metadata["canonical_pdf_relpath"] = linked[relative]["pdf"]
        metadata["canonical_pages"] = linked[relative]["pages"]
    if source.suffix.lower() == ".pdf":
        blocks, page_info = parse_pdf_blocks(source, engine=engine, max_pages=max_pages)
    elif source.suffix.lower() in IMAGE_SUFFIXES:
        blocks, page_info = parse_image_blocks(source, engine=engine)
    elif source.suffix.lower() == ".docx":
        from src.ingestion.docx_chunker import split_docx_by_heading1
        from src.ingestion.document_blocks import DocumentBlock

        sections = split_docx_by_heading1(str(source), metadata["domain"])
        blocks = [
            DocumentBlock(
                index,
                0,
                "text",
                section.page_content,
                [0, 0, 1, 1],
                "native",
                details={"section_title": section.metadata.get("section_title", source.stem)},
            )
            for index, section in enumerate(sections, 1)
        ]
        metadata["logical_page_kind"] = "docx_section"
        page_info = {
            "total_pages": None,
            "processed_pages": None,
            "sections": len(sections),
            "page_modes": ["native"],
            "truncated": False,
        }
    else:
        raise ValueError("入库入口支持PDF、DOCX或常见图片")
    chunks, reviews = build_structured_chunks(blocks, metadata)
    return {
        "source": metadata,
        "pages": page_info,
        "chunks": chunks,
        "review_items": reviews,
        "block_count": len(blocks),
        "blocks": [block.to_dict() for block in blocks],
    }


def save_preview(documents, skipped=None, output_dir=None):
    folder = Path(
        output_dir
        or ROOT
        / "data/ocr_runs"
        / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    )
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {
        "version": PARSER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "database_published": False,
        "documents": [],
        "skipped": skipped or [],
    }
    chunk_rows = []
    for document in documents:
        item = {key: value for key, value in document.items() if key not in ("chunks", "blocks")}
        item["chunk_ids"] = [chunk.metadata["chunk_id"] for chunk in document["chunks"]]
        manifest["documents"].append(item)
        chunk_rows.extend(
            {"page_content": chunk.page_content, "metadata": chunk.metadata}
            for chunk in document["chunks"]
        )
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (folder / "chunks.jsonl").open("w", encoding="utf-8") as stream:
        for row in chunk_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (folder / "raw_blocks.json").write_text(
        json.dumps(
            [{"source": doc["source"], "blocks": doc["blocks"]} for doc in documents],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return folder


def load_preview(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    chunks = [
        Document(**json.loads(line))
        for line in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {chunk.metadata["chunk_id"]: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise ValueError("预览包含重复切片ID")
    for item in manifest["documents"]:
        if item["pages"]["truncated"]:
            raise ValueError("部分页面预览不能发布；请先完整解析该文档")
        item["chunks"] = [by_id[key] for key in item["chunk_ids"]]
        if not item["chunks"]:
            raise ValueError("质量检查后没有可发布切片，不能覆盖旧版本")
        if file_sha256(item["source"]["source"]) != item["source"]["source_sha256"]:
            raise ValueError("预览后源文件发生变化，请重新解析")
        for chunk in item["chunks"]:
            if (
                chunk.metadata.get("quality_status") != "approved"
                or chunk.metadata["source_sha256"] != item["source"]["source_sha256"]
            ):
                raise ValueError("切片质量状态或来源校验不一致")
    return manifest
