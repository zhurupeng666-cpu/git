from pathlib import Path
import hashlib
from langchain_core.documents import Document
from src.ingestion.pdf_chunker import (
    split_pdf_by_page
)
from src.ingestion.docx_chunker import (
    split_docx_by_heading1
)

def generate_chunk_id(
        chunk: Document
) -> str:

    """
    根据 Chunk 的来源位置生成稳定唯一 ID。
    """

    metadata = chunk.metadata

    source = metadata.get(
        "source",
        ""
    )

    chunk_type = metadata.get(
        "chunk_type",
        ""
    )


    # ==========================================
    # PDF
    # ==========================================

    if chunk_type == "pdf_page":

        locator = str(
            metadata.get(
                "page_number",
                ""
            )
        )


    # ==========================================
    # DOCX
    # ==========================================

    elif chunk_type == "heading1_section":

        locator = str(
            metadata.get(
                "section_index",
                ""
            )
        )
    else:
        locator = ""

    raw_id = (
        f"{source}"
        f"|{chunk_type}"
        f"|{locator}"
    )


    hash_value = hashlib.sha1(
        raw_id.encode("utf-8")
    ).hexdigest()[:16]


    return (
        f"{chunk_type}_{hash_value}"
    )

def build_knowledge_base_chunks(
        root_path: str
) -> list[Document]:

    """
    遍历整个企业知识库，
    根据文件类型选择对应的结构化切分方式，
    最终返回统一的 LangChain Document 列表。
    """
    root_path = Path(root_path)

    all_chunks = []
    # ========================================================
    # 遍历整个知识库
    # ========================================================

    for file_path in root_path.rglob("*"):

        if not file_path.is_file():
            continue

        # ====================================================
        # 获取业务域
        #
        # knowledge_base/
        # ├── equipment/
        # ├── process/
        # ├── quality/
        # ...
        #
        # 第一层目录即 domain
        # ====================================================

        relative_path = file_path.relative_to(
            root_path
        )

        domain = relative_path.parts[0]

        # ====================================================
        # PDF
        # ====================================================

        if file_path.suffix.lower() == ".pdf":
            chunks = split_pdf_by_page(
                file_path=str(file_path),
                domain=domain
            )

            all_chunks.extend(
                chunks
            )


        # ====================================================
        # DOCX
        # ====================================================

        elif file_path.suffix.lower() == ".docx":
            chunks = split_docx_by_heading1(
                file_path=str(file_path),
                domain=domain
            )

            all_chunks.extend(
                chunks
            )

    # ============================================================
    # 为所有 Chunk 添加稳定唯一 ID
    # ============================================================

    for chunk in all_chunks:
        chunk.metadata["chunk_id"] = (
            generate_chunk_id(
                chunk
            )
        )

    return all_chunks
