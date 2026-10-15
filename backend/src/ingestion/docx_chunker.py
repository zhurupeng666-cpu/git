from pathlib import Path
from docx import Document as WordDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from langchain_core.documents import Document


# ============================================================
# 按 Word 原始顺序读取段落和表格
# ============================================================

def iter_block_items(document):
    for child in document.element.body.iterchildren():

        # 普通段落 / 标题
        if isinstance(child, CT_P):

            yield Paragraph(
                child,
                document
            )

        # 表格
        elif isinstance(child, CT_Tbl):

            yield Table(
                child,
                document
            )


# ============================================================
# 表格转换成文本
# ============================================================

def table_to_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells = [
            cell.text.strip()
            for cell in row.cells
        ]
        row_text = " | ".join(cells)
        rows.append(
            row_text
        )

    return "\n".join(rows)


# ============================================================
# DOCX结构化切分
# ============================================================

def split_docx_by_heading1(
        file_path: str,
        domain: str
) -> list[Document]:

    file_path = Path(file_path)

    doc = WordDocument(
        str(file_path)
    )

    chunks = []

    current_title = None
    current_parts = []

    section_index = 0

    # 是否正在跳过目录
    skip_catalog = False


    # ========================================================
    # 保存当前章节
    # ========================================================

    def save_current_section():

        nonlocal section_index
        nonlocal current_title
        nonlocal current_parts

        if current_title is None:
            return

        text = "\n".join(
            current_parts
        ).strip()

        if not text:
            return
        section_index += 1

        chunk = Document(
            page_content=text,
            metadata={
                "source":
                    str(file_path),

                "file_name":
                    file_path.name,

                "file_type":
                    "docx",

                "domain":
                    domain,

                "section_index":
                    section_index,

                "section_title":
                    current_title,

                "chunk_type":
                    "heading1_section",
            }
        )
        chunks.append(
            chunk
        )


    # ========================================================
    # 按真实顺序遍历 Word
    # ========================================================

    for block in iter_block_items(doc):

        # ====================================================
        # Paragraph
        # ====================================================

        if isinstance(block, Paragraph):

            text = block.text.strip()

            if not text:
                continue


            style_name = block.style.name


            # ------------------------------------------------
            # Heading 1
            # ------------------------------------------------

            if style_name == "Heading 1":

                # ============================================
                # 目录页
                # ============================================

                catalog_keywords = [
                    "内容导航",
                    "条款索引",
                    "目录",
                ]

                if any(
                        keyword in text
                        for keyword in catalog_keywords
                ):

                    # 先保存之前的数据
                    save_current_section()

                    current_title = None
                    current_parts = []

                    skip_catalog = True

                    continue


                # ============================================
                # 下一个真正的 Heading 1
                # ============================================

                # 如果之前正在跳过目录
                if skip_catalog:

                    skip_catalog = False

                # 保存上一个主题
                save_current_section()

                # 创建新主题
                current_title = text
                current_parts = [
                    text
                ]

                continue


            # ------------------------------------------------
            # 跳过目录内部的正文、表格等内容
            # ------------------------------------------------

            if skip_catalog:
                continue


            # ------------------------------------------------
            # 已进入一个真实知识主题
            # ------------------------------------------------

            if current_title is not None:

                current_parts.append(
                    text
                )


        # ====================================================
        # Table
        # ====================================================

        elif isinstance(block, Table):

            # 目录中的大表格不要放进知识库
            if skip_catalog:
                continue


            if current_title is not None:

                table_text = table_to_text(
                    block
                )

                if table_text:

                    current_parts.append(
                        table_text
                    )


    # ========================================================
    # 文档结束，保存最后一个章节
    # ========================================================

    save_current_section()


    return chunks