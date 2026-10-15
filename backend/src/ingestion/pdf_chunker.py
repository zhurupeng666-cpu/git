from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document


def is_catalog_page(text: str) -> bool:

    """
    判断当前 PDF 页是否为目录页。

    不在整页正文中模糊搜索关键词，
    而是检查前若干行是否存在明确的目录标题，
    避免正文中出现“目录”“索引”等词时被误删。
    """

    catalog_titles = {
        "内容导航与条款索引",
        "内容导航",
        "条款索引",
        "目录",
    }

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # 目录标题通常位于页面顶部
    for line in lines[:10]:

        if line in catalog_titles:
            return True

    return False

def split_pdf_by_page(
        file_path: str,
        domain: str
) -> list[Document]:

    """
    按 PDF 页进行基础 Chunk 切分。

    当前知识库中的 PDF 页面平均约 600~900 字符，
    且每页基本对应一个完整知识主题，
    因此暂时采用：

        1 Page = 1 Base Chunk

    后续如发现超长页，再增加二次切分策略。
    """

    file_path = Path(file_path)
    reader = PdfReader(
        str(file_path)
    )

    chunks = []

    # =========================================
    # 按页读取
    # =========================================

    for page_index, page in enumerate(
        reader.pages
    ):

        text = page.extract_text()

        # 有些PDF页可能提取不到文本
        if text is None:
            continue

        text = text.strip()

        # 跳过目录页
        if is_catalog_page(text):
            continue

        # 跳过真正的空白页
        if not text:
            continue


        # =====================================
        # 创建统一 LangChain Document
        # =====================================

        chunk = Document(
            page_content=text,
            metadata={

                "source":
                    str(file_path),

                "file_name":
                    file_path.name,

                "file_type":
                    "pdf",

                "domain":
                    domain,


                # -----------------------------
                # Python内部页索引
                # 从0开始
                # -----------------------------

                "page_index":
                    page_index,


                # -----------------------------
                # 用户看到的页码
                # 从1开始
                # -----------------------------

                "page_number":
                    page_index + 1,


                # -----------------------------
                # 当前Chunk类型
                # -----------------------------

                "chunk_type":
                    "pdf_page",

            }

        )


        chunks.append(
            chunk
        )


    return chunks