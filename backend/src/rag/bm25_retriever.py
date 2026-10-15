import os
from pathlib import Path

import jieba
import psycopg
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi


load_dotenv()
DB_URL = os.getenv("DATABASE_URL")


class BM25Retriever:

    def __init__(self, db_url: str = DB_URL):
        self.db_url = db_url

        # 加载半导体专业词典
        project_root = Path(__file__).resolve().parents[2]
        dictionary_path = (
            project_root
            / "data"
            / "dictionaries"
            / "semiconductor_terms.txt"
        )

        if dictionary_path.exists():
            jieba.load_userdict(str(dictionary_path))
            print("已加载半导体专业词典：", dictionary_path)
        else:
            print("警告：未找到半导体专业词典：", dictionary_path)

        # 从 PostgreSQL 加载知识库
        self.documents = self._load_documents()

        print(
            f"BM25 Corpus Chunk数量：{len(self.documents)}"
        )

        # BM25 索引使用：文件名 + 章节标题 + 正文
        self.tokenized_corpus = [
            self._tokenize(
                document["content"]
            )
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

    def _load_documents(self):
        sql = """
        SELECT
            chunk_id,
            content,
            metadata
        FROM rag_chunks
        ORDER BY chunk_id
        """

        documents = []

        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)

                for chunk_id, content, metadata in cur.fetchall():
                    documents.append(
                        {
                            "chunk_id": chunk_id,
                            "content": content,
                            "metadata": metadata
                        }
                    )

        return documents

    @staticmethod
    def _build_index_text(document):
        """构建 BM25 使用的检索文本。"""

        metadata = document.get("metadata") or {}

        file_name = metadata.get(
            "file_name",
            ""
        )

        section_title = metadata.get(
            "section_title",
            ""
        )

        content = document.get(
            "content",
            ""
        )

        # 去掉 .pdf、.docx 后缀，只保留有检索意义的文件名
        if file_name:
            file_name = Path(file_name).stem

        # DOCX 正文开头可能已经带有章节标题，避免重复计算词频
        if (
            section_title
            and content.startswith(section_title)
        ):
            content = content[
                len(section_title):
            ].lstrip()

        parts = []

        if file_name:
            parts.append(file_name)

        if section_title:
            parts.append(section_title)

        if content:
            parts.append(content)

        return "\n".join(parts)

    @staticmethod
    def _tokenize(text: str):
        text = text.lower()

        # jieba 精确模式，专业词典已在初始化时加载
        tokens = jieba.lcut(text)

        return [
            token.strip()
            for token in tokens
            if token.strip()
        ]

    def search(
        self,
        query: str,
        top_k: int = 5
    ):
        query_tokens = self._tokenize(query)

        # 对全部 Chunk 计算 BM25 分数
        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for index in ranked_indices:
            document = self.documents[index]

            results.append(
                {
                    "chunk_id": document["chunk_id"],
                    "content": document["content"],
                    "metadata": document["metadata"],
                    "bm25_score": float(scores[index])
                }
            )

        return results