from src.rag.index_snapshot import RefreshableBM25
import jieba
import psycopg

from rank_bm25 import BM25Okapi


class KeywordBM25Retriever:

    def __init__(
        self,
        db_url: str,
        user_dict_path: str = "data/dictionaries/semiconductor_terms.txt"
    ):
        self.db_url = db_url

        jieba.load_userdict(user_dict_path)

        self.documents = self._load_documents()

        self.tokenized_corpus = [
            self._tokenize(
                self._build_index_text(document)
            )
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )
        self.index_snapshot = RefreshableBM25(self.db_url, self._load_documents, self._tokenize,
                                              self._build_index_text, self.documents, self.bm25)


    def _load_documents(self):
        documents = []

        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        r.chunk_id,
                        r.content,
                        r.metadata,
                        e.keywords
                    FROM rag_chunks r
                    LEFT JOIN rag_chunk_expansions e
                        ON r.chunk_id = e.chunk_id
                    ORDER BY r.chunk_id
                """)

                rows = cur.fetchall()

        for row in rows:
            documents.append(
                {
                    "chunk_id": row[0],
                    "content": row[1],
                    "metadata": row[2],
                    "keywords": row[3] or []
                }
            )

        return documents


    @staticmethod
    def _tokenize(text: str):
        text = text.lower()

        tokens = jieba.lcut(text)

        return [
            token.strip()
            for token in tokens
            if token.strip()
        ]


    @staticmethod
    def _build_index_text(document):
        content = document["content"]
        keywords = document["keywords"]

        keyword_text = " ".join(keywords)

        return (
            content
            + "\n"
            + keyword_text
        )


    def search(
        self,
        query: str,
        top_k: int = 10
    ):
        # 查询全程固定快照，发布后最多在轮询间隔内自动发现新索引。
        documents, model = self.index_snapshot.get()
        if not documents:
            return []
        query_tokens = self._tokenize(query)

        scores = model.get_scores(
            query_tokens
        )

        top_indices = (
            scores
            .argsort()[::-1][:top_k]
        )

        results = []

        for index in top_indices:
            document = documents[index]

            results.append(
                {
                    "chunk_id": document["chunk_id"],
                    "content": document["content"],
                    "metadata": document["metadata"],
                    "score": float(scores[index])
                }
            )

        return results