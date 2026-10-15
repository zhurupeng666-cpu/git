from src.config.request_limits import statement_options
from src.embedding.bge_embedding import BGEEmbedding
from src.api.stream_control import check_cancelled
import psycopg


class VectorRetriever:
    def __init__(
        self,
        db_url
    ):
        self.db_url = db_url
        # 正文与扩展问题共用权重和推理名额，不能绕过本地模型资源限制。
        self.model = BGEEmbedding()

    def search(
        self,
        query,
        top_k=5
    ):

        # 1. query向量化
        query_embedding = self.model.embed_text(query)

        vector = (
            "["
            +
            ",".join(
                map(
                    str,
                    query_embedding.tolist()
                )
            )
            +
            "]"
        )

        # 2. pgvector相似度搜索
        sql = """
        SELECT
            chunk_id,
            content,
            metadata,
            embedding <=> %s::vector AS distance

        FROM rag_chunks

        ORDER BY
            embedding <=> %s::vector

        LIMIT %s
        """

        check_cancelled()
        # 每次查询独立事务，避免并发线程共用连接和长期idle in transaction。
        with psycopg.connect(self.db_url, connect_timeout=5,
                              options=statement_options("RAG_QUERY_TIMEOUT_MS", "15000")) as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    vector,
                    vector,
                    top_k
                )
            )

            results = cur.fetchall()

        check_cancelled()
        return results