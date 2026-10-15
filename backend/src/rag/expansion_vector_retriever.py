from src.config.request_limits import statement_options
import psycopg

from src.embedding.bge_embedding import BGEEmbedding


class ExpansionVectorRetriever:

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.embedding_model = BGEEmbedding()

    def search(
        self,
        query: str,
        top_k: int = 10,
        view_type: str | None = None
    ):
        query_embedding = self.embedding_model.embed_text(query)

        vector_text = (
            "["
            + ",".join(
                str(float(value))
                for value in query_embedding
            )
            + "]"
        )

        if view_type is None:
            view_filter = ""
            params = (
                vector_text,
                top_k
            )

        else:
            view_filter = "WHERE v.view_type = %s"
            params = (
                vector_text,
                view_type,
                top_k
            )

        sql = f"""
            WITH view_results AS (
                SELECT
                    v.chunk_id,
                    v.view_type,
                    v.view_index,
                    r.content,
                    r.metadata,
                    v.embedding <=> %s::vector AS distance
                FROM rag_chunk_expansion_vectors v
                JOIN rag_chunks r
                    ON v.chunk_id = r.chunk_id
                {view_filter}
            ),

            best_view_per_chunk AS (
                SELECT DISTINCT ON (chunk_id)
                    chunk_id,
                    view_type,
                    view_index,
                    content,
                    metadata,
                    distance
                FROM view_results
                ORDER BY
                    chunk_id,
                    distance
            )

            SELECT
                chunk_id,
                view_type,
                view_index,
                content,
                metadata,
                distance
            FROM best_view_per_chunk
            ORDER BY distance
            LIMIT %s
        """

        with psycopg.connect(self.db_url, connect_timeout=5,
                              options=statement_options("RAG_QUERY_TIMEOUT_MS", "15000")) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    params
                )

                rows = cur.fetchall()

        results = []

        for row in rows:
            results.append(
                {
                    "chunk_id": row[0],
                    "view_type": row[1],
                    "view_index": row[2],
                    "content": row[3],
                    "metadata": row[4],
                    "distance": float(row[5])
                }
            )

        return results