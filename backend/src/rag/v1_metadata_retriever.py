import psycopg

from src.embedding.bge_embedding import (
    BGEEmbedding
)

from src.rag.domain_router import (
    DomainRouter
)


class V1MetadataRetriever:

    def __init__(
            self,
            db_url: str
    ):

        self.db_url = db_url

        # Query Embedding
        self.embedding_model = BGEEmbedding()

        # Top-2 Domain Router
        self.domain_router = DomainRouter()


    def search(
            self,
            query: str,
            top_k: int = 5
    ):

        # ====================================================
        # 1. Domain Router
        # ====================================================

        domains = self.domain_router.route(
            query
        )

        primary_domain = domains[0]
        secondary_domain = domains[1]


        # ====================================================
        # 2. Query Embedding
        # ====================================================

        query_embedding = (
            self.embedding_model.embed_text(
                query
            )
        )


        vector_string = (
            "["
            + ",".join(
                map(
                    str,
                    query_embedding.tolist()
                )
            )
            + "]"
        )


        # ====================================================
        # 3. Metadata Filter + Dense Retrieval
        # ====================================================

        sql = """
        SELECT
            chunk_id,
            content,
            metadata,
            embedding <=> %s::vector AS distance

        FROM rag_chunks

        WHERE
            metadata->>'domain' IN (%s, %s)

        ORDER BY
            embedding <=> %s::vector

        LIMIT %s
        """


        with psycopg.connect(
            self.db_url
        ) as conn:

            with conn.cursor() as cur:

                cur.execute(
                    sql,
                    (
                        vector_string,
                        primary_domain,
                        secondary_domain,
                        vector_string,
                        top_k
                    )
                )

                rows = cur.fetchall()


        # ====================================================
        # 4. 返回结果
        # ====================================================

        results = []


        for row in rows:

            chunk_id = row[0]
            content = row[1]
            metadata = row[2]
            distance = float(
                row[3]
            )

            results.append(
                {
                    "chunk_id":
                        chunk_id,

                    "content":
                        content,

                    "metadata":
                        metadata,

                    "distance":
                        distance,

                    "similarity":
                        1 - distance,
                }
            )


        return {
            "primary_domain":
                primary_domain,

            "secondary_domain":
                secondary_domain,

            "results":
                results,
        }