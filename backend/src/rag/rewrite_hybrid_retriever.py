from src.rag.retrieval_timing import retrieval_stage
from collections import defaultdict
from src.rag.vector_retriever import VectorRetriever
from src.rag.bm25_retriever import BM25Retriever
from src.rag.query_rewriter import QueryRewriter


class RewriteHybridRetriever:

    def __init__(
            self,
            db_url: str,
            rrf_k: int = 10,
            dense_weight: float = 0.5,
            bm25_weight: float = 0.5,
            original_weight: float = 0.5,
            rewrite_weight: float = 0.5
    ):

        self.dense_retriever = VectorRetriever(
            db_url
        )

        self.bm25_retriever = BM25Retriever(
            db_url
        )

        self.query_rewriter = QueryRewriter()

        self.rrf_k = rrf_k

        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        self.original_weight = original_weight
        self.rewrite_weight = rewrite_weight


    def search(
            self,
            query: str,
            top_k: int = 5,
            candidate_k: int = 20,
            context = None

    ):

        # ====================================================
        # 1. Query Rewrite
        # ====================================================

        with retrieval_stage("query_rewrite"):
            rewritten_query = (
                self.query_rewriter.rewrite(
                    query=query,
                    context=context
                )
            )


        # ====================================================
        # 2. Original Query Retrieval
        # ====================================================

        with retrieval_stage("original_dense"):
            original_dense = (
                self.dense_retriever.search(
                    query=query,
                    top_k=candidate_k
                )
            )

        with retrieval_stage("original_bm25"):
            original_bm25 = (
                self.bm25_retriever.search(
                    query=query,
                    top_k=candidate_k
                )
            )


        # ====================================================
        # 3. Rewritten Query Retrieval
        # ====================================================

        with retrieval_stage("rewrite_dense"):
            rewrite_dense = (
                self.dense_retriever.search(
                    query=rewritten_query,
                    top_k=candidate_k
                )
            )

        with retrieval_stage("rewrite_bm25"):
            rewrite_bm25 = (
                self.bm25_retriever.search(
                    query=rewritten_query,
                    top_k=candidate_k
                )
            )


        # ====================================================
        # 4. RRF
        # ====================================================

        rrf_scores = defaultdict(
            float
        )

        document_map = {}


        # ====================================================
        # Original Dense
        # ====================================================

        for rank, item in enumerate(
                original_dense,
                start=1
        ):

            chunk_id = item[0]

            rrf_scores[
                chunk_id
            ] += (
                self.original_weight
                * self.dense_weight
                / (
                    self.rrf_k
                    + rank
                )
            )

            document_map[
                chunk_id
            ] = {
                "chunk_id":
                    item[0],

                "content":
                    item[1],

                "metadata":
                    item[2]
            }


        # ====================================================
        # Original BM25
        # ====================================================

        for rank, item in enumerate(
                original_bm25,
                start=1
        ):

            chunk_id = item[
                "chunk_id"
            ]

            rrf_scores[
                chunk_id
            ] += (
                self.original_weight
                * self.bm25_weight
                / (
                    self.rrf_k
                    + rank
                )
            )

            if chunk_id not in document_map:

                document_map[
                    chunk_id
                ] = {
                    "chunk_id":
                        item["chunk_id"],

                    "content":
                        item["content"],

                    "metadata":
                        item["metadata"]
                }


        # ====================================================
        # Rewrite Dense
        # ====================================================

        for rank, item in enumerate(
                rewrite_dense,
                start=1
        ):

            chunk_id = item[0]

            rrf_scores[
                chunk_id
            ] += (
                self.rewrite_weight
                * self.dense_weight
                / (
                    self.rrf_k
                    + rank
                )
            )

            if chunk_id not in document_map:

                document_map[
                    chunk_id
                ] = {
                    "chunk_id":
                        item[0],

                    "content":
                        item[1],

                    "metadata":
                        item[2]
                }


        # ====================================================
        # Rewrite BM25
        # ====================================================

        for rank, item in enumerate(
                rewrite_bm25,
                start=1
        ):

            chunk_id = item[
                "chunk_id"
            ]

            rrf_scores[
                chunk_id
            ] += (
                self.rewrite_weight
                * self.bm25_weight
                / (
                    self.rrf_k
                    + rank
                )
            )

            if chunk_id not in document_map:

                document_map[
                    chunk_id
                ] = {
                    "chunk_id":
                        item["chunk_id"],

                    "content":
                        item["content"],

                    "metadata":
                        item["metadata"]
                }


        # ====================================================
        # 5. 最终排序
        # ====================================================

        ranked_chunk_ids = sorted(

            rrf_scores.keys(),

            key=lambda chunk_id:
                rrf_scores[
                    chunk_id
                ],

            reverse=True
        )


        # ====================================================
        # 6. Top-K
        # ====================================================

        results = []


        for chunk_id in ranked_chunk_ids[
            :top_k
        ]:

            item = document_map[
                chunk_id
            ].copy()

            item[
                "rrf_score"
            ] = rrf_scores[
                chunk_id
            ]

            results.append(
                item
            )


        return {
            "original_query":
                query,

            "rewritten_query":
                rewritten_query,

            "results":
                results
        }