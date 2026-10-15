from collections import defaultdict

from src.rag.vector_retriever import VectorRetriever
from src.rag.bm25_retriever import BM25Retriever


class HybridRetriever:

    def __init__(
            self,
            db_url: str,
            rrf_k: int = 60,
            dense_weight: float = 0.5,
            bm25_weight: float = 0.5
    ):

        # ====================================================
        # Dense Retriever
        # ====================================================

        self.dense_retriever = VectorRetriever(
            db_url
        )


        # ====================================================
        # BM25 Retriever
        # ====================================================

        self.bm25_retriever = BM25Retriever(
            db_url
        )


        # ====================================================
        # RRF 参数
        # ====================================================

        self.rrf_k = rrf_k


        # ====================================================
        # Dense / BM25 权重
        # ====================================================

        self.dense_weight = dense_weight

        self.bm25_weight = bm25_weight


    def search(
            self,
            query: str,
            top_k: int = 5,
            candidate_k: int = 20
    ):

        # ====================================================
        # 1. Dense Retrieval
        # ====================================================

        dense_results = (
            self.dense_retriever.search(
                query=query,
                top_k=candidate_k
            )
        )


        # ====================================================
        # 2. BM25 Retrieval
        # ====================================================

        bm25_results = (
            self.bm25_retriever.search(
                query=query,
                top_k=candidate_k
            )
        )


        # ====================================================
        # 3. RRF 分数
        # ====================================================

        rrf_scores = defaultdict(
            float
        )

        document_map = {}


        # ====================================================
        # 4. Dense Ranking
        #
        # VectorRetriever 返回：
        #
        # (
        #     chunk_id,
        #     content,
        #     metadata,
        #     distance
        # )
        # ====================================================

        for rank, item in enumerate(
                dense_results,
                start=1
        ):

            chunk_id = item[0]

            content = item[1]

            metadata = item[2]

            distance = float(
                item[3]
            )


            # ================================================
            # Weighted RRF
            #
            # Dense：
            #
            # dense_weight / (rrf_k + rank)
            # ================================================

            rrf_scores[
                chunk_id
            ] += (
                self.dense_weight
                / (
                    self.rrf_k
                    + rank
                )
            )


            document_map[
                chunk_id
            ] = {

                "chunk_id":
                    chunk_id,

                "content":
                    content,

                "metadata":
                    metadata,

                "similarity":
                    1.0 - distance,

                "dense_rank":
                    rank,

                "bm25_rank":
                    None
            }


        # ====================================================
        # 5. BM25 Ranking
        # ====================================================

        for rank, item in enumerate(
                bm25_results,
                start=1
        ):

            chunk_id = item[
                "chunk_id"
            ]


            # ================================================
            # Weighted RRF
            #
            # BM25：
            #
            # bm25_weight / (rrf_k + rank)
            # ================================================

            rrf_scores[
                chunk_id
            ] += (
                self.bm25_weight
                / (
                    self.rrf_k
                    + rank
                )
            )


            # ================================================
            # Dense 和 BM25 都召回
            # ================================================

            if chunk_id in document_map:

                document_map[
                    chunk_id
                ][
                    "bm25_rank"
                ] = rank

                document_map[
                    chunk_id
                ][
                    "bm25_score"
                ] = item[
                    "bm25_score"
                ]


            # ================================================
            # 只有 BM25 召回
            # ================================================

            else:

                document_map[
                    chunk_id
                ] = {

                    "chunk_id":
                        chunk_id,

                    "content":
                        item[
                            "content"
                        ],

                    "metadata":
                        item[
                            "metadata"
                        ],

                    "similarity":
                        None,

                    "bm25_score":
                        item[
                            "bm25_score"
                        ],

                    "dense_rank":
                        None,

                    "bm25_rank":
                        rank
                }


        # ====================================================
        # 6. 根据 Weighted RRF 排序
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
        # 7. 返回 Top-K
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


        return results