from src.rag.rewrite_hybrid_retriever import RewriteHybridRetriever
from src.rag.reranker import BGEReranker


class RerankRetriever:

    def __init__(
        self,
        db_url: str,
        rerank_k: int = 50
    ):
        # V2 最终检索参数
        self.hybrid_retriever = RewriteHybridRetriever(
            db_url=db_url,
            rrf_k=10,
            dense_weight=0.4,
            bm25_weight=0.6,
            original_weight=0.5,
            rewrite_weight=0.5
        )

        self.reranker = BGEReranker()

        # V2 先召回多少个候选交给 Reranker
        self.rerank_k = rerank_k

    def search(
        self,
        query: str,
        top_k: int = 5
    ):
        # V2：Hybrid Retrieval + Query Rewrite + RRF
        hybrid_result = self.hybrid_retriever.search(
            query=query,
            top_k=self.rerank_k,
            candidate_k=self.rerank_k
        )

        candidates = hybrid_result["results"]

        # V3：使用原始用户 Query 对候选 Chunk 重新排序
        reranked_results = self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=top_k
        )

        return {
            "original_query": hybrid_result["original_query"],
            "rewritten_query": hybrid_result["rewritten_query"],
            "results": reranked_results
        }