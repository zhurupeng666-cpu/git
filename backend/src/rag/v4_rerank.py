import os
import logging
from src.rag.retrieval_timing import retrieval_stage

from src.rag.rewrite_hybrid_retriever import RewriteHybridRetriever
from src.rag.expansion_vector_retriever import ExpansionVectorRetriever
from src.rag.keyword_bm25_retriever import KeywordBM25Retriever
from src.rag.reranker import BGEReranker


class V4Rerank:
    """V4完整检索：三路召回 → 聚合去重 → BGE重排 → Consensus融合。"""

    def __init__(self, db_url: str):
        # 第一路：V2混合检索（原始/改写Query × Dense/BM25）
        self.v2_retriever = RewriteHybridRetriever(
            db_url=db_url,
            rrf_k=10,
            dense_weight=0.4,
            bm25_weight=0.6,
            original_weight=0.5,
            rewrite_weight=0.5
        )

        # 第二路：基于Synthetic Questions的向量检索
        self.question_retriever = ExpansionVectorRetriever(db_url)

        # 第三路：融合正文与扩展关键词的BM25检索
        self.keyword_retriever = KeywordBM25Retriever(db_url)

        # BGE-Reranker-v2-m3负责候选精排
        self.reranker = BGEReranker()

        # 检索共识融合参数
        self.alpha = 0.3
        self.rrf_k = 10
        # 默认按召回共识选Top20精排；配置0可恢复全部候选模式。
        self.rerank_limit = int(os.getenv("RERANK_MAX_CANDIDATES", "20"))
        if self.rerank_limit < 0:
            raise ValueError("RERANK_MAX_CANDIDATES不能小于0")

    @staticmethod
    def _add_results(candidate_map: dict, results: list, source: str):
        """按chunk_id合并候选，并保留各检索路线中的排名。"""

        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]

            if chunk_id not in candidate_map:
                candidate_map[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "sources": []
                }

            # 同一个Chunk可同时被多条检索路线召回
            candidate_map[chunk_id]["sources"].append({
                "source": source,
                "rank": rank
            })

    def _consensus_score(self, document: dict) -> float:
        """计算Chunk在三条检索路线中的倒数排名共识分。"""

        score = sum(
            1 / (self.rrf_k + item["rank"])
            for item in document["sources"]
        )
        return score / 3

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """执行BGE重排，并融合Retrieval Consensus。"""

        if not candidates:
            return []

        # 对全部候选进行BGE重排，而不是提前截断为Top5
        reranked = self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=len(candidates)
        )

        if len(reranked) != len(candidates):
            raise ValueError("BGE Reranker返回的候选数量不完整")

        # 保留召回阶段的完整Chunk和来源信息
        candidate_map = {doc["chunk_id"]: doc for doc in candidates}
        final_results = []

        for rank, item in enumerate(reranked, start=1):
            chunk_id = item["chunk_id"]
            document = candidate_map[chunk_id].copy()

            # BGE的最终排名转换为倒数排名分数
            rerank_score = 1 / (self.rrf_k + rank)

            # 计算三路检索共识分
            consensus_score = self._consensus_score(document)

            # 最终融合：BGE 70% + Consensus 30%
            final_score = (
                (1 - self.alpha) * rerank_score
                + self.alpha * consensus_score
            )

            document["rerank_rank"] = rank
            document["consensus_score"] = consensus_score
            document["final_score"] = final_score
            final_results.append(document)

        # 根据融合分数重新排序，返回最终TopK
        final_results.sort(key=lambda x: x["final_score"], reverse=True)
        return final_results[:top_k]

    def search(
            self,
            query: str,
            top_k: int = 5,
            v2_k: int = 50,
            question_k: int = 25,
            keyword_k: int = 25,
            context=None
    ) -> dict:
        """
        执行V4完整检索流程。
        :param query: 当前用户问题
        :param top_k: 最终返回文档数量
        :param v2_k: 混合检索候选数量
        :param question_k: Synthetic Question召回数量
        :param keyword_k: BM25召回数量
        :param context: LangGraph短期历史上下文
        :return: 重排序后的Chunk结果
        """


        if not query or not query.strip():
            raise ValueError("检索问题不能为空")
        if min(top_k, v2_k, question_k, keyword_k) < 1:
            raise ValueError("检索数量必须大于等于1")

        # 1. V2混合检索（Original/Rewrite × Dense/BM25）
        with retrieval_stage("hybrid_retrieval"):
            v2_response = self.v2_retriever.search(
                query=query,
                top_k=v2_k,
                candidate_k=v2_k,
                context=context
            )

        # V2返回的是字典，真正的Chunk列表位于results字段
        v2_results = v2_response["results"]

        # 2. Synthetic Question Dense检索
        with retrieval_stage("question_dense"):
            question_results = self.question_retriever.search(
                query=query,
                top_k=question_k,
                view_type="question"
            )

        # 3. Keyword Enhanced BM25检索
        with retrieval_stage("keyword_bm25"):
            keyword_results = self.keyword_retriever.search(
                query=query,
                top_k=keyword_k
            )

        # 4. 三路候选合并去重
        candidate_map = {}
        self._add_results(candidate_map, v2_results, source="v2")
        self._add_results(candidate_map, question_results, source="question_dense")
        self._add_results(candidate_map, keyword_results, source="keyword_bm25")

        candidates = list(candidate_map.values())

        # 5. 召回仍保留所有候选，CPU只对共识排名靠前的候选执行昂贵的BGE精排。
        # 分数相同时优先任一路中的高排名；Python稳定排序保留原有顺序。
        rerank_candidates = candidates
        if self.rerank_limit:
            rerank_candidates = sorted(
                candidates,
                key=lambda item: (
                    self._consensus_score(item),
                    -min(source["rank"] for source in item["sources"])
                ),
                reverse=True
            )[:max(top_k, self.rerank_limit)]
        logging.getLogger(__name__).info(
            "Retrieval candidates=%s rerank_candidates=%s",
            len(candidates), len(rerank_candidates)
        )
        with retrieval_stage("rerank"):
            top_documents = self._rerank(query, rerank_candidates, top_k)

        return {
            "query": query,
            "candidate_count": len(candidates),
            "top_documents": top_documents
        }
