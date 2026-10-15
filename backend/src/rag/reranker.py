import torch
import os
from src.config.model_cache import model_path
from src.api.stream_control import check_cancelled
from src.config.local_inference import configure_runtime, inference_slot
from FlagEmbedding import FlagReranker


class BGEReranker:

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3"
    ):
        configure_runtime()
        self.batch_size = int(os.getenv("RERANK_BATCH_SIZE", "4"))
        if self.batch_size < 1:
            raise ValueError("RERANK_BATCH_SIZE必须大于0")
        # GPU 可用时使用 FP16，提高推理速度
        use_fp16 = torch.cuda.is_available()

        self.model = FlagReranker(
            model_path(model_name),
            use_fp16=use_fp16
        )

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5
    ):
        if not documents:
            return []

        # 为每个候选 Chunk 构造 Query + Passage 文本对
        pairs = [
            [
                query,
                document["content"]
            ]
            for document in documents
        ]

        scores = []
        # 显式拆批：客户端取消后不继续遍历剩余候选，单个原生张量运算无法强制打断。
        for offset in range(0, len(pairs), self.batch_size):
            check_cancelled()
            batch = pairs[offset:offset + self.batch_size]
            with inference_slot():
                batch_scores = self.model.compute_score(batch, batch_size=self.batch_size)
            if not isinstance(batch_scores, list):
                batch_scores = [batch_scores]
            if len(batch_scores) != len(batch):
                raise ValueError("重排批次返回的分数数量不匹配")
            scores.extend(batch_scores)

        results = []

        for document, score in zip(
            documents,
            scores
        ):
            result = document.copy()
            result["rerank_score"] = float(score)

            results.append(result)

        # Reranker 分数越高，相关性越强
        results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return results[:top_k]