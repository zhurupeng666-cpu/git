
import asyncio
from ragas.embeddings.base import BaseRagasEmbedding


class RagasBGEAdapter(BaseRagasEmbedding):
    """将现有BGE-M3向量模型适配为RAGAS Embedding接口。"""

    def __init__(self, bge_embedding):
        super().__init__()
        self.bge = bge_embedding

    def embed_text(self, text: str, **kwargs) -> list[float]:
        """同步：单条文本向量化。"""
        return self.bge.embed_text(text).tolist()

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        """异步：单条文本向量化。"""
        return await asyncio.to_thread(self.embed_text, text)

    def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
        """同步：批量文本向量化。"""
        return self.bge.embed_documents(texts).tolist()

    async def aembed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
        """异步：批量文本向量化。"""
        return await asyncio.to_thread(self.embed_texts, texts)
