from sentence_transformers import SentenceTransformer
from threading import Lock
from hashlib import sha256
import os
from src.embedding.query_vector_cache import QueryVectorCache
from src.config.model_cache import model_path
from src.config.local_inference import configure_runtime, inference_slot
from src.api.stream_control import check_cancelled

class BGEEmbedding:
    # V4正文检索与扩展问题检索复用同一份权重，避免每个Worker加载两份BGE。
    shared_model = None
    model_lock = Lock()
    query_cache = QueryVectorCache(int(os.getenv("QUERY_VECTOR_CACHE_SIZE", "512")))

    def __init__(self):
        configure_runtime()
        with self.model_lock:
            check_cancelled()
            if type(self).shared_model is None:
                type(self).shared_model = SentenceTransformer(model_path("BAAI/bge-m3"), local_files_only=True)
            self.model = type(self).shared_model

    def embed_text(
            self,
            text: str
    ):
        # 只复用完全相同的编码输入；模型实例参与key，防止更换模型后误用旧向量。
        # 缓存key不保留查询明文，不缓存检索结果、用户历史或答案。
        key = (id(self.model), sha256(text.encode("utf-8")).digest())
        def compute():
            with inference_slot():
                return self.model.encode(text, normalize_embeddings=True)
        return self.query_cache.get_or_compute(key, compute)

    def embed_documents(
            self,
            texts: list[str]
    ):
        with inference_slot():
            return self.model.encode(texts, normalize_embeddings=True,
                                     batch_size=16, show_progress_bar=True)