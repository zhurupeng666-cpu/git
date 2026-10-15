"""发布后按版本刷新进程内BM25快照；一次查询固定使用同一组文档和分数。"""

import logging
import os
from threading import Lock
from time import monotonic
import psycopg
from rank_bm25 import BM25Okapi


class RefreshableBM25:
    def __init__(self, db_url, loader, tokenizer, text_builder, documents, model):
        self.db_url = db_url
        self.loader = loader
        self.tokenizer = tokenizer
        self.text_builder = text_builder
        self.snapshot = (documents, model)
        self.revision = None
        self.checked_at = 0
        self.lock = Lock()
        self.interval = float(os.getenv("KNOWLEDGE_INDEX_POLL_SECONDS", "5"))
        if self.interval < 0:
            raise ValueError("索引检查间隔不能小于0")

    def get(self):
        if monotonic() - self.checked_at < self.interval:
            return self.snapshot
        with self.lock:
            if monotonic() - self.checked_at < self.interval:
                return self.snapshot
            with psycopg.connect(
                self.db_url, connect_timeout=5, options="-c statement_timeout=5000"
            ) as conn:
                revision = conn.execute(
                    "SELECT count(*),max(metadata->>'published_at') FROM public.rag_chunks"
                ).fetchone()
            if revision != self.revision:
                documents = self.loader()
                corpus = [self.tokenizer(self.text_builder(document)) for document in documents]
                model = BM25Okapi(corpus) if corpus else None
                # 整体替换引用，其他查询持有旧快照也不会发生索引和正文错位。
                self.snapshot = (documents, model)
                self.revision = revision
                logging.getLogger(__name__).info(
                    "Knowledge BM25 snapshot refreshed: documents=%d", len(documents)
                )
            self.checked_at = monotonic()
            return self.snapshot
