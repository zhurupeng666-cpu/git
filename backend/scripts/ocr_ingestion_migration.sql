-- OCR文档版本记录；不改变原有切片、业务及认证表。
CREATE TABLE IF NOT EXISTS public.rag_document_versions (
    document_id TEXT PRIMARY KEY,
    source_relpath TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    ingest_signature TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    chunk_count INTEGER NOT NULL CHECK(chunk_count > 0),
    review_count INTEGER NOT NULL DEFAULT 0 CHECK(review_count >= 0),
    published_at TIMESTAMPTZ NOT NULL
);
