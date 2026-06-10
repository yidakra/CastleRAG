"""Indexing utilities for CastleRAG."""

from castlerag.index.io import (
    load_aux_records,
    load_clip_records,
    load_embedding_cache,
    load_event_summary_records,
    load_transcript_windows,
    write_embedding_cache,
    write_json,
    write_jsonl_records,
)
from castlerag.index.pipeline import (
    build_bm25_artifact,
    build_qdrant_index,
    cache_dense_embeddings,
    discover_chunk_artifacts,
    filter_records,
    load_chunk_records,
)
from castlerag.index.qdrant import (
    bootstrap_collection,
    build_point_batches,
    create_collection,
    create_payload_indexes,
    get_client,
    record_to_qdrant_point,
    upsert_batch,
)
from castlerag.index.transcript_lexical import (
    BM25IndexBundle,
    build_bm25_index,
    load_bm25_index,
)

__all__ = [
    "BM25IndexBundle",
    "bootstrap_collection",
    "build_bm25_artifact",
    "build_bm25_index",
    "build_point_batches",
    "build_qdrant_index",
    "cache_dense_embeddings",
    "create_collection",
    "create_payload_indexes",
    "discover_chunk_artifacts",
    "filter_records",
    "get_client",
    "load_aux_records",
    "load_bm25_index",
    "load_chunk_records",
    "load_clip_records",
    "load_embedding_cache",
    "load_event_summary_records",
    "load_transcript_windows",
    "record_to_qdrant_point",
    "upsert_batch",
    "write_embedding_cache",
    "write_json",
    "write_jsonl_records",
]
