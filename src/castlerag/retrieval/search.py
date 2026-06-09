"""Query encoding, modality-scoped Qdrant search, and RRF score fusion.

Dual-path transcript retrieval is mandatory (SPEC §4.3):
  1. BM25 lexical scoring
  2. OmniEmbed dense retrieval
  → merged with Reciprocal Rank Fusion (k=60)

Dense multimodal retrieval runs separate filtered Qdrant searches per
source_type, then fuses with RRF before merging with the transcript lane.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from castlerag.schemas import EvalQuestion, RetrievalHit
from castlerag.routing.question_router import RouteHints


def reciprocal_rank_fusion(
    ranked_lists: List[List[RetrievalHit]],
    k: int = 60,
) -> List[RetrievalHit]:
    """Fuse multiple ranked lists into one using RRF(k).

    RRF score for a document d = sum_r 1/(k + rank_r(d))
    """
    raise NotImplementedError("Implemented in issue #7")


def retrieve(
    question: EvalQuestion,
    hints: RouteHints,
    qdrant_client: Any,
    collection_name: str,
    bm25_index: Any,
    embed_client: Any,
    retrieval_cfg: Any,
) -> List[RetrievalHit]:
    """Full dual-path retrieval for one question.

    Returns a ranked list of RetrievalHits (at most max_evidence_rows),
    ready for reranking.
    """
    raise NotImplementedError("Implemented in issue #7")
