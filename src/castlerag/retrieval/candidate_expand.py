"""Expansion from candidate videos to frame packs and linked evidence packs."""

from __future__ import annotations

from typing import Any, Dict, List

from castlerag.schemas import RetrievalHit


def expand_candidates(
    hits: List[RetrievalHit],
    max_candidate_videos: int = 4,
    frames_per_candidate: int = 32,
) -> List[Dict[str, Any]]:
    """Collapse retrieval hits into up to max_candidate_videos evidence packs.

    Each pack includes sampled frame paths, linked transcript windows,
    event summaries, OCR spans, and auxiliary notes for the reranker.
    """
    raise NotImplementedError("Implemented in issue #8")
