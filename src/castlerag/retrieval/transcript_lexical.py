"""Transcript BM25 retrieval with answer-option and metadata bonus scoring.

BM25 scoring formula (SPEC §4.3):
  base BM25 score over window text x query
  + answer option overlap bonus
  + phrase-match bonus
  + day bonus
  + person bonus
  + room bonus
  + temporal-keyword bonus

Mandatory dual-path: this module provides the BM25 lane only.
Dense transcript retrieval runs through embed/omniembed.py + retrieval/search.py.
The two lanes are merged with RRF in retrieval/search.py before reranking.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from castlerag.schemas import RetrievalHit, TranscriptWindow

# Temporal keywords for bonus scoring (derived from WDL pattern)
_TEMPORAL_KEYWORDS = frozenset([
    "before", "after", "while", "during", "then", "when", "next",
    "previously", "later", "first", "last", "finally", "once",
])


def score_windows(
    bm25_index: Any,
    windows: List[TranscriptWindow],
    query: str,
    choices: Dict[str, str],
    day_hint: Optional[str] = None,
    person_hint: Optional[str] = None,
    room_hint: Optional[str] = None,
    top_k: int = 30,
) -> List[RetrievalHit]:
    """Score transcript windows with BM25 + metadata bonuses and return top-k hits."""
    raise NotImplementedError("Implemented in issue #7")
