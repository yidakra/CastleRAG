"""Transcript BM25 retrieval with answer-option and metadata bonus scoring."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from castlerag.schemas import RetrievalHit, TranscriptWindow

_TOKEN_RE = re.compile(r"\b\w+\b")

# Temporal keywords for bonus scoring (derived from WDL pattern)
_TEMPORAL_KEYWORDS = frozenset(
    [
        "before",
        "after",
        "while",
        "during",
        "then",
        "when",
        "next",
        "previously",
        "later",
        "first",
        "last",
        "finally",
        "once",
    ]
)


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
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    base_scores = np.asarray(bm25_index.bm25.get_scores(query_tokens), dtype=np.float32)
    query_lower = query.lower()
    answer_tokens = set()
    answer_phrases: List[str] = []
    for choice in choices.values():
        answer_tokens.update(_tokenize(choice))
        phrase = choice.strip().lower()
        if len(phrase.split()) > 1:
            answer_phrases.append(phrase)

    query_temporal = _TEMPORAL_KEYWORDS.intersection(query_tokens)
    scored: List[tuple[float, TranscriptWindow]] = []
    for idx, window in enumerate(windows):
        transcript_lower = window.transcript_text.lower()
        transcript_tokens = set(_tokenize(window.transcript_text))
        score = float(base_scores[idx])

        # Option overlap helps exact lexical matches on answer words and entities.
        score += 0.15 * len(answer_tokens.intersection(transcript_tokens))

        # Phrase matches are a strong signal for exact quoted or named spans.
        if query_lower in transcript_lower:
            score += 1.0
        score += 0.4 * sum(1 for phrase in answer_phrases if phrase in transcript_lower)

        if day_hint and window.day == day_hint:
            score += 0.75
        if person_hint and (
            (
                window.participant_id
                and window.participant_id.lower() == person_hint.lower()
            )
            or person_hint.lower() in transcript_lower
        ):
            score += 0.75
        if room_hint and (
            (window.room and window.room.lower() == room_hint.lower())
            or room_hint.lower() in transcript_lower
        ):
            score += 0.5

        if query_temporal and query_temporal.intersection(transcript_tokens):
            score += 0.5

        scored.append((score, window))

    ranked = sorted(
        scored,
        key=lambda item: (
            -item[0],
            item[1].absolute_start,
            item[1].transcript_window_id,
        ),
    )[:top_k]
    return [
        RetrievalHit(
            rank=rank,
            score=score,
            point_id=f"lexical:{window.transcript_window_id}",
            record_id=window.transcript_window_id,
            source_type="transcript_window",
            modality="text",
            day=window.day,
            camera_id=window.camera_id,
            participant_id=window.participant_id,
            absolute_start=window.absolute_start,
            absolute_end=window.absolute_end,
            transcript_text=window.transcript_text,
        )
        for rank, (score, window) in enumerate(ranked, start=1)
    ]


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())
