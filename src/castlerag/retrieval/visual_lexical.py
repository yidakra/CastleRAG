"""Visual-text BM25 retrieval over captions, OCR and scene graphs.

Query-time counterpart of :mod:`castlerag.index.visual_lexical`.  Scores
the per-clip / per-event visual documents with BM25 plus the same style of
answer-option and metadata bonuses the transcript lane uses, and returns
``RetrievalHit`` rows for ``main_clip`` / ``main_event_summary`` so they
fuse with the dense multimodal lanes on ``record_id``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from castlerag.config import RetrievalConfig
from castlerag.schemas import RetrievalHit

_TOKEN_RE = re.compile(r"\b\w+\b")

# Defaults used when the retrieval config predates the lane (e.g. the
# SimpleNamespace configs in older tests). Read from RetrievalConfig so there
# is a single source of truth; configs/base.yaml documents the same values
# and a test guards against drift.
_DEFAULT_RETRIEVAL = RetrievalConfig()
DEFAULT_VISUAL_TEXT_TOP_K: int = _DEFAULT_RETRIEVAL.visual_text_top_k
DEFAULT_VISUAL_TEXT_ROUTE_WEIGHTS: Dict[str, float] = dict(
    _DEFAULT_RETRIEVAL.visual_text_route_weights
)


# Function words that make almost any caption "match" a question; the evidence
# gate ignores them so it tests for shared content terms only.
_STOPWORDS = frozenset(
    "a an the is are was were be been do does did what which who whom whose "
    "where when why how of on in at to for with by from and or it its this that "
    "these those there here his her their our your my he she they we you i s "
    "no not yes has have had can will would could should may might any some "
    "as if then than so but".split()
)


def _doc_features(visual_index: Any, docs: List[Any]) -> tuple[List[str], List[set]]:
    """Return per-doc (lowercase text, token set), computed once per index.

    The corpus is static for the lifetime of a loaded index, so the features
    are cached on the bundle after the first query instead of re-tokenising
    every document on every request.
    """
    cached = getattr(visual_index, "_scorer_features", None)
    if cached is not None and len(cached[0]) == len(docs):
        return cached
    lowers = [doc.text.lower() for doc in docs]
    token_sets = [set(_tokenize(doc.text)) for doc in docs]
    try:
        visual_index._scorer_features = (lowers, token_sets)
    except (AttributeError, TypeError):  # read-only fake indexes in tests
        pass
    return lowers, token_sets


def score_visual_docs(
    visual_index: Any,
    query: str,
    choices: Mapping[str, str],
    day_hint: Optional[str] = None,
    person_hint: Optional[str] = None,
    room_hint: Optional[str] = None,
    top_k: int = DEFAULT_VISUAL_TEXT_TOP_K,
    exclude_cameras: Optional[Sequence[str]] = None,
) -> List[RetrievalHit]:
    """Score visual-text docs with BM25 + bonuses and return the top-k hits.

    ``exclude_cameras`` are skipped *before* ranking and truncation, so a
    rejected camera can never consume lane slots (the dense lanes filter
    server-side, before ``limit``, and this lane must match).

    ``visual_index`` is a :class:`~castlerag.index.visual_lexical.VisualBM25IndexBundle`
    (or anything exposing ``bm25`` and ``docs``).  An index with no documents
    yields no hits.
    """
    docs = list(getattr(visual_index, "docs", None) or [])
    bm25 = getattr(visual_index, "bm25", None)
    if not docs or bm25 is None or top_k <= 0:
        return []
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    base_scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float32)
    query_lower = query.lower()
    answer_tokens: set[str] = set()
    answer_phrases: List[str] = []
    for choice in choices.values():
        # Function words in a choice ("a Bosch dishwasher", "in the kitchen")
        # must not count as overlap, or the evidence gate below is defeated.
        answer_tokens.update(set(_tokenize(choice)) - _STOPWORDS)
        phrase = choice.strip().lower()
        # A phrase of pure function words ("in the kitchen") would substring-
        # match most captions; it only counts if it carries a content token.
        if len(phrase.split()) > 1 and set(_tokenize(phrase)) - _STOPWORDS:
            answer_phrases.append(phrase)

    excluded = set(exclude_cameras or ())
    content_query_tokens = set(query_tokens) - _STOPWORDS
    lowers, token_sets = _doc_features(visual_index, docs)
    scored: List[tuple[float, Any]] = []
    for idx, doc in enumerate(docs):
        if excluded and doc.camera_id in excluded:
            continue
        text_lower = lowers[idx]
        doc_tokens = token_sets[idx]
        # rank_bm25 yields 0 or slightly negative scores for terms present in
        # most of a small corpus, so the sign of the BM25 score is not a usable
        # evidence test; clamp it and gate on actual term overlap instead.
        score = max(0.0, float(base_scores[idx]))

        # Answer-option overlap: brand names, labels and prices show up in
        # OCR verbatim, so a single shared token is a strong signal here.
        answer_overlap = len(answer_tokens.intersection(doc_tokens))
        phrase_hits = sum(1 for phrase in answer_phrases if phrase in text_lower)
        # A raw-substring match of an all-stopword query is not evidence; it
        # only counts when the query carries a content token (in which case
        # the first clause already holds).
        has_lexical_evidence = (
            bool(content_query_tokens & doc_tokens)
            or answer_overlap > 0
            or phrase_hits > 0
        )
        if not has_lexical_evidence:
            # The metadata bonuses below must not promote an unrelated
            # same-day / same-person doc into the lane, where RRF would hand
            # it a vote purely for being ranked.
            continue
        score += 0.15 * answer_overlap
        if query_lower in text_lower:
            score += 1.0
        score += 0.4 * phrase_hits

        if day_hint and doc.day == day_hint:
            score += 0.75
        if person_hint and (
            (doc.participant_id and doc.participant_id.lower() == person_hint.lower())
            or person_hint.lower() in text_lower
        ):
            score += 0.75
        if room_hint and (
            (doc.room and doc.room.lower() == room_hint.lower())
            or room_hint.lower() in text_lower
        ):
            score += 0.5

        scored.append((score, doc))

    ranked = sorted(
        scored,
        key=lambda item: (-item[0], item[1].absolute_start, item[1].record_id),
    )[:top_k]
    return [
        RetrievalHit(
            rank=rank,
            score=score,
            point_id=f"visual_lexical:{doc.record_id}",
            record_id=doc.record_id,
            source_type=doc.source_type,
            modality=doc.modality,
            day=doc.day,
            camera_id=doc.camera_id,
            participant_id=doc.participant_id,
            room=doc.room,
            hour=doc.hour,
            start_seconds=doc.start_seconds,
            end_seconds=doc.end_seconds,
            absolute_start=doc.absolute_start,
            absolute_end=doc.absolute_end,
            transcript_text=doc.transcript_text,
            event_summary=doc.event_summary,
            ocr_text=doc.ocr_text,
            asset_path=doc.asset_path,
            sampled_frame_paths=list(doc.sampled_frame_paths),
        )
        for rank, (score, doc) in enumerate(ranked, start=1)
    ]


def visual_lane_top_k(retrieval_cfg: Any) -> int:
    """Return the configured lane size, tolerating configs without the key."""
    return int(getattr(retrieval_cfg, "visual_text_top_k", DEFAULT_VISUAL_TEXT_TOP_K))


def visual_lane_weight(route: str, retrieval_cfg: Any) -> float:
    """Return the RRF weight of the visual-text lane for ``route``.

    Reads ``retrieval_cfg.visual_text_route_weights`` when present and falls
    back to :data:`DEFAULT_VISUAL_TEXT_ROUTE_WEIGHTS`; unknown routes get 1.0.
    """
    configured = getattr(retrieval_cfg, "visual_text_route_weights", None) or {}
    if route in configured:
        return float(configured[route])
    return float(DEFAULT_VISUAL_TEXT_ROUTE_WEIGHTS.get(route, 1.0))


def _tokenize(text: str) -> List[str]:
    """Lowercase and split text into word tokens using the module regex."""
    return _TOKEN_RE.findall(text.lower())
