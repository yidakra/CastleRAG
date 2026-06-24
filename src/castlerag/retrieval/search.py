"""Query encoding, modality-scoped Qdrant search, and RRF score fusion."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from castlerag.retrieval.filters import build_filter
from castlerag.retrieval.transcript_lexical import score_windows
from castlerag.routing.question_router import RouteHints
from castlerag.schemas import EvalQuestion, RetrievalHit


def reciprocal_rank_fusion(
    ranked_lists: List[List[RetrievalHit]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[RetrievalHit]:
    """Fuse multiple ranked lists into one using weighted RRF(k).

    Each list contributes ``w / (k + rank)`` to a hit's score. When
    ``weights`` is None all lists are weighted equally (w=1.0). Pass
    route-aware weights to boost trusted sources (e.g. BM25 for speech
    questions, video lanes for visual questions).
    """
    by_record: Dict[str, RetrievalHit] = {}
    scores = defaultdict(float)
    max_raw: Dict[str, float] = {}

    for i, ranked in enumerate(ranked_lists):
        w = (weights[i] if i < len(weights) else 1.0) if weights is not None else 1.0
        for rank, hit in enumerate(ranked, start=1):
            scores[hit.record_id] += w / (k + rank)
            existing = by_record.get(hit.record_id)
            if existing is None or hit.score > existing.score:
                by_record[hit.record_id] = hit
            # Preserve the best raw cosine similarity seen for this record across
            # all query variants and modality lanes. Seed from the existing value
            # (not 0.0) so a genuinely negative cosine is not clamped upward.
            if hit.raw_score is not None:
                prev = max_raw.get(hit.record_id)
                max_raw[hit.record_id] = (
                    hit.raw_score if prev is None else max(prev, hit.raw_score)
                )

    fused = []
    for record_id, hit in by_record.items():
        update: Dict[str, object] = {"score": scores[record_id]}
        if record_id in max_raw:
            update["raw_score"] = max_raw[record_id]
        fused.append(hit.model_copy(update=update))

    fused.sort(
        key=lambda hit: (
            -hit.score,
            hit.absolute_start if hit.absolute_start is not None else float("inf"),
            hit.record_id,
        )
    )
    return [
        hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(fused, start=1)
    ]


def retrieve(
    question: EvalQuestion,
    hints: RouteHints,
    qdrant_client: Any,
    collection_name: str,
    bm25_index: Any,
    embed_client: Any,
    retrieval_cfg: Any,
) -> List[RetrievalHit]:
    """Full dual-path retrieval for one question."""
    query_variants = _query_variants(question, hints)
    transcript_bm25 = score_windows(
        bm25_index=bm25_index,
        windows=bm25_index.windows,
        query=question.query,
        choices=question.answers,
        day_hint=hints.day,
        person_hint=hints.participant,
        room_hint=hints.room,
        top_k=retrieval_cfg.transcript_top_k,
    )
    # The dense lanes hard-exclude rejected cameras server-side, but BM25 runs
    # locally and is fused in via RRF — so drop excluded cameras here too, or
    # they leak back through the transcript lane.
    if hints.exclude_cameras:
        _excluded = set(hints.exclude_cameras)
        transcript_bm25 = [
            hit for hit in transcript_bm25 if hit.camera_id not in _excluded
        ]
    query_vectors = np.asarray(
        embed_client.embed_texts(query_variants), dtype=np.float32
    )
    if query_vectors.ndim != 2:
        raise ValueError(
            f"Expected 2D query embedding matrix, got shape {query_vectors.shape}"
        )

    # Per-variant fusion weights: choices-expanded (index 1 for MCQ) is
    # down-weighted because appending all four answer options shifts the
    # embedding away from the core semantic/visual query signal.
    variant_weights = _query_variant_weights(question, hints)

    # For speech-heavy questions, boost BM25 — it captures exact lexical
    # matches that dense embeddings may spread across synonyms.
    bm25_w = 2.0 if hints.route == "speech_text" else 1.0

    transcript_dense_lists = [
        _dense_search(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=retrieval_cfg.transcript_top_k,
            source_type="transcript_window",
            modality="text",
            day=hints.day,
            participant_id=hints.participant,
            # room is deliberately NOT a hard dense filter: ego clips/windows
            # carry room=None (only fixed cameras set it), so filtering dense
            # retrieval by hints.room zeroes out all ego evidence in ego scope
            # (issue #50). Room stays a soft signal via BM25 (room_hint above)
            # and the reranker.
            exclude_camera_ids=hints.exclude_cameras,
        )
        for query_vector in query_vectors
    ]
    # No intermediate cap here — let _collapse_hits apply the route budget
    # after all lanes are fused. Capping early discards high-quality
    # transcript hits before they can compete with multimodal evidence.
    transcript_lane = reciprocal_rank_fusion(
        [transcript_bm25, *transcript_dense_lists],
        k=retrieval_cfg.rrf_k,
        weights=[bm25_w, *variant_weights],
    )

    multimodal_lists: List[List[RetrievalHit]] = []
    multimodal_weights: List[float] = []
    multimodal_specs = [
        ("main_event_summary", "text", retrieval_cfg.event_summary_top_k),
        ("main_clip", "video", retrieval_cfg.video_top_k),
        ("aux_photo", "image", retrieval_cfg.photo_top_k),
        ("aux_video", "video", retrieval_cfg.aux_video_top_k),
        ("aux_heartrate", "text", retrieval_cfg.heartrate_top_k),
        ("aux_gaze", "text", retrieval_cfg.gaze_top_k),
        ("aux_thermal", "image", retrieval_cfg.thermal_top_k),
    ]
    for source_type, modality, limit in multimodal_specs:
        for qi, query_vector in enumerate(query_vectors):
            hits = _dense_search(
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                source_type=source_type,
                modality=modality,
                day=hints.day,
                participant_id=hints.participant,
                # room intentionally omitted as a hard filter; see transcript
                # lane above and issue #50.
                exclude_camera_ids=hints.exclude_cameras,
            )
            if hits:
                hits = _apply_score_thresholds(
                    hits, retrieval_cfg.modality_score_thresholds
                )
                if hits:
                    multimodal_lists.append(hits)
                    multimodal_weights.append(variant_weights[qi])

    multimodal_lane = reciprocal_rank_fusion(
        multimodal_lists,
        k=retrieval_cfg.rrf_k,
        weights=multimodal_weights if multimodal_weights else None,
    )

    # Route-aware final merge: boost the lane that carries the strongest
    # signal for this question type.
    if hints.route == "speech_text":
        final_weights: Optional[List[float]] = [2.0, 1.0]  # transcript dominates
    elif hints.route == "static_visual":
        final_weights = [1.0, 2.0]  # visual dominates
    else:
        final_weights = None  # equal weighting for temporal / mixed

    merged = reciprocal_rank_fusion(
        [transcript_lane, multimodal_lane],
        k=retrieval_cfg.rrf_k,
        weights=final_weights,
    )
    return _collapse_hits(merged, hints, retrieval_cfg)


def _query_variants(question: EvalQuestion, hints: RouteHints) -> List[str]:
    """Return query variants: bare, choices-expanded, and optionally entity-focused.

    The choices-expanded variant is skipped for free-form (open) questions: there
    the choices are blank placeholders, so embedding "Choices: A . B . C . D ."
    only injects noise into the dense query and drags retrieval off-topic.
    """
    variants = [question.query]
    if not question.is_free_form():
        variants.append(
            f"{question.query} Choices: "
            f"A {question.answers['a']}. "
            f"B {question.answers['b']}. "
            f"C {question.answers['c']}. "
            f"D {question.answers['d']}."
        )
    if hints.llm_key_entities:
        entity_str = " ".join(hints.llm_key_entities)
        variants.append(f"{question.query} {entity_str}")
    return variants


def _query_variant_weights(question: EvalQuestion, hints: RouteHints) -> List[float]:
    """Return per-variant fusion weights matching the order from _query_variants.

    Choices-expanded is down-weighted (0.7) because appending all four MCQ
    options shifts the embedding away from the core semantic/visual query.
    Entity-focused is slightly down-weighted (0.9) — useful but narrower than
    the bare query.
    """
    weights = [1.0]  # bare query
    if not question.is_free_form():
        weights.append(0.7)  # choices-expanded
    if hints.llm_key_entities:
        weights.append(0.9)  # entity-focused
    return weights


def _apply_score_thresholds(
    hits: List[RetrievalHit],
    thresholds: Dict[str, float],
) -> List[RetrievalHit]:
    """Drop hits below per-modality minimum similarity scores."""
    if not thresholds:
        return hits
    return [h for h in hits if h.score >= thresholds.get(h.modality, 0.0)]


def _dense_search(
    *,
    qdrant_client: Any,
    collection_name: str,
    query_vector: List[float],
    limit: int,
    source_type: str,
    modality: str,
    day: Optional[str] = None,
    camera_id: Optional[str] = None,
    participant_id: Optional[str] = None,
    room: Optional[str] = None,
    time_range_start_ms: Optional[int] = None,
    time_range_end_ms: Optional[int] = None,
    has_speech: Optional[bool] = None,
    exclude_camera_ids: Optional[Sequence[str]] = None,
) -> List[RetrievalHit]:
    """Run one filtered dense Qdrant search and normalize the results."""
    query_filter = build_filter(
        day=day,
        camera_id=camera_id,
        participant_id=participant_id,
        room=room,
        modality=modality,
        source_type=source_type,
        time_range_start_ms=time_range_start_ms,
        time_range_end_ms=time_range_end_ms,
        has_speech=has_speech,
        exclude_camera_ids=exclude_camera_ids,
    )
    response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    points = getattr(response, "points", response)
    hits: List[RetrievalHit] = []
    for rank, point in enumerate(points, start=1):
        payload = dict(getattr(point, "payload", {}) or {})
        raw = float(getattr(point, "score"))
        hits.append(
            RetrievalHit(
                rank=rank,
                score=raw,
                raw_score=raw,
                point_id=str(getattr(point, "id", payload.get("point_id", ""))),
                record_id=str(payload["record_id"]),
                source_type=str(payload["source_type"]),
                modality=str(payload["modality"]),
                day=payload.get("day"),
                camera_id=payload.get("camera_id"),
                participant_id=payload.get("participant_id"),
                room=payload.get("room"),
                hour=payload.get("hour"),
                start_seconds=payload.get("start_seconds"),
                end_seconds=payload.get("end_seconds"),
                absolute_start=payload.get("absolute_start"),
                absolute_end=payload.get("absolute_end"),
                transcript_text=payload.get("transcript_text"),
                event_summary=payload.get("event_summary"),
                ocr_text=payload.get("ocr_text"),
                asset_path=payload.get("asset_path"),
                sampled_frame_paths=payload.get("sampled_frame_paths") or [],
            )
        )
    return hits


def _collapse_hits(
    hits: List[RetrievalHit],
    hints: RouteHints,
    retrieval_cfg: Any,
) -> List[RetrievalHit]:
    """Apply per-source budgets and re-rank hits according to route priority."""
    transcript_budget = min(
        retrieval_cfg.transcript_top_k,
        hints.evidence_profile.transcript_budget,
    )
    max_candidate_videos = min(
        retrieval_cfg.max_candidate_videos,
        hints.evidence_profile.candidate_video_budget,
    )
    max_aux_images = min(
        retrieval_cfg.max_aux_images,
        hints.evidence_profile.auxiliary_image_budget,
    )
    max_rows = min(
        retrieval_cfg.max_evidence_rows,
        hints.evidence_profile.max_evidence_rows,
    )

    transcript_count = 0
    candidate_count = 0
    aux_image_count = 0
    kept: List[RetrievalHit] = []

    ordered_hits = sorted(hits, key=lambda hit: (_route_priority(hints, hit), hit.rank))

    for hit in ordered_hits:
        if len(kept) >= max_rows:
            break
        if hit.source_type == "transcript_window":
            if transcript_count >= transcript_budget:
                continue
            transcript_count += 1
        elif hit.source_type in {"main_clip", "main_event_summary"}:
            if candidate_count >= max_candidate_videos:
                continue
            candidate_count += 1
        elif hit.modality == "image" and hit.source_type.startswith("aux_"):
            if aux_image_count >= max_aux_images:
                continue
            aux_image_count += 1

        kept.append(hit)

    return [
        hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(kept, start=1)
    ]
def _route_priority(hints: RouteHints, hit: RetrievalHit) -> int:
    """Return a hit's sort priority from its position in the route's source list."""
    try:
        return hints.evidence_profile.source_priority.index(hit.source_type)
    except ValueError:
        return len(hints.evidence_profile.source_priority)
