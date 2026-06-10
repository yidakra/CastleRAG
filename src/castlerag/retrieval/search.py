"""Query encoding, modality-scoped Qdrant search, and RRF score fusion."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from castlerag.retrieval.filters import build_filter
from castlerag.retrieval.transcript_lexical import score_windows
from castlerag.routing.question_router import RouteHints
from castlerag.schemas import EvalQuestion, RetrievalHit

_TRANSCRIPT_ROUTE_BUDGETS = {
    "static_visual": 10,
    "speech_text": 30,
    "temporal": 30,
    "mixed": 30,
}


def reciprocal_rank_fusion(
    ranked_lists: List[List[RetrievalHit]],
    k: int = 60,
) -> List[RetrievalHit]:
    """Fuse multiple ranked lists into one using RRF(k)."""
    by_record: Dict[str, RetrievalHit] = {}
    scores = defaultdict(float)

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            scores[hit.record_id] += 1.0 / (k + rank)
            existing = by_record.get(hit.record_id)
            if existing is None or hit.score > existing.score:
                by_record[hit.record_id] = hit

    fused = []
    for record_id, hit in by_record.items():
        fused.append(hit.model_copy(update={"score": scores[record_id]}))

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
    query_variants = _query_variants(question)
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
    query_vectors = np.asarray(
        embed_client.embed_texts(query_variants), dtype=np.float32
    )
    if query_vectors.ndim != 2:
        raise ValueError(
            f"Expected 2D query embedding matrix, got shape {query_vectors.shape}"
        )

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
            room=hints.room,
        )
        for query_vector in query_vectors
    ]
    transcript_lane = reciprocal_rank_fusion(
        [transcript_bm25, *transcript_dense_lists],
        k=retrieval_cfg.rrf_k,
    )[: _transcript_budget(hints.route, retrieval_cfg.transcript_top_k)]

    multimodal_lists: List[List[RetrievalHit]] = []
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
        for query_vector in query_vectors:
            hits = _dense_search(
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                source_type=source_type,
                modality=modality,
                day=hints.day,
                participant_id=hints.participant,
                room=hints.room,
            )
            if hits:
                multimodal_lists.append(hits)

    multimodal_lane = reciprocal_rank_fusion(multimodal_lists, k=retrieval_cfg.rrf_k)
    merged = reciprocal_rank_fusion(
        [transcript_lane, multimodal_lane], k=retrieval_cfg.rrf_k
    )
    return _collapse_hits(merged, hints, retrieval_cfg)


def _query_variants(question: EvalQuestion) -> List[str]:
    return [
        question.query,
        (
            f"{question.query} Choices: "
            f"A {question.answers['a']}. "
            f"B {question.answers['b']}. "
            f"C {question.answers['c']}. "
            f"D {question.answers['d']}."
        ),
    ]


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
        hits.append(
            RetrievalHit(
                rank=rank,
                score=float(getattr(point, "score")),
                point_id=str(getattr(point, "id", payload.get("point_id", ""))),
                record_id=str(payload["record_id"]),
                source_type=str(payload["source_type"]),
                modality=str(payload["modality"]),
                day=payload.get("day"),
                camera_id=payload.get("camera_id"),
                participant_id=payload.get("participant_id"),
                absolute_start=payload.get("absolute_start"),
                absolute_end=payload.get("absolute_end"),
                transcript_text=payload.get("transcript_text"),
                event_summary=payload.get("event_summary"),
                ocr_text=payload.get("ocr_text"),
                asset_path=payload.get("asset_path"),
            )
        )
    return hits


def _collapse_hits(
    hits: List[RetrievalHit],
    hints: RouteHints,
    retrieval_cfg: Any,
) -> List[RetrievalHit]:
    transcript_budget = _transcript_budget(hints.route, retrieval_cfg.transcript_top_k)
    max_candidate_videos = retrieval_cfg.max_candidate_videos
    max_aux_images = retrieval_cfg.max_aux_images
    max_rows = retrieval_cfg.max_evidence_rows

    transcript_count = 0
    candidate_count = 0
    aux_image_count = 0
    kept: List[RetrievalHit] = []

    ordered_hits = sorted(
        hits, key=lambda hit: (_route_priority(hints.route, hit), hit.rank)
    )

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


def _transcript_budget(route: str, default_budget: int) -> int:
    return min(default_budget, _TRANSCRIPT_ROUTE_BUDGETS.get(route, default_budget))


def _route_priority(route: str, hit: RetrievalHit) -> int:
    if route == "speech_text":
        return 0 if hit.source_type == "transcript_window" else 1
    if route == "static_visual":
        if hit.source_type in {"main_clip", "main_event_summary"} or (
            hit.modality == "image" and hit.source_type.startswith("aux_")
        ):
            return 0
        return 1
    if route == "temporal":
        if hit.source_type in {"transcript_window", "main_event_summary"}:
            return 0
        if hit.source_type == "main_clip":
            return 1
        return 2
    return 0
