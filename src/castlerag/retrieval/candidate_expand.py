"""Expansion from retrieval hits to route-aware evidence packs."""

from __future__ import annotations

from collections import OrderedDict
from typing import List

from castlerag.routing.question_router import QuestionRoute
from castlerag.schemas import EvidencePack, RetrievalHit


def expand_candidates(
    hits: List[RetrievalHit],
    route: QuestionRoute,
    max_candidate_videos: int = 4,
    frames_per_candidate: int = 32,
) -> List[EvidencePack]:
    """Collapse retrieval hits into up to max_candidate_videos evidence packs.

    Each pack includes sampled frame paths, linked transcript windows,
    event summaries, OCR spans, and auxiliary notes for the reranker.
    """
    del frames_per_candidate

    primary_hits = [
        hit
        for hit in hits
        if hit.source_type in {"main_clip", "main_event_summary", "transcript_window"}
    ]
    if not primary_hits:
        primary_hits = hits

    packs: List[EvidencePack] = []
    used_row_ids: set[str] = set()
    for primary in primary_hits:
        if len(packs) >= max_candidate_videos:
            break
        bundle = _bundle_rows(primary, hits)
        evidence_rows = []
        seen_local: set[str] = set()
        for row in bundle:
            if row.record_id in seen_local:
                continue
            seen_local.add(row.record_id)
            evidence_rows.append(row)
            used_row_ids.add(row.record_id)

        packs.append(
            EvidencePack(
                pack_id=f"pack_{primary.record_id}",
                route=route,
                primary_hit=primary,
                retrieval_score=primary.score,
                evidence_rows=evidence_rows,
                transcript_evidence=_collect_transcripts(evidence_rows),
                event_summaries=_collect_event_summaries(evidence_rows),
                ocr_spans=_collect_ocr(evidence_rows),
                frame_descriptions=_collect_frame_descriptions(evidence_rows),
                auxiliary_notes=_collect_aux_notes(evidence_rows),
            )
        )

    if not packs and hits:
        primary = hits[0]
        packs.append(
            EvidencePack(
                pack_id=f"pack_{primary.record_id}",
                route=route,
                primary_hit=primary,
                retrieval_score=primary.score,
                evidence_rows=[primary],
                transcript_evidence=_collect_transcripts([primary]),
                event_summaries=_collect_event_summaries([primary]),
                ocr_spans=_collect_ocr([primary]),
                frame_descriptions=_collect_frame_descriptions([primary]),
                auxiliary_notes=_collect_aux_notes([primary]),
            )
        )

    return packs


def _bundle_rows(primary: RetrievalHit, hits: List[RetrievalHit]) -> List[RetrievalHit]:
    """Return hits that share context with or temporally overlap the primary hit."""
    rows: "OrderedDict[str, RetrievalHit]" = OrderedDict()
    rows[primary.record_id] = primary
    for hit in hits:
        if hit.record_id == primary.record_id:
            continue
        if _same_context(primary, hit) or _overlaps(primary, hit):
            rows[hit.record_id] = hit
    return list(rows.values())


def _same_context(left: RetrievalHit, right: RetrievalHit) -> bool:
    """Return True if both hits share the same day, camera, and participant."""
    return (
        left.day == right.day
        and left.camera_id == right.camera_id
        and left.participant_id == right.participant_id
    )


def _overlaps(left: RetrievalHit, right: RetrievalHit, slack_ms: int = 30_000) -> bool:
    """Return True if the two hits' time spans overlap within the given slack window."""
    if left.absolute_start is None or left.absolute_end is None:
        return False
    if right.absolute_start is None or right.absolute_end is None:
        return False
    return not (
        right.absolute_start > left.absolute_end + slack_ms
        or right.absolute_end < left.absolute_start - slack_ms
    )


def _collect_transcripts(rows: List[RetrievalHit]) -> List[str]:
    """Return deduplicated transcript texts from the given hits."""
    return _unique_values([row.transcript_text for row in rows if row.transcript_text])


def _collect_event_summaries(rows: List[RetrievalHit]) -> List[str]:
    """Return deduplicated event summary strings from the given hits."""
    return _unique_values([row.event_summary for row in rows if row.event_summary])


def _collect_ocr(rows: List[RetrievalHit]) -> List[str]:
    """Return deduplicated OCR text strings from the given hits."""
    return _unique_values([row.ocr_text for row in rows if row.ocr_text])


def _collect_frame_descriptions(rows: List[RetrievalHit]) -> List[str]:
    """Return deduplicated clip asset path strings from main_clip hits."""
    values = []
    for row in rows:
        if row.source_type == "main_clip" and row.asset_path:
            values.append(f"clip asset: {row.asset_path}")
    return _unique_values(values)


def _collect_aux_notes(rows: List[RetrievalHit]) -> List[str]:
    """Return deduplicated auxiliary-source notes prefixed with their source type."""
    values = []
    for row in rows:
        if not row.source_type.startswith("aux_"):
            continue
        note = (
            row.event_summary
            or row.ocr_text
            or row.transcript_text
            or row.asset_path
        )
        if note:
            values.append(f"{row.source_type}: {note}")
    return _unique_values(values)


def _unique_values(values: List[str]) -> List[str]:
    """Return values with duplicates removed while preserving insertion order."""
    return list(OrderedDict((value, None) for value in values).keys())
