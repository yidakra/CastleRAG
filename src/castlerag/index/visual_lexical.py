"""BM25 index over per-clip captions, OCR spans and scene-graph text.

The transcript BM25 lane (``transcripts.pkl``) only sees what was *said*.
Object-level and on-screen-text questions ("what brand is the fridge",
"what is written on the apron") are answered by what the caption/OCR pass
*saw*, which today is reachable only through the dense clip vector, where
the transcript text dilutes it (issue #50, modality gap).

This module builds a second, CPU-only lexical index over the visual text
that already exists in ``clips.jsonl`` / ``events.jsonl``:

* one document per clip: ``clip_caption + ocr_text + scene_graph_text``
* one document per event summary: ``event_summary + aggregated_ocr_text``

Transcript text is deliberately **not** part of these documents.  No
re-annotation is needed; the artifact is rebuilt from all loaded records
on every ``castlerag index`` run, exactly like the transcript index, so a
day/camera-scoped ingest keeps earlier days searchable.

See retrieval/visual_lexical.py for query-time scoring with bonuses.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from castlerag.schemas import ClipRecord, EventSummaryRecord

VISUAL_TEXT_INDEX_NAME = "visual_text.pkl"

_TOKEN_RE = re.compile(r"\b\w+\b")


class VisualTextDoc(BaseModel):
    """One lexical document derived from a clip or event summary record.

    Carries enough metadata to rebuild a ``RetrievalHit`` at query time
    without touching Qdrant.  ``text`` is the indexed surface; the other
    text fields are kept separately so hits can expose them individually.
    """

    record_id: str
    source_type: str  # "main_clip" | "main_event_summary"
    modality: str  # "video" for clips, "text" for event summaries
    day: str
    camera_id: str
    participant_id: Optional[str] = None
    room: Optional[str] = None
    hour: Optional[int] = None
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    absolute_start: int
    absolute_end: int
    text: str
    clip_caption: Optional[str] = None
    ocr_text: Optional[str] = None
    scene_graph_text: Optional[str] = None
    event_summary: Optional[str] = None
    transcript_text: Optional[str] = None
    asset_path: Optional[str] = None
    sampled_frame_paths: List[str] = Field(default_factory=list)


@dataclass
class VisualBM25IndexBundle:
    """Persistable visual-text BM25 bundle."""

    bm25: BM25Okapi
    docs: List[VisualTextDoc]
    tokenized_corpus: List[List[str]]


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer shared with the transcript BM25 index."""
    return _TOKEN_RE.findall(text.lower())


def _join(parts: Sequence[Optional[str]]) -> str:
    """Join the non-empty text parts with a separator BM25 tokenises away."""
    return " | ".join(p.strip() for p in parts if p and p.strip())


def clip_visual_text(record: ClipRecord) -> str:
    """Return the indexed visual surface of a clip (no transcript)."""
    return _join((record.clip_caption, record.ocr_text, record.scene_graph_text))


def event_visual_text(record: EventSummaryRecord) -> str:
    """Return the indexed visual surface of an event summary."""
    return _join((record.event_summary, record.aggregated_ocr_text))


def build_visual_docs(
    clips: Sequence[ClipRecord],
    events: Sequence[EventSummaryRecord] = (),
) -> List[VisualTextDoc]:
    """Turn clip and event records into lexical documents, skipping empty ones.

    Records without any caption/OCR/scene-graph text (e.g. placeholders or
    clips the caption pass has not reached yet) contribute nothing to the
    index, so they can never surface through this lane.
    """
    docs: List[VisualTextDoc] = []
    for clip in clips:
        text = clip_visual_text(clip)
        if not text:
            continue
        docs.append(
            VisualTextDoc(
                record_id=clip.clip_id,
                source_type="main_clip",
                modality="video",
                day=clip.day,
                camera_id=clip.camera_id,
                participant_id=clip.participant_id,
                room=clip.room,
                hour=clip.hour,
                start_seconds=clip.start_seconds,
                end_seconds=clip.end_seconds,
                absolute_start=clip.absolute_start,
                absolute_end=clip.absolute_end,
                text=text,
                clip_caption=clip.clip_caption,
                ocr_text=clip.ocr_text,
                scene_graph_text=clip.scene_graph_text,
                transcript_text=clip.transcript_text,
                asset_path=clip.retrieval_clip_path or clip.source_video_path,
                sampled_frame_paths=list(clip.sampled_frame_paths),
            )
        )
    for event in events:
        text = event_visual_text(event)
        if not text:
            continue
        docs.append(
            VisualTextDoc(
                record_id=event.event_summary_id,
                source_type="main_event_summary",
                modality="text",
                day=event.day,
                camera_id=event.camera_id,
                participant_id=event.participant_id,
                room=event.room,
                absolute_start=event.absolute_start,
                absolute_end=event.absolute_end,
                text=text,
                ocr_text=event.aggregated_ocr_text,
                event_summary=event.event_summary,
            )
        )
    return docs


def build_visual_bm25_index(
    clips: Sequence[ClipRecord],
    events: Sequence[EventSummaryRecord],
    out_path: Path,
) -> VisualBM25IndexBundle:
    """Build and persist the visual-text BM25 index; return it for immediate use.

    ``rank_bm25`` needs at least one document, so an empty corpus is stored
    as an empty bundle whose ``bm25`` is ``None``; the scorer treats that as
    "no hits".
    """
    docs = build_visual_docs(clips, events)
    tokenized_corpus = [_tokenize(doc.text) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    bundle = VisualBM25IndexBundle(
        bm25=bm25, docs=docs, tokenized_corpus=tokenized_corpus
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "docs": [doc.model_dump() for doc in docs],
        "tokenized_corpus": tokenized_corpus,
    }
    with out_path.open("wb") as fh:
        pickle.dump(payload, fh)
    return bundle


def load_visual_bm25_index(index_path: Path) -> VisualBM25IndexBundle:
    """Load a persisted visual-text BM25 index from disk."""
    with index_path.open("rb") as fh:
        payload = pickle.load(fh)
    docs = [VisualTextDoc.model_validate(doc) for doc in payload["docs"]]
    tokenized_corpus = payload["tokenized_corpus"]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    return VisualBM25IndexBundle(
        bm25=bm25, docs=docs, tokenized_corpus=tokenized_corpus
    )


def load_visual_bm25_index_if_present(
    cache_dir: Path,
) -> Optional[VisualBM25IndexBundle]:
    """Load ``visual_text.pkl`` from ``cache_dir`` or return None when absent.

    The lane is optional: deployments indexed before it existed keep working
    unchanged until ``castlerag index`` is re-run.  A pickle that exists but
    fails to load is a real error and is raised, not hidden.
    """
    path = Path(cache_dir) / VISUAL_TEXT_INDEX_NAME
    if not path.exists():
        return None
    return load_visual_bm25_index(path)
