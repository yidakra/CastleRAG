"""I/O helpers for indexing artifacts.

This module handles:
- loading preprocessed JSONL record files into typed schema objects
- persisting cached embedding matrices to disk
- loading cached embedding matrices back for Qdrant upsert
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, TypeVar

import numpy as np
from pydantic import BaseModel

from castlerag.schemas import (
    AuxRecord,
    ClipRecord,
    EventSummaryRecord,
    TranscriptWindow,
)

RecordT = TypeVar("RecordT", bound=BaseModel)


def _read_jsonl_models(path: Path, model_type: type[RecordT]) -> List[RecordT]:
    """Load typed Pydantic models from a JSONL file."""
    rows: List[RecordT] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(model_type.model_validate_json(line))
    return rows


def load_transcript_windows(path: Path) -> List[TranscriptWindow]:
    """Load transcript windows from JSONL."""
    return _read_jsonl_models(path, TranscriptWindow)


def load_clip_records(path: Path) -> List[ClipRecord]:
    """Load main clip records from JSONL."""
    return _read_jsonl_models(path, ClipRecord)


def load_event_summary_records(path: Path) -> List[EventSummaryRecord]:
    """Load event-summary records from JSONL."""
    return _read_jsonl_models(path, EventSummaryRecord)


def load_aux_records(path: Path) -> List[AuxRecord]:
    """Load auxiliary records from JSONL."""
    return _read_jsonl_models(path, AuxRecord)


def write_jsonl_records(records: Iterable[BaseModel], path: Path) -> Path:
    """Persist typed records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
    return path


def write_embedding_cache(
    record_ids: Sequence[str],
    vectors: np.ndarray,
    path: Path,
) -> Path:
    """Persist embedding cache as an NPZ bundle."""
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
    if len(record_ids) != vectors.shape[0]:
        raise ValueError(
            "record_ids length "
            f"({len(record_ids)}) must match vectors rows ({vectors.shape[0]})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        record_ids=np.asarray(record_ids, dtype=str),
        vectors=np.asarray(vectors, dtype=np.float32),
    )
    return path


def load_embedding_cache(path: Path) -> tuple[List[str], np.ndarray]:
    """Load an NPZ embedding cache bundle."""
    with np.load(path, allow_pickle=False) as payload:
        record_ids = payload["record_ids"].tolist()
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
    return list(record_ids), vectors


def write_json(path: Path, payload: dict) -> Path:
    """Write a small JSON metadata artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path
