"""BM25 transcript index creation and query-time scoring.

Stores transcript utterance windows.  Optimised for exact and near-exact
overlap with question text, answer choices, people, days, rooms, and
temporal markers.

See retrieval/transcript_lexical.py for query-time scoring with bonuses.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

from castlerag.schemas import TranscriptWindow


def build_bm25_index(
    windows: List[TranscriptWindow],
    out_path: Path,
) -> Any:
    """Build and persist a BM25 index over transcript windows.

    Returns the in-memory index object for immediate use.
    """
    raise NotImplementedError("Implemented in issue #6")


def load_bm25_index(index_path: Path) -> Any:
    """Load a persisted BM25 index from disk."""
    raise NotImplementedError("Implemented in issue #6")
