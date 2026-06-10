"""BM25 transcript index creation and query-time scoring.

Stores transcript utterance windows.  Optimised for exact and near-exact
overlap with question text, answer choices, people, days, rooms, and
temporal markers.

See retrieval/transcript_lexical.py for query-time scoring with bonuses.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from rank_bm25 import BM25Okapi

from castlerag.schemas import TranscriptWindow


@dataclass
class BM25IndexBundle:
    """Persistable transcript BM25 bundle."""

    bm25: BM25Okapi
    windows: List[TranscriptWindow]
    tokenized_corpus: List[List[str]]


_TOKEN_RE = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer for transcript BM25 indexing."""
    return _TOKEN_RE.findall(text.lower())


def build_bm25_index(
    windows: List[TranscriptWindow],
    out_path: Path,
) -> Any:
    """Build and persist a BM25 index over transcript windows.

    Returns the in-memory index object for immediate use.
    """
    tokenized_corpus = [_tokenize(window.transcript_text) for window in windows]
    bm25 = BM25Okapi(tokenized_corpus)
    bundle = BM25IndexBundle(
        bm25=bm25,
        windows=windows,
        tokenized_corpus=tokenized_corpus,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "windows": [window.model_dump() for window in windows],
        "tokenized_corpus": tokenized_corpus,
    }
    with out_path.open("wb") as fh:
        pickle.dump(payload, fh)
    return bundle


def load_bm25_index(index_path: Path) -> Any:
    """Load a persisted BM25 index from disk."""
    with index_path.open("rb") as fh:
        payload = pickle.load(fh)

    windows = [TranscriptWindow.model_validate(window) for window in payload["windows"]]
    tokenized_corpus = payload["tokenized_corpus"]
    bm25 = BM25Okapi(tokenized_corpus)
    return BM25IndexBundle(
        bm25=bm25,
        windows=windows,
        tokenized_corpus=tokenized_corpus,
    )
