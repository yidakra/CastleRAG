"""Artifact discovery and dense-index build orchestration for issue #5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

from castlerag import __version__
from castlerag.config import CastleRAGConfig
from castlerag.embed.omniembed import OmniEmbedClient
from castlerag.index.io import (
    load_aux_records,
    load_clip_records,
    load_embedding_cache,
    load_event_summary_records,
    load_transcript_windows,
    write_embedding_cache,
    write_json,
)
from castlerag.index.qdrant import (
    bootstrap_collection,
    build_point_batches,
    upsert_batch,
)
from castlerag.index.transcript_lexical import build_bm25_index
from castlerag.schemas import (
    AuxRecord,
    ClipRecord,
    EventSummaryRecord,
    TranscriptWindow,
)

Record = TranscriptWindow | ClipRecord | EventSummaryRecord | AuxRecord

_TRANSCRIPT_FILES = ("transcripts.jsonl",)
_CLIP_FILES = ("clips.jsonl",)
_EVENT_FILES = ("events.jsonl",)
_AUX_FILES = ("aux.jsonl",)

_CACHE_TRANSCRIPTS = "transcripts.npz"
_CACHE_EVENTS = "events.npz"
_CACHE_CLIPS = "clips.npz"
_CACHE_AUX_TEXT = "aux_text.npz"
_CACHE_AUX_IMAGE = "aux_image.npz"
_CACHE_AUX_VIDEO = "aux_video.npz"


@dataclass
class ChunkArtifacts:
    transcripts: List[Path]
    clips: List[Path]
    events: List[Path]
    aux: List[Path]


@dataclass
class LoadedArtifacts:
    transcripts: List[TranscriptWindow]
    clips: List[ClipRecord]
    events: List[EventSummaryRecord]
    aux: List[AuxRecord]


@dataclass
class CacheArtifact:
    name: str
    path: Path
    record_ids: List[str]
    vectors: np.ndarray
    records: List[Record]


def discover_chunk_artifacts(chunks_dir: Path) -> ChunkArtifacts:
    """Recursively discover preprocessed chunk JSONL artifacts."""

    def _glob(names: Sequence[str]) -> List[Path]:
        """Glob chunks_dir for each filename pattern and return sorted unique paths."""
        matches: List[Path] = []
        for name in names:
            matches.extend(sorted(chunks_dir.rglob(name)))
        return sorted(set(matches))

    return ChunkArtifacts(
        transcripts=_glob(_TRANSCRIPT_FILES),
        clips=_glob(_CLIP_FILES),
        events=_glob(_EVENT_FILES),
        aux=_glob(_AUX_FILES),
    )


def load_chunk_records(chunks_dir: Path) -> LoadedArtifacts:
    """Load all discovered chunk artifacts into typed record lists."""
    artifacts = discover_chunk_artifacts(chunks_dir)
    transcripts = [
        row for path in artifacts.transcripts for row in load_transcript_windows(path)
    ]
    clips = [row for path in artifacts.clips for row in load_clip_records(path)]
    events = [
        row for path in artifacts.events for row in load_event_summary_records(path)
    ]
    aux = [row for path in artifacts.aux for row in load_aux_records(path)]
    return LoadedArtifacts(transcripts=transcripts, clips=clips, events=events, aux=aux)


def filter_records(
    records: LoadedArtifacts,
    cfg: CastleRAGConfig,
    day: Optional[int] = None,
) -> LoadedArtifacts:
    """Filter loaded artifacts to the configured camera scope and optional day."""
    day_label = f"day{day}" if day is not None else None

    def _camera_allowed(camera_id: Optional[str], camera_type: Optional[str]) -> bool:
        """Return True if the camera is within the configured scope."""
        if cfg.dataset.camera_scope == "all":
            return True
        if camera_type == "fixed":
            return False
        if camera_id is None:
            return True
        return camera_id in set(cfg.dataset.ego_cameras)

    return LoadedArtifacts(
        transcripts=[
            row
            for row in records.transcripts
            if (day_label is None or row.day == day_label)
            and _camera_allowed(row.camera_id, row.camera_type)
        ],
        clips=[
            row
            for row in records.clips
            if (day_label is None or row.day == day_label)
            and _camera_allowed(row.camera_id, row.camera_type)
        ],
        events=[
            row
            for row in records.events
            if (day_label is None or row.day == day_label)
            and _camera_allowed(row.camera_id, row.camera_type)
        ],
        aux=[
            row
            for row in records.aux
            if (day_label is None or row.day == day_label)
            and _camera_allowed(row.camera_id, row.camera_type)
        ],
    )


def build_bm25_artifact(records: LoadedArtifacts, out_dir: Path) -> Path:
    """Build the transcript BM25 artifact from normalized transcript windows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transcripts.pkl"
    build_bm25_index(records.transcripts, out_path)
    return out_path


def cache_dense_embeddings(
    records: LoadedArtifacts,
    cfg: CastleRAGConfig,
    embed_client: OmniEmbedClient,
    modality: str | None = None,
    day: Optional[int] = None,
    force: bool = False,
) -> List[Path]:
    """Write restart-safe dense embedding caches for all available record groups."""
    cache_dir = Path(cfg.embedding.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    scoped = filter_records(records, cfg, day=day)
    suffix = _cache_suffix(day)
    day_label = f"day{day}" if day is not None else None

    cache_paths: List[Path] = []
    if modality in (None, "transcript"):
        cache_paths.append(
            _cache_records(
                name="transcripts",
                records=scoped.transcripts,
                cache_path=cache_dir / f"transcripts{suffix}.npz",
                embed_fn=embed_client.embed_texts,
                batch_size=cfg.embedding.batch_sizes.transcript,
                payload_fn=lambda row: row.transcript_text,
                record_id_fn=lambda row: row.transcript_window_id,
                force=force,
            )
        )

    if modality in (None, "event_summary"):
        event_records = [row for row in scoped.events if row.event_summary]
        cache_paths.append(
            _cache_records(
                name="events",
                records=event_records,
                cache_path=cache_dir / f"events{suffix}.npz",
                embed_fn=embed_client.embed_texts,
                batch_size=cfg.embedding.batch_sizes.event_summary,
                payload_fn=lambda row: row.event_summary or "",
                record_id_fn=lambda row: row.event_summary_id,
                force=force,
            )
        )

    if modality in (None, "video"):
        # vLLM's /v1/embeddings is text-only, so the clip's textual surface
        # (caption + transcript + OCR) gates inclusion rather than the presence
        # of extracted frames.  embed_videos() already prepends "Query: " when
        # it falls back to embed_texts(), so the payload is the raw text.
        clip_records = [
            row
            for row in scoped.clips
            if row.clip_caption or row.transcript_text or row.ocr_text
        ]
        cache_paths.append(
            _cache_records(
                name="clips",
                records=clip_records,
                cache_path=cache_dir / f"clips{suffix}.npz",
                embed_fn=embed_client.embed_videos,
                batch_size=cfg.embedding.batch_sizes.video,
                payload_fn=lambda row: (
                    " | ".join(
                        s for s in (
                            row.clip_caption,
                            row.transcript_text,
                            row.ocr_text,
                        ) if s
                    ) or "Video clip"
                ),
                record_id_fn=lambda row: row.clip_id,
                force=force,
            )
        )
        aux_video_records = [
            row
            for row in scoped.aux
            if row.modality == "video" and _aux_video_frames(row)
        ]
        cache_paths.append(
            _cache_records(
                name="aux_video",
                records=aux_video_records,
                cache_path=cache_dir / f"aux_video{suffix}.npz",
                embed_fn=embed_client.embed_videos,
                batch_size=cfg.embedding.batch_sizes.video,
                # Frame paths reach the text-only backend as a list, which
                # falls back to the literal "Video clip" string and collapses
                # every aux_video vector onto a single point.  Embed the
                # record id instead so each row gets a distinct vector.
                payload_fn=_aux_video_payload,
                record_id_fn=lambda row: row.clip_id,
                force=force,
            )
        )

    if modality in (None, "image"):
        aux_image_records = [
            row for row in scoped.aux if row.modality == "image" and row.asset_path
        ]
        cache_paths.append(
            _cache_records(
                name="aux_image",
                records=aux_image_records,
                cache_path=cache_dir / f"aux_image{suffix}.npz",
                embed_fn=embed_client.embed_images,
                batch_size=cfg.embedding.batch_sizes.image,
                payload_fn=lambda row: row.asset_path or "",
                record_id_fn=lambda row: row.clip_id,
                force=force,
            )
        )

    if modality in (None, "text"):
        aux_text_records = [
            row for row in scoped.aux if row.modality == "text" and row.summary_text
        ]
        cache_paths.append(
            _cache_records(
                name="aux_text",
                records=aux_text_records,
                cache_path=cache_dir / f"aux_text{suffix}.npz",
                embed_fn=embed_client.embed_texts,
                batch_size=cfg.embedding.batch_sizes.event_summary,
                payload_fn=lambda row: row.summary_text or "",
                record_id_fn=lambda row: row.clip_id,
                force=force,
            )
        )

    summary = {
        "model": cfg.embedding.model,
        "backend": cfg.embedding.backend,
        "generated_files": [str(path) for path in cache_paths if path.exists()],
        "dim": embed_client.dim,
        "day": day_label,
        "version": __version__,
    }
    write_json(cache_dir / f"manifest{suffix}.json", summary)
    return [path for path in cache_paths if path.exists()]


def load_dense_caches(
    cache_dir: Path,
    records: LoadedArtifacts,
    *,
    pattern: str = "*.npz",
) -> List[CacheArtifact]:
    """Load available dense embedding caches and join them back to typed records.

    ``pattern`` restricts which cache files are read; the default loads every
    ``*.npz`` under ``cache_dir``.  Pass ``*_day{N}.npz`` when only the
    day-N subset should be upserted.
    """
    index = _record_index(records)
    artifacts: List[CacheArtifact] = []
    cache_paths = sorted(cache_dir.glob(pattern))
    for path in cache_paths:
        record_ids, vectors = load_embedding_cache(path)
        typed_records = [
            index[record_id] for record_id in record_ids if record_id in index
        ]
        if len(typed_records) != len(record_ids):
            missing = [record_id for record_id in record_ids if record_id not in index]
            raise KeyError(f"Missing records for cached ids in {path}: {missing[:5]}")
        artifacts.append(
            CacheArtifact(
                name=path.stem,
                path=path,
                record_ids=record_ids,
                vectors=vectors,
                records=typed_records,
            )
        )
    return artifacts


def build_qdrant_index(
    cfg: CastleRAGConfig,
    records: LoadedArtifacts,
    recreate: bool = False,
    day: Optional[int] = None,
) -> tuple[int, List[Path]]:
    """Bootstrap Qdrant and upsert dense caches.

    When ``day`` is set, ``records`` is filtered to that day and only the
    matching ``*_day{N}.npz`` cache files are upserted, so incremental
    re-runs (e.g. after adding a new day's chunks) do not re-upsert the
    days already in the collection.
    """
    cache_dir = Path(cfg.embedding.cache_dir)
    scoped = filter_records(records, cfg, day=day)
    pattern = f"*_day{day}.npz" if day is not None else "*.npz"
    cache_artifacts = load_dense_caches(cache_dir, scoped, pattern=pattern)
    if not cache_artifacts:
        raise FileNotFoundError(f"No embedding caches found under {cache_dir}")

    vector_size = _discover_vector_size(cache_artifacts)
    client = bootstrap_collection(
        host=cfg.qdrant.host,
        port=cfg.qdrant.port,
        collection_name=cfg.qdrant.collection,
        vector_size=vector_size,
        distance=cfg.qdrant.distance,
        on_disk_payload=cfg.qdrant.on_disk_payload,
        recreate=recreate,
    )

    # Qdrant's REST POST body limit defaults to 32 MB; chunk upserts so a
    # 3584-dim clip artifact (~64 KB/point with payload) stays well under it.
    upsert_chunk = 256
    for artifact in cache_artifacts:
        payload_rows = build_point_batches(
            artifact.records,
            model_version=cfg.version,
            model_name=cfg.embedding.model,
            build_id=f"castle-index-{__version__}",
        )
        point_ids = [row.point_id for row in payload_rows]
        payloads = [row.model_dump(exclude_none=True) for row in payload_rows]
        for start in range(0, len(point_ids), upsert_chunk):
            stop = start + upsert_chunk
            upsert_batch(
                client=client,
                collection_name=cfg.qdrant.collection,
                point_ids=point_ids[start:stop],
                vectors=artifact.vectors[start:stop].tolist(),
                payloads=payloads[start:stop],
            )

    return vector_size, [artifact.path for artifact in cache_artifacts]


def _cache_records(
    *,
    name: str,
    records: Sequence[Record],
    cache_path: Path,
    embed_fn: Callable[[list], np.ndarray],
    batch_size: int,
    payload_fn: Callable[[Record], str | List[str]],
    record_id_fn: Callable[[Record], str],
    force: bool,
) -> Path:
    """Cache one homogeneous record set to an NPZ bundle."""
    if cache_path.exists() and not force:
        return cache_path
    if not records:
        return cache_path

    payloads = [payload_fn(record) for record in records]
    record_ids = [record_id_fn(record) for record in records]
    vectors = _batched_embed(embed_fn, payloads, batch_size)
    if len(record_ids) != vectors.shape[0]:
        raise ValueError(
            f"{name} cache size mismatch: "
            f"record_ids={len(record_ids)} vectors={vectors.shape[0]}"
        )
    return write_embedding_cache(record_ids, vectors, cache_path)


def _batched_embed(
    embed_fn: Callable[[list], np.ndarray],
    payloads: Sequence[str | List[str]],
    batch_size: int,
) -> np.ndarray:
    """Embed a homogeneous payload list in deterministic contiguous batches."""
    if not payloads:
        return np.zeros((0, 0), dtype=np.float32)
    batches: List[np.ndarray] = []
    for start in range(0, len(payloads), batch_size):
        batch = list(payloads[start : start + batch_size])
        vectors = np.asarray(embed_fn(batch), dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"Embedding batch must be 2D, got shape {vectors.shape}")
        batches.append(vectors)
    return np.concatenate(batches, axis=0)


def _record_index(records: LoadedArtifacts) -> dict[str, Record]:
    """Build a record-id-to-Record mapping from all artifact lists."""
    index: dict[str, Record] = {}
    for row in records.transcripts:
        index[row.transcript_window_id] = row
    for row in records.clips:
        index[row.clip_id] = row
    for row in records.events:
        index[row.event_summary_id] = row
    for row in records.aux:
        index[row.clip_id] = row
    return index


def _discover_vector_size(cache_artifacts: Iterable[CacheArtifact]) -> int:
    """Return the embedding dimension from the first non-empty cache artifact."""
    for artifact in cache_artifacts:
        if artifact.vectors.size and artifact.vectors.ndim == 2:
            return int(artifact.vectors.shape[1])
    raise ValueError("Could not discover vector size from empty caches")


def _aux_video_payload(record: Record) -> str:
    """Render an aux video record as a unique text payload for text-only embed.

    The text-only vLLM backend turns list payloads into the literal
    ``"Video clip"`` string and collapses every aux video vector onto the
    same point.  Returning a distinct per-record string (with the summary,
    owner, and id) gives each row its own dense vector.
    """
    if not isinstance(record, AuxRecord):
        return "Aux video clip"
    parts = [
        f"Aux video {record.source_type}",
        record.summary_text or "",
        record.aux_owner or "",
        record.clip_id,
    ]
    return " | ".join(s for s in parts if s)


def _aux_video_frames(record: Record) -> List[str]:
    """Return sampled frame paths from an AuxRecord's raw_features, or an empty list."""
    if not isinstance(record, AuxRecord):
        return []
    if not isinstance(record.raw_features, dict):
        return []
    frame_paths = record.raw_features.get("sampled_frame_paths")
    if not isinstance(frame_paths, list):
        return []
    return [str(path) for path in frame_paths if isinstance(path, str)]


def _cache_suffix(day: Optional[int]) -> str:
    """Return a day-qualified cache filename suffix, or an empty string."""
    return f"_day{day}" if day is not None else ""
