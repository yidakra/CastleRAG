"""Tests for indexing and embedding helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from castlerag.config import CastleRAGConfig
from castlerag.embed.omniembed import OmniEmbedClient, format_query_text, make_point_id
from castlerag.index.io import (
    load_aux_records,
    load_clip_records,
    load_embedding_cache,
    load_event_summary_records,
    load_transcript_windows,
    write_embedding_cache,
    write_jsonl_records,
)
from castlerag.index.pipeline import (
    build_qdrant_index,
    cache_dense_embeddings,
    discover_chunk_artifacts,
    filter_records,
    load_chunk_records,
)
from castlerag.index.qdrant import (
    _BOOL_INDEX_FIELDS,
    _INTEGER_INDEX_FIELDS,
    _KEYWORD_INDEX_FIELDS,
    bootstrap_collection,
    build_point_batches,
    create_collection,
    create_payload_indexes,
    get_client,
    record_to_qdrant_point,
    upsert_batch,
)
from castlerag.index.transcript_lexical import build_bm25_index, load_bm25_index
from castlerag.schemas import (
    AuxRecord,
    ClipRecord,
    EventSummaryRecord,
    TranscriptSegment,
    TranscriptWindow,
)


def _transcript_window() -> TranscriptWindow:
    return TranscriptWindow(
        transcript_window_id="tx_0001",
        day="day1",
        camera_id="Allie",
        camera_type="ego",
        participant_id="Allie",
        room=None,
        hour=8,
        transcript_text="hello from the kitchen",
        transcript_segments=[TranscriptSegment(start=0.0, end=2.0, text="hello")],
        has_speech=True,
        transcript_char_len=22,
        absolute_start=1_672_531_200_000,
        absolute_end=1_672_531_202_000,
    )


def _clip_record() -> ClipRecord:
    return ClipRecord(
        clip_id="clip_0001",
        parent_source_id="vid_08",
        day="day1",
        hour=8,
        camera_id="Allie",
        camera_type="ego",
        participant_id="Allie",
        start_seconds=0.0,
        end_seconds=30.0,
        absolute_start=1_672_531_200_000,
        absolute_end=1_672_531_230_000,
        source_video_path="/data/main/day1/Allie/video/08.mp4",
        retrieval_clip_path="/data/derived/clips/day1/Allie/08/clip_0001.mp4",
        sampled_frame_paths=["/tmp/0001.jpg", "/tmp/0002.jpg"],
        transcript_text="hello",
        clip_caption="Allie enters the kitchen",
        ocr_text="EXIT",
        has_speech=True,
    )


def _event_record() -> EventSummaryRecord:
    return EventSummaryRecord(
        event_summary_id="evt_0001",
        day="day1",
        camera_id="Allie",
        camera_type="ego",
        participant_id="Allie",
        absolute_start=1_672_531_200_000,
        absolute_end=1_672_531_320_000,
        member_clip_ids=["clip_0001", "clip_0002", "clip_0003", "clip_0004"],
        event_summary="Allie walks into the kitchen and opens the fridge.",
        aggregated_ocr_text="EXIT",
    )


def _config(tmp_path: Path) -> CastleRAGConfig:
    return CastleRAGConfig.model_validate(
        {
            "preprocessing": {"chunks_dir": str(tmp_path / "chunks")},
            "embedding": {
                "cache_dir": str(tmp_path / "embeddings"),
                "backend": "transformers",
                "batch_sizes": {
                    "transcript": 2,
                    "event_summary": 2,
                    "image": 2,
                    "video": 2,
                },
            },
            "qdrant": {"collection": "castle_test"},
            "version": "0.1.0",
        }
    )


def test_make_point_id_deterministic():
    assert make_point_id("m1", "main_clip", "clip_1", "video") == make_point_id(
        "m1", "main_clip", "clip_1", "video"
    )


def test_bm25_index_roundtrip(tmp_path: Path):
    windows = [
        _transcript_window(),
        _transcript_window().model_copy(
            update={
                "transcript_window_id": "tx_0002",
                "transcript_text": "fridge door opens",
                "absolute_start": 1_672_531_205_000,
                "absolute_end": 1_672_531_207_000,
            }
        ),
    ]
    index_path = tmp_path / "transcripts.pkl"
    bundle = build_bm25_index(windows, index_path)
    assert index_path.exists()
    kitchen_scores = bundle.bm25.get_scores(["kitchen"])
    assert kitchen_scores[0] >= kitchen_scores[1]

    loaded = load_bm25_index(index_path)
    assert len(loaded.windows) == 2
    assert loaded.windows[0].transcript_window_id == "tx_0001"
    fridge_scores = loaded.bm25.get_scores(["fridge"])
    assert fridge_scores[1] >= fridge_scores[0]


def test_record_to_qdrant_point_transcript():
    point = record_to_qdrant_point(_transcript_window(), model_version="omniembed-v1")
    assert point.record_id == "tx_0001"
    assert point.source_type == "transcript_window"
    assert point.modality == "text"
    assert point.transcript_text == "hello from the kitchen"


def test_record_to_qdrant_point_clip():
    point = record_to_qdrant_point(_clip_record(), model_version="omniembed-v1")
    assert point.record_id == "clip_0001"
    assert point.source_type == "main_clip"
    assert point.modality == "video"
    assert point.sampled_frame_paths == ["/tmp/0001.jpg", "/tmp/0002.jpg"]


def test_record_to_qdrant_point_event_summary():
    point = record_to_qdrant_point(_event_record(), model_version="omniembed-v1")
    assert point.record_id == "evt_0001"
    assert point.source_type == "main_event_summary"
    assert point.event_summary is not None


def test_record_to_qdrant_point_aux_text():
    aux = AuxRecord(
        clip_id="aux_hr_0001",
        source_type="aux_heartrate",
        modality="text",
        day="day1",
        participant_id="Allie",
        absolute_start=1_672_531_200_000,
        absolute_end=1_672_531_260_000,
        summary_text="Heartrate rising to 92 bpm",
    )
    point = record_to_qdrant_point(aux, model_version="omniembed-v1")
    assert point.record_id == "aux_hr_0001"
    assert point.event_summary == "Heartrate rising to 92 bpm"


def test_record_to_qdrant_point_rejects_unknown_type():
    try:
        record_to_qdrant_point(object(), model_version="omniembed-v1")  # type: ignore[arg-type]
    except TypeError as exc:
        assert "Unsupported record type" in str(exc)
    else:
        raise AssertionError("Expected TypeError for unsupported record type")


def test_build_point_batches_mixed_records():
    points = build_point_batches(
        [_transcript_window(), _clip_record(), _event_record()],
        model_version="omniembed-v1",
        model_name="Tevatron/OmniEmbed-v0.1-multivent",
    )
    assert len(points) == 3
    assert all(
        point.model_name == "Tevatron/OmniEmbed-v0.1-multivent" for point in points
    )


def test_write_and_load_transcript_windows(tmp_path: Path):
    path = tmp_path / "transcripts.jsonl"
    write_jsonl_records([_transcript_window()], path)
    loaded = load_transcript_windows(path)
    assert len(loaded) == 1
    assert loaded[0].transcript_window_id == "tx_0001"


def test_write_and_load_clip_records(tmp_path: Path):
    path = tmp_path / "clips.jsonl"
    write_jsonl_records([_clip_record()], path)
    loaded = load_clip_records(path)
    assert len(loaded) == 1
    assert loaded[0].clip_id == "clip_0001"


def test_write_and_load_event_summary_records(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    write_jsonl_records([_event_record()], path)
    loaded = load_event_summary_records(path)
    assert len(loaded) == 1
    assert loaded[0].event_summary_id == "evt_0001"


def test_write_and_load_aux_records(tmp_path: Path):
    aux = AuxRecord(
        clip_id="aux_hr_0001",
        source_type="aux_heartrate",
        modality="text",
        day="day1",
        participant_id="Allie",
        absolute_start=1_672_531_200_000,
        absolute_end=1_672_531_260_000,
        summary_text="Heartrate rising to 92 bpm",
    )
    path = tmp_path / "aux.jsonl"
    write_jsonl_records([aux], path)
    loaded = load_aux_records(path)
    assert len(loaded) == 1
    assert loaded[0].clip_id == "aux_hr_0001"


def test_embedding_cache_roundtrip(tmp_path: Path):
    cache_path = tmp_path / "embeddings.npz"
    record_ids = ["r1", "r2"]
    vectors = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    write_embedding_cache(record_ids, vectors, cache_path)
    loaded_ids, loaded_vectors = load_embedding_cache(cache_path)
    assert loaded_ids == record_ids
    assert np.array_equal(loaded_vectors, vectors)


def test_embedding_cache_rejects_non_2d(tmp_path: Path):
    cache_path = tmp_path / "bad.npz"
    vectors = np.asarray([1.0, 2.0], dtype=np.float32)
    try:
        write_embedding_cache(["r1", "r2"], vectors, cache_path)
    except ValueError as exc:
        assert "2D" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-2D vectors")


def test_discover_and_load_chunk_records(tmp_path: Path):
    chunk_root = tmp_path / "chunks" / "day1" / "Allie"
    write_jsonl_records([_transcript_window()], chunk_root / "transcripts.jsonl")
    write_jsonl_records([_clip_record()], chunk_root / "clips.jsonl")
    write_jsonl_records([_event_record()], chunk_root / "events.jsonl")
    aux = AuxRecord(
        clip_id="aux_photo_0001",
        source_type="aux_photo",
        modality="image",
        day="day1",
        participant_id="Allie",
        absolute_start=1_672_531_200_000,
        absolute_end=1_672_531_200_001,
        asset_path="/tmp/photo.jpg",
    )
    write_jsonl_records([aux], chunk_root / "aux.jsonl")

    artifacts = discover_chunk_artifacts(tmp_path / "chunks")
    assert len(artifacts.transcripts) == 1
    assert len(artifacts.clips) == 1
    assert len(artifacts.events) == 1
    assert len(artifacts.aux) == 1

    loaded = load_chunk_records(tmp_path / "chunks")
    assert len(loaded.transcripts) == 1
    assert len(loaded.clips) == 1
    assert len(loaded.events) == 1
    assert len(loaded.aux) == 1


def test_filter_records_scopes_to_ego_and_day(tmp_path: Path):
    cfg = _config(tmp_path)
    day2_window = _transcript_window().model_copy(update={"day": "day2"})
    exo_clip = _clip_record().model_copy(
        update={"clip_id": "clip_exo", "camera_id": "Kitchen", "camera_type": "fixed"}
    )
    records = load_chunk_records(tmp_path / "missing")
    records.transcripts = [_transcript_window(), day2_window]
    records.clips = [_clip_record(), exo_clip]
    records.events = [_event_record()]
    records.aux = []

    scoped = filter_records(records, cfg, day=1)
    assert [row.transcript_window_id for row in scoped.transcripts] == ["tx_0001"]
    assert [row.clip_id for row in scoped.clips] == ["clip_0001"]


def test_cache_dense_embeddings_writes_expected_npz_files(tmp_path: Path):
    class FakeEmbedClient:
        def __init__(self) -> None:
            self.dim = 3

        def embed_texts(self, payloads: list[str]) -> np.ndarray:
            return np.asarray(
                [[float(len(text)), 1.0, 0.0] for text in payloads], dtype=np.float32
            )

        def embed_images(self, payloads: list[str]) -> np.ndarray:
            return np.asarray(
                [[float(idx), 2.0, 0.0] for idx, _ in enumerate(payloads)],
                dtype=np.float32,
            )

        def embed_videos(self, payloads: list[list[str]]) -> np.ndarray:
            return np.asarray(
                [[float(len(frames)), 3.0, 0.0] for frames in payloads],
                dtype=np.float32,
            )

    cfg = _config(tmp_path)
    records = load_chunk_records(tmp_path / "missing")
    records.transcripts = [_transcript_window()]
    records.clips = [_clip_record()]
    records.events = [_event_record()]
    records.aux = [
        AuxRecord(
            clip_id="aux_text_0001",
            source_type="aux_heartrate",
            modality="text",
            day="day1",
            participant_id="Allie",
            absolute_start=1_672_531_200_000,
            absolute_end=1_672_531_260_000,
            summary_text="92 bpm",
        ),
        AuxRecord(
            clip_id="aux_image_0001",
            source_type="aux_photo",
            modality="image",
            day="day1",
            participant_id="Allie",
            absolute_start=1_672_531_200_000,
            absolute_end=1_672_531_200_001,
            asset_path="/tmp/photo.jpg",
        ),
        AuxRecord(
            clip_id="aux_video_0001",
            source_type="aux_video",
            modality="video",
            day="day1",
            participant_id="Allie",
            absolute_start=1_672_531_200_000,
            absolute_end=1_672_531_230_000,
            asset_path="/tmp/video.mp4",
            raw_features={"sampled_frame_paths": ["/tmp/f1.jpg", "/tmp/f2.jpg"]},
        ),
    ]

    paths = cache_dense_embeddings(records, cfg, FakeEmbedClient())
    names = {path.name for path in paths}
    assert {
        "transcripts.npz",
        "events.npz",
        "clips.npz",
        "aux_text.npz",
        "aux_image.npz",
        "aux_video.npz",
    }.issubset(names)
    assert (Path(cfg.embedding.cache_dir) / "manifest.json").exists()


def test_cache_dense_embeddings_uses_day_scoped_filenames(tmp_path: Path):
    class FakeEmbedClient:
        def __init__(self) -> None:
            self.dim = 2

        def embed_texts(self, payloads: list[str]) -> np.ndarray:
            return np.asarray(
                [[1.0, float(idx)] for idx, _ in enumerate(payloads)], dtype=np.float32
            )

        def embed_images(self, payloads: list[str]) -> np.ndarray:
            return np.asarray(
                [[2.0, float(idx)] for idx, _ in enumerate(payloads)], dtype=np.float32
            )

        def embed_videos(self, payloads: list[list[str]]) -> np.ndarray:
            return np.asarray(
                [[3.0, float(len(frames))] for frames in payloads], dtype=np.float32
            )

    cfg = _config(tmp_path)
    records = load_chunk_records(tmp_path / "missing")
    records.transcripts = [
        _transcript_window(),
        _transcript_window().model_copy(update={"day": "day2"}),
    ]
    records.clips = [
        _clip_record(),
        _clip_record().model_copy(update={"clip_id": "clip_day2", "day": "day2"}),
    ]
    records.events = [
        _event_record(),
        _event_record().model_copy(
            update={"event_summary_id": "evt_day2", "day": "day2"}
        ),
    ]
    records.aux = []

    paths = cache_dense_embeddings(
        records, cfg, FakeEmbedClient(), modality="transcript", day=1
    )
    assert [path.name for path in paths] == ["transcripts_day1.npz"]
    loaded_ids, _ = load_embedding_cache(paths[0])
    assert loaded_ids == ["tx_0001"]


def test_cache_dense_embeddings_per_day_calls_are_incremental(tmp_path: Path):
    """Per-day re-runs must skip existing days and embed only the new one."""

    class CountingEmbedClient:
        def __init__(self) -> None:
            self.dim = 2
            self.text_calls: list[list[str]] = []

        def embed_texts(self, payloads: list[str]) -> np.ndarray:
            self.text_calls.append(list(payloads))
            return np.asarray(
                [[1.0, float(idx)] for idx, _ in enumerate(payloads)], dtype=np.float32
            )

        def embed_images(self, payloads: list[str]) -> np.ndarray:
            return np.asarray(
                [[2.0, float(idx)] for idx, _ in enumerate(payloads)], dtype=np.float32
            )

        def embed_videos(self, payloads: list[list[str]]) -> np.ndarray:
            return np.asarray(
                [[3.0, float(len(frames))] for frames in payloads], dtype=np.float32
            )

    cfg = _config(tmp_path)
    records = load_chunk_records(tmp_path / "missing")
    records.transcripts = [
        _transcript_window(),
        _transcript_window().model_copy(
            update={"transcript_window_id": "tx_0002", "day": "day2"}
        ),
    ]
    records.clips = []
    records.events = []
    records.aux = []

    client = CountingEmbedClient()
    cache_dense_embeddings(
        records, cfg, client, modality="transcript", day=1
    )
    assert len(client.text_calls) == 1
    assert client.text_calls[0] == [records.transcripts[0].transcript_text]

    # A second invocation for the same day must be a complete no-op — the
    # cache file already exists and _cache_records short-circuits.
    cache_dense_embeddings(
        records, cfg, client, modality="transcript", day=1
    )
    assert len(client.text_calls) == 1

    # Embedding day 2 should re-invoke the embedder for the day-2 record only,
    # writing a distinct day2-suffixed NPZ.
    paths = cache_dense_embeddings(
        records, cfg, client, modality="transcript", day=2
    )
    assert len(client.text_calls) == 2
    assert client.text_calls[1] == [records.transcripts[1].transcript_text]
    assert [path.name for path in paths] == ["transcripts_day2.npz"]
    assert (Path(cfg.embedding.cache_dir) / "transcripts_day1.npz").exists()
    assert (Path(cfg.embedding.cache_dir) / "transcripts_day2.npz").exists()


def test_build_qdrant_index_upserts_all_cached_artifacts(tmp_path: Path, monkeypatch):
    class FakeEmbedClient:
        def __init__(self) -> None:
            self.dim = 2

        def embed_texts(self, payloads: list[str]) -> np.ndarray:
            return np.asarray(
                [[1.0, float(idx)] for idx, _ in enumerate(payloads)], dtype=np.float32
            )

        def embed_images(self, payloads: list[str]) -> np.ndarray:
            return np.asarray(
                [[2.0, float(idx)] for idx, _ in enumerate(payloads)], dtype=np.float32
            )

        def embed_videos(self, payloads: list[list[str]]) -> np.ndarray:
            return np.asarray(
                [[3.0, float(len(frames))] for frames in payloads], dtype=np.float32
            )

    cfg = _config(tmp_path)
    records = load_chunk_records(tmp_path / "missing")
    records.transcripts = [_transcript_window()]
    records.clips = [_clip_record()]
    records.events = [_event_record()]
    records.aux = [
        AuxRecord(
            clip_id="aux_text_0001",
            source_type="aux_heartrate",
            modality="text",
            day="day1",
            participant_id="Allie",
            absolute_start=1_672_531_200_000,
            absolute_end=1_672_531_260_000,
            summary_text="92 bpm",
        )
    ]
    cache_dense_embeddings(records, cfg, FakeEmbedClient())

    class FakeQdrantClient:
        pass

    captured: dict[str, object] = {}

    def _bootstrap(**kwargs):
        captured["vector_size"] = kwargs["vector_size"]
        return FakeQdrantClient()

    def _upsert_batch(**kwargs):
        payloads = captured.setdefault("payloads", [])
        assert isinstance(payloads, list)
        payloads.append(kwargs)

    monkeypatch.setattr("castlerag.index.pipeline.bootstrap_collection", _bootstrap)
    monkeypatch.setattr("castlerag.index.pipeline.upsert_batch", _upsert_batch)

    vector_size, cache_paths = build_qdrant_index(cfg, records, recreate=True)
    assert vector_size == 2
    assert len(cache_paths) == 4
    payload_batches = captured["payloads"]
    assert isinstance(payload_batches, list)
    assert len(payload_batches) == 4


def test_build_qdrant_index_day_filter_only_upserts_matching_day(
    tmp_path: Path, monkeypatch
):
    """`build_qdrant_index(day=N)` reads only the day-N caches and upserts only
    the day-N records, leaving existing days in the collection untouched."""

    class FakeEmbedClient:
        def __init__(self) -> None:
            self.dim = 2

        def embed_texts(self, payloads: list[str]) -> np.ndarray:
            return np.asarray(
                [[1.0, float(idx)] for idx, _ in enumerate(payloads)], dtype=np.float32
            )

        def embed_images(self, payloads: list[str]) -> np.ndarray:
            return np.asarray(
                [[2.0, float(idx)] for idx, _ in enumerate(payloads)], dtype=np.float32
            )

        def embed_videos(self, payloads: list[list[str]]) -> np.ndarray:
            return np.asarray(
                [[3.0, float(len(frames))] for frames in payloads], dtype=np.float32
            )

    cfg = _config(tmp_path)
    records = load_chunk_records(tmp_path / "missing")
    records.transcripts = [
        _transcript_window(),
        _transcript_window().model_copy(
            update={"transcript_window_id": "tx_0002", "day": "day2"}
        ),
    ]
    records.clips = []
    records.events = []
    records.aux = []

    client = FakeEmbedClient()
    cache_dense_embeddings(records, cfg, client, modality="transcript", day=1)
    cache_dense_embeddings(records, cfg, client, modality="transcript", day=2)
    cache_dir = Path(cfg.embedding.cache_dir)
    assert (cache_dir / "transcripts_day1.npz").exists()
    assert (cache_dir / "transcripts_day2.npz").exists()

    captured_upserts: list[dict] = []

    class FakeQdrantClient:
        pass

    monkeypatch.setattr(
        "castlerag.index.pipeline.bootstrap_collection",
        lambda **kwargs: FakeQdrantClient(),
    )
    monkeypatch.setattr(
        "castlerag.index.pipeline.upsert_batch",
        lambda **kwargs: captured_upserts.append(kwargs),
    )

    build_qdrant_index(cfg, records, recreate=True, day=2)

    # Exactly one upsert batch, scoped to the day-2 transcript record only.
    assert len(captured_upserts) == 1
    ids = captured_upserts[0]["payloads"]
    assert len(ids) == 1
    assert ids[0]["record_id"] == "tx_0002"


def test_format_query_text():
    assert format_query_text("What happened?") == "Query: What happened?"


def test_embed_texts_uses_openai_embeddings_shape():
    class _EmbeddingRow:
        def __init__(self, embedding: list[float]) -> None:
            self.embedding = embedding

    class _EmbeddingsAPI:
        def __init__(self) -> None:
            self.last_input = None

        def create(self, model: str, input: list[str]):  # noqa: A002
            self.last_input = input
            return type(
                "Resp",
                (),
                {"data": [_EmbeddingRow([1.0, 2.0]), _EmbeddingRow([3.0, 4.0])]},
            )()

    fake_api = _EmbeddingsAPI()
    fake_client = type("Client", (), {"embeddings": fake_api})()

    client = OmniEmbedClient()
    client._client = fake_client
    vectors = client.embed_texts(["alpha", "beta"])
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, 2)
    assert fake_api.last_input == ["Query: alpha", "Query: beta"]
    assert client.dim == 2


def test_embed_images_delegates_to_client_method():
    class FakeClient:
        def embed_images(self, image_paths: list[str]) -> list[list[float]]:
            assert image_paths == ["a.jpg", "b.jpg"]
            return [[1.0, 0.0], [0.0, 1.0]]

    client = OmniEmbedClient()
    client._client = FakeClient()
    vectors = client.embed_images(["a.jpg", "b.jpg"])
    assert vectors.shape == (2, 2)


# ---------------------------------------------------------------------------
# qdrant.py — collection management
# ---------------------------------------------------------------------------


def test_get_client_raises_import_error_when_qdrant_client_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _block_qdrant(name, *args, **kwargs):
        if name == "qdrant_client":
            raise ImportError("no module named qdrant_client")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_qdrant)
    try:
        get_client()
    except ImportError as exc:
        assert "qdrant-client not installed" in str(exc)
    else:
        raise AssertionError("Expected ImportError")


def test_create_collection_calls_create_on_client():
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    create_collection(mock_client, "test_col", vector_size=128, recreate=False)
    mock_client.create_collection.assert_called_once()
    call_kwargs = mock_client.create_collection.call_args.kwargs
    assert call_kwargs["collection_name"] == "test_col"


def test_create_collection_recreate_deletes_first():
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    create_collection(mock_client, "test_col", vector_size=64, recreate=True)
    mock_client.delete_collection.assert_called_once_with("test_col")
    mock_client.create_collection.assert_called_once()


def test_create_collection_recreate_swallows_not_found():
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.delete_collection.side_effect = Exception("collection not found")
    create_collection(mock_client, "test_col", vector_size=64, recreate=True)
    mock_client.create_collection.assert_called_once()


def test_create_collection_recreate_reraises_other_errors():
    from unittest.mock import MagicMock

    import pytest

    mock_client = MagicMock()
    mock_client.delete_collection.side_effect = Exception("disk full")
    with pytest.raises(Exception, match="disk full"):
        create_collection(mock_client, "test_col", vector_size=64, recreate=True)


def test_create_payload_indexes_calls_all_fields():
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    create_payload_indexes(mock_client, "test_col")
    total_expected = (
        len(_KEYWORD_INDEX_FIELDS)
        + len(_INTEGER_INDEX_FIELDS)
        + len(_BOOL_INDEX_FIELDS)
    )
    assert mock_client.create_payload_index.call_count == total_expected


def test_upsert_batch_calls_client_upsert():
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    upsert_batch(
        mock_client,
        "test_col",
        point_ids=["id1", "id2"],
        vectors=[[0.1, 0.2], [0.3, 0.4]],
        payloads=[{"k": "v1"}, {"k": "v2"}],
    )
    mock_client.upsert.assert_called_once()
    assert mock_client.upsert.call_args.kwargs["collection_name"] == "test_col"


def test_upsert_batch_raises_on_length_mismatch():
    from unittest.mock import MagicMock

    import pytest

    mock_client = MagicMock()
    with pytest.raises(ValueError, match="equal-length"):
        upsert_batch(
            mock_client,
            "test_col",
            point_ids=["id1"],
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            payloads=[{"k": "v1"}],
        )


def test_bootstrap_collection_returns_client():
    from unittest.mock import MagicMock, patch

    mock_client = MagicMock()
    with patch("castlerag.index.qdrant.get_client", return_value=mock_client):
        result = bootstrap_collection(
            host="localhost",
            port=6333,
            collection_name="castle",
            vector_size=128,
        )
    assert result is mock_client
    mock_client.create_collection.assert_called_once()
    total_expected = (
        len(_KEYWORD_INDEX_FIELDS)
        + len(_INTEGER_INDEX_FIELDS)
        + len(_BOOL_INDEX_FIELDS)
    )
    assert mock_client.create_payload_index.call_count == total_expected
