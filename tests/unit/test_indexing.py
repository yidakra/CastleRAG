"""Tests for indexing and embedding helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from castlerag.embed.omniembed import OmniEmbedClient, format_query_text, make_point_id
from castlerag.index.qdrant import build_point_batches, record_to_qdrant_point
from castlerag.index.transcript_lexical import build_bm25_index, load_bm25_index
from castlerag.schemas import AuxRecord, ClipRecord, EventSummaryRecord, TranscriptSegment, TranscriptWindow


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


def test_build_point_batches_mixed_records():
    points = build_point_batches(
        [_transcript_window(), _clip_record(), _event_record()],
        model_version="omniembed-v1",
        model_name="Tevatron/OmniEmbed-v0.1-multivent",
    )
    assert len(points) == 3
    assert all(point.model_name == "Tevatron/OmniEmbed-v0.1-multivent" for point in points)


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
            return type("Resp", (), {"data": [_EmbeddingRow([1.0, 2.0]), _EmbeddingRow([3.0, 4.0])]})()

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
