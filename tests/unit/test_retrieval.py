"""Tests for routing and retrieval logic."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from castlerag.retrieval.search import reciprocal_rank_fusion, retrieve
from castlerag.retrieval.transcript_lexical import score_windows
from castlerag.routing.question_router import RouteEvidenceProfile, route_question
from castlerag.schemas import (
    EvalQuestion,
    RetrievalHit,
    TranscriptSegment,
    TranscriptWindow,
)


def _question() -> EvalQuestion:
    return EvalQuestion(
        question_id="q1",
        query="What did Allie say after breakfast in the kitchen?",
        answers={
            "a": "She went to work",
            "b": "She cooked soup",
            "c": "She called Bjorn",
            "d": "She left the house",
        },
    )


def _windows() -> list[TranscriptWindow]:
    return [
        TranscriptWindow(
            transcript_window_id="tw_1",
            day="day1",
            camera_id="Allie",
            camera_type="ego",
            participant_id="Allie",
            room="Kitchen",
            hour=8,
            transcript_text=(
                "After breakfast Allie said she would call Bjorn from the "
                "kitchen."
            ),
            transcript_segments=[
                TranscriptSegment(start=0.0, end=4.0, text="After breakfast")
            ],
            has_speech=True,
            transcript_char_len=66,
            absolute_start=1_672_531_200_000,
            absolute_end=1_672_531_215_000,
        ),
        TranscriptWindow(
            transcript_window_id="tw_2",
            day="day1",
            camera_id="Allie",
            camera_type="ego",
            participant_id="Allie",
            room="Office",
            hour=9,
            transcript_text="Allie quietly walked into the office.",
            transcript_segments=[],
            has_speech=True,
            transcript_char_len=37,
            absolute_start=1_672_531_300_000,
            absolute_end=1_672_531_315_000,
        ),
    ]


def test_route_question_extracts_route_and_hints():
    hints = route_question(
        question="On day 1, what did Allie say before entering the kitchen?",
        choices={"a": "hello", "b": "bye", "c": "thanks", "d": "nothing"},
    )
    assert hints.route == "temporal"
    assert hints.day == "day1"
    assert hints.participant == "Allie"
    assert hints.room == "Kitchen"
    assert hints.has_speech_cue is True
    assert hints.has_temporal_cue is True


def test_score_windows_prefers_exact_overlap_and_hints():
    class FakeBM25:
        def get_scores(self, tokens: list[str]) -> np.ndarray:
            return np.asarray([2.0, 1.0], dtype=np.float32)

    bundle = SimpleNamespace(bm25=FakeBM25())
    hits = score_windows(
        bm25_index=bundle,
        windows=_windows(),
        query=_question().query,
        choices=_question().answers,
        day_hint="day1",
        person_hint="Allie",
        room_hint="Kitchen",
        top_k=2,
    )
    assert hits[0].record_id == "tw_1"
    assert hits[0].score > hits[1].score


def test_reciprocal_rank_fusion_merges_on_record_id():
    hit_a = RetrievalHit(
        rank=1,
        score=1.0,
        point_id="p1",
        record_id="r1",
        source_type="main_clip",
        modality="video",
    )
    hit_b = RetrievalHit(
        rank=2,
        score=0.8,
        point_id="p2",
        record_id="r2",
        source_type="main_clip",
        modality="video",
    )
    hit_c = RetrievalHit(
        rank=1,
        score=0.9,
        point_id="x1",
        record_id="r1",
        source_type="transcript_window",
        modality="text",
    )
    fused = reciprocal_rank_fusion([[hit_a, hit_b], [hit_c]], k=60)
    assert fused[0].record_id == "r1"
    assert fused[1].record_id == "r2"
    assert fused[0].rank == 1


def test_retrieve_fuses_transcript_and_multimodal_hits():
    class FakeBM25:
        def get_scores(self, tokens: list[str]) -> np.ndarray:
            return np.asarray([3.0, 1.0], dtype=np.float32)

    class FakeEmbedClient:
        def embed_texts(self, texts: list[str]) -> np.ndarray:
            assert len(texts) == 2
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    class FakePoint:
        def __init__(self, pid: str, score: float, payload: dict) -> None:
            self.id = pid
            self.score = score
            self.payload = payload

    class FakeQdrantClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def query_points(self, **kwargs):
            self.calls.append(kwargs)
            source_type = next(
                condition.match.value
                for condition in kwargs["query_filter"].must
                if condition.key == "source_type"
            )
            if source_type == "transcript_window":
                points = [
                    FakePoint(
                        "pt_tx_1",
                        0.9,
                        {
                            "record_id": "tw_1",
                            "source_type": "transcript_window",
                            "modality": "text",
                            "day": "day1",
                            "camera_id": "Allie",
                            "participant_id": "Allie",
                            "absolute_start": 1_672_531_200_000,
                            "absolute_end": 1_672_531_215_000,
                            "transcript_text": "Allie said she would call Bjorn.",
                        },
                    )
                ]
            elif source_type == "main_clip":
                points = [
                    FakePoint(
                        "pt_clip_1",
                        0.8,
                        {
                            "record_id": "clip_1",
                            "source_type": "main_clip",
                            "modality": "video",
                            "day": "day1",
                            "camera_id": "Allie",
                            "participant_id": "Allie",
                            "absolute_start": 1_672_531_200_000,
                            "absolute_end": 1_672_531_230_000,
                            "event_summary": "Allie speaks in the kitchen.",
                            "asset_path": "/tmp/clip.mp4",
                        },
                    )
                ]
            else:
                points = []
            return SimpleNamespace(points=points)

    bm25_bundle = SimpleNamespace(bm25=FakeBM25(), windows=_windows())
    retrieval_cfg = SimpleNamespace(
        transcript_top_k=30,
        event_summary_top_k=20,
        video_top_k=20,
        photo_top_k=16,
        aux_video_top_k=8,
        heartrate_top_k=8,
        gaze_top_k=8,
        thermal_top_k=8,
        rrf_k=60,
        max_candidate_videos=4,
        frames_per_candidate=32,
        max_aux_images=16,
        max_evidence_rows=50,
    )
    hints = route_question(_question().query, _question().answers)
    qdrant = FakeQdrantClient()
    hits = retrieve(
        question=_question(),
        hints=hints,
        qdrant_client=qdrant,
        collection_name="castle_test",
        bm25_index=bm25_bundle,
        embed_client=FakeEmbedClient(),
        retrieval_cfg=retrieval_cfg,
    )
    assert hits
    assert hits[0].record_id == "tw_1"
    assert any(hit.source_type == "main_clip" for hit in hits)
    assert len(hits) <= 50


def test_retrieve_consumes_router_budget_profile_without_reparsing():
    class FakeBM25:
        def get_scores(self, tokens: list[str]) -> np.ndarray:
            return np.asarray([5.0, 4.0], dtype=np.float32)

    class FakeEmbedClient:
        def embed_texts(self, texts: list[str]) -> np.ndarray:
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    class FakePoint:
        def __init__(self, pid: str, score: float, payload: dict) -> None:
            self.id = pid
            self.score = score
            self.payload = payload

    class FakeQdrantClient:
        def query_points(self, **kwargs):
            source_type = next(
                condition.match.value
                for condition in kwargs["query_filter"].must
                if condition.key == "source_type"
            )
            if source_type == "transcript_window":
                return SimpleNamespace(
                    points=[
                        FakePoint(
                            "pt_tx_3",
                            0.95,
                            {
                                "record_id": "tw_2",
                                "source_type": "transcript_window",
                                "modality": "text",
                                "day": "day1",
                                "camera_id": "Allie",
                                "participant_id": "Allie",
                                "absolute_start": 1_672_531_300_000,
                                "absolute_end": 1_672_531_315_000,
                                "transcript_text": (
                                    "Allie quietly walked into the office."
                                ),
                            },
                        )
                    ]
                )
            if source_type == "main_clip":
                return SimpleNamespace(
                    points=[
                        FakePoint(
                            "pt_clip_2",
                            0.7,
                            {
                                "record_id": "clip_2",
                                "source_type": "main_clip",
                                "modality": "video",
                                "day": "day1",
                                "camera_id": "Allie",
                                "participant_id": "Allie",
                                "absolute_start": 1_672_531_320_000,
                                "absolute_end": 1_672_531_350_000,
                                "asset_path": "/tmp/clip_2.mp4",
                            },
                        )
                    ]
                )
            return SimpleNamespace(points=[])

    bm25_bundle = SimpleNamespace(bm25=FakeBM25(), windows=_windows())
    retrieval_cfg = SimpleNamespace(
        transcript_top_k=30,
        event_summary_top_k=20,
        video_top_k=20,
        photo_top_k=16,
        aux_video_top_k=8,
        heartrate_top_k=8,
        gaze_top_k=8,
        thermal_top_k=8,
        rrf_k=60,
        max_candidate_videos=4,
        frames_per_candidate=32,
        max_aux_images=16,
        max_evidence_rows=50,
    )
    hints = route_question(
        "What color shirt was Allie wearing in the kitchen?",
        {"a": "Blue", "b": "Black", "c": "White", "d": "Red"},
    )
    hints.evidence_profile = RouteEvidenceProfile(
        transcript_budget=1,
        candidate_video_budget=4,
        frames_per_candidate_video=32,
        auxiliary_image_budget=16,
        max_evidence_rows=50,
        source_priority=("main_clip", "transcript_window"),
    )
    hits = retrieve(
        question=_question(),
        hints=hints,
        qdrant_client=FakeQdrantClient(),
        collection_name="castle_test",
        bm25_index=bm25_bundle,
        embed_client=FakeEmbedClient(),
        retrieval_cfg=retrieval_cfg,
    )
    assert sum(1 for hit in hits if hit.source_type == "transcript_window") == 1
    assert hits[0].source_type == "main_clip"
