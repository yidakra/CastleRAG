"""Unit tests for src/castlerag/eval/run_eval.py — error paths and wiring."""

from __future__ import annotations

import builtins
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from castlerag.config import CastleRAGConfig, load_config
from castlerag.eval.run_eval import (
    EvalPipeline,
    IndexArtifactReport,
    PipelineDependencyError,
    RerankResult,
    RetrievalHit,
    _build_default_pipeline,
    _build_vllm_chat_client,
    _coerce_rerank_result,
    _ensure_qdrant_collection_ready,
    _ensure_vllm_runtime_ready,
    _flatten_reranked_evidence,
    _qdrant_collection_count,
    _qdrant_collection_exists,
    run_eval,
)
from castlerag.routing.question_router import RouteHints
from castlerag.schemas import (
    EvalQuestion,
    EvidencePack,
    Prediction,
    RerankedEvidencePack,
    RerankerOutput,
)

run_eval_module = importlib.import_module("castlerag.eval.run_eval")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hit(record_id: str, rank: int = 1, score: float = 0.9) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        score=score,
        point_id=f"pt_{record_id}",
        record_id=record_id,
        source_type="transcript_window",
        modality="text",
        day="day1",
        camera_id="Allie",
        participant_id="Allie",
        absolute_start=1_672_531_200_000,
        absolute_end=1_672_531_215_000,
        transcript_text="Allie said hello.",
    )


def _make_question(qid: str = "q1") -> EvalQuestion:
    return EvalQuestion(
        question_id=qid,
        query="What did Allie do?",
        answers={"a": "A", "b": "B", "c": "C", "d": "D"},
    )


def _make_reranked_pack(hit: RetrievalHit) -> RerankedEvidencePack:
    pack = EvidencePack(
        pack_id=f"pack_{hit.record_id}",
        route="speech_text",
        primary_hit=hit,
        evidence_rows=[hit],
    )
    reranker_output = RerankerOutput(
        relevance=3,
        support={"a": 2, "b": 1, "c": 0, "d": 0},
        keep=True,
        rationale="Relevant.",
    )
    return RerankedEvidencePack(
        pack=pack,
        reranker_output=reranker_output,
        final_rerank_score=0.8,
    )


def _make_artifact_report(tmp_path: Path) -> IndexArtifactReport:
    return IndexArtifactReport(
        bm25_path=tmp_path / "transcripts.pkl",
        chunks_dir=tmp_path / "chunks",
        cache_dir=tmp_path / "embeddings",
        chunk_files={"transcripts": [], "clips": [], "events": [], "aux": []},
        embedding_caches={
            "transcripts": [],
            "events": [],
            "clips": [],
            "aux_text": [],
            "aux_image": [],
            "aux_video": [],
        },
    )


def _questions_dict(qids: list | None = None) -> Dict[str, EvalQuestion]:
    if qids is None:
        qids = ["q1"]
    return {qid: _make_question(qid) for qid in qids}


def _default_route(query: str, answers: dict) -> RouteHints:
    return RouteHints(route="speech_text", day="day1", participant="Allie")


def _default_retrieve(question: EvalQuestion, hints: RouteHints) -> List[RetrievalHit]:
    return [_make_hit(f"{question.question_id}_1")]


def _default_rerank(
    question: EvalQuestion,
    hints: RouteHints,
    candidate_packs: list,
) -> RerankResult:
    return RerankResult(
        route="speech_text",
        support_priors={"a": 0.5, "b": 0.0, "c": 0.0, "d": 0.0},
        evidence_rows=[],
    )


def _default_generate(
    question: EvalQuestion,
    hints: RouteHints,
    evidence_rows: List[RetrievalHit],
    support_priors: Dict[str, float],
) -> Prediction:
    return Prediction(question_id=question.question_id, predicted_answer="a")


def _default_pipeline() -> EvalPipeline:
    return EvalPipeline(
        route=_default_route,
        retrieve=_default_retrieve,
        rerank=_default_rerank,
        generate=_default_generate,
    )


# ---------------------------------------------------------------------------
# run_eval error paths
# ---------------------------------------------------------------------------


class TestRunEvalErrorPaths:
    """Test that stage failures are wrapped as PipelineDependencyError."""

    def test_retrieve_not_implemented_raises_dependency_error(self, tmp_path: Path):
        def _bad_retrieve(question, hints):
            raise NotImplementedError("retrieval not done yet")

        pipeline = EvalPipeline(
            route=_default_route,
            retrieve=_bad_retrieve,
            rerank=_default_rerank,
            generate=_default_generate,
        )
        with pytest.raises(PipelineDependencyError, match="retrieval.*q1"):
            run_eval(_questions_dict(), out_dir=tmp_path / "out", pipeline=pipeline)

    def test_retrieve_pipeline_dependency_error_is_re_wrapped(self, tmp_path: Path):
        def _bad_retrieve(question, hints):
            raise PipelineDependencyError("Qdrant offline")

        pipeline = EvalPipeline(
            route=_default_route,
            retrieve=_bad_retrieve,
            rerank=_default_rerank,
            generate=_default_generate,
        )
        with pytest.raises(
            PipelineDependencyError,
            match="retrieval dependency failed for question q1",
        ):
            run_eval(_questions_dict(), out_dir=tmp_path / "out", pipeline=pipeline)

    def test_retrieve_generic_exception_is_wrapped(self, tmp_path: Path):
        def _bad_retrieve(question, hints):
            raise RuntimeError("unexpected I/O failure")

        pipeline = EvalPipeline(
            route=_default_route,
            retrieve=_bad_retrieve,
            rerank=_default_rerank,
            generate=_default_generate,
        )
        with pytest.raises(
            PipelineDependencyError, match="retrieval failed for question q1"
        ):
            run_eval(_questions_dict(), out_dir=tmp_path / "out", pipeline=pipeline)

    def test_rerank_not_implemented_raises_dependency_error(self, tmp_path: Path):
        def _bad_rerank(question, hints, packs):
            raise NotImplementedError("reranker not done yet")

        pipeline = EvalPipeline(
            route=_default_route,
            retrieve=_default_retrieve,
            rerank=_bad_rerank,
            generate=_default_generate,
        )
        with pytest.raises(PipelineDependencyError, match="reranking.*q1"):
            run_eval(_questions_dict(), out_dir=tmp_path / "out", pipeline=pipeline)

    def test_rerank_pipeline_dependency_error_is_re_wrapped(self, tmp_path: Path):
        def _bad_rerank(question, hints, packs):
            raise PipelineDependencyError("LLM endpoint down")

        pipeline = EvalPipeline(
            route=_default_route,
            retrieve=_default_retrieve,
            rerank=_bad_rerank,
            generate=_default_generate,
        )
        with pytest.raises(
            PipelineDependencyError,
            match="reranking dependency failed for question q1",
        ):
            run_eval(_questions_dict(), out_dir=tmp_path / "out", pipeline=pipeline)

    def test_rerank_generic_exception_is_wrapped(self, tmp_path: Path):
        def _bad_rerank(question, hints, packs):
            raise ValueError("shape mismatch")

        pipeline = EvalPipeline(
            route=_default_route,
            retrieve=_default_retrieve,
            rerank=_bad_rerank,
            generate=_default_generate,
        )
        with pytest.raises(
            PipelineDependencyError, match="reranking failed for question q1"
        ):
            run_eval(_questions_dict(), out_dir=tmp_path / "out", pipeline=pipeline)

    def test_generate_not_implemented_raises_dependency_error(self, tmp_path: Path):
        def _bad_generate(question, hints, evidence_rows, support_priors):
            raise NotImplementedError("generation not done yet")

        pipeline = EvalPipeline(
            route=_default_route,
            retrieve=_default_retrieve,
            rerank=_default_rerank,
            generate=_bad_generate,
        )
        with pytest.raises(PipelineDependencyError, match="generation.*q1"):
            run_eval(_questions_dict(), out_dir=tmp_path / "out", pipeline=pipeline)

    def test_generate_pipeline_dependency_error_is_re_wrapped(self, tmp_path: Path):
        def _bad_generate(question, hints, evidence_rows, support_priors):
            raise PipelineDependencyError("VLLM server crashed")

        pipeline = EvalPipeline(
            route=_default_route,
            retrieve=_default_retrieve,
            rerank=_default_rerank,
            generate=_bad_generate,
        )
        with pytest.raises(
            PipelineDependencyError,
            match="generation dependency failed for question q1",
        ):
            run_eval(_questions_dict(), out_dir=tmp_path / "out", pipeline=pipeline)

    def test_generate_generic_exception_is_wrapped(self, tmp_path: Path):
        def _bad_generate(question, hints, evidence_rows, support_priors):
            raise OSError("disk full")

        pipeline = EvalPipeline(
            route=_default_route,
            retrieve=_default_retrieve,
            rerank=_default_rerank,
            generate=_bad_generate,
        )
        with pytest.raises(
            PipelineDependencyError, match="generation failed for question q1"
        ):
            run_eval(_questions_dict(), out_dir=tmp_path / "out", pipeline=pipeline)

    def test_no_questions_selected_raises_value_error(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No questions selected"):
            run_eval({}, out_dir=tmp_path / "out", pipeline=_default_pipeline())


# ---------------------------------------------------------------------------
# _flatten_reranked_evidence
# ---------------------------------------------------------------------------


class TestFlattenRerankedEvidence:
    def test_empty_kept_packs_falls_back_to_fallback_hits(self):
        reranked = RerankResult(
            route="speech_text",
            kept_packs=[],
        )
        fallback = [_make_hit("fb_1"), _make_hit("fb_2"), _make_hit("fb_3")]
        result = _flatten_reranked_evidence(
            reranked, fallback_hits=fallback, max_rows=5
        )
        assert result == fallback

    def test_empty_kept_packs_respects_max_rows(self):
        reranked = RerankResult(route="speech_text", kept_packs=[])
        fallback = [_make_hit(f"fb_{i}") for i in range(10)]
        result = _flatten_reranked_evidence(
            reranked, fallback_hits=fallback, max_rows=3
        )
        assert len(result) == 3
        assert result == fallback[:3]

    def test_deduplication_across_packs(self):
        hit_shared = _make_hit("shared_1")
        hit_unique = _make_hit("unique_1")
        pack1 = _make_reranked_pack(hit_shared)
        # Second pack has both the shared hit and a unique hit — shared should dedup
        pack2_base = EvidencePack(
            pack_id="pack_2",
            route="speech_text",
            primary_hit=hit_shared,
            evidence_rows=[hit_shared, hit_unique],
        )
        reranker_out = RerankerOutput(
            relevance=2,
            support={"a": 1, "b": 0, "c": 0, "d": 0},
            keep=True,
            rationale="ok",
        )
        pack2 = RerankedEvidencePack(
            pack=pack2_base,
            reranker_output=reranker_out,
            final_rerank_score=0.5,
        )
        reranked = RerankResult(
            route="speech_text",
            kept_packs=[pack1, pack2],
        )
        result = _flatten_reranked_evidence(
            reranked, fallback_hits=[], max_rows=10
        )
        ids = [h.record_id for h in result]
        # shared_1 should appear exactly once
        assert ids.count("shared_1") == 1
        assert "unique_1" in ids

    def test_max_rows_caps_output(self):
        hits = [_make_hit(f"h{i}") for i in range(8)]
        packs = [_make_reranked_pack(h) for h in hits]
        reranked = RerankResult(route="speech_text", kept_packs=packs)
        result = _flatten_reranked_evidence(
            reranked, fallback_hits=[], max_rows=3
        )
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _coerce_rerank_result
# ---------------------------------------------------------------------------


class TestCoerceRerankResult:
    def test_passthrough_for_rerank_result(self):
        rr = RerankResult(route="speech_text")
        assert _coerce_rerank_result(rr, "speech_text") is rr

    def test_dict_list_with_support_scores(self):
        hit = _make_hit("rec_1")
        pack_dicts = [
            {
                "support": {"a": 4, "b": 2, "c": 1, "d": 0},
                "evidence_rows": [hit],
                "primary_hit": None,
            }
        ]
        result = _coerce_rerank_result(pack_dicts, "speech_text")
        assert isinstance(result, RerankResult)
        assert result.route == "speech_text"
        # support[a] = max(0, 4/4) = 1.0
        assert result.support_priors["a"] == pytest.approx(1.0)
        assert result.support_priors["b"] == pytest.approx(0.5)
        assert hit in result.evidence_rows

    def test_dict_list_without_retrieval_hits_gives_empty_evidence(self):
        pack_dicts = [
            {
                "support": {"a": 0, "b": 0, "c": 0, "d": 0},
                "evidence_rows": None,
                "primary_hit": None,
            }
        ]
        result = _coerce_rerank_result(pack_dicts, "temporal")
        assert isinstance(result, RerankResult)
        assert result.evidence_rows == []

    def test_dict_list_non_retrieval_hit_entries_are_skipped(self):
        pack_dicts = [
            {
                "support": None,
                "evidence_rows": [{"not": "a RetrievalHit"}],
                "primary_hit": None,
            }
        ]
        result = _coerce_rerank_result(pack_dicts, "mixed")
        assert isinstance(result, RerankResult)
        assert result.evidence_rows == []


# ---------------------------------------------------------------------------
# _build_default_pipeline
# ---------------------------------------------------------------------------


class TestBuildDefaultPipeline:
    def test_returns_eval_pipeline_with_all_callables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cfg = load_config(
            override_path=Path(
                "/Users/adeliev/Development/CastleRAG/.claude/worktrees/"
                "agent-a4206a53948faec0a/configs/base.yaml"
            )
        )

        mock_bm25 = MagicMock()
        mock_qdrant = MagicMock()
        mock_report = _make_artifact_report(tmp_path)
        mock_llm_client = MagicMock()

        monkeypatch.setattr(
            run_eval_module,
            "_prepare_default_runtime",
            lambda c: (mock_bm25, mock_qdrant, mock_report),
        )
        monkeypatch.setattr(
            run_eval_module,
            "_build_vllm_chat_client",
            lambda: mock_llm_client,
        )
        monkeypatch.setattr(
            run_eval_module,
            "OmniEmbedClient",
            lambda **kwargs: MagicMock(),
        )

        pipeline = _build_default_pipeline(cfg)
        assert isinstance(pipeline, EvalPipeline)
        assert callable(pipeline.route)
        assert callable(pipeline.retrieve)
        assert callable(pipeline.rerank)
        assert callable(pipeline.generate)


# ---------------------------------------------------------------------------
# _ensure_qdrant_collection_ready
# ---------------------------------------------------------------------------


class TestEnsureQdrantCollectionReady:
    def _cfg(self) -> CastleRAGConfig:
        cfg = CastleRAGConfig()
        cfg.qdrant.host = "localhost"
        cfg.qdrant.port = 6333
        cfg.qdrant.collection = "test_collection"
        return cfg

    def test_collection_not_found_raises_dependency_error(self):
        client = MagicMock()
        client.collection_exists.return_value = False
        with pytest.raises(PipelineDependencyError, match="does not exist"):
            _ensure_qdrant_collection_ready(client, self._cfg())

    def test_empty_collection_raises_dependency_error(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        count_response = SimpleNamespace(count=0)
        client.count.return_value = count_response
        with pytest.raises(PipelineDependencyError, match="is empty"):
            _ensure_qdrant_collection_ready(client, self._cfg())

    def test_healthy_collection_passes(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        count_response = SimpleNamespace(count=42)
        client.count.return_value = count_response
        # Should not raise
        _ensure_qdrant_collection_ready(client, self._cfg())

    def test_collection_exists_check_exception_raises_dependency_error(self):
        client = MagicMock()
        client.collection_exists.side_effect = ConnectionError("refused")
        with pytest.raises(PipelineDependencyError, match="could not reach Qdrant"):
            _ensure_qdrant_collection_ready(client, self._cfg())

    def test_count_exception_raises_dependency_error(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.count.side_effect = RuntimeError("count failed")
        with pytest.raises(PipelineDependencyError, match="could not inspect Qdrant"):
            _ensure_qdrant_collection_ready(client, self._cfg())


# ---------------------------------------------------------------------------
# _qdrant_collection_exists
# ---------------------------------------------------------------------------


class TestQdrantCollectionExists:
    def test_collection_exists_via_collection_exists_method_true(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        assert _qdrant_collection_exists(client, "mycol") is True

    def test_collection_exists_via_collection_exists_method_false(self):
        client = MagicMock()
        client.collection_exists.return_value = False
        assert _qdrant_collection_exists(client, "mycol") is False

    def test_get_collection_path_exists(self):
        # client has no collection_exists but has get_collection
        client = SimpleNamespace(
            get_collection=lambda name: object()
        )
        assert _qdrant_collection_exists(client, "mycol") is True

    def test_get_collection_not_found_returns_false(self):
        def _get_collection(name):
            raise Exception("not found: collection does not exist")

        client = SimpleNamespace(get_collection=_get_collection)
        assert _qdrant_collection_exists(client, "mycol") is False

    def test_get_collection_other_error_reraises(self):
        def _get_collection(name):
            raise RuntimeError("network timeout")

        client = SimpleNamespace(get_collection=_get_collection)
        with pytest.raises(RuntimeError, match="network timeout"):
            _qdrant_collection_exists(client, "mycol")

    def test_no_known_method_returns_true(self):
        # Neither collection_exists nor get_collection — assume exists
        client = SimpleNamespace()
        assert _qdrant_collection_exists(client, "mycol") is True


# ---------------------------------------------------------------------------
# _qdrant_collection_count
# ---------------------------------------------------------------------------


class TestQdrantCollectionCount:
    def test_int_response_returned_directly(self):
        client = MagicMock()
        client.count.return_value = 99
        assert _qdrant_collection_count(client, "col") == 99

    def test_object_with_count_attribute(self):
        client = MagicMock()
        client.count.return_value = SimpleNamespace(count=42)
        assert _qdrant_collection_count(client, "col") == 42

    def test_object_without_count_attribute_returns_none(self):
        client = MagicMock()
        client.count.return_value = SimpleNamespace(total=100)  # no .count
        assert _qdrant_collection_count(client, "col") is None

    def test_no_count_method_returns_none(self):
        client = SimpleNamespace()  # no count attribute
        assert _qdrant_collection_count(client, "col") is None


# ---------------------------------------------------------------------------
# _ensure_vllm_runtime_ready
# ---------------------------------------------------------------------------


class TestEnsureVllmRuntimeReady:
    def _cfg(self) -> CastleRAGConfig:
        cfg = CastleRAGConfig()
        cfg.embedding.backend = "vllm"
        return cfg

    def test_missing_vllm_base_url_raises_dependency_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        with pytest.raises(PipelineDependencyError, match="VLLM_BASE_URL is not set"):
            _ensure_vllm_runtime_ready(self._cfg())

    def test_missing_openai_package_raises_dependency_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")

        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        with pytest.raises(PipelineDependencyError, match="openai package is required"):
            _ensure_vllm_runtime_ready(self._cfg())


# ---------------------------------------------------------------------------
# _build_vllm_chat_client
# ---------------------------------------------------------------------------


class TestBuildVllmChatClient:
    def test_no_vllm_base_url_raises_dependency_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        with pytest.raises(PipelineDependencyError, match="VLLM_BASE_URL is not set"):
            _build_vllm_chat_client()

    def test_missing_openai_package_raises_dependency_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")

        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        with pytest.raises(PipelineDependencyError, match="openai package is required"):
            _build_vllm_chat_client()

    def test_with_valid_url_and_openai_returns_client(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")

        mock_openai_instance = MagicMock()
        mock_openai_cls = MagicMock(return_value=mock_openai_instance)

        mock_openai_module = MagicMock()
        mock_openai_module.OpenAI = mock_openai_cls

        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "openai":
                return mock_openai_module
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        client = _build_vllm_chat_client()
        assert client is mock_openai_instance
        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:8000", api_key="not-needed"
        )
