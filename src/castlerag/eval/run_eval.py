"""Full benchmark loop and smoke-test orchestration.

Per question (SPEC §7.4):
  1. route and extract hints
  2. retrieve route-aware candidates
  3. rerank candidate evidence packs
  4. generate final choice
  5. save prediction plus evidence trace

This module owns orchestration and artifact I/O only. Model internals remain in
the retrieval/rerank/generation modules so concurrent work can land safely.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from castlerag.config import CastleRAGConfig, load_config
from castlerag.embed.omniembed import OmniEmbedClient
from castlerag.eval.io import (
    compute_accuracy,
    export_submission,
    select_questions,
    write_evidence_traces,
    write_predictions,
)
from castlerag.generation.answer import generate_answer
from castlerag.index import get_client, load_bm25_index
from castlerag.rerank.llm_reranker import rerank_candidates
from castlerag.retrieval.search import retrieve as retrieve_evidence
from castlerag.routing.question_router import RouteHints, route_question
from castlerag.schemas import EvalQuestion, Prediction, RerankResult, RetrievalHit


class PipelineDependencyError(RuntimeError):
    """Raised when the eval runner reaches a missing pipeline dependency."""


@dataclass(frozen=True)
class EvalOutputPaths:
    predictions: Path
    evidence_traces: Path
    submissions: Path
    metrics: Path


@dataclass(frozen=True)
class EvalRunResult:
    predictions: Dict[str, Prediction]
    traces: List[dict]
    output_paths: EvalOutputPaths
    accuracy: Optional[float] = None


@dataclass(frozen=True)
class EvalPipeline:
    route: Callable[[str, Dict[str, str]], RouteHints]
    retrieve: Callable[[EvalQuestion, RouteHints], List[RetrievalHit]]
    rerank: Callable[
        [EvalQuestion, RouteHints, List[dict]],
        RerankResult | List[dict],
    ]
    generate: Callable[
        [EvalQuestion, RouteHints, List[RetrievalHit], Dict[str, float]], Prediction
    ]


def run_eval(
    questions: Dict[str, EvalQuestion],
    config_path: Optional[Path] = None,
    answers_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    *,
    predictions_path: Optional[Path] = None,
    question_ids: Optional[Iterable[str]] = None,
    max_questions: Optional[int] = None,
    pipeline: Optional[EvalPipeline] = None,
) -> EvalRunResult:
    """Run the full prediction loop and write output files.

    The runner writes rich predictions, evidence traces, official submission
    export, and metrics when a ground-truth key is available.
    """
    cfg = load_config(override_path=config_path)
    selected = select_questions(
        questions,
        question_ids=question_ids,
        limit=max_questions,
    )
    if not selected:
        raise ValueError("No questions selected for evaluation")

    outputs = _resolve_output_paths(
        cfg,
        out_dir=out_dir,
        predictions_path=predictions_path,
    )
    active_pipeline = pipeline or _build_default_pipeline(cfg)

    predictions: Dict[str, Prediction] = {}
    traces: List[dict] = []
    for question in selected.values():
        hints = active_pipeline.route(question.query, question.answers)
        try:
            retrieved = active_pipeline.retrieve(question, hints)
        except NotImplementedError as exc:
            raise _stage_error("retrieval", question.question_id, exc) from exc

        candidate_packs = _candidate_packs_from_hits(retrieved)
        try:
            reranked = active_pipeline.rerank(question, hints, candidate_packs)
        except NotImplementedError as exc:
            raise _stage_error("reranking", question.question_id, exc) from exc

        rerank_result = _coerce_rerank_result(reranked, hints.route)
        evidence_rows = _flatten_reranked_evidence(
            rerank_result,
            fallback_hits=retrieved,
            max_rows=cfg.retrieval.max_evidence_rows,
        )
        support_priors = _aggregate_support_priors(rerank_result)
        try:
            prediction = active_pipeline.generate(
                question,
                hints,
                evidence_rows,
                support_priors,
            )
        except NotImplementedError as exc:
            raise _stage_error("generation", question.question_id, exc) from exc

        predictions[question.question_id] = prediction
        traces.append(
            {
                "question_id": question.question_id,
                "route": hints.route,
                "retrieved_count": len(retrieved),
                "reranked_count": len(rerank_result.kept_packs),
                "top_evidence_ids": prediction.top_evidence_ids,
                "support_priors": prediction.support_priors or support_priors,
                "predicted_answer": prediction.predicted_answer,
            }
        )

    write_predictions(predictions, outputs.predictions)
    write_evidence_traces(traces, outputs.evidence_traces)
    export_submission(predictions, outputs.submissions)

    accuracy: Optional[float] = None
    if answers_path is not None:
        accuracy = compute_accuracy(selected, predictions, answers_path)
        outputs.metrics.parent.mkdir(parents=True, exist_ok=True)
        outputs.metrics.write_text(
            json.dumps(
                {
                    "accuracy": accuracy,
                    "num_questions": len(selected),
                    "predictions_path": str(outputs.predictions),
                    "submission_path": str(outputs.submissions),
                },
                indent=2,
            )
        )

    return EvalRunResult(
        predictions=predictions,
        traces=traces,
        output_paths=outputs,
        accuracy=accuracy,
    )


def _resolve_output_paths(
    cfg: CastleRAGConfig,
    *,
    out_dir: Optional[Path],
    predictions_path: Optional[Path],
) -> EvalOutputPaths:
    default_dir = Path(cfg.outputs.dir)
    if out_dir is not None:
        target_dir = out_dir
    elif predictions_path is not None:
        target_dir = predictions_path.parent
    else:
        target_dir = default_dir
    predictions = predictions_path or target_dir / Path(cfg.outputs.predictions).name
    return EvalOutputPaths(
        predictions=predictions,
        evidence_traces=target_dir / Path(cfg.outputs.evidence_traces).name,
        submissions=target_dir / Path(cfg.outputs.submissions).name,
        metrics=target_dir / Path(cfg.outputs.metrics).name,
    )


def _build_default_pipeline(cfg: CastleRAGConfig) -> EvalPipeline:
    bm25_path = Path(cfg.embedding.cache_dir) / "transcripts.pkl"
    if not bm25_path.exists():
        raise PipelineDependencyError(
            "BM25 transcript index not found. Run `castlerag index` first."
        )

    bm25_index = load_bm25_index(bm25_path)
    qdrant_client = get_client(cfg.qdrant.host, cfg.qdrant.port)
    embed_client = OmniEmbedClient(
        model=cfg.embedding.model,
        backend=cfg.embedding.backend,
        vllm_base_url=_vllm_base_url(),
        vllm_tensor_parallel=cfg.embedding.vllm_tensor_parallel,
        vllm_gpu_memory_utilization=cfg.embedding.vllm_gpu_memory_utilization,
    )
    generation_client = _build_vllm_chat_client()

    def _retrieve(question: EvalQuestion, hints: RouteHints) -> List[RetrievalHit]:
        return retrieve_evidence(
            question=question,
            hints=hints,
            qdrant_client=qdrant_client,
            collection_name=cfg.qdrant.collection,
            bm25_index=bm25_index,
            embed_client=embed_client,
            retrieval_cfg=cfg.retrieval,
        )

    def _rerank(
        question: EvalQuestion,
        hints: RouteHints,
        candidate_packs: List[dict],
    ) -> RerankResult:
        return rerank_candidates(
            question=question,
            hints=hints,
            candidate_packs=candidate_packs,
            llm_client=generation_client,
            top_k=cfg.reranking.top_k,
            min_relevance=cfg.reranking.min_relevance,
        )

    def _generate(
        question: EvalQuestion,
        hints: RouteHints,
        evidence_rows: List[RetrievalHit],
        support_priors: Dict[str, float],
    ) -> Prediction:
        return generate_answer(
            question=question,
            hints=hints,
            evidence_rows=evidence_rows,
            support_priors=support_priors,
            llm_client=generation_client,
            max_evidence_rows=cfg.retrieval.max_evidence_rows,
        )

    return EvalPipeline(
        route=route_question,
        retrieve=_retrieve,
        rerank=_rerank,
        generate=_generate,
    )


def _candidate_packs_from_hits(hits: List[RetrievalHit]) -> List[dict]:
    return [
        {
            "rank": rank,
            "record_id": hit.record_id,
            "primary_hit": hit,
            "evidence_rows": [hit],
        }
        for rank, hit in enumerate(hits, start=1)
    ]


def _flatten_reranked_evidence(
    reranked: RerankResult,
    *,
    fallback_hits: List[RetrievalHit],
    max_rows: int,
) -> List[RetrievalHit]:
    if not reranked.kept_packs:
        return fallback_hits[:max_rows]

    rows: List[RetrievalHit] = []
    seen: set[str] = set()
    for pack in reranked.kept_packs:
        candidates = pack.pack.evidence_rows or [pack.pack.primary_hit]
        for hit in candidates:
            if hit.record_id in seen:
                continue
            seen.add(hit.record_id)
            rows.append(hit)
            if len(rows) >= max_rows:
                return rows
    return rows or fallback_hits[:max_rows]


def _aggregate_support_priors(reranked: RerankResult) -> Dict[str, float]:
    return dict(reranked.support_priors)


def _coerce_rerank_result(
    reranked: RerankResult | List[dict],
    route: str,
) -> RerankResult:
    if isinstance(reranked, RerankResult):
        return reranked

    support = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
    evidence_rows: List[RetrievalHit] = []
    seen: set[str] = set()
    for pack in reranked:
        raw_support = pack.get("support")
        if isinstance(raw_support, dict):
            for key in support:
                value = raw_support.get(key)
                if isinstance(value, (int, float)):
                    support[key] = max(support[key], float(value) / 4.0)
        for hit in pack.get("evidence_rows") or [pack.get("primary_hit")]:
            if isinstance(hit, RetrievalHit) and hit.record_id not in seen:
                evidence_rows.append(hit)
                seen.add(hit.record_id)
    return RerankResult(
        route=route,  # type: ignore[arg-type]
        support_priors=support if any(support.values()) else {},
        evidence_rows=evidence_rows,
    )


def _stage_error(
    stage: str,
    question_id: str,
    exc: Exception,
) -> PipelineDependencyError:
    return PipelineDependencyError(
        f"{stage} is not implemented for question {question_id}: {exc}"
    )


def _vllm_base_url() -> Optional[str]:
    return os.getenv("VLLM_BASE_URL")


def _build_vllm_chat_client() -> Any:
    base_url = _vllm_base_url()
    if not base_url:
        raise PipelineDependencyError(
            "VLLM_BASE_URL is not set. Start the Qwen3-VL vLLM endpoint first."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise PipelineDependencyError(
            "openai package is required to talk to the vLLM endpoint."
        ) from exc
    return OpenAI(base_url=base_url, api_key="not-needed")
