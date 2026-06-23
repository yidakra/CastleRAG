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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from castlerag.config import CastleRAGConfig, load_config
from castlerag.embed.omniembed import OmniEmbedClient
from castlerag.eval.io import (
    accuracy_breakdown,
    compute_diversity_metrics,
    export_submission,
    select_questions,
    write_evidence_traces,
    write_predictions,
)
from castlerag.generation.answer import generate_answer
from castlerag.index import get_client, load_bm25_index
from castlerag.rerank.llm_reranker import rerank_candidates
from castlerag.retrieval.candidate_expand import expand_candidates
from castlerag.retrieval.search import retrieve as retrieve_evidence
from castlerag.routing.question_router import RouteHints, route_question
from castlerag.schemas import EvalQuestion, Prediction, RerankResult, RetrievalHit

try:  # pragma: no cover - optional dependency typing only
    from qdrant_client.http.exceptions import (
        ResponseHandlingException,
        UnexpectedResponse,
    )
except ImportError:  # pragma: no cover - qdrant-client may be absent in some envs
    ResponseHandlingException = UnexpectedResponse = ()  # type: ignore[assignment]


class PipelineDependencyError(RuntimeError):
    """Raised when the eval runner reaches a missing pipeline dependency."""


@dataclass(frozen=True)
class EvalOutputPaths:
    """Resolved file paths for all eval output artefacts."""

    predictions: Path
    evidence_traces: Path
    submissions: Path
    metrics: Path


@dataclass(frozen=True)
class EvalRunResult:
    """Return value of run_eval: predictions, traces, paths, and optional metrics."""

    predictions: Dict[str, Prediction]
    traces: List[dict]
    output_paths: EvalOutputPaths
    accuracy: Optional[float] = None
    diversity: Optional[Dict[str, Any]] = None
    # Number of correctly-answered and ground-truth-graded questions. These let
    # callers display ``n_correct/n_graded`` rather than guessing a denominator.
    n_correct: Optional[int] = None
    n_graded: Optional[int] = None


@dataclass(frozen=True)
class IndexArtifactReport:
    """Inventory of local index artefacts used for dependency diagnostics."""

    bm25_path: Path
    chunks_dir: Path
    cache_dir: Path
    chunk_files: Dict[str, List[Path]]
    embedding_caches: Dict[str, List[Path]]


@dataclass(frozen=True)
class EvalPipeline:
    """Pluggable pipeline of callables used by run_eval (enables test injection)."""

    route: Callable[[str, Dict[str, str]], RouteHints]
    retrieve: Callable[[EvalQuestion, RouteHints], List[RetrievalHit]]
    rerank: Callable[
        [EvalQuestion, RouteHints, List[dict]],
        RerankResult | List[dict],
    ]
    generate: Callable[
        [EvalQuestion, RouteHints, List[RetrievalHit], Dict[str, float]], Prediction
    ]


@dataclass(frozen=True)
class QuestionResult:
    """All per-question pipeline outputs (used by run_eval and the dashboard)."""

    prediction: Prediction
    hints: RouteHints
    retrieved: List[RetrievalHit]
    evidence_rows: List[RetrievalHit]
    support_priors: Dict[str, float]
    rerank_result: RerankResult


def run_question(
    pipeline: EvalPipeline,
    cfg: CastleRAGConfig,
    question: EvalQuestion,
    exclude_cameras: Sequence[str] = (),
    *,
    generate_prediction: bool = True,
) -> QuestionResult:
    """Run one question through route -> retrieve -> rerank -> generate.

    Returns the prediction together with the retrieval hits, reranked evidence,
    and support priors so callers (the eval loop and the UI ``RagEngine``) can
    both reuse the exact same per-question path.

    ``exclude_cameras`` hard-excludes those camera ids from dense retrieval (the
    UI refine loop passes the angles the reviewer rejected); empty on the eval
    path, so benchmark behaviour is unchanged.

    ``generate_prediction=False`` skips the MCQ answer generator entirely and
    returns a stub prediction. The free-form UI uses this: it answers open
    questions with its own choice-free generator, so running the multiple-choice
    generator here would only waste an LLM call and force a meaningless letter.
    """
    try:
        hints = pipeline.route(question.query, question.answers)
    except PipelineDependencyError as exc:
        raise _stage_dependency_error("routing", question.question_id, exc) from exc
    except NotImplementedError as exc:
        raise _stage_error("routing", question.question_id, exc) from exc
    except Exception as exc:
        raise _stage_failure_error("routing", question.question_id, exc) from exc

    if exclude_cameras:
        hints.exclude_cameras = tuple(exclude_cameras)

    try:
        retrieved = pipeline.retrieve(question, hints)
    except PipelineDependencyError as exc:
        raise _stage_dependency_error("retrieval", question.question_id, exc) from exc
    except NotImplementedError as exc:
        raise _stage_error("retrieval", question.question_id, exc) from exc
    except Exception as exc:
        raise _stage_failure_error("retrieval", question.question_id, exc) from exc

    candidate_packs = expand_candidates(
        retrieved,
        route=hints.route,
        max_candidate_videos=cfg.retrieval.max_candidate_videos,
        frames_per_candidate=cfg.retrieval.frames_per_candidate,
    )
    try:
        reranked = pipeline.rerank(question, hints, candidate_packs)
    except PipelineDependencyError as exc:
        raise _stage_dependency_error("reranking", question.question_id, exc) from exc
    except NotImplementedError as exc:
        raise _stage_error("reranking", question.question_id, exc) from exc
    except Exception as exc:
        raise _stage_failure_error("reranking", question.question_id, exc) from exc

    rerank_result = _coerce_rerank_result(reranked, hints.route)
    evidence_rows = _flatten_reranked_evidence(
        rerank_result,
        fallback_hits=retrieved,
        max_rows=cfg.retrieval.max_evidence_rows,
    )
    support_priors = _aggregate_support_priors(rerank_result)
    if not generate_prediction:
        prediction = Prediction(
            question_id=question.question_id,
            predicted_answer="a",  # placeholder; the caller supplies the real answer
            route=hints.route,
            support_priors=support_priors or None,
            top_evidence_ids=[hit.record_id for hit in evidence_rows],
            raw_answer_text="",
            is_supported=bool(evidence_rows)
            and max(support_priors.values(), default=0.0) > 0.0,
        )
    else:
        try:
            prediction = pipeline.generate(
                question, hints, evidence_rows, support_priors
            )
        except PipelineDependencyError as exc:
            raise _stage_dependency_error(
                "generation", question.question_id, exc
            ) from exc
        except NotImplementedError as exc:
            raise _stage_error("generation", question.question_id, exc) from exc
        except Exception as exc:
            raise _stage_failure_error(
                "generation", question.question_id, exc
            ) from exc

    return QuestionResult(
        prediction=prediction,
        hints=hints,
        retrieved=retrieved,
        evidence_rows=evidence_rows,
        support_priors=support_priors,
        rerank_result=rerank_result,
    )


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
    use_wandb: bool = False,
) -> EvalRunResult:
    """Run the full prediction loop and write output files.

    The runner writes rich predictions, evidence traces, official submission
    export, and metrics for every run. Accuracy is populated only when a
    ground-truth key is available; diversity, counts, and paths are always
    written to outputs.metrics.
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

    from castlerag.eval.wandb_logger import WandbLogger

    wb = (
        WandbLogger(cfg, n_questions=len(selected))
        if (use_wandb or cfg.wandb.enabled)
        else None
    )

    predictions: Dict[str, Prediction] = {}
    traces: List[dict] = []
    for question in selected.values():
        result = run_question(active_pipeline, cfg, question)
        predictions[question.question_id] = result.prediction
        traces.append(
            {
                "question_id": question.question_id,
                "route": result.hints.route,
                "retrieved_count": len(result.retrieved),
                "reranked_count": len(result.rerank_result.kept_packs),
                "top_evidence_ids": result.prediction.top_evidence_ids,
                "top_evidence_cameras": sorted(
                    {
                        h.camera_id
                        for h in result.evidence_rows
                        if h.camera_id is not None
                    }
                ),
                "support_priors": (
                    result.prediction.support_priors or result.support_priors
                ),
                "predicted_answer": result.prediction.predicted_answer,
                "is_supported": result.prediction.is_supported,
            }
        )
        if wb is not None:
            wb.log_question(question, result)

    write_predictions(predictions, outputs.predictions)
    write_evidence_traces(traces, outputs.evidence_traces)
    export_submission(predictions, outputs.submissions)

    diversity = compute_diversity_metrics(traces)

    # Compute accuracy from an explicit answer key, or fall back to embedded
    # ground_truth (populated by the CSV loader) when available.
    accuracy: Optional[float] = None
    n_correct: Optional[int] = None
    n_graded: Optional[int] = None
    has_gt = answers_path is not None or any(
        q.ground_truth is not None for q in selected.values()
    )
    if has_gt:
        n_correct, n_graded = accuracy_breakdown(selected, predictions, answers_path)
        accuracy = n_correct / n_graded if n_graded > 0 else 0.0

    outputs.metrics.parent.mkdir(parents=True, exist_ok=True)
    outputs.metrics.write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "num_questions": len(selected),
                "diversity": diversity,
                "predictions_path": str(outputs.predictions),
                "submission_path": str(outputs.submissions),
            },
            indent=2,
        )
    )

    if wb is not None:
        wb.log_summary(accuracy, diversity, len(selected))
        wb.log_artifacts(
            [outputs.predictions, outputs.submissions, outputs.evidence_traces]
        )
        wb.finish()

    return EvalRunResult(
        predictions=predictions,
        traces=traces,
        output_paths=outputs,
        accuracy=accuracy,
        diversity=diversity,
        n_correct=n_correct,
        n_graded=n_graded,
    )


def _resolve_output_paths(
    cfg: CastleRAGConfig,
    *,
    out_dir: Optional[Path],
    predictions_path: Optional[Path],
) -> EvalOutputPaths:
    """Resolve output file paths from config, applying any caller-supplied overrides."""
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
    """Construct the default EvalPipeline wired to BM25, Qdrant, and vLLM clients."""
    bm25_index, qdrant_client, artifact_report = _prepare_default_runtime(cfg)
    embed_client = OmniEmbedClient(
        model=cfg.embedding.model,
        backend=cfg.embedding.backend,
        vllm_base_url=_omniembed_base_url(cfg),
        vllm_tensor_parallel=cfg.embedding.vllm_tensor_parallel,
        vllm_gpu_memory_utilization=cfg.embedding.vllm_gpu_memory_utilization,
    )
    generation_client = _build_vllm_chat_client()

    def _retrieve(question: EvalQuestion, hints: RouteHints) -> List[RetrievalHit]:
        """Retrieve evidence hits for a question using dense and BM25 search."""
        try:
            return retrieve_evidence(
                question=question,
                hints=hints,
                qdrant_client=qdrant_client,
                collection_name=cfg.qdrant.collection,
                bm25_index=bm25_index,
                embed_client=embed_client,
                retrieval_cfg=cfg.retrieval,
            )
        except (
            ConnectionError,
            TimeoutError,
            OSError,
            ResponseHandlingException,
            UnexpectedResponse,
        ) as exc:  # pragma: no cover - guarded in run_eval tests
            raise PipelineDependencyError(
                _dependency_failure_message(
                    "retrieval",
                    f"dense retrieval/BM25 query failed: {exc}",
                    artifact_report,
                )
            ) from exc

    def _rerank(
        question: EvalQuestion,
        hints: RouteHints,
        candidate_packs: List[dict],
    ) -> RerankResult:
        """Rerank candidate evidence packs using the LLM reranker."""
        return rerank_candidates(
            question=question,
            hints=hints,
            candidate_packs=candidate_packs,
            llm_client=generation_client,
            top_k=cfg.reranking.top_k,
            min_relevance=cfg.reranking.min_relevance,
            model=cfg.reranking.model,
        )

    def _generate(
        question: EvalQuestion,
        hints: RouteHints,
        evidence_rows: List[RetrievalHit],
        support_priors: Dict[str, float],
    ) -> Prediction:
        """Generate a final answer prediction from evidence rows and support priors."""
        return generate_answer(
            question=question,
            hints=hints,
            evidence_rows=evidence_rows,
            support_priors=support_priors,
            llm_client=generation_client,
            model=cfg.generation.model,
            max_evidence_rows=cfg.retrieval.max_evidence_rows,
            shuffle_choices=cfg.generation.shuffle_choices,
        )

    return EvalPipeline(
        route=route_question,
        retrieve=_retrieve,
        rerank=_rerank,
        generate=_generate,
    )


def _flatten_reranked_evidence(
    reranked: RerankResult,
    *,
    fallback_hits: List[RetrievalHit],
    max_rows: int,
) -> List[RetrievalHit]:
    """Return deduplicated evidence hits from kept rerank packs, up to max_rows."""
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
    """Return a plain dict copy of the support priors from a RerankResult."""
    return dict(reranked.support_priors)


def _coerce_rerank_result(
    reranked: RerankResult | List[dict],
    route: str,
) -> RerankResult:
    """Coerce a raw list-of-dicts rerank output into a canonical RerankResult."""
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
    """Return a PipelineDependencyError for a stage that raised NotImplementedError."""
    return PipelineDependencyError(
        f"{stage} is not implemented for question {question_id}: {exc}"
    )


def _stage_dependency_error(
    stage: str,
    question_id: str,
    exc: PipelineDependencyError,
) -> PipelineDependencyError:
    """Wrap a dependency failure in a named stage as a PipelineDependencyError."""
    return PipelineDependencyError(
        f"{stage} dependency failed for question {question_id}: {exc}"
    )


def _stage_failure_error(
    stage: str,
    question_id: str,
    exc: Exception,
) -> PipelineDependencyError:
    """Return a PipelineDependencyError for an unexpected exception in a named stage."""
    return PipelineDependencyError(
        f"{stage} failed for question {question_id}: {exc}"
    )


def _vllm_base_url() -> Optional[str]:
    """Return the VLLM_BASE_URL environment variable value, or None if unset."""
    return os.getenv("VLLM_BASE_URL")


def _omniembed_base_url(cfg: Optional[CastleRAGConfig] = None) -> Optional[str]:
    """Return the base URL the OmniEmbed embeddings client should connect to.

    Resolution order (issue #54):
      1. ``OMNIEMBED_BASE_URL`` env var — explicit per-run override.
      2. ``cfg.embedding.base_url`` — declared endpoint for two-endpoint
         deployments (UI / eval / answer), so the embed client never depends on
         remembering an env var.
      3. ``VLLM_BASE_URL`` — back-compat fallback for single-endpoint / query-
         cache setups where embeddings and generation share one server.

    Without (1) or (2), a two-endpoint run would POST query embeddings to the
    generation server and 404 on any query-cache miss.
    """
    cfg_url = cfg.embedding.base_url if cfg is not None else None
    return os.getenv("OMNIEMBED_BASE_URL") or cfg_url or _vllm_base_url()


def _prepare_default_runtime(
    cfg: CastleRAGConfig,
) -> tuple[Any, Any, IndexArtifactReport]:
    """Load and validate index artifacts; return BM25, Qdrant client, and report."""
    artifact_report = _discover_index_artifacts(cfg)
    bm25_path = artifact_report.bm25_path
    if not bm25_path.exists():
        raise PipelineDependencyError(
            _dependency_failure_message(
                "indexing",
                f"BM25 transcript index not found at {bm25_path}",
                artifact_report,
            )
        )

    try:
        bm25_index = load_bm25_index(bm25_path)
    except Exception as exc:
        raise PipelineDependencyError(
            _dependency_failure_message(
                "indexing",
                f"failed to load BM25 transcript index at {bm25_path}: {exc}",
                artifact_report,
            )
        ) from exc
    windows = getattr(bm25_index, "windows", None)
    if not windows:
        raise PipelineDependencyError(
            _dependency_failure_message(
                "indexing",
                f"BM25 transcript index at {bm25_path} is empty",
                artifact_report,
            )
        )

    try:
        qdrant_client = get_client(cfg.qdrant.host, cfg.qdrant.port)
    except ImportError as exc:
        raise PipelineDependencyError(
            "retrieval dependency missing: qdrant-client is not installed."
        ) from exc

    _ensure_qdrant_collection_ready(qdrant_client, cfg)
    _ensure_vllm_runtime_ready(cfg)
    return bm25_index, qdrant_client, artifact_report


def _discover_index_artifacts(cfg: CastleRAGConfig) -> IndexArtifactReport:
    """Scan cache and chunks directories and return a report of found artifact paths."""
    cache_dir = Path(cfg.embedding.cache_dir)
    chunks_dir = Path(cfg.preprocessing.chunks_dir)
    return IndexArtifactReport(
        bm25_path=cache_dir / "transcripts.pkl",
        chunks_dir=chunks_dir,
        cache_dir=cache_dir,
        chunk_files={
            "transcripts": sorted(chunks_dir.rglob("transcripts.jsonl"))
            if chunks_dir.exists()
            else [],
            "clips": sorted(chunks_dir.rglob("clips.jsonl"))
            if chunks_dir.exists()
            else [],
            "events": sorted(chunks_dir.rglob("events.jsonl"))
            if chunks_dir.exists()
            else [],
            "aux": sorted(chunks_dir.rglob("aux.jsonl")) if chunks_dir.exists() else [],
        },
        embedding_caches={
            "transcripts": sorted(cache_dir.glob("transcripts*.npz"))
            if cache_dir.exists()
            else [],
            "events": (
                sorted(cache_dir.glob("events*.npz")) if cache_dir.exists() else []
            ),
            "clips": (
                sorted(cache_dir.glob("clips*.npz")) if cache_dir.exists() else []
            ),
            "aux_text": sorted(cache_dir.glob("aux_text*.npz"))
            if cache_dir.exists()
            else [],
            "aux_image": sorted(cache_dir.glob("aux_image*.npz"))
            if cache_dir.exists()
            else [],
            "aux_video": sorted(cache_dir.glob("aux_video*.npz"))
            if cache_dir.exists()
            else [],
        },
    )


def _ensure_qdrant_collection_ready(client: Any, cfg: CastleRAGConfig) -> None:
    """Raise PipelineDependencyError if the Qdrant collection is missing or empty."""
    collection_name = cfg.qdrant.collection
    host = cfg.qdrant.host
    port = cfg.qdrant.port
    try:
        exists = _qdrant_collection_exists(client, collection_name)
    except Exception as exc:
        raise PipelineDependencyError(
            "retrieval dependency missing: could not reach Qdrant at "
            f"{host}:{port} while checking collection {collection_name!r}: {exc}"
        ) from exc
    if not exists:
        raise PipelineDependencyError(
            "retrieval dependency missing: Qdrant collection "
            f"{collection_name!r} does not exist on {host}:{port}. "
            "Run `castlerag index` to create and populate it."
        )

    try:
        point_count = _qdrant_collection_count(client, collection_name)
    except Exception as exc:
        raise PipelineDependencyError(
            "retrieval dependency missing: could not inspect Qdrant collection "
            f"{collection_name!r} on {host}:{port}: {exc}"
        ) from exc
    if point_count == 0:
        raise PipelineDependencyError(
            "retrieval dependency missing: Qdrant collection "
            f"{collection_name!r} on {host}:{port} is empty. "
            "Run `castlerag index` to upsert dense points."
        )


def _qdrant_collection_exists(client: Any, collection_name: str) -> bool:
    """Return True if the named collection exists in the Qdrant client."""
    if hasattr(client, "collection_exists"):
        return bool(client.collection_exists(collection_name))
    if hasattr(client, "get_collection"):
        try:
            client.get_collection(collection_name)
            return True
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" in msg or "does not exist" in msg:
                return False
            raise
    return True


def _qdrant_collection_count(client: Any, collection_name: str) -> Optional[int]:
    """Return the approximate point count for a Qdrant collection, or None on error."""
    if not hasattr(client, "count"):
        return None
    response = client.count(collection_name=collection_name, exact=False)
    if isinstance(response, int):
        return response
    count = getattr(response, "count", None)
    if isinstance(count, int):
        return count
    return None


def _ensure_vllm_runtime_ready(cfg: CastleRAGConfig) -> None:
    """Raise PipelineDependencyError if VLLM_BASE_URL is unset or openai is missing."""
    needed_stages: List[str] = ["reranking", "generation"]
    if cfg.embedding.backend == "vllm":
        needed_stages.insert(0, "retrieval")
    base_url = _vllm_base_url()
    if not base_url:
        raise PipelineDependencyError(
            "runtime dependency missing: VLLM_BASE_URL is not set. "
            "The following stages require a local vLLM endpoint: "
            f"{', '.join(needed_stages)}."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise PipelineDependencyError(
            "runtime dependency missing: openai package is required for the "
            f"vLLM-backed {', '.join(needed_stages)} stages."
        ) from exc
    OpenAI(base_url=base_url, api_key="not-needed")


def _dependency_failure_message(
    stage: str,
    detail: str,
    artifact_report: IndexArtifactReport,
) -> str:
    """Build a diagnostic error message with artifact counts for a failing stage."""
    chunk_counts = ", ".join(
        f"{name}={len(paths)}" for name, paths in artifact_report.chunk_files.items()
    )
    cache_counts = ", ".join(
        f"{name}={len(paths)}"
        for name, paths in artifact_report.embedding_caches.items()
    )
    return (
        f"{stage} dependency missing: {detail}. "
        f"Local chunk artifacts under {artifact_report.chunks_dir}: {chunk_counts}. "
        f"Local embedding caches under {artifact_report.cache_dir}: {cache_counts}. "
        "Run `castlerag preprocess` and `castlerag index` against a small "
        "egocentric subset first."
    )


def _build_vllm_chat_client() -> Any:
    """Instantiate and return an OpenAI client pointed at the local vLLM endpoint."""
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
