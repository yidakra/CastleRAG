"""Optional Weights & Biases logging for CastleRAG eval runs.

Importing this module is always safe — if ``wandb`` is not installed the
logger is a no-op and nothing is logged.  Enable logging by setting
``cfg.wandb.enabled = true`` in config or passing ``use_wandb=True`` to
``run_eval``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

log = logging.getLogger("castlerag.eval.wandb")

if TYPE_CHECKING:
    from castlerag.config import CastleRAGConfig
    from castlerag.eval.run_eval import QuestionResult
    from castlerag.schemas import EvalQuestion


def _wandb() -> Any:
    try:
        import wandb  # type: ignore[import]

        return wandb
    except ImportError:
        return None


def _stringify_keys(value: Any) -> Any:
    """Recursively coerce dict keys to ``str`` so wandb's summary encoder is safe.

    wandb builds dotted summary paths by concatenating nested keys as strings; a
    dict with int keys (e.g. the camera-count histogram) raises ``TypeError`` mid
    encode. Lists/tuples are walked too; scalars pass through unchanged.
    """
    if isinstance(value, dict):
        return {str(k): _stringify_keys(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_keys(v) for v in value]
    return value


def _flat_config(cfg: "CastleRAGConfig", n_questions: int) -> Dict[str, Any]:
    """Flatten the parts of cfg that are meaningful hyperparameters for a run."""
    return {
        "model/generation": cfg.generation.model,
        "model/reranker": cfg.reranking.model,
        "retrieval/rrf_k": cfg.retrieval.rrf_k,
        "retrieval/transcript_top_k": cfg.retrieval.transcript_top_k,
        "retrieval/max_evidence_rows": cfg.retrieval.max_evidence_rows,
        "reranking/top_k": cfg.reranking.top_k,
        "reranking/min_relevance": cfg.reranking.min_relevance,
        "reranking/relevance_weight": cfg.reranking.relevance_weight,
        "ui/score_mode": cfg.ui.score_mode,
        "n_questions": n_questions,
    }


class WandbLogger:
    """Thin wrapper around wandb for CastleRAG eval runs.

    All public methods are no-ops when wandb is not installed or the run
    failed to initialise.
    """

    def __init__(
        self,
        cfg: "CastleRAGConfig",
        n_questions: int,
        run_name: Optional[str] = None,
    ) -> None:
        w = _wandb()
        if w is None:
            self._active = False
            return

        entity = cfg.wandb.entity or None
        name = run_name or cfg.wandb.run_name or None
        try:
            self._run = w.init(
                project=cfg.wandb.project,
                entity=entity,
                name=name,
                config=_flat_config(cfg, n_questions),
                reinit="finish_previous",
            )
            self._table = w.Table(
                columns=[
                    "question_id",
                    "query",
                    "predicted",
                    "ground_truth",
                    "correct",
                    "route",
                    "n_evidence_cameras",
                ]
            )
            self._active = True
            self._n_correct = 0
            self._n_graded = 0
        except Exception as exc:  # pragma: no cover
            # Surface the cause instead of silently disabling — a swallowed
            # init error (bad entity, offline node, auth) is otherwise invisible
            # and looks like "wandb just didn't log".
            log.warning("wandb init failed (%s): %s", type(exc).__name__, exc)
            self._active = False

    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """True only when a wandb run was successfully initialised."""
        return self._active

    def log_question(
        self,
        question: "EvalQuestion",
        result: "QuestionResult",
    ) -> None:
        if not self._active:
            return
        import wandb  # type: ignore[import]

        gt = question.ground_truth
        predicted = result.prediction.predicted_answer
        is_correct = (predicted == gt) if gt is not None else None
        n_cameras = len(
            {h.camera_id for h in result.evidence_rows if h.camera_id is not None}
        )
        self._table.add_data(
            question.question_id,
            question.query,
            predicted,
            gt,
            is_correct,
            result.hints.route,
            n_cameras,
        )
        log_dict: Dict[str, Any] = {
            "question/route": result.hints.route,
            "question/n_evidence_cameras": n_cameras,
            "question/n_retrieved": len(result.retrieved),
        }
        if is_correct is not None:
            self._n_graded += 1
            if is_correct:
                self._n_correct += 1
            log_dict["accuracy/running"] = self._n_correct / self._n_graded
        wandb.log(log_dict)

    def log_summary(
        self,
        accuracy: Optional[float],
        diversity: Optional[Dict[str, Any]],
        n_questions: int,
        support_split: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._active:
            return
        import wandb  # type: ignore[import]

        wandb.log({"predictions": self._table})
        summary: Dict[str, Any] = {"n_questions": n_questions}
        if accuracy is not None:
            summary["accuracy"] = accuracy
        for k, v in (diversity or {}).items():
            summary[f"diversity/{k}"] = v
        if support_split:
            # Flatten the evidence-backed vs guessed split into scalar summary
            # keys so the guess rate and per-subset accuracy are first-class
            # metrics in the run (not buried in an artifact).
            if support_split.get("unsupported_rate") is not None:
                summary["support/unsupported_rate"] = support_split["unsupported_rate"]
            if support_split.get("num_unsupported") is not None:
                summary["support/num_unsupported"] = support_split["num_unsupported"]
            sup = support_split.get("supported") or {}
            guess = support_split.get("unsupported") or {}
            if sup.get("accuracy") is not None:
                summary["support/supported_accuracy"] = sup["accuracy"]
            if guess.get("accuracy") is not None:
                summary["support/guessed_accuracy"] = guess["accuracy"]
        for k, v in summary.items():
            # wandb's summary encoder builds dotted paths and concatenates each
            # nested key onto the path as a string, so a nested dict with
            # non-string keys (e.g. diversity's int-keyed camera_count_distribution)
            # crashes it. Coerce keys to str before handing the value over.
            wandb.run.summary[k] = _stringify_keys(v)  # type: ignore[union-attr]

    def log_artifacts(self, paths: List[Path]) -> None:
        if not self._active:
            return
        import wandb  # type: ignore[import]

        artifact = wandb.Artifact("eval_outputs", type="eval")
        for p in paths:
            if p.exists():
                artifact.add_file(str(p))
        wandb.run.log_artifact(artifact)  # type: ignore[union-attr]

    def finish(self) -> None:
        if not self._active:
            return
        import wandb  # type: ignore[import]

        wandb.finish()
        self._active = False
