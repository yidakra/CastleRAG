"""Evaluation helpers and orchestration for CastleRAG."""

from castlerag.eval.io import (
    compute_accuracy,
    export_submission,
    load_predictions,
    load_questions,
    select_questions,
    write_evidence_traces,
    write_predictions,
)
from castlerag.eval.run_eval import (
    EvalOutputPaths,
    EvalPipeline,
    EvalRunResult,
    PipelineDependencyError,
    run_eval,
)

__all__ = [
    "EvalOutputPaths",
    "EvalPipeline",
    "EvalRunResult",
    "PipelineDependencyError",
    "compute_accuracy",
    "export_submission",
    "load_predictions",
    "load_questions",
    "run_eval",
    "select_questions",
    "write_evidence_traces",
    "write_predictions",
]
