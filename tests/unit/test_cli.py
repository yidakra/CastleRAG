"""Tests for src/castlerag/cli.py"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from castlerag.cli import app
from castlerag.eval.run_eval import (
    EvalOutputPaths,
    EvalRunResult,
    PipelineDependencyError,
)
from castlerag.schemas import Prediction

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "castlerag" in result.output.lower()


def test_preprocess_dry_run():
    result = runner.invoke(app, ["preprocess", "--dry-run"])
    assert result.exit_code == 0
    assert "dry-run" in result.output


def test_preprocess_dry_run_shows_config():
    result = runner.invoke(app, ["preprocess", "--dry-run"])
    assert result.exit_code == 0
    # Should show clip size, stride, scope
    assert "30" in result.output
    assert "ego" in result.output


def test_preprocess_not_yet_implemented():
    result = runner.invoke(app, ["preprocess"])
    assert result.exit_code != 0


def test_embed_dry_run():
    result = runner.invoke(app, ["embed", "--dry-run"])
    assert result.exit_code == 0
    assert "dry-run" in result.output


def test_embed_shows_model():
    result = runner.invoke(app, ["embed", "--dry-run"])
    assert "OmniEmbed" in result.output or "omniembed" in result.output.lower()


def test_embed_not_yet_implemented():
    result = runner.invoke(app, ["embed"])
    assert result.exit_code != 0


def test_index_dry_run():
    result = runner.invoke(app, ["index", "--dry-run"])
    assert result.exit_code == 0
    assert "dry-run" in result.output


def test_index_not_yet_implemented():
    result = runner.invoke(app, ["index"])
    assert result.exit_code != 0


def test_answer_runs_eval_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    q_file = tmp_path / "questions.json"
    q_file.write_text(
        json.dumps(
            {
                "q1": {
                    "query": "What did Allie do?",
                    "answers": {"a": "A", "b": "B", "c": "C", "d": "D"},
                }
            }
        )
    )
    out_path = tmp_path / "predictions.json"

    def _fake_run_eval(questions, **kwargs):
        assert list(questions) == ["q1"]
        assert kwargs["predictions_path"] == out_path
        return EvalRunResult(
            predictions={"q1": Prediction(question_id="q1", predicted_answer="a")},
            traces=[],
            output_paths=EvalOutputPaths(
                predictions=out_path,
                evidence_traces=tmp_path / "evidence_traces.jsonl",
                submissions=tmp_path / "submissions.json",
                metrics=tmp_path / "metrics.json",
            ),
        )

    monkeypatch.setattr("castlerag.cli.run_eval", _fake_run_eval)
    result = runner.invoke(app, ["answer", str(q_file), "--out", str(out_path)])
    assert result.exit_code == 0
    assert "predicted : 1 questions" in result.output


def test_eval_with_predictions(tmp_path: Path):
    q_file = tmp_path / "questions.json"
    q_file.write_text(
        json.dumps(
            {
                "q1": {
                    "query": "Test question?",
                    "answers": {"a": "A", "b": "B", "c": "C", "d": "D"},
                }
            }
        )
    )
    pred_file = tmp_path / "predictions.json"
    pred_file.write_text(json.dumps({"q1": "b"}))

    result = runner.invoke(
        app,
        [
            "eval",
            str(q_file),
            str(pred_file),
            "--config",
            str(Path(__file__).parent.parent.parent / "configs" / "base.yaml"),
        ],
    )
    # Should succeed (no answer key, so just writes submission)
    assert result.exit_code == 0
    # Submission should have been written
    assert (
        tmp_path / "submissions.json"
    ).exists() or "submission" in result.output.lower()


def test_smoke_test_runs_first_five_questions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    q_file = tmp_path / "questions.json"
    q_file.write_text(
        json.dumps(
            {
                f"q{i}": {
                    "query": f"Test {i}?",
                    "answers": {"a": "A", "b": "B", "c": "C", "d": "D"},
                }
                for i in range(1, 7)
            }
        )
    )

    def _fake_run_eval(questions, **kwargs):
        assert kwargs["max_questions"] == 5
        assert kwargs["out_dir"] == Path("outputs") / "smoke_test"
        return EvalRunResult(
            predictions={
                f"q{i}": Prediction(question_id=f"q{i}", predicted_answer="a")
                for i in range(1, 6)
            },
            traces=[],
            output_paths=EvalOutputPaths(
                predictions=Path("outputs") / "smoke_test" / "predictions.json",
                evidence_traces=Path("outputs")
                / "smoke_test"
                / "evidence_traces.jsonl",
                submissions=Path("outputs") / "smoke_test" / "submissions.json",
                metrics=Path("outputs") / "smoke_test" / "metrics.json",
            ),
        )

    monkeypatch.setattr("castlerag.cli.run_eval", _fake_run_eval)
    result = runner.invoke(app, ["smoke-test", str(q_file)])
    assert result.exit_code == 0
    assert "predicted : 5 questions" in result.output


def test_smoke_test_surfaces_pipeline_dependency_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    q_file = tmp_path / "questions.json"
    q_file.write_text(
        json.dumps(
            {
                "q1": {
                    "query": "Test?",
                    "answers": {"a": "A", "b": "B", "c": "C", "d": "D"},
                }
            }
        )
    )

    def _fake_run_eval(questions, **kwargs):
        raise PipelineDependencyError("generation is not implemented for question q1")

    monkeypatch.setattr("castlerag.cli.run_eval", _fake_run_eval)
    result = runner.invoke(app, ["smoke-test", str(q_file)])
    assert result.exit_code == 1
    assert "generation is not implemented for question q1" in result.output
