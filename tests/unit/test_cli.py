"""Tests for src/castlerag/cli.py"""

import json
from pathlib import Path

from typer.testing import CliRunner

from castlerag.cli import app

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


def test_answer_not_yet_implemented(tmp_path: Path):
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
    result = runner.invoke(app, ["answer", str(q_file)])
    assert result.exit_code != 0


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


def test_smoke_test_not_yet_implemented(tmp_path: Path):
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
    result = runner.invoke(app, ["smoke-test", str(q_file)])
    assert result.exit_code != 0
