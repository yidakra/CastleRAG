"""Typer CLI entrypoint for CastleRAG.

Commands: preprocess, embed, index, retrieve, answer, eval, smoke-test
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from castlerag.config import CastleRAGConfig, load_config

app = typer.Typer(
    name="castlerag",
    help="CastleRAG: multimodal RAG for the CASTLE 2024 challenge (EgoVis 2026)",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_snellius_path() -> Path:
    """Resolve snellius.yaml — package-installed copy first, then project root."""
    pkg_relative = Path(__file__).parent / "configs" / "snellius.yaml"
    if pkg_relative.exists():
        return pkg_relative
    return Path(__file__).parent.parent.parent / "configs" / "snellius.yaml"


def _resolve_config(config: Optional[Path], snellius: bool) -> CastleRAGConfig:
    override: Optional[Path] = None
    if snellius:
        override = _default_snellius_path()
    elif config is not None:
        override = config
    return load_config(override_path=override)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def preprocess(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Override config YAML"),
    snellius: bool = typer.Option(False, "--snellius", help="Apply configs/snellius.yaml overlay"),
    day: Optional[int] = typer.Option(None, "--day", min=1, max=4, help="Process a single day (1-4)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print actions without writing files"),
) -> None:
    """Discover CASTLE files, create 30-second clips, extract 1 fps frames, normalize transcripts."""
    cfg = _resolve_config(config, snellius)
    days = [day] if day is not None else cfg.dataset.days
    console.print(f"[bold]castlerag preprocess[/bold]  days={days}  dry_run={dry_run}")
    console.print(f"  dataset root  : {cfg.dataset.root}")
    console.print(f"  camera scope  : {cfg.dataset.camera_scope}  "
                  f"({len(cfg.dataset.ego_cameras)} ego cameras)")
    console.print(f"  clip / stride : {cfg.preprocessing.clip_seconds}s / "
                  f"{cfg.preprocessing.stride_seconds}s  @ {cfg.preprocessing.fps} fps")
    if dry_run:
        console.print("[yellow]dry-run: no files written[/yellow]")
        return
    console.print("[red]preprocess not yet implemented — see issues #3, #4[/red]")
    raise typer.Exit(1)


@app.command()
def embed(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    snellius: bool = typer.Option(False, "--snellius"),
    modality: Optional[str] = typer.Option(
        None, "--modality",
        help="Filter by modality: transcript | video | image | event_summary",
    ),
    day: Optional[int] = typer.Option(None, "--day", min=1, max=4),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Encode derived chunks with OmniEmbed (vLLM) and cache embeddings to disk."""
    cfg = _resolve_config(config, snellius)
    console.print(f"[bold]castlerag embed[/bold]  modality={modality or 'all'}  "
                  f"day={day if day is not None else 'all'}")
    console.print(f"  model   : {cfg.embedding.model}")
    console.print(f"  backend : {cfg.embedding.backend}")
    if dry_run:
        console.print("[yellow]dry-run: no embeddings written[/yellow]")
        return
    console.print("[red]embed not yet implemented — see issue #5[/red]")
    raise typer.Exit(1)


@app.command()
def index(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    snellius: bool = typer.Option(False, "--snellius"),
    create_collection: bool = typer.Option(
        False, "--create-collection", help="(Re)create the Qdrant collection"
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Create Qdrant collection + payload indexes and upsert all evidence points."""
    cfg = _resolve_config(config, snellius)
    console.print(f"[bold]castlerag index[/bold]  collection={cfg.qdrant.collection}")
    console.print(f"  Qdrant : {cfg.qdrant.host}:{cfg.qdrant.port}")
    if dry_run:
        console.print("[yellow]dry-run: no Qdrant writes[/yellow]")
        return
    console.print("[red]index not yet implemented — see issue #6[/red]")
    raise typer.Exit(1)


@app.command()
def retrieve(
    question: str = typer.Argument(..., help="Question text"),
    choice_a: str = typer.Option(..., "--a", help="Choice A text"),
    choice_b: str = typer.Option(..., "--b", help="Choice B text"),
    choice_c: str = typer.Option(..., "--c", help="Choice C text"),
    choice_d: str = typer.Option(..., "--d", help="Choice D text"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    snellius: bool = typer.Option(False, "--snellius"),
) -> None:
    """Retrieve route-aware evidence for a single question (debug / inspection)."""
    cfg = _resolve_config(config, snellius)
    console.print("[bold]castlerag retrieve[/bold]")
    console.print(f"  question : {question[:80]}")
    console.print("[red]retrieval not yet implemented — see issue #7[/red]")
    raise typer.Exit(1)


@app.command()
def answer(
    questions_path: Path = typer.Argument(..., help="Official CASTLE questions JSON"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    snellius: bool = typer.Option(False, "--snellius"),
    question_id: Optional[str] = typer.Option(None, "--id", help="Run only this question id"),
    out: Optional[Path] = typer.Option(None, "--out", help="Output predictions path"),
) -> None:
    """Run full retrieve → rerank → generate pipeline on CASTLE questions."""
    cfg = _resolve_config(config, snellius)
    out_path = out or Path(cfg.outputs.predictions)
    console.print("[bold]castlerag answer[/bold]")
    console.print(f"  questions : {questions_path}")
    console.print(f"  model     : {cfg.generation.model}")
    console.print(f"  output    : {out_path}")
    console.print("[red]answer pipeline not yet implemented — see issues #8–#14[/red]")
    raise typer.Exit(1)


@app.command(name="eval")
def eval_cmd(
    questions_path: Path = typer.Argument(..., help="Official CASTLE questions JSON"),
    predictions_path: Path = typer.Argument(..., help="Predictions JSON"),
    answers_path: Optional[Path] = typer.Option(
        None, "--answers", help="Ground-truth answer key JSON"
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Evaluate predictions against ground truth and export submission JSON."""
    from castlerag.eval.io import (
        compute_accuracy,
        export_submission,
        load_predictions,
        load_questions,
    )

    cfg = _resolve_config(config, False)
    questions = load_questions(questions_path)
    predictions = load_predictions(predictions_path)
    console.print(
        f"[bold]castlerag eval[/bold]  "
        f"{len(questions)} questions, {len(predictions)} predictions"
    )

    if answers_path:
        accuracy = compute_accuracy(questions, predictions, answers_path)
        n_correct = int(round(accuracy * len(questions)))
        console.print(
            f"  accuracy : [green]{accuracy:.4f}[/green]  "
            f"({n_correct}/{len(questions)})"
        )
    else:
        console.print("[yellow]No answer key provided — exporting submission only.[/yellow]")

    sub_path = Path(cfg.outputs.submissions)
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    export_submission(predictions, sub_path)
    console.print(f"  submission written to [cyan]{sub_path}[/cyan]")


@app.command(name="smoke-test")
def smoke_test(
    questions_path: Path = typer.Argument(..., help="CASTLE questions JSON"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    n: int = typer.Option(5, "--n", help="Number of questions to run (default 5)"),
) -> None:
    """5-question end-to-end smoke test (issue #15)."""
    cfg = _resolve_config(config, False)
    console.print(f"[bold]castlerag smoke-test[/bold]  n={n}")
    console.print("[red]smoke-test not yet implemented — depends on issues #3–#14[/red]")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
