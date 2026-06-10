"""Typer CLI entrypoint for CastleRAG.

Commands: preprocess, embed, index, retrieve, answer, eval, smoke-test
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from castlerag.config import CastleRAGConfig, load_config
from castlerag.embed.omniembed import OmniEmbedClient
from castlerag.eval import PipelineDependencyError, load_questions, run_eval
from castlerag.index import get_client, load_bm25_index
from castlerag.index.pipeline import (
    build_bm25_artifact,
    build_qdrant_index,
    cache_dense_embeddings,
    filter_records,
    load_chunk_records,
)
from castlerag.retrieval.search import retrieve as retrieve_evidence
from castlerag.routing.question_router import route_question
from castlerag.schemas import EvalQuestion

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


def _count_records(records: object) -> int:
    return sum(
        len(getattr(records, field))
        for field in ("transcripts", "clips", "events", "aux")
    )


def _vllm_base_url() -> Optional[str]:
    return os.getenv("VLLM_BASE_URL")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def preprocess(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Override config YAML"
    ),
    snellius: bool = typer.Option(
        False, "--snellius", help="Apply configs/snellius.yaml overlay"
    ),
    day: Optional[int] = typer.Option(
        None, "--day", min=1, max=4, help="Process a single day (1-4)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print actions without writing files"
    ),
) -> None:
    """Discover CASTLE files and build normalized preprocessing artifacts."""
    cfg = _resolve_config(config, snellius)
    days = [day] if day is not None else cfg.dataset.days
    console.print(f"[bold]castlerag preprocess[/bold]  days={days}  dry_run={dry_run}")
    console.print(f"  dataset root  : {cfg.dataset.root}")
    console.print(
        f"  camera scope  : {cfg.dataset.camera_scope}  "
        f"({len(cfg.dataset.ego_cameras)} ego cameras)"
    )
    console.print(
        f"  clip / stride : {cfg.preprocessing.clip_seconds}s / "
        f"{cfg.preprocessing.stride_seconds}s  @ {cfg.preprocessing.fps} fps"
    )
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
        None,
        "--modality",
        help="Filter by modality: transcript | event_summary | text | image | video",
    ),
    day: Optional[int] = typer.Option(None, "--day", min=1, max=4),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Encode derived chunks with OmniEmbed (vLLM) and cache embeddings to disk."""
    cfg = _resolve_config(config, snellius)
    console.print(
        f"[bold]castlerag embed[/bold]  modality={modality or 'all'}  "
        f"day={day if day is not None else 'all'}"
    )
    console.print(f"  model   : {cfg.embedding.model}")
    console.print(f"  backend : {cfg.embedding.backend}")
    if dry_run:
        console.print("[yellow]dry-run: no embeddings written[/yellow]")
        return
    records = load_chunk_records(Path(cfg.preprocessing.chunks_dir))
    scoped = filter_records(records, cfg, day=day)
    if _count_records(scoped) == 0:
        console.print("[red]No chunk records found — run preprocess first.[/red]")
        raise typer.Exit(1)
    embed_client = OmniEmbedClient(
        model=cfg.embedding.model,
        backend=cfg.embedding.backend,
        vllm_base_url=_vllm_base_url(),
        vllm_tensor_parallel=cfg.embedding.vllm_tensor_parallel,
        vllm_gpu_memory_utilization=cfg.embedding.vllm_gpu_memory_utilization,
    )
    paths = cache_dense_embeddings(
        records, cfg, embed_client, modality=modality, day=day
    )
    console.print(f"  caches  : {len(paths)} written under {cfg.embedding.cache_dir}")
    if embed_client.dim is not None:
        console.print(f"  dim     : {embed_client.dim}")


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
    records = load_chunk_records(Path(cfg.preprocessing.chunks_dir))
    scoped = filter_records(records, cfg)
    if _count_records(scoped) == 0:
        console.print("[red]No chunk records found — run preprocess first.[/red]")
        raise typer.Exit(1)
    cache_dir = Path(cfg.embedding.cache_dir)
    if not any(cache_dir.glob("*.npz")):
        embed_client = OmniEmbedClient(
            model=cfg.embedding.model,
            backend=cfg.embedding.backend,
            vllm_base_url=_vllm_base_url(),
            vllm_tensor_parallel=cfg.embedding.vllm_tensor_parallel,
            vllm_gpu_memory_utilization=cfg.embedding.vllm_gpu_memory_utilization,
        )
        cache_dense_embeddings(records, cfg, embed_client)
    bm25_path = build_bm25_artifact(scoped, Path(cfg.embedding.cache_dir))
    vector_size, cache_paths = build_qdrant_index(
        cfg,
        scoped,
        recreate=create_collection,
    )
    console.print(f"  BM25    : {bm25_path}")
    console.print(f"  dense   : {len(cache_paths)} cache bundles upserted")
    console.print(f"  dim     : {vector_size}")


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
    bm25_path = Path(cfg.embedding.cache_dir) / "transcripts.pkl"
    if not bm25_path.exists():
        console.print(
            "[red]BM25 transcript index not found — run `castlerag index` first.[/red]"
        )
        raise typer.Exit(1)
    hints = route_question(
        question=question,
        choices={"a": choice_a, "b": choice_b, "c": choice_c, "d": choice_d},
    )
    eval_question = EvalQuestion(
        question_id="adhoc",
        query=question,
        answers={"a": choice_a, "b": choice_b, "c": choice_c, "d": choice_d},
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
    hits = retrieve_evidence(
        question=eval_question,
        hints=hints,
        qdrant_client=qdrant_client,
        collection_name=cfg.qdrant.collection,
        bm25_index=bm25_index,
        embed_client=embed_client,
        retrieval_cfg=cfg.retrieval,
    )
    console.print(f"  route    : {hints.route}")
    console.print(f"  evidence : {len(hits)} hits")
    for hit in hits[:10]:
        console.print(
            f"  [{hit.rank}] {hit.source_type} {hit.record_id} "
            f"score={hit.score:.4f} day={hit.day} camera={hit.camera_id}"
        )


@app.command()
def answer(
    questions_path: Path = typer.Argument(..., help="Official CASTLE questions JSON"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    snellius: bool = typer.Option(False, "--snellius"),
    question_id: Optional[str] = typer.Option(
        None, "--id", help="Run only this question id"
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="Output predictions path"),
) -> None:
    """Run full retrieve → rerank → generate pipeline on CASTLE questions."""
    cfg = _resolve_config(config, snellius)
    out_path = out or Path(cfg.outputs.predictions)
    override = _default_snellius_path() if snellius else config
    console.print("[bold]castlerag answer[/bold]")
    console.print(f"  questions : {questions_path}")
    console.print(f"  model     : {cfg.generation.model}")
    console.print(f"  output    : {out_path}")
    questions = load_questions(questions_path)
    try:
        result = run_eval(
            questions,
            config_path=override,
            question_ids=[question_id] if question_id else None,
            predictions_path=out_path,
        )
    except (PipelineDependencyError, NotImplementedError, ValueError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"  predicted : {len(result.predictions)} questions")
    console.print(f"  traces    : {result.output_paths.evidence_traces}")
    console.print(f"  submit    : {result.output_paths.submissions}")


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
    from castlerag.eval.io import compute_accuracy, export_submission, load_predictions

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
        console.print(
            "[yellow]No answer key provided — exporting submission only.[/yellow]"
        )

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
    questions = load_questions(questions_path)
    out_dir = Path(cfg.outputs.dir) / "smoke_test"
    try:
        result = run_eval(
            questions,
            config_path=config,
            out_dir=out_dir,
            max_questions=n,
        )
    except (PipelineDependencyError, NotImplementedError, ValueError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"  predicted : {len(result.predictions)} questions")
    console.print(f"  output    : {result.output_paths.predictions}")


if __name__ == "__main__":
    app()
