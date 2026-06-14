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
    """Load and return the runtime config, applying any CLI override path."""
    override: Optional[Path] = None
    if snellius:
        override = _default_snellius_path()
    elif config is not None:
        override = config
    return load_config(override_path=override)


def _count_records(records: object) -> int:
    """Return the total number of records across all record-list fields."""
    return sum(
        len(getattr(records, field))
        for field in ("transcripts", "clips", "events", "aux")
    )


def _vllm_base_url() -> Optional[str]:
    """Return the VLLM_BASE_URL environment variable, or None if unset."""
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
    caption: bool = typer.Option(
        False,
        "--caption",
        help="Run per-clip caption + OCR (requires VLLM_BASE_URL)",
    ),
    events: bool = typer.Option(
        False,
        "--events",
        help="Compress clip groups into event summaries (requires VLLM_BASE_URL)",
    ),
    aux: bool = typer.Option(
        False,
        "--aux",
        help="Normalize auxiliary modalities (photo, thermal, video)",
    ),
    skip_base: bool = typer.Option(
        False,
        "--skip-base",
        help="Skip the base windowing/frame-extraction/transcript pass. "
        "Useful when rerunning --caption/--events over already-extracted artifacts.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print actions without writing files"
    ),
) -> None:
    """Discover CASTLE files and build normalized preprocessing artifacts.

    Base phase (default): windows, 1 fps frames, transcript normalization.
    --skip-base : skip the base pass entirely (assumes chunks already exist).
    --caption   : annotate each clip with visual caption + OCR  (GPU / vLLM)
    --events    : compress 4-clip groups into event summaries   (GPU / vLLM)
    --aux       : normalize photo, thermal, auxiliary video     (CPU)
    """
    from castlerag.dataset.layout import discover_hours
    from castlerag.dataset.transcripts import load_raw_segments, merge_into_windows
    from castlerag.index.io import write_jsonl_records
    from castlerag.preprocess.media import extract_frames_1fps, get_video_duration
    from castlerag.preprocess.windows import iter_windows, mark_placeholder_windows
    from castlerag.schemas import ClipRecord

    cfg = _resolve_config(config, snellius)
    days_list = [day] if day is not None else cfg.dataset.days
    console.print(
        f"[bold]castlerag preprocess[/bold]  days={days_list}  "
        f"caption={caption}  events={events}  aux={aux}  dry_run={dry_run}"
    )
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

    if (caption or events) and not _vllm_base_url():
        console.print(
            "[red]--caption and --events require VLLM_BASE_URL to be set.[/red]"
        )
        raise typer.Exit(1)

    root = Path(cfg.dataset.root)
    chunks_dir = Path(cfg.preprocessing.chunks_dir)
    frames_dir = Path(cfg.preprocessing.frames_dir)

    # ------------------------------------------------------------------ #
    # Base phase: windowing + frames + transcript normalization            #
    # ------------------------------------------------------------------ #
    n_clips = n_windows = 0
    if skip_base:
        console.print(
            "  base          : skipped (--skip-base); reusing existing chunks"
        )
        # Per-asset skip happens inside the loop below; with skip_base we bypass
        # the loop entirely so caption/events run directly over existing files.
    for asset in [] if skip_base else discover_hours(
        root=root,
        ego_cameras=cfg.dataset.ego_cameras,
        exo_cameras=cfg.dataset.exo_cameras,
        days=days_list,
        hours=cfg.dataset.hours,
        camera_scope=cfg.dataset.camera_scope,
    ):
        if asset.missing_video:
            continue

        out_dir = chunks_dir / asset.day / asset.camera_id / f"{asset.hour:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            duration = get_video_duration(asset.video_path)
        except Exception as exc:
            console.print(
                f"[yellow]  skip {asset.day}/{asset.camera_id}/{asset.hour:02d}"
                f" — ffprobe failed: {exc}[/yellow]"
            )
            continue

        windows = list(
            iter_windows(
                video_path=asset.video_path,
                camera_id=asset.camera_id,
                day=asset.day,
                hour=asset.hour,
                duration_seconds=duration,
                clip_seconds=cfg.preprocessing.clip_seconds,
                stride_seconds=cfg.preprocessing.stride_seconds,
            )
        )

        frame_base = frames_dir / asset.day / asset.camera_id / f"{asset.hour:02d}"
        for w in windows:
            clip_frame_dir = frame_base / str(w.clip_index)
            extract_frames_1fps(
                asset.video_path,
                clip_frame_dir,
                w.start_seconds,
                w.end_seconds,
                fps=cfg.preprocessing.fps,
            )

        windows = mark_placeholder_windows(windows, frame_base)

        # Relative ms base: day offset + hour offset (calendar dates TBD)
        day_index = int(asset.day.lstrip("day")) - 1
        base_ms = (day_index * 86400 + asset.hour * 3600) * 1000

        clips: list[ClipRecord] = []
        for w in windows:
            clip_frame_dir = frame_base / str(w.clip_index)
            frame_paths = sorted(clip_frame_dir.glob("*.jpg"))
            clip_id = (
                f"{asset.day}_{asset.camera_id}_{asset.hour:02d}_{w.clip_index:04d}"
            )
            clips.append(
                ClipRecord(
                    clip_id=clip_id,
                    parent_source_id=(
                        f"{asset.day}_{asset.camera_id}_{asset.hour:02d}"
                    ),
                    day=asset.day,
                    hour=asset.hour,
                    camera_id=asset.camera_id,
                    camera_type=asset.camera_type,
                    participant_id=asset.participant_id,
                    room=asset.room,
                    start_seconds=w.start_seconds,
                    end_seconds=w.end_seconds,
                    absolute_start=base_ms + int(w.start_seconds * 1000),
                    absolute_end=base_ms + int(w.end_seconds * 1000),
                    source_video_path=str(asset.video_path),
                    sampled_frame_paths=[str(p) for p in frame_paths],
                    is_placeholder=w.is_placeholder,
                )
            )

        tw_list = []
        if asset.transcript_path and asset.transcript_path.exists():
            segments = load_raw_segments(asset.transcript_path)
            tw_list = merge_into_windows(
                segments=segments,
                base_unix_ms=base_ms,
                camera_id=asset.camera_id,
                camera_type=asset.camera_type,
                participant_id=asset.participant_id,
                room=asset.room,
                day=asset.day,
                hour=asset.hour,
            )

        write_jsonl_records(clips, out_dir / "clips.jsonl")
        write_jsonl_records(tw_list, out_dir / "transcripts.jsonl")
        n_clips += len(clips)
        n_windows += len(tw_list)

    if not skip_base:
        console.print(
            f"  base          : {n_clips} clips, {n_windows} transcript windows written"
        )

    # Scope caption/events to the same days as the base pass.  Without this,
    # --skip-base --day N would still pick up other days' chunks via rglob.
    day_chunk_roots = [chunks_dir / f"day{d}" for d in days_list]

    def _iter_clip_paths():
        for root in day_chunk_roots:
            if root.exists():
                yield from sorted(root.rglob("clips.jsonl"))

    # ------------------------------------------------------------------ #
    # Caption / OCR phase                                                  #
    # ------------------------------------------------------------------ #
    if caption:
        from castlerag.index.io import load_clip_records
        from castlerag.preprocess.caption_ocr import annotate_clip

        vllm_url = _vllm_base_url()
        n_annotated = 0
        for clips_path in _iter_clip_paths():
            clip_records = load_clip_records(clips_path)
            updated: list[ClipRecord] = []
            for cr in clip_records:
                frames = [Path(p) for p in cr.sampled_frame_paths]
                try:
                    ann = annotate_clip(
                        clip_id=cr.clip_id,
                        frame_paths=frames,
                        transcript_text=cr.transcript_text,
                        model_name=cfg.generation.model,
                        vllm_base_url=vllm_url,
                    )
                    updated.append(
                        cr.model_copy(
                            update={
                                "clip_caption": ann.clip_caption,
                                "ocr_text": ann.ocr_text,
                            }
                        )
                    )
                except Exception as exc:
                    console.print(
                        f"[yellow]  caption skipped {cr.clip_id}: {exc}[/yellow]"
                    )
            write_jsonl_records(updated, clips_path)
            n_annotated += len(updated)
        console.print(f"  caption/OCR   : {n_annotated} clips annotated")

    # ------------------------------------------------------------------ #
    # Event compression phase                                              #
    # ------------------------------------------------------------------ #
    if events:
        from castlerag.index.io import load_clip_records
        from castlerag.preprocess.event_compress import compress_clips_to_event

        vllm_url = _vllm_base_url()
        n_events = 0
        for clips_path in _iter_clip_paths():
            clip_records = [
                cr for cr in load_clip_records(clips_path) if not cr.is_placeholder
            ]
            clip_records.sort(key=lambda c: c.absolute_start)
            event_records = []
            for i in range(0, len(clip_records) - 3, 4):
                group = clip_records[i : i + 4]
                if len(group) < 4:
                    break
                try:
                    ev = compress_clips_to_event(
                        clips=group,
                        model_name=cfg.generation.model,
                        vllm_base_url=vllm_url,
                    )
                    event_records.append(ev)
                except Exception as exc:
                    console.print(f"[yellow]  event compress skipped: {exc}[/yellow]")
            if event_records:
                ev_path = clips_path.parent / "events.jsonl"
                write_jsonl_records(event_records, ev_path)
                n_events += len(event_records)
        console.print(f"  events        : {n_events} event summaries written")

    # ------------------------------------------------------------------ #
    # Auxiliary phase                                                      #
    # ------------------------------------------------------------------ #
    if aux:
        from castlerag.preprocess.auxiliary import (
            iter_aux_video_records,
            iter_photo_records,
            iter_thermal_records,
        )

        aux_records = []
        for day_num in days_list:
            day_label = f"day{day_num}"
            for cam in cfg.dataset.ego_cameras:
                aux_records.extend(iter_photo_records(root, cam, day_label))
            aux_records.extend(iter_thermal_records(root, day_label))
            aux_records.extend(iter_aux_video_records(root, day_label))

        if aux_records:
            aux_out = chunks_dir / "aux.jsonl"
            aux_out.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl_records(aux_records, aux_out)
        console.print(f"  aux           : {len(aux_records)} auxiliary records written")


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
    day: Optional[int] = typer.Option(
        None,
        "--day",
        min=1,
        max=4,
        help="Embed and upsert only the day-N records.  Existing days in the "
        "Qdrant collection are left untouched; BM25 is still rebuilt from all "
        "available transcripts.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Create Qdrant collection + payload indexes and upsert evidence points.

    With ``--day N`` the embed + upsert paths are scoped to that day, so the
    canonical incremental flow is::

        castlerag preprocess --day 2 --caption --events --aux
        castlerag index      --day 2

    BM25 is always rebuilt from the full transcript scope so retrieval keeps
    matching previously-ingested days.
    """
    cfg = _resolve_config(config, snellius)
    scope = f"day{day}" if day is not None else "all-days"
    console.print(
        f"[bold]castlerag index[/bold]  collection={cfg.qdrant.collection}  "
        f"scope={scope}"
    )
    console.print(f"  Qdrant : {cfg.qdrant.host}:{cfg.qdrant.port}")
    if dry_run:
        console.print("[yellow]dry-run: no Qdrant writes[/yellow]")
        return
    records = load_chunk_records(Path(cfg.preprocessing.chunks_dir))
    scoped_all = filter_records(records, cfg)
    if _count_records(scoped_all) == 0:
        console.print("[red]No chunk records found — run preprocess first.[/red]")
        raise typer.Exit(1)
    if day is not None and _count_records(filter_records(records, cfg, day=day)) == 0:
        console.print(
            f"[red]No chunk records found for day {day} — "
            f"run `castlerag preprocess --day {day}` first.[/red]"
        )
        raise typer.Exit(1)
    # Always invoke cache_dense_embeddings — _cache_records short-circuits
    # per cache file, so existing artifacts are not re-embedded.
    embed_client = OmniEmbedClient(
        model=cfg.embedding.model,
        backend=cfg.embedding.backend,
        vllm_base_url=_vllm_base_url(),
        vllm_tensor_parallel=cfg.embedding.vllm_tensor_parallel,
        vllm_gpu_memory_utilization=cfg.embedding.vllm_gpu_memory_utilization,
    )
    cache_dense_embeddings(records, cfg, embed_client, day=day)
    # BM25 rebuilds from the full record scope so day-1 retrieval keeps
    # working after a day-2 incremental ingest.
    bm25_path = build_bm25_artifact(scoped_all, Path(cfg.embedding.cache_dir))
    vector_size, cache_paths = build_qdrant_index(
        cfg,
        records,
        recreate=create_collection,
        day=day,
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
    if len(questions) < n:
        console.print(
            "[red]Smoke test requires at least "
            f"{n} questions, but {questions_path} only contains {len(questions)}.[/red]"
        )
        raise typer.Exit(1)
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
