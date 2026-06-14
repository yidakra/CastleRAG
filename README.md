# CastleRAG

Multimodal RAG system for verifiable multiple-choice QA over the CASTLE 2024 dataset,
targeting the [CASTLE Challenge @ EgoVis 2026](https://castle-challenge.github.io/).

## Overview

CastleRAG builds an offline evidence memory from ego-camera video, transcripts,
and auxiliary sensor data, then answers CASTLE multiple-choice questions using
a retrieve → rerank → generate pipeline.

**Key technical decisions (fixed):**

| Component | Choice |
|---|---|
| Generation / reranking model | `Qwen/Qwen3-VL-8B-Instruct` via vLLM |
| Ablation baseline | `OpenGVLab/InternVL3-8B` |
| Retrieval embedding | `Tevatron/OmniEmbed-v0.1-multivent` |
| Transcript retrieval | Dual-path: BM25 + OmniEmbed dense, merged with RRF |
| Video scope (baseline) | 10 egocentric cameras only |
| Clip policy | 30 s clips, 30 s stride, 1 fps sampled frames |
| Evidence budget | Top 50 rows to generator |

## Setup

```bash
git clone <repo>
cd CastleRAG

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

For GPU inference (Snellius or local GPU host):
```bash
pip install -e ".[inference]"
```

## Dataset

The CASTLE 2024 dataset is available on Hugging Face (`castle-challenge/castle2024`).
Download it to `/data/castle2024` (or set `dataset.root` in `configs/base.yaml`).

Expected layout:
```text
/data/castle2024/
  main/
    day{1..4}/
      {camera}/
        video/{HH}.mp4
        transcript/{HH}.json
        metadata/{HH}.*.csv
  auxiliary/
    heartrate/
    gaze/
    photo/
    thermal/
    video/
```

## Pipeline commands

```bash
# Base preprocessing — windowing, 1 fps frames, transcript normalization (CPU)
castlerag preprocess --snellius --day 1

# Per-clip caption + OCR (GPU, requires VLLM_BASE_URL)
export VLLM_BASE_URL=http://localhost:8000/v1
castlerag preprocess --caption --snellius --day 1

# Compress 4-clip groups into 2-minute event summaries (GPU)
castlerag preprocess --events --snellius --day 1

# Normalize auxiliary modalities — photo, thermal, video (CPU)
castlerag preprocess --aux --snellius

# Encode derived chunks with OmniEmbed (GPU)
castlerag embed --snellius --modality transcript --day 1
castlerag embed --snellius --modality video --day 1

# Create Qdrant collection and upsert evidence points
castlerag index --snellius --create-collection

# Run full pipeline on CASTLE questions
castlerag answer questions.json --snellius

# Evaluate predictions against ground truth
castlerag eval questions.json predictions.json --answers ground_truth.json

# 5-question smoke test against a live vLLM endpoint
castlerag smoke-test questions.json --n 5
```

## Local offline smoke test

Verify pipeline wiring without any models or running services:

```bash
python scripts/smoke_local.py
```

Uses in-memory Qdrant, random-vector embeddings, and a stub LLM to confirm all
code paths connect end-to-end. Exit 0 = all 5 synthetic questions produce a
valid `a|b|c|d` prediction.

With a real local model endpoint (Ollama, llama.cpp, local vLLM):

```bash
VLLM_BASE_URL=http://localhost:11434/v1 python scripts/smoke_local.py --real
```

## Running on Snellius

See [docs/snellius.md](docs/snellius.md) for the full end-to-end setup
guide (account, venv, dataset download, vLLM servers, SLURM chain, and
incremental ingest with `--day N`).  The TL;DR for the SLURM chain:

Edit `configs/snellius.yaml` to set your account and scratch paths, then:

```bash
# Submit in dependency order (replace <job_id> with the id returned by sbatch):

# 1. Base preprocessing — windowing, 1 fps frames, transcript normalization (CPU)
JOB1=$(sbatch --parsable scripts/slurm/preprocess_main.slurm)

# 2a. Per-clip caption + OCR — array job, one task per day (GPU, needs VLLM_BASE_URL)
export VLLM_BASE_URL=http://<vllm-host>:8000/v1
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} scripts/slurm/caption_ocr.slurm)

# 2b. Auxiliary modality normalization — runs in parallel with caption_ocr (CPU)
JOB2B=$(sbatch --parsable --dependency=afterok:${JOB1} scripts/slurm/preprocess_aux.slurm)

# 3. Event compression — depends on caption_ocr (GPU, needs VLLM_BASE_URL)
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} scripts/slurm/compress_events.slurm)

# 4. Embedding and indexing
sbatch --dependency=afterok:${JOB3} scripts/slurm/index_transcripts.slurm
sbatch --dependency=afterok:${JOB3} scripts/slurm/embed_text.slurm
sbatch --dependency=afterok:${JOB3} scripts/slurm/embed_video.slurm
sbatch --dependency=afterok:${JOB3} scripts/slurm/embed_images.slurm
sbatch --dependency=afterok:${JOB3} scripts/slurm/index_qdrant.slurm
```

Estimated cost: ~EUR 115–204 for the full ego-only evidence build
(60–106 GPU-hours × 128 SBU/h × EUR 15/1000 SBU; see `SPEC.md §3.5`).

## Tests

```bash
pytest
```

## Configuration

- `configs/base.yaml` — local development defaults
- `configs/snellius.yaml` — Snellius HPC overrides (deep-merged on top of base)

Override any field via `--config path/to/custom.yaml`.

## Spec

See `SPEC.md` for the full technical specification.
