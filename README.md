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
# Discover files, create 30 s clips, extract 1 fps frames, normalize transcripts
castlerag preprocess --snellius --day 1

# Encode derived chunks with OmniEmbed (requires GPU)
castlerag embed --snellius --modality video --day 1

# Create Qdrant collection and upsert evidence points
castlerag index --snellius --create-collection

# Evaluate predictions
castlerag eval questions.json predictions.json --answers ground_truth.json

# 5-question smoke test (issue #15)
castlerag smoke-test questions.json --n 5
```

## Running on Snellius

Edit `configs/snellius.yaml` to set your account and scratch paths, then:

```bash
# Submit in dependency order (replace <job_id> with the id returned by sbatch):
sbatch scripts/slurm/preprocess_main.slurm
sbatch scripts/slurm/embed_text.slurm
sbatch scripts/slurm/embed_video.slurm
sbatch scripts/slurm/embed_images.slurm
sbatch scripts/slurm/index_qdrant.slurm

# Not yet implemented — do not submit until the referenced issues are resolved:
# scripts/slurm/caption_ocr.slurm      (issue #4)
# scripts/slurm/compress_events.slurm  (issue #4)
# scripts/slurm/index_transcripts.slurm (issues #5, #6)
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
