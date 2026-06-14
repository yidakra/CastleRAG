# Running CastleRAG on Snellius

End-to-end setup guide for the SURF Snellius HPC cluster.  Assumes you already
have a Snellius account, SSH access, and a project SBU budget you can charge
to.  No prior CastleRAG context required.

## 1. Prerequisites

- Snellius login (`ssh <user>@snellius.surf.nl`)
- A project account with GPU SBUs (the `gpu_a100` partition is what every
  SLURM script in this repo targets; one quarter-node = 1 A100 + 18 CPU
  cores + 120 GiB RAM)
- About **600 GiB** of scratch space.  `/scratch` is per-user, fast, and
  purged after 14 days of inactivity — do *not* keep anything important
  there long-term.
- The `2024` software stack (the SLURM scripts `module load 2024` and pin
  Python 3.11 + CUDA 12.6 from it)

```bash
ssh <user>@snellius.surf.nl
mkdir -p /scratch/$USER/castle2024 /scratch/$USER/castle_derived /scratch/$USER/castle_outputs ~/code
cd ~/code
git clone git@github.com:yidakra/CastleRAG.git
cd CastleRAG
```

## 2. One-time environment

Build the venv on a login node — the SLURM scripts pick it up from `${HOME}/castlerag_venv` by default.

```bash
module purge
module load 2024
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.6.0
module load FFmpeg/6.0-GCCcore-12.3.0

python -m venv ~/castlerag_venv
source ~/castlerag_venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,inference]"
```

Smoke check:

```bash
castlerag --help
python scripts/smoke_local.py     # offline wiring check, no GPU needed
```

`smoke_local.py` returns 0 if every code path connects end-to-end with stub
models and an in-memory Qdrant; it's the fastest way to verify your install
before queueing a real job.

## 3. Dataset

Download the CASTLE 2024 release once, into scratch:

```bash
# pip install huggingface_hub if it isn't already in your venv
huggingface-cli download CASTLE-Dataset/CASTLE2024 \
    --repo-type dataset \
    --local-dir /scratch/$USER/castle2024
```

The dataset is around 450 GiB; the download can run for several hours.  If
you only want to validate the pipeline first, restrict yourself to one day
by editing `configs/snellius.yaml` (see §5).

Expected layout (the loader fails fast if it's wrong):

```text
/scratch/$USER/castle2024/
  main/day{1..4}/{camera}/{video|transcript|metadata}/...
  auxiliary/{heartrate|gaze|photo|thermal|video}/...
```

## 4. Configure your account in snellius.yaml

`configs/snellius.yaml` already points scratch paths at `/scratch/$USER/...`,
so the only mandatory edits are:

```yaml
slurm:
  account: "your-snellius-project-account"   # e.g. EINF-1234
  mail_user: "you@example.com"
```

`configs/base.yaml` is deep-merged underneath; `snellius.yaml` only carries
the overrides.  Don't duplicate keys between the two files — change them in
one place.

You can also restrict the dataset scope here to make the first run cheap:

```yaml
dataset:
  days: [1]              # work on day 1 only the first time through
  hours: [8, 9, 10, 11, 12, 13]
```

## 5. Start the vLLM inference server

CastleRAG needs two model endpoints:

| Endpoint     | Model                          | Used by                                  |
|--------------|--------------------------------|------------------------------------------|
| Embedding    | `Tevatron/OmniEmbed-v0.1-multivent` | `castlerag preprocess --caption/--events`, `castlerag embed` |
| Generation   | `Qwen/Qwen3-VL-8B-Instruct`    | `castlerag preprocess --caption/--events` (vision-language), `castlerag answer`, `castlerag smoke-test` |

On Snellius's gpu_a100 quarter-node (1 A100 80 GiB), both fit individually
with room to spare.  The simplest pattern is **one vLLM process per node
running both models is not possible** — they have different served names
and tokenisers — so start them as **two long-running interactive jobs** on
two separate nodes:

```bash
# Terminal A — OmniEmbed
srun --account=$ACCOUNT --partition=gpu_a100 --gres=gpu:1 --cpus-per-task=18 \
     --mem=120G --time=24:00:00 --pty bash
module purge && module load 2024 CUDA/12.6.0 Python/3.11.3-GCCcore-12.3.0
source ~/castlerag_venv/bin/activate
vllm serve Tevatron/OmniEmbed-v0.1-multivent \
    --task embedding \
    --port 8200 \
    --served-model-name omniembed \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1
# (leave running; note the hostname — e.g. gcn123)

# Terminal B — Qwen3-VL
srun --account=$ACCOUNT --partition=gpu_a100 --gres=gpu:1 --cpus-per-task=18 \
     --mem=120G --time=24:00:00 --pty bash
module purge && module load 2024 CUDA/12.6.0 Python/3.11.3-GCCcore-12.3.0
source ~/castlerag_venv/bin/activate
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --port 8201 \
    --served-model-name qwen3vl \
    --gpu-memory-utilization 0.88 \
    --max-model-len 16384 \
    --trust-remote-code
# (leave running; note the hostname — e.g. gcn456)
```

The batch jobs in §6 must be able to reach both endpoints from the compute
node they land on.  Snellius compute nodes share an internal network, so
`http://<vllm-hostname>:8200/v1` is reachable from any other compute node.
Export this in your shell before submitting jobs:

```bash
export VLLM_EMBED_URL=http://<embed-host>:8200/v1
export VLLM_BASE_URL=http://<gen-host>:8201/v1
```

**A100-poor alternative.**  If you only get one GPU at a time, use the
**hot-swap pattern** documented for the 2× A2 host: pre-compute every query
vector via OmniEmbed once, save them to an NPZ, then shut OmniEmbed down and
serve the cache from disk while Qwen3-VL takes over the GPU.

```bash
# Once, before evaluation:
OMNIEMBED_URL=$VLLM_EMBED_URL \
    python scripts/precompute_queries.py /scratch/$USER/questions.json \
        --out /scratch/$USER/castle_derived/embeddings/query_cache.npz
# Then for `castlerag answer`:
export OMNIEMBED_QUERY_CACHE=/scratch/$USER/castle_derived/embeddings/query_cache.npz
# OmniEmbedClient serves dense queries from the NPZ, no embed server needed
```

## 6. Run the offline build via SLURM

`scripts/slurm/` contains nine batch scripts wired into a dependency chain
that mirrors `castlerag preprocess → embed → index`.  Submit them in order:

```bash
ACCOUNT=EINF-1234   # whatever you set in snellius.yaml

# 1. Base preprocessing — windowing, 1 fps frames, transcript normalization
JOB1=$(sbatch --parsable --account=$ACCOUNT scripts/slurm/preprocess_main.slurm)

# 2a. Per-clip caption + OCR (needs Qwen3-VL at $VLLM_BASE_URL)
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} --account=$ACCOUNT \
              scripts/slurm/caption_ocr.slurm)

# 2b. Auxiliary modalities (CPU; runs in parallel with 2a)
JOB2B=$(sbatch --parsable --dependency=afterok:${JOB1} --account=$ACCOUNT \
               scripts/slurm/preprocess_aux.slurm)

# 3. Event compression (Qwen3-VL again)
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} --account=$ACCOUNT \
              scripts/slurm/compress_events.slurm)

# 4. OmniEmbed dense passes (each needs $VLLM_EMBED_URL)
JOB4A=$(sbatch --parsable --dependency=afterok:${JOB3} --account=$ACCOUNT \
               scripts/slurm/index_transcripts.slurm)
JOB4B=$(sbatch --parsable --dependency=afterok:${JOB3} --account=$ACCOUNT \
               scripts/slurm/embed_text.slurm)
JOB4C=$(sbatch --parsable --dependency=afterok:${JOB3} --account=$ACCOUNT \
               scripts/slurm/embed_video.slurm)
JOB4D=$(sbatch --parsable --dependency=afterok:${JOB2B} --account=$ACCOUNT \
               scripts/slurm/embed_images.slurm)

# 5. Qdrant indexing (CPU; start Qdrant on the compute node first — see below)
sbatch --dependency=afterok:${JOB4A}:${JOB4B}:${JOB4C}:${JOB4D} --account=$ACCOUNT \
       scripts/slurm/index_qdrant.slurm
```

Each preprocess/embed script is an array job over `--array=1-4` (one task
per day).  To rebuild only day 2, override the array on `sbatch`:

```bash
sbatch --array=2 --account=$ACCOUNT scripts/slurm/preprocess_main.slurm
```

Total cost for the full ego-only build is roughly **60–106 A100-hours**
(see SPEC §3.5) — between EUR 115 and EUR 204 at 128 SBU/h × EUR 15/1000 SBU.

### Qdrant on the indexing node

`index_qdrant.slurm` assumes Qdrant is already listening on `localhost:6333`
when the job runs.  Easiest pattern: start Qdrant inside the batch script
itself so it lives only for the lifetime of the job.  Edit
`scripts/slurm/index_qdrant.slurm` to add, just before the `castlerag index`
line:

```bash
# Start Qdrant in the background
${HOME}/qdrant/qdrant &
QDRANT_PID=$!
sleep 5
trap 'kill $QDRANT_PID 2>/dev/null' EXIT
```

(Download the qdrant binary once into `~/qdrant/` — single static binary,
no sudo required.)

## 7. Incremental ingest (preferred after the first run)

The `--day N` flag is symmetric across `preprocess`, `embed`, and `index`,
so adding a new day to an existing collection is a three-step incremental
run rather than a full rebuild:

```bash
sbatch --array=5 --account=$ACCOUNT scripts/slurm/preprocess_main.slurm
sbatch --array=5 --account=$ACCOUNT --dependency=afterok:<job_id> \
       scripts/slurm/embed_text.slurm
# … etc for caption_ocr, compress_events, embed_video
# Finally:
sbatch --account=$ACCOUNT --dependency=afterok:<embed_jobs> \
       --export=ALL,CASTLE_DAY=5 scripts/slurm/index_qdrant.slurm
```

Or interactively, on one node with both vLLM endpoints live:

```bash
castlerag preprocess --config configs/snellius.yaml --day 5 --caption --events --aux
castlerag embed      --config configs/snellius.yaml --day 5
castlerag index      --config configs/snellius.yaml --day 5
```

`castlerag index --day N` only embeds and upserts day-N artifacts; existing
days in the Qdrant collection are left intact (UUIDv5 point ids are
deterministic per `record_id`, so day-N records produce distinct new
points).  BM25 still rebuilds from the full transcript corpus, so day-1
retrieval keeps working.

## 8. Evaluation

After indexing finishes, run the eval:

```bash
castlerag answer \
    --config configs/snellius.yaml \
    /scratch/$USER/questions.json \
    --out /scratch/$USER/castle_outputs/predictions.json
```

The eval writes `predictions.json`, `evidence_traces.jsonl`,
`submissions.json`, and `metrics.json` under `outputs.dir`.  `submissions.json`
is the format the challenge expects.

Smoke a single batch first to catch config / endpoint issues without burning
the whole run:

```bash
castlerag smoke-test /scratch/$USER/questions.json --n 5 \
    --config configs/snellius.yaml
```

## 9. Tips and pitfalls

- **Scratch purge.**  `/scratch` files untouched for 14 days are deleted.
  Move anything you want to keep (predictions, eval outputs) to `~` or to
  a project filesystem before then.
- **Module hygiene.**  Always `module purge` at the top of your shell and
  every batch script.  Stray modules from `~/.bashrc` will load CUDA 11 over
  the 12.6 the venv was built against and break vLLM with cryptic errors.
- **vLLM startup time.**  Both models take ~3–4 minutes to load on first
  request.  Curl `/v1/models` once before submitting jobs so the warm-up
  cost is paid before the array fan-out.
- **Qdrant memory.**  3884 points (one day) need < 1 GiB; the full
  multi-day index sits comfortably in 16 GiB.  `--mem=120G` is generous.
- **Job logs.**  Every script writes to `logs/%x_%A_%a.out`; `tail -f` the
  most recent file to watch progress in real time.
- **Hot-swap escape hatch.**  If you ever have to share a single GPU
  between OmniEmbed and Qwen3-VL, the `OMNIEMBED_QUERY_CACHE` env var
  combined with `scripts/precompute_queries.py` lets you run eval without
  a live embed server.
