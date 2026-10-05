# Ingesting all four CASTLE days on Snellius

This runbook takes days 1 to 4 into the one Qdrant collection
`castle_multimodal_v1`, with every camera (10 ego plus the 5 fixed room
cameras), using `scripts/slurm/ingest_day.slurm`. Nothing here has been run end
to end yet; the numbers are estimates.

All commands use `--account=gisr109364`. The account is per user, so run
`accinfo` and use the one it shows for you.

## 0. Constraints

- There is no project space. Everything lives on `/scratch-shared/$USER`,
  which deletes files 14 days after they were last used.
- Home is 200 GB and mostly full. Only small derived artifacts can go there.
- The dataset comes down in two waves: days 1+2, then days 3+4, into
  `/scratch-shared/$USER/castle2024`. Each day has to be preprocessed,
  embedded and indexed while its raw video is still on disk.
- The budget is about 98k SBU, shared with two teammates.

## 1. How the job works

`ingest_day.slurm` is the day-1 fixed-camera job from PR #61 with the day made
a parameter. `fixedcams_day1.slurm` still exists as a wrapper that runs it with
`DAY=1`, so the commands in `docs/fixedcams_reingest.md` keep working.

`DAY` is passed with `--export=ALL,DAY=N` (default 1). Everything follows from
it: the raw video path `castle2024/main/dayN`, the chunk dir
`castle_derived/chunks/dayN`, every `--day N` flag, the Qdrant counts
(`--day dayN`), the backup dir name, and the log prefix
`logs/ingest_dayN_<what>_<job>.log`. The default `HOURS` is `8..20` for days
1-3 and `8..18` for day 4 (from `src/castlerag/ui/youtube_mirror.csv`).
Directives in `#SBATCH` lines can't see `DAY`, so pass
`--job-name=castle-ingest-dayN` too. The main log is then
`logs/castle-ingest-dayN_<job>.out`.

One job runs on 3 A100s (two Qwen3-VL servers for caption/events, OmniEmbed
for embedding) and does, in order:

1. State check. Qdrant starts on `/scratch-shared/$USER/qdrant_storage`. The
   job aborts if the collection has fewer than `MIN_POINTS` points (default
   25000, meaning day 1 must already be in). It counts the day's points,
   resolves `CAMS`, scope-checks the cameras with `preprocess --dry-run`, and
   checks that raw video exists for each one.
2. Rollback material. It copies `transcripts.pkl`, `visual_text.json`, and,
   if the day already has caches, `*_dayN.npz` and `manifest_dayN.json` into
   `castle_derived/embeddings_backup_pre_ingest_dayN_<job>/`, then verifies the
   copy. It takes a Qdrant snapshot into `/scratch-shared/$USER/qdrant_snapshots/`.
   It also records sha1 checksums of every chunk file of that day's cameras
   that it is not ingesting. It aborts if the day has points in Qdrant but no
   caches on disk.
3. `castlerag preprocess --day N --camera C --hour H ... --caption --events`,
   about 10 camera x hour-group workers.
4. Validation gate. Each camera must have at least 98 % of clips captioned,
   events, no corrupt lines, and the right `camera_type`/`room` (fixed:
   `fixed`/`<cam>`; ego: `ego`/none). The other cameras' checksums must be
   unchanged.
5. `castlerag embed --day N --modality {transcript,event_summary,video}`.
   This is incremental, so it embeds only new record ids.
6. `castlerag index --day N`, without `--create-collection`. This step
   upserts the day's points and rebuilds `transcripts.pkl` and
   `visual_text.json` from the chunk records of every day on disk.
7. Verification. It writes `logs/ingest_dayN_counts_after_<job>.txt` and fails
   if any requested camera still has 0 `main_clip` points.

Knobs (`--export=ALL,...`): `DAY`, `CAMS` (default `auto`: every in-scope
camera with raw video and 0 `main_clip` points that day), `BOOTSTRAP`,
`MIN_POINTS`, `SKIP_BASE`, `HOURS`, `SPLIT`, `TARGET_WORKERS`, `SNAPSHOT`,
`CONF` (default `configs/snellius_fixedcams.yaml`, which has
`camera_scope: "all"`).

### One day = three chained jobs

A full day is 131-181 camera-hours. That is 30-45 h of wall time on 3 GPUs,
more than the job's 20 h limit. Captioning also can't resume: a timed-out job
loses its captions. So each day runs as three jobs of about 50-62
camera-hours, at roughly 8-15 h each, chained with `afterok`:

| Group | Cameras |
|---|---|
| `FIXED` | Kitchen Living1 Living2 Meeting Reading |
| `EGO_A` | Allie Bjorn Cathal Florian Klaus |
| `EGO_B` | Luca Onanong Stevan Tien Werner (day 4: no Tien) |

Each job is a complete additive ingest of its own group. The second and third
jobs checksum the chunks written by the earlier ones. If a job fails, the
later ones never start (`afterok`). Fix the cause and resubmit the failed
group with the same command. If the base pass (frames and chunks) already
finished for that group, add `SKIP_BASE=1`. If it only ran out of time, raise
`--time` on the resubmit (`gpu_a100` allows more than 20 h) or split the group
with `HOURS=`.

## 2. One-time setup on the new account

These are the same steps as `docs/snellius.md` §1-2 (venv `~/castlerag_venv`
on the 2024 module stack, Qdrant binary at `~/qdrant/qdrant`). The jobs run
with `HF_HUB_OFFLINE=1`, so download the models into the scratch HF cache once
on the login node:

```bash
export HF_HOME=/scratch-shared/$USER/hf_cache
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct
huggingface-cli download Tevatron/Qwen2.5-Omni-7B-Thinker     # OmniEmbed base
huggingface-cli download Tevatron/OmniEmbed-v0.1-multivent    # OmniEmbed LoRA
huggingface-cli download Qwen/Qwen2.5-Omni-7B                 # OmniEmbed processor
```

The model ids are the defaults of `scripts/omniembed_server.py` and of the
`vllm serve` lines in the job. The HF cache is on scratch too. Every job reads
it, so it stays in use while you keep running jobs.

If `~/castle_archives/*/RESTORE.md` holds an older day-1 index, you can
restore it (`sha256sum -c SHA256SUMS`, then the `tar` lines in RESTORE.md) and
skip day 1 in §3. Then run `ingest_day.slurm` with `DAY=1` and the default
`CAMS=auto`, which ingests only the cameras that are missing.

## 3. Wave 1: days 1 and 2

```bash
export HF_HOME=/scratch-shared/$USER/hf_cache
huggingface-cli download CASTLE-Dataset/CASTLE2024 --repo-type dataset \
    --local-dir /scratch-shared/$USER/castle2024 \
    --include "main/day1/*" "main/day2/*"
ls /scratch-shared/$USER/castle2024/main/day1      # 15 camera dirs expected (no Bao)
myquota                                            # scratch headroom: see §5 for sizes
```

The jobs never run `--aux`, so the `auxiliary/` tree is not needed.

Submit from the repo root, so that `SLURM_SUBMIT_DIR` is the repo:

```bash
cd ~/CastleRAG
A=gisr109364
FIXED="Kitchen Living1 Living2 Meeting Reading"
EGO_A="Allie Bjorn Cathal Florian Klaus"
EGO_B="Luca Onanong Stevan Tien Werner"
S=scripts/slurm/ingest_day.slurm

# Day 1 on an empty scratch: the first job creates the collection (BOOTSTRAP=1),
# the next two accept the then-partial collection (MIN_POINTS=1).
D1A=$(sbatch --parsable --account=$A --job-name=castle-ingest-day1 \
        --export=ALL,DAY=1,BOOTSTRAP=1,CAMS="$FIXED" $S)
D1B=$(sbatch --parsable --account=$A --job-name=castle-ingest-day1 --dependency=afterok:$D1A \
        --export=ALL,DAY=1,MIN_POINTS=1,CAMS="$EGO_A" $S)
D1C=$(sbatch --parsable --account=$A --job-name=castle-ingest-day1 --dependency=afterok:$D1B \
        --export=ALL,DAY=1,MIN_POINTS=1,CAMS="$EGO_B" $S)

# Day 2 (day 1 complete, so the default MIN_POINTS=25000 holds)
D2A=$(sbatch --parsable --account=$A --job-name=castle-ingest-day2 --dependency=afterok:$D1C \
        --export=ALL,DAY=2,CAMS="$FIXED" $S)
D2B=$(sbatch --parsable --account=$A --job-name=castle-ingest-day2 --dependency=afterok:$D2A \
        --export=ALL,DAY=2,CAMS="$EGO_A" $S)
D2C=$(sbatch --parsable --account=$A --job-name=castle-ingest-day2 --dependency=afterok:$D2B \
        --export=ALL,DAY=2,CAMS="$EGO_B" $S)
squeue -u $USER
```

The jobs must run one at a time. They share the Qdrant storage, the embedding
caches and `transcripts.pkl`, so keep the `afterok` chain. Do not submit two
days side by side.

After each day:

```bash
tail -n 40 logs/castle-ingest-day1_${D1C}.out
cat logs/ingest_day1_counts_after_${D1C}.txt     # all 15 cameras with main_clip points
```

The 40-question day-1 eval (`smoke_day1_roomfix.slurm`, see
`docs/fixedcams_reingest.md` §5) works once day 1 is done:

```bash
sbatch --account=$A --dependency=afterok:$D1C \
       --export=ALL,CONF=configs/snellius_fixedcams.yaml scripts/slurm/smoke_day1_roomfix.slurm
```

## 4. Wave 2: days 3 and 4

First do §5 for days 1 and 2: archive them, then free the space. Then:

```bash
huggingface-cli download CASTLE-Dataset/CASTLE2024 --repo-type dataset \
    --local-dir /scratch-shared/$USER/castle2024 \
    --include "main/day3/*" "main/day4/*"

D3A=$(sbatch --parsable --account=$A --job-name=castle-ingest-day3 \
        --export=ALL,DAY=3,CAMS="$FIXED" $S)
D3B=$(sbatch --parsable --account=$A --job-name=castle-ingest-day3 --dependency=afterok:$D3A \
        --export=ALL,DAY=3,CAMS="$EGO_A" $S)
D3C=$(sbatch --parsable --account=$A --job-name=castle-ingest-day3 --dependency=afterok:$D3B \
        --export=ALL,DAY=3,CAMS="$EGO_B" $S)

# Day 4 has no Tien stream (and a Bao stream, see §7)
D4A=$(sbatch --parsable --account=$A --job-name=castle-ingest-day4 --dependency=afterok:$D3C \
        --export=ALL,DAY=4,CAMS="$FIXED" $S)
D4B=$(sbatch --parsable --account=$A --job-name=castle-ingest-day4 --dependency=afterok:$D4A \
        --export=ALL,DAY=4,CAMS="$EGO_A" $S)
D4C=$(sbatch --parsable --account=$A --job-name=castle-ingest-day4 --dependency=afterok:$D4B \
        --export=ALL,DAY=4,CAMS="Luca Onanong Stevan Werner" $S)
```

If you pass a camera that has no raw video for the day, the job aborts in the
state check before any GPU work. That is what would happen with Tien on day 4.

## 5. What to keep, what to delete

Keep these. They are expensive to rebuild and small next to video and frames:

| Artifact | Where it is | Why |
|---|---|---|
| Chunk JSONLs (`clips.jsonl`, `events.jsonl`, `transcripts.jsonl` per camera/hour) | `castle_derived/chunks/dayN/` | Hold the captions, OCR and events, which are most of the GPU cost. **Every later `index` rebuilds `transcripts.pkl` and `visual_text.json` from the chunks of all days on disk**, so deleting days 1-2 chunks before wave 2 would silently drop them from both BM25 lanes. |
| Embedding caches `*_dayN.npz`, `manifest_dayN.json` | `castle_derived/embeddings/` | Re-index without re-embedding. Rollback. |
| `transcripts.pkl`, `visual_text.json` | `castle_derived/embeddings/` | The BM25 and visual-text lanes the UI and eval load. |
| Qdrant storage (or a snapshot) | `qdrant_storage/storage/`, `qdrant_snapshots/` | The index itself. |

These have to stay on scratch while the later days are ingested (the job reads
them there). To survive the 14-day purge, also copy them to home after each
wave:

```bash
# login node, while no castle job is running (the Qdrant tar must be consistent)
DAYS="1 2" bash scripts/archive_bugb.sh        # -> ~/castle_archives/castle_days12_<date>/
DAYS="1 2 3 4" bash scripts/archive_bugb.sh    # after wave 2
```

The script tars the whole `chunks/` tree, the listed days' caches with
`transcripts.pkl`/`visual_text.json`, and the Qdrant storage. It writes
`SHA256SUMS` and `RESTORE.md`. Before writing anything it checks that home has
room and aborts if not. **Sizes have not been measured on this account yet.**
After day 1, run

```bash
du -sh /scratch-shared/$USER/castle_derived/chunks/day1 \
       /scratch-shared/$USER/castle_derived/embeddings \
       /scratch-shared/$USER/qdrant_storage /scratch-shared/$USER/qdrant_snapshots
```

and check that four days of it fit in home (`myquota`). If they don't, keep
only the newest archive (delete older `~/castle_archives/*` once the new
`sha256sum -c SHA256SUMS` passes). You can also drop `qdrant_storage.tar`,
because the index can be rebuilt from the chunks and caches with
`castlerag index --day N` per day without re-embedding. Old
`embeddings_backup_pre_ingest_day*_<job>/` dirs and older Qdrant snapshots can
go once the next job of the chain has passed.

Delete before wave 2, once days 1 and 2 are indexed, verified and archived:

```bash
rm -rf /scratch-shared/$USER/castle2024/main/day1 /scratch-shared/$USER/castle2024/main/day2
rm -rf /scratch-shared/$USER/castle_derived/frames_1fps/day1 /scratch-shared/$USER/castle_derived/frames_1fps/day2
```

- Raw video can be downloaded again.
- Sampled frames are the bulk of the space: about 330 GB for day-1 ego, plus
  about 170 GB expected for the fixed cameras. They can be re-extracted from
  raw video. A day's frames are needed until that day's `embed --modality
  video` has run.
- Deleting frames has a cost. Qdrant payloads keep `sampled_frame_paths`, and
  the visual reranker and answer generation send those frames to Qwen3-VL. For
  days whose frames are gone, missing files are skipped (`frame_encoding.py`),
  so those days fall back to text-only evidence (captions, OCR, transcripts)
  at query time. If scratch has room, keep the frames of the days you will
  demo.

## 6. Compute estimate

Per day, for all 15 cameras on 3 A100s at 128 SBU per GPU-hour. This scales
from the day-1 estimate by camera-hours, about 91 SBU per camera-hour:

| Day | Camera-hours (ego + fixed) | Estimated SBU | Wall time, 3 jobs |
|---|---|---|---|
| 1 | 119 + 62 = 181 | ~16.5k | ~43 h |
| 2 | 104 + 61 = 165 | ~15.1k | ~39 h |
| 3 | 118 + 62 = 180 | ~16.4k | ~43 h |
| 4 | 87 + 53 = 140 | ~12.8k | ~33 h |
| **Total** | 666 | **~60.8k** | ~158 h |

That is about 62 % of the ~98k SBU shared with two teammates. Agree on the
split before wave 1. Restoring a day-1 archive instead of re-ingesting saves
~16.5k. A failed and resubmitted group costs its share again, roughly
4.5k-5.7k SBU per group. The day-4 figure includes Bao's 9 hours (see §7).
Embedding and the index upsert are included and small (under an hour per job).

## 7. Open points

- **Bao on day 4.** Bao's ego stream exists only on day 4 (9 hours).
  `configs/snellius_me.yaml` and `configs/snellius_fixedcams.yaml` leave Bao
  out of `ego_cameras`, so `preprocess --camera Bao` is rejected by the scope
  check and `CAMS=auto` doesn't pick Bao up. Ingesting Bao needs Bao added to
  `ego_cameras` in both configs. That also changes `known_participants` at
  query time, because questions that name Bao would then apply a dense
  participant filter. Decide before day 4.
- The 14-day purge counts from "last use". Whether reads by the index step
  count as use for the day-1/2 chunks has not been checked, so archive after
  each wave rather than rely on it.
- The per-group wall times are scaled from the day-1 ego run and the
  fixed-camera estimate, not measured. Check the first day-1 job's duration
  (`grep "Phase 1 done" logs/castle-ingest-day1_<job>.out`) before the rest of
  the chain gets far.
