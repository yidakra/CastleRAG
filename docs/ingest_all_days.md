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
`MIN_POINTS`, `SKIP_BASE`, `SKIP_PREPROCESS`, `HOURS`, `SPLIT`, `TARGET_WORKERS`, `SNAPSHOT`,
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
| `EGO_B` | Luca Onanong Stevan Tien Werner (day 4: Bao instead of Tien) |

Each job is a complete additive ingest of its own group. The second and third
jobs checksum the chunks written by the earlier ones. If a job fails, the
later ones never start (`afterok`). Fix the cause and resubmit the failed
group with the same command. If the base pass (frames and chunks) already
finished for that group, add `SKIP_BASE=1`. If the whole Phase 1 (base,
captions and events) finished and the job failed later, at validation, embed
or index, add `SKIP_PREPROCESS=1` instead: it starts no caption servers and
goes straight to validation, embedding and indexing, so the captioning hours
aren't spent again. If it only ran out of time, raise
`--time` on the resubmit (`gpu_a100` allows more than 20 h) or split the group
with `HOURS=`.

## 2. One-time setup on the new account

These are the same steps as `docs/snellius.md` §1-2 (venv `~/castlerag_venv`
on the 2024 module stack, Qdrant binary at `~/qdrant/qdrant`). The jobs run
with `HF_HUB_OFFLINE=1`, so download the models into the scratch HF cache once
on the login node:

```bash
export HF_HOME=/scratch-shared/$USER/hf_cache
hf download Qwen/Qwen3-VL-8B-Instruct
hf download Tevatron/Qwen2.5-Omni-7B-Thinker     # OmniEmbed base
hf download Tevatron/OmniEmbed-v0.1-multivent    # OmniEmbed LoRA
hf download Qwen/Qwen2.5-Omni-7B                 # OmniEmbed processor
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
cd ~/code/CastleRAG
DL=$(sbatch --parsable --account=gisr109364 --export=ALL,DAYS="1 2" scripts/slurm/download_castle.slurm)
# ~4.4 TB, roughly 3 h. The ingest chain below waits for $DL (afterok).
```

The jobs never run `--aux`, so the `auxiliary/` tree is not needed.

Submit from the repo root, so that `SLURM_SUBMIT_DIR` is the repo:

```bash
cd ~/code/CastleRAG
A=gisr109364
FIXED="Kitchen Living1 Living2 Meeting Reading"
EGO_A="Allie Bjorn Cathal Florian Klaus"
EGO_B="Luca Onanong Stevan Tien Werner"
S=scripts/slurm/ingest_day.slurm

# Day 1 on an empty scratch: the first job creates the collection (BOOTSTRAP=1),
# the next two accept the then-partial collection (MIN_POINTS=1).
D1A=$(sbatch --parsable --account=$A --job-name=castle-ingest-day1 --dependency=afterok:$DL \
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

Once the download job has finished, check what arrived:

```bash
tail -n 5 logs/castle-download_${DL}.out           # mp4 count and size per day
ls /scratch-shared/$USER/castle2024/main/day1      # 15 camera dirs expected (no Bao)
myquota                                            # scratch headroom: see §5 for sizes
```

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
# Same variables as wave 1 (a new login shell won't have them).
cd ~/code/CastleRAG
A=gisr109364
FIXED="Kitchen Living1 Living2 Meeting Reading"
EGO_A="Allie Bjorn Cathal Florian Klaus"
EGO_B="Luca Onanong Stevan Tien Werner"
S=scripts/slurm/ingest_day.slurm

# Download days 3+4 (~3.8 TB). D3A waits for it (afterok).
DL=$(sbatch --parsable --account=$A --export=ALL,DAYS="3 4",AUX=0 scripts/slurm/download_castle.slurm)

D3A=$(sbatch --parsable --account=$A --job-name=castle-ingest-day3 --dependency=afterok:$DL \
        --export=ALL,DAY=3,CAMS="$FIXED" $S)
D3B=$(sbatch --parsable --account=$A --job-name=castle-ingest-day3 --dependency=afterok:$D3A \
        --export=ALL,DAY=3,CAMS="$EGO_A" $S)
D3C=$(sbatch --parsable --account=$A --job-name=castle-ingest-day3 --dependency=afterok:$D3B \
        --export=ALL,DAY=3,CAMS="$EGO_B" $S)

# Day 4 has no Tien stream but does have Bao's (Bao exists only on day 4)
D4A=$(sbatch --parsable --account=$A --job-name=castle-ingest-day4 --dependency=afterok:$D3C \
        --export=ALL,DAY=4,CAMS="$FIXED" $S)
D4B=$(sbatch --parsable --account=$A --job-name=castle-ingest-day4 --dependency=afterok:$D4A \
        --export=ALL,DAY=4,CAMS="$EGO_A" $S)
D4C=$(sbatch --parsable --account=$A --job-name=castle-ingest-day4 --dependency=afterok:$D4B \
        --export=ALL,DAY=4,CAMS="Luca Onanong Stevan Werner Bao" $S)
```

If you pass a camera that has no raw video for the day, the job aborts in the
state check before any GPU work. That is what would happen with Tien on day 4.

## 5. What to keep, what to delete

Keep these. They are expensive to rebuild and small next to video and frames
(frames themselves are covered below):

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

### Frames: thin, don't delete

Frames are read again at question time: the reranker sends 4 frames per clip
and the answer generator 8 to Qwen3-VL. Deleting a day's frames silently turns
that day text-only. So keep frames for every day, but thin them once the day's
captioning and events are done (both read the full set):

```bash
python scripts/thin_frames.py --config configs/snellius_fixedcams.yaml --day 1          # dry run
python scripts/thin_frames.py --config configs/snellius_fixedcams.yaml --day 1 --apply
```

It keeps the 8 evenly spaced frames per clip that the generator would pick
(the reranker's 4 are drawn from those), deletes the other ~22, and rewrites
`sampled_frame_paths` in the chunk JSONLs (with a `.prethin` copy). Readers
sample from the frames that still exist (`frame_encoding.available_frames`),
so already-indexed days need no re-index. That cuts frames from ~500 GB to
~130 GB per day.

### Back up each day off scratch

The 14-day purge and the loss of a login are both real risks. What costs GPU
time to rebuild (~15k SBU a day) is the captions, chunks and embeddings, so
after a day is ingested and thinned, back those up to the team's **private**
Hugging Face dataset repo (CASTLE's terms forbid distributing derivative
works, so the job refuses a public repo):

```bash
# once: create the private dataset repo on huggingface.co, then on a login node
hf auth login                     # write token; the job never handles it
sbatch --account=gisr109364 --export=ALL,DAY=1 scripts/slurm/upload_artifacts.slurm
```

Per day it uploads the chunks, the embedding caches and manifest, plus the
current `transcripts.pkl` and `visual_text.json`. It's resumable: files whose
content is already on the Hub are skipped.
Restore = download, untar into `castle_derived/`, `castlerag index --day N`
(CPU only, no re-embedding).

Frames are not uploaded by default. A free HF account has 100 GB of private
storage and thinned frames are ~130 GB per day, while the frames are the cheap
part: `preprocess/media.py::extract_frames_1fps` cuts them on CPU from the raw
video with deterministic names (`-ss <clip start>`, `fps=1`, `%04d.jpg`). On a
plan with room (PRO: 1 TB private) add `FRAMES=1` to upload them as one tar
per camera-hour.

That makes scratch the only copy of the frames, and scratch is cleared every
14 days, so expect to lose them. Without frames a day's answers are text-only.
Before anything that needs image-grounded answers (paper eval, demo), check
and rebuild:

```bash
python scripts/regen_frames.py --config configs/snellius_fixedcams.yaml --day 1   # dry run: N missing
# if frames are missing: download the day's video again, then
DL=$(sbatch --parsable --account=gisr109364 --export=ALL,DAYS="1",AUX=0 scripts/slurm/download_castle.slurm)
sbatch --account=gisr109364 --dependency=afterok:$DL --export=ALL,DAY=1 scripts/slurm/regen_frames.slurm
```

It re-extracts each clip that lists a missing frame and moves only the listed
(thinned) frames into place; with the same FFmpeg module they are
byte-identical to the originals. Re-runs skip complete clips.

A second copy on a laptop costs nothing (`scripts/pull_artifacts.sh`, run
locally; `FRAMES=1` adds the frames, ~130 GB per day).

### Delete before wave 2

Once days 1 and 2 are indexed, thinned and backed up, delete only their raw
video (it can be downloaded again):

```bash
rm -rf /scratch-shared/$USER/castle2024/main/day1 /scratch-shared/$USER/castle2024/main/day2
```

Space on scratch (8 TiB quota): wave 1 peaks at ~5.4 TB (4.4 TB video + frames),
wave 2 at ~5 TB (3.8 TB video + thinned frames of all days), and after deleting
days 3-4 video about 0.5 TB of thinned frames remain.

### Re-flag placeholders on days ingested before #68

Clips that are blank or show the CASTLE test card are *placeholders*: they are
not captioned, embedded or indexed, and event summaries skip them. Days 1-3
were ingested with the old rule, which flagged by stillness. That marked real
footage of idle fixed-camera rooms as placeholders (5,395 clips, so those
stretches got no event summaries) and missed some test-card clips. To bring
an ingested day up to the current rule:

```bash
cd ~/code/CastleRAG
python scripts/reflag_placeholders.py --config configs/snellius_fixedcams.yaml --day 1           # dry run
python scripts/reflag_placeholders.py --config configs/snellius_fixedcams.yaml --day 1 --apply   # ~1 h CPU, run as a job
ALL="$FIXED $EGO_A $EGO_B"   # day 4: Bao instead of Tien
sbatch --account=$A --job-name=castle-reflag-day1 \
       --export=ALL,DAY=1,CAMS="$ALL",SKIP_BASE=1,CAPTION=0,MIN_POINTS=1 $S
```

The job rebuilds only the event summaries (captions are kept, `CAPTION=0`),
embeds the new ones and re-indexes the day. Its index step deletes the clips
that are now placeholders and the events that re-grouping replaced
(`prune_stale_points`), so the day's points match its chunk files again. The
re-flag reads the frames still on disk; thinned days keep 8 per clip, enough
for the >80 % rule. Clips with no frames left keep their old flag.

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

- **Bao on day 4 (decided: ingest).** Bao's ego stream exists only on day 4
  (9 hours), and Bao is now in `ego_cameras` in both Snellius configs, so the
  day-4 `EGO_B` group includes Bao. At query time this is safe: since #64 the
  dense participant filter keys on the (participant, day) pairs actually present
  in the indexed transcript windows, so a question naming Bao about days 1-3
  drops the filter instead of matching nothing.
- The 14-day purge counts from "last use". Whether reads by the index step
  count as use for the day-1/2 chunks has not been checked, so archive after
  each wave rather than rely on it.
- The per-group wall times are scaled from the day-1 ego run and the
  fixed-camera estimate, not measured. Check the first day-1 job's duration
  (`grep "Phase 1 done" logs/castle-ingest-day1_<job>.out`) before the rest of
  the chain gets far.
