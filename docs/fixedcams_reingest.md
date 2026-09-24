# Fixed room cameras in the day-1 index (#50 Bug B): runbook

This runbook adds the 5 fixed room cameras (Kitchen, Living1, Living2, Meeting,
Reading) to the existing Qdrant collection `castle_multimodal_v1` for day 1.
It does not rebuild the ego index. It then re-runs the 40-question day-1 eval
and checks the zero-evidence list from issue #50.

Nothing here has been run yet. All commands are for Snellius. The account is
`gpuuva082`; change it if yours is different.

## 0. What we already know

- The ego-only day-1 index had **25,920** points. PR #59 says a fixed-camera
  ingest was landed on 2026-06-23 and took the collection to **39,432** points,
  with "all five room cameras present". The config and scripts for that run
  were never committed. `configs/snellius_me.yaml` has always been
  `camera_scope: "ego"`.
- W&B runs from 2026-06-25 (`jazcvsvd`, `pcia4tw8`) cite `day1_Kitchen_*`,
  `day1_Living1_*`, `day1_Living2_*` and `day1_Meeting_*` evidence. No run has
  ever cited a `Reading` id. So Reading is either missing from the index or
  present but never retrieved. Step 1 tells us which.
- `/scratch-shared` deletes files after about 14 days without access. The June
  index may be gone. `scripts/archive_bugb.sh` (PR #58) may have saved a copy
  under `~/castle_archives/bugb_day1_*`.
- A query-time bug, fixed on this branch, hid fixed-camera evidence no matter
  what was in the index. The router's participant hint (for example "Werner"
  in "logo on Werner's apron") was applied as a hard Qdrant `must` filter on
  every dense lane. Fixed cameras have `participant_id=None`, so any question
  that names a person could only get that person's ego camera back. The CSV
  anchors for the apron and t-shirt questions are `Kitchen Day 1 12:16` and
  `Kitchen Day 1 14:44:46`: the answer is on a fixed camera that the filter
  removed. This is the same kind of failure as the room filter fixed in #53.

**So the first eval to run is on this branch against whatever index exists
(step 5a).** Only ingest the cameras that step 1 reports as missing.

## 1. What changed on `feat/fixedcams-reingest`

| Area | Change | Why |
|---|---|---|
| `configs/snellius_fixedcams.yaml` | Copy of `snellius_me.yaml` with `camera_scope: "all"` | `load_config` merges only one override file, so it cannot be layered on `snellius_me.yaml`. A test checks the two files differ only in scope. |
| `preprocess --camera X --hour H` | Limits base, caption and events to the listed cameras and hours. Rejects cameras outside `camera_scope`. | Before this, caption and events ran `rglob` over every `clips.jsonl` of the day. A fixed-camera run would have re-captioned and rewritten all ego clips. That was also the cross-camera race behind `resume2_day1.slurm`'s `chunks_iso/` workaround. |
| `_cache_records` (index/pipeline.py) | Incremental per record id: keeps cached rows, embeds only new ids, keeps out-of-scope rows, and fails on a vector-dim mismatch | Before this, an existing `clips_day1.npz` made `embed`/`index` skip the whole modality, so fixed clips were silently never embedded. This also fixes #43. |
| `load_dense_caches(scope=...)` | Drops cached rows outside the current scope. Still raises `KeyError` for stale ids. | Without this, an ego-scope `index --day 1` run after the fixed ingest would fail with `KeyError` on the fixed ids. |
| `build_filter(participant_includes_fixed=True)`, used by `_dense_search` | Participant becomes a top-level `should`: `participant_id == X OR camera_type == "fixed"` | Stops the participant hint from filtering out fixed cameras (see §0). On an ego-only index the results are the same as before. |
| `retrieve(known_participants=cfg.dataset.ego_cameras)` | Drops the dense participant filter when the name is not an indexed ego camera (for example "Bao", who has no day-1 ego stream) | Otherwise such questions match zero ego points. BM25 still uses the name as a soft bonus. |
| Router | `"reading area"` maps to `Reading` | The coat-of-arms question says "reading area". This is only a soft BM25 hint. |
| Reranker pack | Adds a `Room: <room> (fixed room camera)` line when `room` is set | Fixed-camera packs used to read "Participant: N/A" with no location. Ego prompts are byte-identical. |
| UI `padding_roster` | Pads with the fixed cameras too when `camera_scope == "all"` | All 5 fixed cameras are in `youtube_mirror.csv` for day 1 (Kitchen, Living1, Living2 and Meeting have 13 h each; Reading has 10 h), so they embed. |
| `scripts/qdrant_camera_counts.py` | Read-only count of points per `camera_type` and per camera × `source_type`, plus the list of missing fixed cameras | Checks the index state before and after the ingest |
| `scripts/slurm/fixedcams_day1.slurm` | One job: state check, backup and snapshot, preprocess (camera × hour workers), embed per modality, additive index, verify | See §3 |
| `scripts/compare_bugb_eval.py` | Scores a smoke run against the 8 questions from #50, including whether the CSV anchor camera appears in the evidence | See §5 |
| `smoke_day1_roomfix.slurm`, `_smoke_wandb.slurm`, `ui_live.slurm` | `CONF` can be overridden with `--export` | Lets these jobs run with the fixed-camera config |

Point ids stay deterministic: `uuid5(version|source_type|record_id|modality)`.
Fixed-camera record ids contain the camera name (`day1_Kitchen_08_0000`), so
they never collide with ego ids. Re-upserting an ego point overwrites it with
identical data. Fixed-camera payloads carry `camera_type="fixed"`,
`room=<camera>` and no `participant_id`. Ego payloads keep `room=None`, as #53
assumed.

## 2. Step 1: find out what state the index is in (CPU only, ~10 min)

```bash
ssh <user>@snellius.surf.nl
cd ~/CastleRAG && git fetch && git checkout feat/fixedcams-reingest
source ~/castlerag_venv/bin/activate && pip install -e . -q    # editable; picks up the branch

ls -d /scratch-shared/$USER/qdrant_storage/storage/collections/castle_multimodal_v1 \
      /scratch-shared/$USER/castle_derived/chunks/day1/* 2>&1 | head -30
ls -d ~/castle_archives/bugb_day1_* 2>/dev/null
ls /scratch-shared/$USER/castle2024/main/day1/{Kitchen,Living1,Living2,Meeting,Reading}/video/ 2>&1 | head
```

Then count points. This needs Qdrant running, so use a short CPU allocation.
The `rome` partition name is a guess; use whichever CPU partition your budget
allows.

```bash
srun --account=gpuuva082 --partition=rome --ntasks=1 --cpus-per-task=4 \
     --mem=16G --time=00:30:00 --pty bash
module purge; module load 2024 Python/3.12.3-GCCcore-13.3.0
source ~/castlerag_venv/bin/activate; cd ~/CastleRAG
QDRANT__STORAGE__STORAGE_PATH=/scratch-shared/$USER/qdrant_storage/storage ~/qdrant/qdrant > /tmp/q.log 2>&1 &
sleep 15
python scripts/qdrant_camera_counts.py --day day1
kill %1
```

What to do next:

| Result | Action |
|---|---|
| Collection missing or < 25k points, archive exists | Restore it: `cd ~/castle_archives/bugb_day1_<date> && cat RESTORE.md`. Follow those steps, then run Step 1 again. |
| Collection missing, no archive | Rebuild ego first (`full_day1.slurm` / `resume2_day1.slurm`, ~20 h+), then continue here with `CAMS=auto`. |
| All 5 fixed cameras have `main_clip` points | **No ingest needed.** Go to §5a. The participant-filter fix is the change being tested. |
| Some fixed cameras show `0 main_clip` (probably Reading) | Ingest only those cameras (§3). `CAMS=auto` selects them. |
| Fixed cameras present but `WARN ... room` lines | Old fixed points without `room`. Re-ingest those cameras with `CAMS="..."` to rewrite the payloads. The point ids are the same. |

## 3. Step 2: additive ingest (only the missing cameras)

One job on 3 A100s. It runs two Qwen3-VL servers for caption and events, and
OmniEmbed for the embed stage. It never passes `--create-collection`.

```bash
cd ~/CastleRAG
# default CAMS=auto -> only fixed cams with 0 main_clip points; exits 0 if none are missing
JOB=$(sbatch --parsable --account=gpuuva082 scripts/slurm/fixedcams_day1.slurm)
# or explicitly:
# JOB=$(sbatch --parsable --account=gpuuva082 --export=ALL,CAMS="Reading" scripts/slurm/fixedcams_day1.slurm)
tail -f logs/castle-fixedcams-day1_${JOB}.out
```

What the job does, in order (from `scripts/slurm/fixedcams_day1.slurm`):

0. Starts Qdrant on the existing storage. Aborts if there are fewer than 25,000
   points. Prints the per-camera counts to `logs/fixed_counts_before_<job>.txt`
   and resolves `CAMS`. Checks the camera names with `preprocess --dry-run`.
   Checks that raw day-1 video exists for each camera.
1. Rollback material. Copies `*_day1.npz`, `manifest_day1.json` and
   `transcripts.pkl` to
   `castle_derived/embeddings_backup_pre_fixedcams_<job>/`. Takes a Qdrant
   snapshot into `/scratch-shared/$USER/qdrant_snapshots/`. Records sha1
   checksums of every ego chunk file.
2. Preprocess: base, caption and events. Each worker handles one camera and a
   group of hours. With `SPLIT=auto` there are about 10 workers in total, so
   Reading alone gets 10 workers. Each worker runs:
   ```bash
   VLLM_BASE_URL=http://localhost:820{1,2}/v1 castlerag preprocess \
       --config configs/snellius_fixedcams.yaml --day 1 \
       --camera <CAM> --hour <H> [--hour <H2> ...] --caption --events
   ```
3. Validation. Each camera must have ≥98 % of clips captioned,
   `camera_type == {"fixed"}`, `room == {<CAM>}`, events present, and no
   corrupt lines. **The ego chunk checksums must be unchanged.**
4. Embed, one command per modality. Only new record ids are embedded:
   ```bash
   VLLM_BASE_URL=http://localhost:8200/v1 castlerag embed --config configs/snellius_fixedcams.yaml --day 1 --modality transcript
   VLLM_BASE_URL=http://localhost:8200/v1 castlerag embed --config configs/snellius_fixedcams.yaml --day 1 --modality event_summary
   VLLM_BASE_URL=http://localhost:8200/v1 castlerag embed --config configs/snellius_fixedcams.yaml --day 1 --modality video
   ```
5. Additive index. This also rebuilds BM25 `transcripts.pkl` with the
   fixed-camera transcripts. Re-upserting the ~26–39k ego points takes a few
   minutes; they are overwritten with identical data.
   ```bash
   castlerag index --config configs/snellius_fixedcams.yaml --day 1      # NO --create-collection
   ```
6. Verification. Writes `logs/fixed_counts_after_<job>.txt`. The job fails if
   any requested camera still has 0 `main_clip` points.

Knobs you can pass with `--export=ALL,...`:

- `CAMS`
- `SKIP_BASE=1`: frames and chunks already exist
- `SPLIT` / `TARGET_WORKERS`
- `HOURS` (default `8..20`)
- `SNAPSHOT=0`
- `MIN_POINTS`
- `CONF`

Do **not** run `--aux` for this. Aux data is per participant and
`chunks/aux.jsonl` is shared.

### Running the stages by hand (interactive, 3-GPU `srun`)

Same stages. Start the servers as in `full_day1.slurm`: gen on :8201/:8202,
OmniEmbed on :8200, and Qdrant on the scratch storage.

```bash
C=configs/snellius_fixedcams.yaml
castlerag preprocess --config $C --day 1 --camera Reading --dry-run          # scope check
VLLM_BASE_URL=http://localhost:8201/v1 castlerag preprocess --config $C --day 1 --camera Reading --hour 8 --hour 10 --hour 12 --hour 14 --hour 16 --hour 18 --hour 20 --caption --events &
VLLM_BASE_URL=http://localhost:8202/v1 castlerag preprocess --config $C --day 1 --camera Reading --hour 9 --hour 11 --hour 13 --hour 15 --hour 17 --hour 19 --caption --events &
wait
for m in transcript event_summary video; do
  VLLM_BASE_URL=http://localhost:8200/v1 castlerag embed --config $C --day 1 --modality $m
done
castlerag index --config $C --day 1
python scripts/qdrant_camera_counts.py --day day1
```

## 4. Compute estimate

All figures are estimates.

- **Size of day 1.** From `youtube_mirror.csv` there are 62 fixed-camera hours
  (Kitchen, Living1, Living2 and Meeting 13 h each; Reading 10 h). The ego
  cameras have 119. At 30 s windows that is about 7.4k fixed clips and about
  1.9k events. The June run added 13,512 points, which matches.
- **Main cost.** Captioning dominates. Each clip needs 3 Qwen3-VL calls
  (caption, OCR, scene graph).
- **What the ego ingest took.** `full_day1.slurm` (3 GPUs, 10 parallel
  cameras) ran out its 12 h limit with only 6 of 10 cameras captioned. It then
  needed `resume_day1`/`resume2_day1` jobs of up to 20 h. That puts the ego
  ingest at roughly 20–30 h of wall time for 119 camera-hours. PR #58 says the
  fixed-camera ingest "is ~12 h".
- **All 5 fixed cameras** (62 camera-hours): about **10–13 h** of wall time on
  3 A100s. That is 30–39 GPU-h, or about **3.8k–5k SBU** at 128 SBU/GPU-h.
- **Reading only** (10 camera-hours, 10 hour-workers): about **2–4 h** of wall
  time, or about 0.8k–1.5k SBU.
- **Embed and index.** Embedding about 10k new text payloads takes under 30
  min. The index upsert takes 10–20 min on scratch storage.
- **Scratch space.** Ego frames used about 330 G, so expect about 170 G more
  for all 5 fixed cameras (about 30 G for Reading). Check with `myquota` first.

The job's `--time=20:00:00` covers the 5-camera case with margin.

## 5. Eval: the 40-question day-1 set

The smoke output directory is fixed
(`/scratch-shared/$USER/castle_outputs/smoke_test/`) and gets overwritten.
**Copy it after every run.**

### 5a. Participant-filter fix only, on the current index (run first)

```bash
J=$(sbatch --parsable --account=gpuuva082 \
      --export=ALL,CONF=configs/snellius_fixedcams.yaml scripts/slurm/smoke_day1_roomfix.slurm)
# after it finishes:
cp -r /scratch-shared/$USER/castle_outputs/smoke_test /scratch-shared/$USER/castle_outputs/smoke_test_participantfix_$J
```

For the W&B-logged variant with the CSV (which has anchors), use
`--export=ALL,N=40,CONF=configs/snellius_fixedcams.yaml scripts/slurm/_smoke_wandb.slurm`.
The retrieval code does not filter by camera scope at query time, so
`snellius_me.yaml` would also work for eval. Use `snellius_fixedcams.yaml` so
the UI padding roster matches.

### 5b. After the ingest (if §3 ran)

```bash
J2=$(sbatch --parsable --account=gpuuva082 --dependency=afterok:${JOB} \
       --export=ALL,CONF=configs/snellius_fixedcams.yaml scripts/slurm/smoke_day1_roomfix.slurm)
cp -r /scratch-shared/$USER/castle_outputs/smoke_test /scratch-shared/$USER/castle_outputs/smoke_test_fixedcams_$J2
```

### 5c. Compare against the #50 list

```bash
python scripts/compare_bugb_eval.py --questions data/smoke_day1.csv \
    --run-dir      /scratch-shared/$USER/castle_outputs/smoke_test_fixedcams_$J2 \
    --baseline-dir /scratch-shared/$USER/castle_outputs/smoke_test_participantfix_$J
```

The script prints overall accuracy, the zero-evidence count (the #50 baseline
was 8/40, and about 11/40 after #53), and how many questions cite at least one
room camera. For each of the 8 target questions it prints `EVID`/`ZERO`,
`OK`/`BAD`, the CSV anchor camera, and whether that anchor appears in the
final evidence.

What to expect, based on the anchors:

| #50 question | Anchor | Expected effect |
|---|---|---|
| fridge brand | Kitchen 12:16 | Needs Kitchen, with no participant in the question. Should already work if Kitchen is indexed. |
| Werner's apron logo | Kitchen 12:16 | Blocked by the participant filter until this branch |
| back of Werner's t-shirt | Kitchen 14:44:46 | Blocked by the participant filter until this branch |
| coat of arms, reading area | (time only, 17:30:50) | Probably needs **Reading**, the camera that was never cited |
| measuring-cup cupboard | Werner ego 17:4x | Ego camera. This is the modality/OCR gap (#50 part 2), not Bug B. |
| duck sculptures, painting medium | Allie ego | Ego camera. Modality gap, not Bug B. |
| buzzers cost | (time only) | Unclear. Probably speech or OCR. |

Post the `compare_bugb_eval.py` output as a comment on #50. Questions in the
"supported but wrong" bucket are outside #50's scope (see the issue comment).

## 6. After it works

- `bash scripts/archive_bugb.sh` on the login node while no job is running.
  The scratch purge will otherwise delete the new state within about 14 days.
- Use `configs/snellius_fixedcams.yaml` for every later `index` run. An `index`
  run under ego-only `snellius_me.yaml` rebuilds `transcripts.pkl` without the
  fixed-camera transcripts. It no longer fails, and the dense points survive.
  Consider switching `snellius_me.yaml` to `camera_scope: "all"` once this is
  validated.
- UI: `sbatch --account=gpuuva082 --export=ALL,CONF=configs/snellius_fixedcams.yaml scripts/slurm/ui_live.slurm`.

## 7. Rollback

- Embedding caches and BM25: copy
  `castle_derived/embeddings_backup_pre_fixedcams_<job>/*` back into
  `castle_derived/embeddings/`.
- Qdrant: restore the snapshot from `/scratch-shared/$USER/qdrant_snapshots/`
  (`PUT /collections/castle_multimodal_v1/snapshots/recover` with
  `{"location": "file:///.../<name>.snapshot"}`). As a quicker option, delete
  only the new points: `POST /collections/castle_multimodal_v1/points/delete`
  with the filter `{"must":[{"key":"camera_id","match":{"any":["Reading"]}}]}`.
- Chunks: the ego chunk files are checksum-verified as unchanged. Deleting
  `chunks/day1/<CAM>/` removes the new camera's chunks.

## 8. Open questions

- The June fixed-camera run was done with scripts that were never committed.
  Nobody knows why Reading never showed up in results. It may be missing
  (step 1 will show), or it may be present but outranked.
- How the June run got past the old "cache file exists, skip" behaviour is
  also unknown. Perhaps the caches were deleted first. If the current `*_day1.npz`
  files predate the fixed points, the incremental cache will simply re-embed
  them. That is correct but costs extra embed time.
- The dataset paths assume `configs/snellius_me.yaml`
  (`/scratch-shared/$USER/castle2024`, `.../castle_derived`) and
  `~/qdrant/qdrant`. The CPU partition name `rome` has not been checked.
