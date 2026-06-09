# CastleRAG Technical Specification

## Scope

CastleRAG is a multimodal RAG system for verifiable multiple-choice question answering over the CASTLE 2024 dataset, targeting the CASTLE Challenge @ EgoVis 2026 benchmark. The initial goal is a working offline pipeline that:

1. reads the official CASTLE dataset layout from Hugging Face,
2. slices long videos into short clips and extracts representative frames,
3. runs offline frame captioning and OCR, then compresses clip evidence into searchable event summaries,
4. normalizes transcripts into short timestamped windows for lexical retrieval,
5. embeds dense multimodal evidence with `Tevatron/OmniEmbed-v0.1-multivent`,
6. indexes dense evidence in Qdrant with filterable payload while keeping transcripts in a separate lexical index,
7. routes each question into static visual, speech/text, temporal, or mixed handling,
8. reranks route-specific evidence packs and generates one of the four answer choices with citations,
9. optionally adapts the answer model with LoRA on CASTLE QA training data,
10. exports predictions and computes accuracy when a ground-truth answer file is available.

This spec still targets a pragmatic end-to-end baseline, but after reading the WDL and MARS reports it now assumes that evidence construction is the main engineering problem. Raw CASTLE video is too large to feed directly into model context at inference time, so the system must build an offline evidence memory first and only expand a small number of candidate videos into frame packs per question.

## Source Constraints

- CASTLE 2024 main data is organized as `main/day{1..4}/{camera}/{video,transcript,metadata}`. Each hour is a separate file such as `08.mp4`, `08.json`, and `08.*.csv`. Missing hours may appear as `.novideo`. The dataset paper states that videos are one-hour, time-aligned segments and that recording gaps inside otherwise present hours are padded with a test-card placeholder.
- CASTLE 2024 auxiliary data is organized separately under `auxiliary/{gaze,heartrate,photo,thermal,video}`.
- Official transcripts are JSON files with `chunks`, each containing `[start, end]` timestamps and text. Timestamps are relative to the enclosing hour.
- OmniEmbed-multivent is a shared embedding model across text, audio, image, and video, built on Qwen2.5-Omni-7B. The model card shows text queries formatted as `Query: ...` and raw media passed through the Qwen Omni processor.
- The Tevatron 2.0 paper recommends `vLLM` for retriever inference efficiency and reports about 3x faster encoding than a standard Transformers stack in both text and multimodal settings.
- Qdrant supports JSON payload, payload filtering, and payload indexes. Fields used for filtering should be indexed explicitly.
- The CASTLE Codabench page describes the question file as a JSON object keyed by question id, with a `query` string and four answer options under `answers`. Submission output is a JSON mapping from question id to `a|b|c|d`. Accuracy is exact-match over all questions.
- WDL, the first-place 2026 system, routes questions into four types: static visual, speech/text, temporal, and mixed. It uses up to 30 transcript chunks, up to 4 candidate videos, 32 frames per candidate video, and up to 16 auxiliary images per question as its high-cost evidence budget.
- MARS, the second-place 2026 system, explicitly converts long raw videos into captions, OCR notes, and compressed summaries offline because direct long-video prompting is infeasible under context and cost limits.

## 1. Repository Structure

Proposed repository layout:

- `README.md`: replace the placeholder Dash text with project setup, dataset expectations, and pipeline commands.
- `SPEC.md`: this spec.
- `pyproject.toml`: package metadata and dependencies.
- `configs/base.yaml`: default local configuration.
- `configs/snellius.yaml`: Snellius-specific paths, SLURM defaults, and Qdrant settings.
- `scripts/slurm/`: batch templates for preprocessing, embedding, indexing, reranking, and evaluation.
- `data/manifests/`: generated manifests for discovered CASTLE assets and derived chunks.
- `data/derived/`: chunk JSONL or Parquet outputs, keyframes, clip manifests, and optional visual summaries.
- `src/castlerag/config.py`: Pydantic config models and config loader.
- `src/castlerag/cli.py`: Typer CLI entrypoint.
- `src/castlerag/dataset/layout.py`: CASTLE path discovery, naming rules, and camera metadata.
- `src/castlerag/dataset/transcripts.py`: transcript JSON parsing and absolute timestamp alignment.
- `src/castlerag/dataset/metadata.py`: hourly sensor CSV loaders from `main/.../metadata`.
- `src/castlerag/preprocess/windows.py`: sliding-window creation for main video chunks.
- `src/castlerag/preprocess/media.py`: ffmpeg-based subclip and keyframe extraction.
- `src/castlerag/preprocess/caption_ocr.py`: offline frame captioning and OCR over representative clip frames.
- `src/castlerag/preprocess/event_compress.py`: compression of adjacent clip evidence into searchable event summaries.
- `src/castlerag/preprocess/auxiliary.py`: photo, auxiliary video, thermal, heartrate, and gaze normalization.
- `src/castlerag/preprocess/visual_summary.py`: offline visual summaries for chunks using the selected open-weight VL model.
- `src/castlerag/schemas.py`: shared typed models for chunk records, retrieval hits, rerank results, and eval items.
- `src/castlerag/embed/omniembed.py`: OmniEmbed processor and batch inference wrappers.
- `src/castlerag/routing/question_router.py`: hint extraction and routing into static visual, speech/text, temporal, and mixed paths.
- `src/castlerag/index/qdrant.py`: collection creation, payload indexes, deterministic ids, and batched upserts.
- `src/castlerag/index/transcript_lexical.py`: transcript lexical index creation and query-time scoring.
- `src/castlerag/retrieval/search.py`: query encoding, modality-scoped Qdrant search, candidate collapse, and score fusion.
- `src/castlerag/retrieval/filters.py`: day, camera, participant, room, time range, and modality filters.
- `src/castlerag/retrieval/transcript_lexical.py`: transcript lexical retrieval and bonus scoring over day/person/room/option overlap.
- `src/castlerag/retrieval/candidate_expand.py`: expansion from candidate videos to frame packs and linked evidence packs.
- `src/castlerag/rerank/llm_reranker.py`: local LLM-as-reranker prompts and scoring.
- `src/castlerag/training/lora_mcqa.py`: LoRA fine-tuning and held-out evaluation for CASTLE multiple-choice answering.
- `src/castlerag/generation/answer.py`: answer generation prompt, citation formatting, and answer extraction.
- `src/castlerag/eval/io.py`: loaders for official questions, local answer keys, and submission export.
- `src/castlerag/eval/run_eval.py`: full benchmark loop and accuracy computation.
- `tests/`: unit and integration tests.

## 2. Data Preprocessing Pipeline

### 2.1 Raw Inputs

Inputs come from two branches:

- Main branch:
  - `main/day{1..4}/{camera}/video/{HH}.mp4`
  - `main/day{1..4}/{camera}/transcript/{HH}.json`
  - `main/day{1..4}/{camera}/metadata/{HH}.*.csv`
- Auxiliary branch:
  - `auxiliary/heartrate/{participant}/...`
  - `auxiliary/gaze/*.csv`
  - `auxiliary/photo/{participant}/*`
  - `auxiliary/thermal/*`
  - `auxiliary/video/{participant}/*`

### 2.2 Canonical Time Model

All derived records must use:

- `day`: `day1` to `day4`
- `hour`: integer 8 to 20 from the source filename
- `start_seconds` and `end_seconds`: offsets within the hour
- `absolute_start` and `absolute_end`: `day + hour + offset`
- `camera_id`: exact folder name, e.g. `Allie`, `Kitchen`, `Living1`
- `camera_type`: `ego` for participant cameras, `fixed` for room cameras
- `participant_id`: participant name for ego cameras, null for fixed cameras

This avoids the `rainrag` assumption that each source file is already a single document-level retrieval unit.

### 2.3 Main Video Windowing

Primary clip units:

- window size: 30 seconds
- stride: 15 seconds
- overlap: 15 seconds

Reasoning:

- 30 seconds is short enough for tractable offline captioning, OCR, and dense video embedding.
- 15-second stride reduces boundary misses for events that cross window edges.
- At 600 hours total, this yields about `600 * 3600 / 15 = 144,000` main clips before filtering.

Filtering rules:

- skip `.novideo` hours completely
- mark windows as `is_placeholder=true` when more than 80% of sampled frames match the CASTLE test-card placeholder
- do not process placeholder windows into the main evidence memory
- keep clips with no transcript if real video exists; these remain visual-only evidence

### 2.4 Transcript Alignment

Transcripts are not kept as one-hour blobs. They are normalized into short lexical retrieval units.

For each hourly transcript JSON:

1. parse `chunks[*].timestamp` and `chunks[*].text`
2. convert timestamps from hour-relative to absolute
3. merge adjacent ASR rows into short utterance windows with two caps:
   - max 15 seconds span
   - max 96 token-equivalent text length
4. preserve raw segment boundaries inside each utterance window
5. store each window with day, stream, time, room, and participant metadata for lexical search

Derived transcript fields per utterance window:

- `transcript_window_id`
- `transcript_text`
- `transcript_segments`
- `speaker_hint`
- `has_speech`
- `transcript_char_len`
- `absolute_start`
- `absolute_end`

### 2.5 Keyframe Sampling

For each retained 30-second main clip:

- extract 8 JPEG keyframes at uniform offsets
- default offsets: `0s, 4s, 8s, 12s, 16s, 20s, 24s, 28s`
- store them under `data/derived/keyframes/{day}/{camera}/{hour}/{clip_id}/`

These keyframes are used for:

- offline captioning
- OCR when text or screens are visible
- debugging and manual evidence inspection
- future frontend playback previews

### 2.6 Subclip Extraction

For each retained main clip:

- extract a 30-second MP4 subclip with audio
- keep original frame rate for archival traceability
- create a retrieval copy when needed for efficient video embedding

Stored paths:

- `source_video_path`
- `retrieval_clip_path`
- `keyframe_paths`

### 2.7 Offline Captioning, OCR, and Event Compression

The offline pipeline must do more than chunk and embed. Following MARS, CastleRAG builds an evidence memory before question answering.

Per 30-second clip:

- input: 8 keyframes plus the transcript text if present
- generate frame-level or clip-level captions emphasizing:
  - people
  - objects
  - actions
  - room cues
  - visible text and screens
- run OCR on frames where text is detected or likely
- produce a compact clip note:
  - `clip_caption`
  - `ocr_text`
  - `transcript_text`
  - `caption_confidence`

Per event-summary window:

- group 4 adjacent clips into a 2-minute event-summary block
- compress the 4 clip notes and aligned transcript windows into an `event_summary`
- use a local reasoning/summarization model for this compression stage
- MARS used DeepSeek; CastleRAG should implement the same stage with an open-weight local summarizer unless policy later allows a hosted model

The event-summary block is the main text artifact used for dense retrieval over long videos. Raw clips remain available for later candidate expansion.

### 2.8 Auxiliary Modality Handling

Auxiliary data is normalized into standalone retrievable records plus optional links back to nearby main clips and event-summary windows. This issue is optional in the first milestone, but the payload schema must be forward-compatible now.

#### Heartrate

- create 60-second summary records per participant
- fields: `bpm_mean`, `bpm_min`, `bpm_max`, `bpm_delta_prev`
- create a text rendering such as `Heartrate for Allie at day2 14:03-14:04: mean 92 bpm, rising from 86 bpm`
- embed as text modality

#### Gaze

- parse each participant CSV
- create 10-second summary records only for intervals with gaze rows
- keep simple first-pass features: row count, mean x/y, std x/y, valid sample ratio
- create a text rendering such as `Gaze session for Bjorn at day1 10:15:00-10:15:10: stable fixation around center-left`
- embed as text modality

This is intentionally simple because the exact gaze columns must be confirmed against the raw files.

#### Photos

- one record per image
- extract timestamp from EXIF when present, otherwise fall back to filename time pattern
- store original path, optional OCR text, and a short visual note
- embed as image modality

#### Thermal

- one record per BMP image
- use file order plus any available metadata for timestamping
- attach an optional thermal note describing visible hot/cold regions
- embed as image modality

#### Auxiliary Video

- one record per video file if duration <= 30 seconds
- otherwise re-window into 30-second clips with 15-second stride
- embed as video modality

### 2.9 Output Format Per Chunk

Main clip records are written as JSONL or Parquet with at least:

- `clip_id`
- `parent_source_id`
- `source_type`
- `modality`
- `day`
- `hour`
- `camera_id`
- `camera_type`
- `participant_id`
- `room`
- `start_seconds`
- `end_seconds`
- `absolute_start`
- `absolute_end`
- `source_video_path`
- `retrieval_clip_path`
- `keyframe_paths`
- `transcript_text`
- `ocr_text`
- `clip_caption`
- `event_summary_id`
- `has_speech`
- `is_placeholder`
- `linked_aux_ids`
- `version`

Event-summary records are written separately with at least:

- `event_summary_id`
- `source_type=main_event_summary`
- `day`
- `camera_id`
- `camera_type`
- `participant_id`
- `room`
- `absolute_start`
- `absolute_end`
- `member_clip_ids`
- `member_transcript_window_ids`
- `event_summary`
- `aggregated_ocr_text`
- `linked_aux_ids`
- `version`

Auxiliary records share the same top-level schema and add modality-specific payload:

- `aux_owner`
- `asset_path`
- `summary_text`
- `raw_features`
- `linked_main_clip_ids`
- `linked_event_summary_ids`

## 3. Indexing Pipeline

### 3.1 Points Written to Qdrant

CastleRAG uses two retrieval stores:

1. transcript lexical index:
   - stores transcript utterance windows only
   - optimized for exact and near-exact overlap with question text, answer choices, people, days, rooms, and temporal markers
2. Qdrant dense multimodal index:
   - stores clip video embeddings
   - stores event-summary text embeddings
   - stores dense auxiliary embeddings for photos, auxiliary videos, thermal, and optional textual auxiliary summaries

This is a deliberate design choice. WDL’s transcript path is lexical, not dense, and MARS also keeps transcripts as searchable text windows. For CASTLE QA, transcript retrieval must preserve exact names, counts, days, rooms, and answer-option wording, which makes a lexical path the safer primary choice.

Approximate point counts for the first build:

- main clip video points: about 144,000
- event-summary dense points: about 36,000
- auxiliary points: expected low tens of thousands
- total Qdrant points: roughly 180,000 to 220,000
- transcript lexical rows: roughly 200,000 to 300,000 depending on utterance-window density

### 3.2 OmniEmbed Batching Strategy

Batch separately by modality:

- event-summary text batches: 64 records
- image batches: 16 records
- video batches: 4 records
- audio batches: only if introduced later as standalone points

Implementation rules:

- use `vLLM` as the default OmniEmbed inference backend
- discover embedding dimensionality from the first successful batch and use it when creating the Qdrant collection
- keep one SLURM array shard per modality and day to simplify retries
- cache intermediate embeddings to disk before Qdrant upsert
- make point ids deterministic: `sha1(model_version + source_type + record_id + modality)`

### 3.3 Qdrant Collection and Payload Schema

Collection name:

- `castle_multimodal_v1`

Payload fields stored with every point:

- `point_id`
- `record_id`
- `parent_source_id`
- `source_type`
- `modality`
- `day`
- `hour`
- `camera_id`
- `camera_type`
- `participant_id`
- `room`
- `start_seconds`
- `end_seconds`
- `absolute_start`
- `absolute_end`
- `duration_seconds`
- `transcript_text`
- `event_summary`
- `clip_caption`
- `ocr_text`
- `asset_path`
- `keyframe_paths`
- `has_speech`
- `is_placeholder`
- `linked_aux_ids`
- `model_name`
- `model_revision`
- `build_id`

Create Qdrant payload indexes for:

- `day`
- `camera_id`
- `camera_type`
- `participant_id`
- `room`
- `modality`
- `source_type`
- `absolute_start`
- `absolute_end`
- `has_speech`

### 3.4 SLURM Job Structure

Jobs:

- `preprocess_main.slurm`: discover files, build 30-second clips, extract subclips and keyframes
- `caption_ocr.slurm`: run frame captioning and OCR on main clips
- `compress_events.slurm`: build 2-minute event summaries from adjacent clips
- `index_transcripts.slurm`: build the transcript lexical index from normalized utterance windows
- `preprocess_aux.slurm`: normalize heartrate, gaze, photo, thermal, and auxiliary video
- `embed_text.slurm`: event-summary and optional auxiliary text records
- `embed_video.slurm`: main-clip video points and auxiliary video points
- `embed_images.slurm`: photos and thermal images
- `index_qdrant.slurm`: collection creation, payload index creation, and batched upsert

Recommended Snellius partition:

- `gpu_a100`

Relevant official Snellius facts:

- `gpu_a100` exposes one GPU as a quarter-node job with 18 CPU cores, 120 GiB RAM, and a billing weight of 128 SBU.

### 3.5 Runtime and SBU Estimate

This section is an estimate, not a measured benchmark. The OmniEmbed model card does not publish CASTLE-scale A100 throughput, so the numbers below are planning assumptions to validate cluster budget.

Assumptions:

- 144,000 main clips
- 36,000 event summaries
- transcript lexical indexing is CPU-heavy but not the main GPU cost
- dense video embedding dominates runtime
- captioning/OCR plus event compression adds a substantial pre-index cost before embedding

Estimated runtime:

- transcript lexical indexing: 2 to 4 CPU-hours
- clip captioning/OCR: 40 to 80 GPU-hours total
- event compression: 20 to 40 GPU-hours total
- dense text and auxiliary embeddings: 2 to 6 GPU-hours total
- dense video embeddings: 100 to 140 GPU-hours total
- full offline evidence-memory build: 162 to 270 GPU-hours total
- on 8 concurrent A100 GPUs: about 21 to 34 wall-clock hours

Estimated Snellius cost:

- 1 A100 GPU-hour = 128 SBU
- 162 to 270 GPU-hours = 20,736 to 34,560 SBU
- official rate is EUR 15 per 1,000 SBU
- total offline build cost estimate = about EUR 311 to EUR 518

Add 10 to 20% headroom for retries, cold starts, and prompt/compression experiments.

## 4. Retrieval and Routing Pipeline

### 4.1 Query Input

The system accepts:

- question text
- four answer choices
- optional explicit filters:
  - `day`
  - `camera_id`
  - `participant_id`
  - `room`
  - `modality`
  - `time_range`

This is an explicit API contract. The first working pipeline should not rely on the LLM to infer those filters from free text.

### 4.2 Query Parsing and Question Routing

Before retrieval, the system runs a lightweight question router that:

- extracts hints:
  - day
  - person
  - room
  - visual/OCR cues
  - speech cues
  - temporal cues
- assigns exactly one route:
  - `static_visual`
  - `speech_text`
  - `temporal`
  - `mixed`

Routing is mandatory. WDL and MARS both report that a single prompt strategy is insufficient for CASTLE.

### 4.3 Transcript Retrieval Strategy

Transcript retrieval uses a separate lexical path rather than OmniEmbed dense search.

Justification:

- WDL explicitly reports lexical scoring over question words, answer options, and day/person/room bonuses.
- Transcript QA in CASTLE often depends on exact strings, quiz numbers, named entities, and room/day references.
- MARS also normalizes transcripts into short timestamped text windows rather than treating them as just another dense modality.

Scoring formula:

- base lexical score over transcript window text and query text
- add overlap score from answer options
- add phrase-match bonus
- add day bonus
- add person bonus
- add room bonus
- add temporal-keyword bonus

Return:

- top 30 transcript windows per question as the global cap
- fewer windows for static-visual routes when transcript evidence is clearly secondary

### 4.4 Dense Multimodal Retrieval

For dense retrieval, encode both:

- `Query: {question}`
- `Query: {question} Choices: A {a}. B {b}. C {c}. D {d}.`

Use the max score across the two query forms during fusion.

Run separate filtered Qdrant searches and fuse them:

- event summaries: top 20
- clip video points: top 20
- photos: top 16
- auxiliary videos: top 8
- heartrate summaries: top 8
- gaze summaries: top 8
- thermal images: top 8

Fusion:

- Reciprocal Rank Fusion with `k=60`

Then collapse hits into up to 4 candidate videos or candidate windows, matching WDL’s high-cost budget.

### 4.5 Filtering

Qdrant filters are applied server-side using payload indexes for:

- exact day
- exact camera
- exact participant
- exact modality
- room
- absolute time overlap

Time overlap rule:

- retrieve points where `absolute_end >= query_start` and `absolute_start <= query_end`

### 4.6 Evidence Budgets by Route

Hard starting-point caps, derived from WDL:

- transcript chunks: up to 30
- candidate videos: up to 4
- frames per candidate video: 32
- auxiliary images: up to 16

Route-specific behavior:

- `static_visual`
  - prioritize candidate videos and auxiliary images
  - expand each of up to 4 candidate videos into 32 sampled frames
  - run OCR-heavy prompting
- `speech_text`
  - prioritize up to 30 transcript chunks
  - use video evidence only when needed to disambiguate speakers, locations, or visible objects
- `temporal`
  - prioritize ordered transcript windows and adjacent event-summary windows
  - sample frames from candidate videos to verify before/after/while relations
- `mixed`
  - combine transcript, video, and auxiliary evidence within the same global caps

### 4.7 Retrieval Output Format

Each hit returned to the reranker contains:

- `rank`
- `score`
- `point_id`
- `record_id`
- `source_type`
- `modality`
- `day`
- `camera_id`
- `participant_id`
- `absolute_start`
- `absolute_end`
- `transcript_text`
- `event_summary`
- `ocr_text`
- `asset_path`

## 5. Reranking

### 5.1 Model Choice

Use one open-weight model for both reranking and generation.

Default:

- `Qwen2.5-VL-7B-Instruct`

Fallback:

- `InternVL2-8B`

The model is run locally on Snellius or an equivalent GPU host.

### 5.2 Candidate Representation

Reranking operates on route-aware evidence packs rather than isolated raw hits. Each pack may include transcript windows, event summaries, OCR spans, sampled frame descriptions, and auxiliary notes.

Format:

```text
Candidate pack {rank}
Route: {route}
Primary source: {source_type}
Day: {day}
Camera: {camera_id}
Participant: {participant_id or N/A}
Time: {absolute_start} to {absolute_end}
Transcript evidence:
{top_transcript_chunks}

Event summary:
{event_summary or "[not available]"}

OCR evidence:
{ocr_text or "[none]"}

Auxiliary evidence:
{aux_summary or "[none]"}
```

This pack structure matches the CASTLE evidence bottleneck reported by both WDL and MARS.

### 5.3 Reranker Prompt Template

```text
You are ranking a route-specific evidence pack for a multiple-choice CASTLE question.

Question:
{question}

Question route:
{route}

Answer choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Evidence pack:
{candidate_text}

Score this candidate on two axes:
1. Evidence relevance from 0 to 4
2. Support for each answer choice from 0 to 4

Return strict JSON:
{
  "relevance": 0-4,
  "support": {"a": 0-4, "b": 0-4, "c": 0-4, "d": 0-4},
  "keep": true|false,
  "rationale": "<<=40 words>"
}
```

### 5.4 Scoring Mechanism

For each candidate pack:

- parse JSON
- compute `final_rerank_score = 0.7 * relevance + 0.3 * max_support`
- discard candidates with `keep=false` or `relevance <= 1`
- retain top 4 candidate packs globally

Also compute answer priors:

- sum `support.a`, `support.b`, `support.c`, `support.d` across kept candidates

These priors are passed to generation as soft evidence, not as the final answer.

### 5.5 Feed Into Generation

Generation receives:

- the routed question type
- top 4 reranked evidence packs
- per-choice cumulative support scores
- the original retrieval scores

## 6. Generation

The default generator for the final pipeline is the LoRA-adapted variant of the selected open-weight model once issue work for LoRA fine-tuning is complete. The non-LoRA checkpoint remains the baseline.

### 6.1 Prompt Template

```text
You answer multiple-choice questions about the CASTLE dataset.

Rules:
- Use only the provided evidence.
- Prefer direct evidence over speculation.
- If evidence is weak, say so briefly but still choose the most supported option.
- Every factual claim used in the decision must cite at least one evidence item.
- Citations must use the format [camera={camera_id} time={day} {start}-{end}] or [aux={source_type} id={record_id}].
- Follow the route-specific instruction block exactly.
- End with exactly one line: FINAL_ANSWER: a|b|c|d

Question route:
{route}

Route-specific instructions:
{route_prompt_block}

Question:
{question}

Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Choice support priors:
{support_summary}

Evidence:
{top_reranked_evidence}
```

Route prompt blocks:

- `static_visual`
  - prioritize frames, OCR text, object counts, colors, brands, and room layout
- `speech_text`
  - prioritize transcript windows and exact spoken content
- `temporal`
  - reconstruct order using timestamps and neighboring evidence
- `mixed`
  - require agreement between transcript and visual evidence before preferring an option

### 6.2 Citation Format

Main video evidence:

- `[camera=Allie time=day2 14:03:00-14:03:30]`

Auxiliary evidence:

- `[aux=photo id=photo_day2_allie_00034]`
- `[aux=heartrate id=hr_day2_allie_1403]`

### 6.3 Handling the 4-Choice Format

The model sees the four options verbatim and must choose one of `a`, `b`, `c`, or `d`.

Post-processing:

- parse the last line with regex `FINAL_ANSWER:\s*([abcd])`
- if parsing fails, fall back to the highest support prior from reranking
- log the raw answer text for auditability

Training target for LoRA:

- the adapted model is optimized to emit only the final multiple-choice answer token sequence, not long free-form explanations
- held-out evaluation must report exact-match answer accuracy on a validation split

### 6.4 Evaluation Output

For challenge submission:

- write `submissions.json`
- format:

```json
{
  "2026_q1": "a",
  "2026_q2": "c"
}
```

## 7. Evaluation

### 7.1 Metric

- accuracy = `correct / total_questions`

No partial credit.

### 7.2 Official Question Loader

Support the official CASTLE question JSON shape:

```json
{
  "2026_q1": {
    "query": "Question text",
    "answers": {
      "a": "First answer",
      "b": "Second answer",
      "c": "Third answer",
      "d": "Fourth answer"
    }
  }
}
```

### 7.3 Local Evaluation Inputs

The eval runner should accept:

- `questions_path`: official question JSON
- `answers_path`: local answer key if available
- `predictions_path`: optional cached predictions
- `train_questions_path`: optional CASTLE QA training split for LoRA runs
- `val_questions_path`: optional held-out split for LoRA evaluation

If `answers_path` is missing:

- still run the full prediction pass
- export `submissions.json`
- do not claim an accuracy number

### 7.4 Full Eval Pass

Per question:

1. load question and options
2. retrieve candidates
3. route question type
4. rerank candidate evidence packs
5. generate final choice
6. save prediction plus evidence trace

Outputs:

- `outputs/predictions.json`
- `outputs/evidence_traces.jsonl`
- `outputs/submissions.json`
- `outputs/metrics.json` when ground truth exists

## 8. What Is Reusable From `rainrag`

### 8.1 Reusable With Minimal Changes

- `src/rainrag/config.py`
  - Reuse the hierarchical Pydantic config pattern.
  - Replace VTT-specific and provider-specific fields with CASTLE dataset, OmniEmbed, SLURM, and local VL model settings.

- `src/rainrag/cli.py`
  - Reuse the Typer command layout and lazy imports.
  - Replace commands with `preprocess`, `embed`, `index`, `retrieve`, `answer`, and `eval`.

- `src/rainrag/index.py`
  - Reuse Qdrant connection handling, deterministic point ids, collection creation, and batched upsert.
  - Extend payload schema and add payload-index creation.

- `src/rainrag` CLI/eval command separation
  - Reuse the pattern of a pipeline CLI plus a separate eval CLI.
  - Replace synthetic eval data generation with official CASTLE QA and LoRA train/val split handling.

- `tests/unit/test_config.py`, `tests/unit/test_index.py`, `tests/unit/test_cli.py`
  - Reuse testing style and expected failure-path coverage.

### 8.2 Reusable Conceptually But Not Directly

- `src/rainrag/query.py`
  - Reuse the orchestration shape: retrieve, rerank, prompt, answer.
  - Replace text-only assumptions, online API providers, and Cohere rerank with lexical transcript retrieval, dense multimodal retrieval, question routing, and local LLM reranking.

- `src/rainrag` hybrid BM25 path
  - The generic pattern is relevant, but the implementation is not.
  - Transcript retrieval in CastleRAG needs answer-option overlap and day/person/room bonuses inspired by WDL, not a generic BM25-only score.

- `src/rainrag/api.py`
  - Reuse request/response model ideas later if an API is exposed.
  - Do not port before the offline pipeline works.

### 8.3 Not Reusable / Direct Conflicts

- `src/rainrag/ingest.py`
  - Conflict: assumes each file becomes one text document or VTT chunk set.
  - CASTLE requires multimodal alignment across hour videos, transcripts, metadata CSVs, and auxiliary assets.

- `src/rainrag/embed.py`
  - Conflict: built around sentence-transformers and text embeddings.
  - CastleRAG needs OmniEmbed multimodal inference through `vLLM`, modality-specific batching, and offline event-memory indexing.

- web metadata, MCP server, Streamlit UI, and journalistic answer shaping
  - Irrelevant for the initial CastleRAG objective.

## 9. Open Questions and Risks

### 9.1 Dataset and Ground Truth

- It is unclear whether the local workspace will include an answer key for the 185 questions. Without it, the pipeline can export predictions but cannot compute accuracy offline.
- Auxiliary timestamp quality needs validation, especially for thermal and personal media.

### 9.2 Throughput and Storage

- OmniEmbed-multivent A100 throughput is not documented in the model card. Runtime and SBU estimates in this spec are planning numbers and must be replaced with measured benchmark logs.
- 8 keyframes per 144,000 windows produces roughly 1.15 million JPEGs. Storage and inode pressure need to be managed carefully.
- Offline captioning, OCR, and event compression are likely to dominate engineering complexity before dense indexing even starts.

### 9.3 Placeholder and Gap Detection

- The dataset includes placeholder padding inside hour videos. If detection is weak, the index will contain many useless windows and retrieval quality will degrade.

### 9.4 Gaze and Metadata Semantics

- The exact meaning of several hourly metadata files and gaze CSV columns is not established by the currently inspected source material. The first implementation should not overfit to guessed semantics.

### 9.5 Reranker Evidence Bottleneck

- The top CASTLE systems explicitly treated evidence selection as the bottleneck.
- WDL uses question-type routing, lexical ASR chunk retrieval, attaching auxiliary images, and candidate frame sampling with concrete budgets.
- MARS uses source selection across transcripts, video, gaze, heartrate, photos, and thermal, plus long-video compression through captioning, OCR, and summaries.
- This means a transcript-only or pure-dense-only port of `rainrag` is very likely to fail on visual, temporal, or auxiliary-modality questions even if the generation model is strong.

### 9.6 Scope Risk

- A full challenge-style agentic planner is out of scope for the first implementation.
- The intended first milestone is a dependable routed retrieval-plus-rerank baseline with citations, LoRA adaptation, and forward-compatible auxiliary support, not a full autonomous multi-agent planner.
