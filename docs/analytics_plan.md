# CastleRAG — Multimodal Analytics Component (Contribution 2)

**Status:** draft for team review · **Author:** initial draft · **Scope:** the "MM
analytics / widgeting" deliverable, designed against the *current merged UI*.

> Meeting-note item 2: *"Plan for widgeting and a strategy to satisfy the MM
> analytics component… what to omit (e.g. heart rate) and how to
> visualize/communicate what we keep, with justification. Deliverable: written
> description of visual items + structure + a new sketch."*

---

## 1. Where this fits

The merged UI is an **investigation view**: ask → claim → three synchronized
camera embeds → per-moment camera-match bar + cross-camera agreement heatmap +
pipeline funnel → human review/refine. That is *per-answer* analytics and it is
strong.

What's missing for contribution 2 is a **corpus-level multimodal analytics
view** — one that *communicates the dataset itself* (who, where, when, doing
what, across which modalities), not just the evidence behind one answer. That is
the gradable "MM analytics component."

**Proposal:** add a second top-level mode — **"Explore"** — beside the current
**"Investigate"** mode, sharing the same shell/theme. Explore is a Plotly
dashboard driven by **precomputed aggregates** (built offline), so it renders
with **no GPU/Qdrant/vLLM** — it works in the offline demo *and* live.

This also gives a clean division of labour: Investigate = the RAG product;
Explore = the analytics contribution. They share data (cameras, rooms, time,
transcripts) but not compute.

---

## 2. What we keep vs. omit (and why)

Decision axis: does the modality **communicate scene/activity understanding**
(the theme of this dataset) at a cost worth paying? Physiological and niche
sensor streams score low; scene/speech/activity streams score high.

| Modality | Decision | Justification |
|---|---|---|
| **Ego video** (11 streams) | **Keep — core** | The participants' POV; the spine of coverage/occupancy/activity. Already mirrored + indexed. |
| **Fixed-room video** (5) | **Keep — analytics only** | Excluded from *retrieval* (TAHAKOM: ego-only suffices for QA), but they give the stable spatial frame that makes room-occupancy and cross-view widgets legible. We already have their embeds. See §5. |
| **Transcripts / speech** | **Keep — core** | Talk-time, conversation density, per-room dialogue. Rich, cheap, already normalized. |
| **Activity segments** | **Keep — headline** | The viewer's `*_segments.csv` give per-camera, per-day **activity labels** (Cooking, Talking, Walking, Using Laptop, …) with timestamps + transcript. Ready-made activity timeline for all 16 streams × 4 days. |
| **OCR / captions / event summaries** | **Keep — support** | Already derived; feed keyword/topic and moment tooltips. No new cost. |
| **Aux photo** | **Light keep** | Cheap, visual — a timeline-linked thumbnail strip. Not a core analytic. |
| **Gaze** | **Defer (stretch)** | Attention is interesting but sparse and expensive to align to the scene frame. At most one showcase overlay later; not core. |
| **Thermal** | **Omit** | Low-res, niche, minimal scene/QA relevance; hard to fold into a coherent narrative. Novelty only. |
| **Heart rate** | **Omit** | Biometric time-series with weak link to "what happened," privacy optics, and little interpretive payoff as a chart. High effort, low signal — the clearest cut. |
| **Aux video** | **Fold in or omit** | Include in the coverage story *only if* cleanly timestamped; otherwise omit. |

**One-line rationale to put in the report:** *we visualize the modalities that
describe the shared scene — video, speech, and activity — and omit
physiological (heart rate) and niche sensor (thermal) streams that add cost and
privacy exposure without advancing scene understanding; gaze is a documented
stretch goal.*

---

## 3. The widgets (Explore dashboard)

All Plotly. Each is driven by a small precomputed aggregate (see §4). Charts
reuse the UI's indigo/light theme (`assets/styles.css` + `figures.py` palette)
so Explore and Investigate read as one system.

1. **Coverage heatmap** — *"the shape of the dataset."*
   Camera (16 rows, ego then fixed) × hour (08–20), faceted/toggled by day.
   Cell = minutes of footage (0 = gap). Plotly `Heatmap`. Immediately shows the
   ego/fixed split, recording gaps (e.g. Tien days 1–3 → Bao day 4), and scale.

2. **Activity timeline** — *headline widget.*
   A Gantt/timeline (`Bar` with `base`, or timeline) of activity segments,
   colored by category, grouped by **room** (default) or **participant**, across
   the day. Straight from `*_segments.csv`. Communicates "what happened where,
   when" at a glance — the core multimodal story.

3. **Room occupancy over time** — stacked area / line of # participants present
   per room across the day, derived from ego-camera room assignment + fixed-cam
   presence. Answers "where were people when." This is where the **fixed
   cameras** earn their keep.

4. **Talk-time & conversation density** — bar of total talk-time per participant
   + a density strip over the day, from transcripts. Ties speech to people/time.

5. **Modality coverage strip** — a compact, honest bar/row of which modalities
   exist and how much, with the omitted ones greyed and labelled "excluded
   (see plan)." Doubles as the in-product justification of §2.

**Per-answer analytics (already built — keep, lightly extend):**
`camera_match_figure` (bar + agreement heatmap) and `pipeline_funnel_figure`
stay. Add a small **route + per-choice support-prior** bar (data already flows
as `route` + `support_priors`) so the Investigate side also reads as "analytics."

---

## 4. Data & build

- **Aggregates are precomputed offline** into a few small JSON/parquet files
  (e.g. `data/analytics/{coverage,activity,occupancy,speech}.json`), built by a
  script (`scripts/build_analytics.py`) from: the YouTube mirror CSV (coverage),
  the `*_segments.csv` (activity + transcript), and the normalized transcript
  windows (speech). The `description` column mixes clean labels with verbose VLM
  captions, so the builder canonicalizes the category (prefer the bolded
  `**Category**` / short label; bucket the rest as "Other").
- Explore reads only these aggregates → **no live backend**, fast, demo-safe.
- New Plotly builders live in `figures.py` (or `figures_explore.py`); the Explore
  layout is a new module wired into `app.py` behind the mode toggle.

---

## 5. Structure / integration

- **Top-bar mode toggle:** `Investigate` (current) ⇄ `Explore` (new).
- Explore layout: a responsive grid — coverage heatmap + modality strip on top,
  activity timeline full-width (the hero), occupancy + talk-time below. Filters:
  day selector, room/participant grouping toggle. Clicking an activity segment or
  occupancy slice can **deep-link into Investigate** (prefill a question / focus
  a moment) — the analytics feed the QA, closing the loop.
- Reuses `MantineProvider` theme + `styles.css` structural classes; no new deps
  beyond what's installed (dash, plotly, dash-mantine-components).

---

## 6. Phasing

1. **P1 (core):** `build_analytics.py` + coverage heatmap + activity timeline +
   the mode toggle. This alone satisfies the MM-analytics requirement.
2. **P2:** occupancy + talk-time + modality strip + Investigate deep-linking.
3. **P3 (stretch):** gaze attention overlay; photo thumbnail strip.

---

## 7. Open decisions for the team

- **Grouping default** for the activity timeline: by **room** or by
  **participant**? (Recommend room — reads as a floor-plan story.)
- **Borderline modalities:** confirm **omit thermal**, **defer gaze**, and
  whether **aux photo** is worth the light-keep.
- **Fixed cameras in Explore:** confirmed useful for occupancy/coverage even
  though excluded from retrieval — OK to source them from the mirror + segments
  without ingesting their video into the index?
- **Does the segment CSV activity taxonomy** (Cooking, Talking, Walking, Using
  Laptop, Leisure Activities, Playing Games, …) match how we want to present
  activities, or do we collapse it to a smaller set?
