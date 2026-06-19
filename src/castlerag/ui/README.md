# CastleRAG UI (claim-verification workspace)

Dash dashboard for verifying one claim at a time against synchronized CASTLE
footage, running end to end on a stub engine **without** the RAG pipeline,
models, Qdrant, or vLLM.

## Run

```bash
pip install -e ".[ui]"
castlerag ui                 # http://127.0.0.1:8050
castlerag ui --debug --port 8060
```

## Layout

A two-column workspace that persists across the whole task:

* **Left — thread of query groups.** Each group shows the question, a short
  answer, the single **claim under review** with a support badge
  (partial → supported), and a ranked list of **evidence moments**. Sending a
  refined query appends a new group (`Refined · n/5`), capped at five
  iterations; the **Ask a new question** bar starts a fresh thread.
* **Right — pinned evidence viewer** for the focused moment: three
  **synchronized camera embeds** (the best ringed in the accent colour), a
  per-camera **confirm / refine / reject** review row with justification fields,
  and a compose box for sending a refined query. When the claim reaches
  *Supported*, a "Search converged" banner replaces the send affordance.

## Architecture

The UI is built with **Dash Mantine Components** (DMC): every visual surface —
top bar, cards, badges, buttons, paper, alerts, the loader — is a themed DMC
component (theme in `app.py:THEME`, wrapped once in a `dmc.MantineProvider`).
`assets/styles.css` is a thin structural layer only (page grid, two-column
layout, scrollable thread, camera/review grids, camera overlays, raw `dcc`
inputs). Value- and `n_submit`-bound text inputs stay as `dcc` controls; the one
chart is Plotly.

| Module | Role | Depends on Dash/DMC? |
|---|---|---|
| `youtube.py` | `(day, camera, hour)` → YouTube embed URL | no |
| `chat.py` | `ChatEngine` protocol + offline `PlaceholderEngine` | no |
| `figures.py` | Plotly per-camera match-score chart | Plotly only |
| `rag_engine.py` / `engine_factory.py` | live `RagEngine` + reachable-backend gating | no |
| `layout.py` / `callbacks.py` / `app.py` | DMC layout, wiring, app factory | yes (DMC) |

The chat layer is a `ChatEngine` protocol with two calls: `answer(question,
choices)` opens an investigation (a `Claim` + ranked `EvidenceMoment`s, each
moment seen from three synchronized `CameraAngle`s), and `refine(claim,
refined_query, iteration)` re-runs retrieval for the same claim as it converges.
The offline `PlaceholderEngine` returns deterministic, structurally valid turns.
The live `RagEngine` wraps `castlerag.eval.run_eval` and implements the same
protocol; `engine_factory.build_engine` selects it when the backend is reachable,
else falls back to the placeholder. `EvidenceRef` mirrors `schemas.RetrievalHit`
(plus `hour` / `start_seconds`) to keep that swap mechanical.

Only the right-column camera grid renders live iframes (at most three at once,
for the focused moment); the left-thread moment thumbnails are static.

## YouTube mirror mapping

The official CASTLE project mirrors every stream on YouTube (one video per
`day / camera / hour`), which the [CASTLE viewer](https://castle-dataset.github.io/castle-viewer/)
embeds. `youtube_mirror.csv` (666 rows: `day,camera,hour,video_id`) is generated
from that viewer's `videos.json`
([CASTLE-Dataset/CASTLE-Dataset.github.io](https://github.com/CASTLE-Dataset/CASTLE-Dataset.github.io)).

Streams: 11 ego (`Allie, Bao, Bjorn, Cathal, Florian, Klaus, Luca, Onanong,
Stevan, Tien, Werner`) + 5 fixed (`Kitchen, Living1, Living2, Meeting,
Reading`); days 1–4, hours 08–20 (with gaps where a stream didn't record). The
three synchronized cameras in the viewer (`Bjorn, Luca, Klaus`) are all real ego
streams, so each resolves at the same `(day, hour)`.

The embed uses `youtube-nocookie.com/embed/<id>?start=<seconds>`, matching the
viewer. Edit the CSV to add or correct rows — no code change. An unmapped triple
falls back to a placeholder video.
