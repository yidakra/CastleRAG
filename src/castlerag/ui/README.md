# CastleRAG UI (placeholder backbone)

Dash dashboard addressing meeting-note item 1: chat + YouTube evidence embeds +
a Plotly analytics shell, running end to end on a stub engine **without** the
RAG pipeline, models, Qdrant, or vLLM.

## Run

```bash
pip install -e ".[ui]"
castlerag ui                 # http://127.0.0.1:8050
castlerag ui --debug --port 8060
```

## Architecture

| Module | Role | Depends on Dash/Plotly? |
|---|---|---|
| `youtube.py` | `(day, camera, hour)` → YouTube embed URL | no |
| `chat.py` | `ChatEngine` protocol + offline `PlaceholderEngine` | no |
| `figures.py` | Plotly figure builders | plotly only |
| `layout.py` / `callbacks.py` / `app.py` | Dash layout, wiring, app factory | yes |

The chat layer is a `ChatEngine` protocol. The offline `PlaceholderEngine`
returns deterministic, structurally valid turns (route, support priors, stub
evidence). A future `RagEngine` wrapping `castlerag.eval.run_eval` implements the
same protocol and is injected via `build_app(engine=...)` — no layout/callback
changes. `EvidenceRef` mirrors `schemas.RetrievalHit` (plus `hour` /
`start_seconds`) to keep that swap mechanical.

## YouTube mirror mapping

The official CASTLE project mirrors every stream on YouTube (one video per
`day / camera / hour`), which the [CASTLE viewer](https://castle-dataset.github.io/castle-viewer/)
embeds. `youtube_mirror.csv` (666 rows: `day,camera,hour,video_id`) is generated
from that viewer's `videos.json`
([CASTLE-Dataset/CASTLE-Dataset.github.io](https://github.com/CASTLE-Dataset/CASTLE-Dataset.github.io)).

Streams: 11 ego (`Allie, Bao, Bjorn, Cathal, Florian, Klaus, Luca, Onanong,
Stevan, Tien, Werner`) + 5 fixed (`Kitchen, Living1, Living2, Meeting,
Reading`); days 1–4, hours 08–20 (with gaps where a stream didn't record).

The embed uses `youtube-nocookie.com/embed/<id>?start=<seconds>`, matching the
viewer. Edit the CSV to add or correct rows — no code change. An unmapped triple
falls back to a placeholder video and is flagged in the evidence caption.

> Note: the real stream names above differ from the placeholder roster in
> `configs/base.yaml` / `routing/question_router.py` (`Celine, Deon, …`); those
> belong to the RAG side and are out of scope for this UI backbone.
