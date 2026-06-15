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

No public YouTube mirror of CASTLE exists; the dataset ships as multi-TB UHD
video on HuggingFace. `youtube_mirror.csv` maps each `day,camera,hour` to a
YouTube `video_id`, seeded with an openly licensed placeholder
(`aqz-KE-bpKQ`, Big Buck Bunny) so embeds render immediately.

To wire the real mirror, edit `youtube_mirror.csv` and replace each `video_id`
with the team's upload — no code change. Unmapped triples fall back to the
placeholder and are flagged in the evidence caption. The embed seeks to the
clip offset via `?start=<seconds>`.
