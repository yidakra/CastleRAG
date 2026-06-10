"""Local smoke test: 5 synthetic questions through the full CastleRAG pipeline.

Usage
-----
# Offline stub mode (no models, no network — verifies pipeline wiring):
    python scripts/smoke_local.py

# Real-model mode via a local vLLM / Ollama OpenAI-compatible endpoint:
    VLLM_BASE_URL=http://localhost:11434/v1 python scripts/smoke_local.py --real

Outputs are written to outputs/smoke_local/.
Exit code 0 = all 5 predictions are valid a|b|c|d.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

_QUESTIONS = {
    "q001": {
        "query": "What colour shirt was Allie wearing in the kitchen on day 1?",
        "answers": {"a": "Blue", "b": "Red", "c": "White", "d": "Black"},
    },
    "q002": {
        "query": "What did Bjorn say to Celine before lunch on day 2?",
        "answers": {
            "a": "See you later",
            "b": "Good morning",
            "c": "Please pass the salt",
            "d": "I am leaving",
        },
    },
    "q003": {
        "query": "In what order did Deon visit the office and the kitchen on day 3?",
        "answers": {
            "a": "Office then kitchen",
            "b": "Kitchen then office",
            "c": "Only the office",
            "d": "Only the kitchen",
        },
    },
    "q004": {
        "query": "How many people were visible in the living room at 14:00 on day 1?",
        "answers": {"a": "One", "b": "Two", "c": "Three", "d": "Four"},
    },
    "q005": {
        "query": "What brand logo was visible on the screen Estella was looking at?",
        "answers": {"a": "Apple", "b": "Google", "c": "Microsoft", "d": "Samsung"},
    },
}

_TRANSCRIPT_TEXTS = [
    "Allie is wearing a blue shirt while preparing breakfast in the kitchen.",
    "Bjorn said to Celine: please pass the salt, before sitting down.",
    "Deon walked from the kitchen to the office and sat at his desk.",
    "Three people were gathered in the living room at two in the afternoon.",
    "Estella was looking at a screen with a large Apple logo on it.",
    "Finn and Greta discussed the schedule for the afternoon.",
    "Harvey washed his hands and returned to the living room.",
    "Isla took notes while sitting in the office.",
    "Jian opened the refrigerator and took out a bottle of water.",
    "The kitchen was empty after everyone went to the living room.",
]

# ---------------------------------------------------------------------------
# Stub clients (no model dependencies)
# ---------------------------------------------------------------------------

_STUB_DIM = 64
_RNG = np.random.default_rng(42)


class _StubEmbedClient:
    """Returns deterministic-ish random unit vectors — wiring test only."""

    dim: int = _STUB_DIM

    def _unit(self, n: int) -> np.ndarray:
        v = _RNG.standard_normal((n, self.dim)).astype(np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return v / norms

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self._unit(len(texts))

    def embed_images(self, paths: List[str]) -> np.ndarray:
        return self._unit(len(paths))

    def embed_videos(self, frame_lists: List[List[str]]) -> np.ndarray:
        return self._unit(len(frame_lists))


class _StubChoice:
    def __init__(self, text: str) -> None:
        self.message = type("M", (), {"content": text})()


class _StubCompletionResponse:
    def __init__(self, text: str) -> None:
        self.choices = [_StubChoice(text)]


class _StubCompletions:
    def create(self, *, model: str, messages: Any, **_: Any) -> _StubCompletionResponse:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        if "Score this candidate" in last_user:
            text = json.dumps(
                {
                    "relevance": 3,
                    "support": {"a": 3, "b": 1, "c": 1, "d": 1},
                    "keep": True,
                    "rationale": "Evidence directly addresses the question.",
                }
            )
        else:
            text = (
                "Based on the retrieved evidence, option A is best supported.\n"
                "FINAL_ANSWER: a"
            )
        return _StubCompletionResponse(text)


class _StubChat:
    def __init__(self) -> None:
        self.completions = _StubCompletions()


class _StubLLMClient:
    """Stub LLM shaped like an OpenAI client (chat.completions.create)."""

    def __init__(self) -> None:
        self.chat = _StubChat()


# ---------------------------------------------------------------------------
# OpenAI-compatible real client (for --real mode)
# ---------------------------------------------------------------------------


def _build_real_clients(
    vllm_url: str,
) -> tuple[Any, Any]:
    """Return (embed_client, llm_client) using a real vLLM/Ollama endpoint."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        sys.exit(
            "openai package required for --real mode: pip install openai\n"
            f"Original error: {exc}"
        )
    from castlerag.embed.omniembed import OmniEmbedClient

    embed = OmniEmbedClient(
        backend="vllm",
        vllm_base_url=vllm_url,
    )
    llm = OpenAI(base_url=vllm_url, api_key="not-needed")
    return embed, llm


# ---------------------------------------------------------------------------
# In-memory index
# ---------------------------------------------------------------------------


def _build_inmemory_index(
    embed_client: Any,
    dim: int,
    tmp_dir: Path,
) -> tuple[Any, Any]:
    """Return (bm25_bundle, qdrant_client) with synthetic records ingested."""
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm

    from castlerag.index.qdrant import build_point_batches, upsert_batch
    from castlerag.index.transcript_lexical import build_bm25_index
    from castlerag.schemas import (
        ClipRecord,
        EventSummaryRecord,
        TranscriptWindow,
    )

    # --- Transcript windows ---
    windows: List[TranscriptWindow] = []
    base_ms = 1_700_000_000_000
    rooms = ["Kitchen", "Living1", "Office", "Hallway", "Living2"]
    for i, text in enumerate(_TRANSCRIPT_TEXTS):
        day = f"day{(i % 4) + 1}"
        cam = ["Allie", "Bjorn", "Celine", "Deon", "Estella"][i % 5]
        windows.append(
            TranscriptWindow(
                transcript_window_id=f"tw_{i:04d}",
                source_type="transcript_window",
                modality="text",
                day=day,
                camera_id=cam,
                camera_type="ego",
                participant_id=cam,
                room=rooms[i % len(rooms)],
                hour=9 + i,
                window_index=i,
                absolute_start=base_ms + i * 15_000,
                absolute_end=base_ms + i * 15_000 + 15_000,
                transcript_text=text,
            )
        )

    # --- Clip records (5 stub clips) ---
    clips: List[ClipRecord] = []
    for i in range(5):
        day = f"day{(i % 4) + 1}"
        cam = ["Allie", "Bjorn", "Celine", "Deon", "Estella"][i]
        clips.append(
            ClipRecord(
                clip_id=f"clip_{i:04d}",
                parent_source_id=f"vid_{i}",
                source_type="main_clip",
                modality="video",
                day=day,
                hour=10 + i,
                camera_id=cam,
                camera_type="ego",
                participant_id=cam,
                start_seconds=0.0,
                end_seconds=30.0,
                absolute_start=base_ms + i * 30_000,
                absolute_end=base_ms + i * 30_000 + 30_000,
                source_video_path=f"/data/main/{day}/{cam}/video/10.mp4",
                event_summary=_TRANSCRIPT_TEXTS[i * 2],
            )
        )

    # --- Event summaries (3) ---
    events: List[EventSummaryRecord] = []
    for i in range(3):
        day = f"day{i + 1}"
        cam = ["Allie", "Bjorn", "Celine"][i]
        events.append(
            EventSummaryRecord(
                event_summary_id=f"ev_{i:04d}",
                source_type="main_event_summary",
                modality="text",
                day=day,
                camera_id=cam,
                camera_type="ego",
                absolute_start=base_ms + i * 120_000,
                absolute_end=base_ms + i * 120_000 + 120_000,
                member_clip_ids=[f"clip_{i:04d}"],
                event_summary=_TRANSCRIPT_TEXTS[i],
            )
        )

    # --- BM25 index (written to temp file) ---
    bm25_path = tmp_dir / "transcripts.pkl"
    bm25_bundle = build_bm25_index(windows, bm25_path)

    # --- In-memory Qdrant ---
    qdrant = QdrantClient(":memory:")
    collection = "smoke_local_v1"
    qdrant.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )

    all_records = [*windows, *clips, *events]
    point_rows = build_point_batches(
        all_records,
        model_version="smoke-stub",
        model_name="stub",
    )
    import uuid as _uuid

    # In-memory Qdrant requires UUID-format point IDs (real Qdrant accepts any string).
    # Convert SHA-1 hex IDs by truncating to 32 chars and formatting as UUID.
    raw_ids = [row.point_id for row in point_rows]
    uuid_ids = [str(_uuid.UUID(hex=pid[:32])) for pid in raw_ids]
    payloads = [row.model_dump(exclude_none=True) for row in point_rows]
    # Override point_id in payload to match the UUID we're actually using.
    for pay, uid in zip(payloads, uuid_ids):
        pay["point_id"] = uid

    vectors = embed_client.embed_texts(
        [r.transcript_text if hasattr(r, "transcript_text") else "stub"
         for r in all_records]
    ).tolist()
    upsert_batch(
        client=qdrant,
        collection_name=collection,
        point_ids=uuid_ids,
        vectors=vectors,
        payloads=payloads,
    )

    return bm25_bundle, qdrant, collection


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


def _build_pipeline(
    embed_client: Any,
    llm_client: Any,
    bm25_bundle: Any,
    qdrant_client: Any,
    collection: str,
) -> Any:
    from castlerag.config import CastleRAGConfig
    from castlerag.eval.run_eval import EvalPipeline
    from castlerag.generation.answer import generate_answer
    from castlerag.rerank.llm_reranker import rerank_candidates
    from castlerag.retrieval.search import retrieve as _retrieve
    from castlerag.routing.question_router import route_question
    from castlerag.schemas import EvalQuestion, Prediction, RetrievalHit

    cfg = CastleRAGConfig()

    def retrieve(question: EvalQuestion, hints: Any) -> List[RetrievalHit]:
        return _retrieve(
            question=question,
            hints=hints,
            qdrant_client=qdrant_client,
            collection_name=collection,
            bm25_index=bm25_bundle,
            embed_client=embed_client,
            retrieval_cfg=cfg.retrieval,
        )

    def rerank(question: EvalQuestion, hints: Any, packs: List[dict]) -> Any:
        return rerank_candidates(
            question=question,
            hints=hints,
            candidate_packs=packs,
            llm_client=llm_client,
            top_k=cfg.reranking.top_k,
            min_relevance=0,
            model="stub",
        )

    def generate(
        question: EvalQuestion,
        hints: Any,
        evidence: List[RetrievalHit],
        support: Dict[str, float],
    ) -> Prediction:
        return generate_answer(
            question=question,
            hints=hints,
            evidence_rows=evidence,
            support_priors=support,
            llm_client=llm_client,
            model="stub",
            max_evidence_rows=cfg.retrieval.max_evidence_rows,
        )

    return EvalPipeline(
        route=route_question,
        retrieve=retrieve,
        rerank=rerank,
        generate=generate,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use a real vLLM endpoint (reads VLLM_BASE_URL env var)",
    )
    parser.add_argument(
        "--vllm-url",
        default=None,
        help="Override VLLM_BASE_URL for --real mode",
    )
    args = parser.parse_args()

    out_dir = Path("outputs/smoke_local")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("CastleRAG local smoke test")
    print("=" * 50)

    if args.real:
        vllm_url = args.vllm_url or os.environ.get("VLLM_BASE_URL")
        if not vllm_url:
            sys.exit(
                "Set VLLM_BASE_URL or pass --vllm-url when using --real mode.\n"
                "Example: VLLM_BASE_URL=http://localhost:11434/v1 "
                "python scripts/smoke_local.py --real"
            )
        print(f"Mode: real  (vLLM endpoint: {vllm_url})")
        embed_client, llm_client = _build_real_clients(vllm_url)
        dim = None  # discovered after first embed call
    else:
        print("Mode: stub  (random embeddings + stub LLM — wiring check only)")
        embed_client = _StubEmbedClient()
        llm_client = _StubLLMClient()  # type: ignore[assignment]
        dim = _STUB_DIM

    with tempfile.TemporaryDirectory(prefix="castlerag_smoke_") as tmp:
        tmp_dir = Path(tmp)
        print("\nBuilding in-memory index...")
        t0 = time.perf_counter()
        bm25, qdrant, collection = _build_inmemory_index(embed_client, dim or _STUB_DIM, tmp_dir)
        print(f"  index ready in {time.perf_counter() - t0:.1f}s")

        pipeline = _build_pipeline(embed_client, llm_client, bm25, qdrant, collection)

        from castlerag.eval.io import load_questions
        from castlerag.eval.run_eval import run_eval
        from castlerag.schemas import EvalQuestion

        questions = {
            qid: EvalQuestion(
                question_id=qid,
                query=data["query"],
                answers=data["answers"],
            )
            for qid, data in _QUESTIONS.items()
        }

        print(f"\nRunning {len(questions)} questions...")
        t1 = time.perf_counter()
        result = run_eval(
            questions=questions,
            out_dir=out_dir,
            pipeline=pipeline,
        )
        elapsed = time.perf_counter() - t1

    print(f"\nCompleted in {elapsed:.1f}s")
    print("-" * 50)

    valid = {"a", "b", "c", "d"}
    all_ok = True
    for qid, pred in result.predictions.items():
        ok = pred.predicted_answer in valid
        if not ok:
            all_ok = False
        status = "OK" if ok else "INVALID"
        print(
            f"  [{status}] {qid}  route={pred.route:14s}  "
            f"answer={pred.predicted_answer}  "
            f"confidence={pred.confidence:.2f}"
        )

    print("-" * 50)
    print(f"Output dir: {out_dir.resolve()}")
    print(f"  predictions:    {result.output_paths.predictions}")
    print(f"  evidence traces:{result.output_paths.evidence_traces}")
    print(f"  submissions:    {result.output_paths.submissions}")

    if all_ok:
        print("\nAll predictions valid. Smoke test PASSED.")
        sys.exit(0)
    else:
        print("\nOne or more invalid predictions. Smoke test FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
