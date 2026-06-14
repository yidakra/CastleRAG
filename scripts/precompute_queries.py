"""Pre-compute OmniEmbed dense vectors for every retrieval query variant.

CASTLE retrieval generates two query variants per question: the bare query
and the query suffixed with the four answer choices.  This script materialises
both for all 185 EgoVis 2026 questions and writes an NPZ keyed by raw text,
so the eval can serve dense vectors from cache while OmniEmbed is offline
(swapped out for Qwen3-VL on the same GPU).

Usage:
    OMNIEMBED_URL=http://localhost:8200/v1 \\
        python scripts/precompute_queries.py /data/castle2024/questions.json \\
            --out /data/castle_derived/embeddings/query_cache.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np


def _query_variants(question: dict) -> List[str]:
    """Mirror retrieval/search._query_variants exactly."""
    answers = question["answers"]
    return [
        question["query"],
        (
            f"{question['query']} Choices: "
            f"A {answers['a']}. B {answers['b']}. "
            f"C {answers['c']}. D {answers['d']}."
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions_path", type=Path)
    parser.add_argument(
        "--url",
        default=os.getenv("OMNIEMBED_URL", "http://localhost:8200/v1"),
        help="OmniEmbed vLLM endpoint (default $OMNIEMBED_URL or :8200/v1)",
    )
    parser.add_argument("--model", default="omniembed")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/data/castle_derived/embeddings/query_cache.npz"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    raw = json.loads(args.questions_path.read_text())
    if isinstance(raw, list):
        questions = {q.get("id") or q["question_id"]: q for q in raw}
    else:
        questions = raw

    texts: List[str] = []
    seen: set[str] = set()
    for q in questions.values():
        for v in _query_variants(q):
            if v not in seen:
                seen.add(v)
                texts.append(v)
    print(f"unique variants to embed: {len(texts)}")

    from castlerag.embed.omniembed import OmniEmbedClient

    client = OmniEmbedClient(
        model=args.model,
        backend="vllm",
        vllm_base_url=args.url,
    )

    vectors: List[np.ndarray] = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        v = client.embed_texts(batch)
        vectors.append(v)
        print(f"  embedded {start + len(batch)}/{len(texts)}", flush=True)
    matrix = np.concatenate(vectors, axis=0)
    print(f"matrix shape: {matrix.shape}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, keys=np.asarray(texts, dtype=object), vectors=matrix)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
