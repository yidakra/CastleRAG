"""Minimal OpenAI-compatible /v1/embeddings server for Tevatron OmniEmbed.

OmniEmbed-v0.1-multivent is a LoRA on Qwen2.5-Omni-7B-Thinker and is NOT servable
by vLLM's embedding task the way castlerag.embed.OmniEmbedClient expects.  This
reproduces the model card's embedding extraction (last-token hidden state of the
Thinker, L2-normalised) behind the OpenAI /v1/embeddings interface so the existing
client works unchanged.  Text-only (the castlerag pipeline only ever text-embeds).

Run:  python scripts/omniembed_server.py --port 8200
Env:  OMNIEMBED_BASE, OMNIEMBED_ADAPTER, OMNIEMBED_PROCESSOR, OMNIEMBED_SERVED_NAME
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration

BASE = os.getenv("OMNIEMBED_BASE", "Tevatron/Qwen2.5-Omni-7B-Thinker")
ADAPTER = os.getenv("OMNIEMBED_ADAPTER", "Tevatron/OmniEmbed-v0.1-multivent")
PROCESSOR = os.getenv("OMNIEMBED_PROCESSOR", "Qwen/Qwen2.5-Omni-7B")
SERVED_NAME = os.getenv("OMNIEMBED_SERVED_NAME", "Tevatron/OmniEmbed-v0.1-multivent")
MAX_BATCH = int(os.getenv("OMNIEMBED_MAX_BATCH", "16"))

_device = "cuda" if torch.cuda.is_available() else "cpu"
print(
    f"[omniembed] loading proc={PROCESSOR} base={BASE} adapter={ADAPTER} dev={_device}",
    flush=True,
)
_processor = AutoProcessor.from_pretrained(PROCESSOR)
_processor.tokenizer.padding_side = "left"
_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    BASE, dtype=torch.bfloat16, attn_implementation="sdpa"
).to(_device).eval()
_model = PeftModel.from_pretrained(_model, ADAPTER).eval()
_model.padding_side = "left"
print("[omniembed] model ready", flush=True)


@torch.no_grad()
def _embed(texts: List[str]) -> List[List[float]]:
    out_vecs: List[List[float]] = []
    for start in range(0, len(texts), MAX_BATCH):
        batch = texts[start : start + MAX_BATCH]
        rendered = []
        for t in batch:
            msg = [{"role": "user", "content": [{"type": "text", "text": t}]}]
            s = _processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            if isinstance(s, list):
                s = s[0]
            rendered.append(s + "<|endoftext|>")
        inputs = _processor.tokenizer(
            rendered, return_tensors="pt", padding="longest"
        ).to(_device)
        model_out = _model(**inputs, return_dict=True, output_hidden_states=True)
        reps = model_out.hidden_states[-1][:, -1]
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        out_vecs.extend(reps.float().cpu().tolist())
    return out_vecs


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None


app = FastAPI()


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model"}]}


@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest):
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    vectors = _embed(texts)
    return {
        "object": "list",
        "model": req.model or SERVED_NAME,
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
