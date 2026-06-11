"""OmniEmbed (Tevatron/OmniEmbed-v0.1-multivent) batch inference wrappers.

Backend: vLLM (default) or HuggingFace Transformers.

Modality batching strategy (SPEC §3.2):
  transcript windows : 128 per batch
  event summaries    : 64 per batch
  images             : 16 per batch
  video clips        : 4 per batch

Query format per OmniEmbed model card:
  text  → "Query: {text}"
  media → raw frames/video via Qwen2.5-Omni processor

Point ids are deterministic: sha1(model_version + source_type + record_id + modality)
"""

from __future__ import annotations

import hashlib
from typing import Any, List, Optional

import numpy as np


def make_point_id(
    model_version: str,
    source_type: str,
    record_id: str,
    modality: str,
) -> str:
    """Return a deterministic hex point id (SHA-1)."""
    key = f"{model_version}|{source_type}|{record_id}|{modality}"
    return hashlib.sha1(key.encode()).hexdigest()


class OmniEmbedClient:
    """Thin wrapper around a vLLM or Transformers OmniEmbed backend.

    Call embed_texts() or embed_images() or embed_videos() to get
    float32 embedding arrays.  Vector dimensionality is discovered from
    the first successful batch and stored in self.dim.
    """

    def __init__(
        self,
        model: str = "Tevatron/OmniEmbed-v0.1-multivent",
        backend: str = "vllm",
        vllm_base_url: Optional[str] = None,
        vllm_tensor_parallel: int = 1,
        vllm_gpu_memory_utilization: float = 0.90,
    ) -> None:
        """Initialise the embedder with model name, backend, and vLLM options."""
        self.model = model
        self.backend = backend
        self.vllm_base_url = vllm_base_url
        self.vllm_tensor_parallel = vllm_tensor_parallel
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.dim: Optional[int] = None  # set after first batch
        self._client: Any = None

    def _ensure_client(self) -> None:
        """Lazily initialise the embedding client on first use."""
        if self._client is not None:
            return
        if self.backend == "vllm":
            self._client = self._init_vllm()
        elif self.backend == "transformers":
            self._client = self._init_transformers()
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

    def _init_vllm(self) -> Any:
        """Return an OpenAI-compatible client for the vLLM embeddings endpoint."""
        if not self.vllm_base_url:
            raise ValueError("vllm_base_url is required when backend='vllm'")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package required for vLLM embedding backend; "
                "pip install castlerag[inference]"
            ) from exc
        return OpenAI(base_url=self.vllm_base_url, api_key="not-needed")

    def _init_transformers(self) -> Any:
        """Return a transformers-based embedding client (stub for local inference)."""
        return object()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed text strings.  Automatically prepends 'Query: ' prefix."""
        self._ensure_client()
        payload = [format_query_text(text) for text in texts]
        if hasattr(self._client, "embeddings"):
            resp = self._client.embeddings.create(model=self.model, input=payload)
            vectors = np.asarray([row.embedding for row in resp.data], dtype=np.float32)
        elif hasattr(self._client, "embed_texts"):
            vectors = np.asarray(self._client.embed_texts(payload), dtype=np.float32)
        else:
            raise NotImplementedError(
                "Configured embedding backend does not support text embeddings"
            )
        if self.dim is None and vectors.size:
            self.dim = int(vectors.shape[1])
        return vectors

    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        """Embed image files (JPEG/PNG).  Preserves source resolution."""
        self._ensure_client()
        if not hasattr(self._client, "embed_images"):
            raise NotImplementedError(
                "Configured embedding backend does not support image embeddings"
            )
        vectors = np.asarray(self._client.embed_images(image_paths), dtype=np.float32)
        if self.dim is None and vectors.size:
            self.dim = int(vectors.shape[1])
        return vectors

    def embed_videos(self, frame_path_lists: List[List[str]]) -> np.ndarray:
        """Embed video clips represented as lists of 1 fps frame paths."""
        self._ensure_client()
        if not hasattr(self._client, "embed_videos"):
            raise NotImplementedError(
                "Configured embedding backend does not support video embeddings"
            )
        vectors = np.asarray(
            self._client.embed_videos(frame_path_lists), dtype=np.float32
        )
        if self.dim is None and vectors.size:
            self.dim = int(vectors.shape[1])
        return vectors


def format_query_text(text: str) -> str:
    """Format text according to the OmniEmbed query convention."""
    return f"Query: {text}"


def prepare_text_inputs(texts: List[str]) -> List[str]:
    """Prepare text inputs for OmniEmbed."""
    return [format_query_text(text) for text in texts]
