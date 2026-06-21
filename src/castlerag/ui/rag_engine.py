"""Real RAG-backed chat engine for the CastleRAG dashboard.

``RagEngine`` runs the actual ``route -> retrieve -> rerank -> generate`` pipeline
(via :func:`castlerag.eval.run_eval.run_question`) for one question and adapts the
output to the dashboard's data model: a :class:`~castlerag.ui.chat.Claim` plus a
ranked list of :class:`~castlerag.ui.chat.EvidenceMoment`, each showing exactly
three synchronized camera angles.

It implements the same :class:`~castlerag.ui.chat.ChatEngine` protocol as the
offline ``PlaceholderEngine`` and is selected by
:func:`castlerag.ui.engine_factory.build_engine` only when the backend infra
(Qdrant + a built index + ``VLLM_BASE_URL``) is reachable.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from castlerag.schemas import EvalQuestion, Prediction, RetrievalHit
from castlerag.ui.chat import (
    _DOT_COLORS,
    CameraAngle,
    ChatTurnResult,
    Claim,
    EvidenceMoment,
    EvidenceRef,
    SupportLevel,
    compose_justification,
    compose_refined_query,
)

log = logging.getLogger(__name__)

_CHOICES = ("a", "b", "c", "d")

# Exactly three synchronized cameras per moment, for now. Tracked for a future
# variable / real-overlapping treatment — see the GitHub follow-up issue. All
# trio logic is isolated in ``_pad_cameras`` so flipping this is a one-site change.
_FIXED_CAMERA_COUNT = 3

# Support-level thresholds over the max per-choice support prior.
_SUPPORTED_AT = 0.7
_PARTIAL_AT = 0.4

_MAX_MOMENTS = 4
_BUCKET_SECONDS = 60  # cluster width when grouping hits into one moment


@dataclass
class RagEngine:
    """Chat engine backed by the real CastleRAG retrieval/generation pipeline."""

    cfg: Any
    pipeline: Any
    ego_cameras: Tuple[str, ...] = ()
    is_live: bool = True
    # Lazily-built OpenAI-compatible vLLM client, reused for review suggestions.
    _client: Any = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_config(cls, cfg: Any = None, mirror: object = None) -> "RagEngine":
        """Build a RagEngine from config, constructing the default pipeline.

        Raises ``PipelineDependencyError`` (or other exceptions) when the backend
        infra is unavailable; the engine factory catches these and falls back.
        """
        import os

        from castlerag.config import load_config
        from castlerag.eval.run_eval import _build_default_pipeline

        # When no cfg is injected, honor CASTLERAG_CONFIG as the override path so the
        # UI picks up host-specific paths (e.g. configs/snellius_me.yaml: scratch
        # cache_dir + the right Qdrant collection). load_config silently skips a
        # missing/unset path, falling back to base.yaml defaults.
        if cfg is None:
            cfg = load_config(override_path=os.getenv("CASTLERAG_CONFIG"))
        pipeline = _build_default_pipeline(cfg)
        ego = tuple(getattr(cfg.dataset, "ego_cameras", ()) or ())
        return cls(cfg=cfg, pipeline=pipeline, ego_cameras=ego)

    # -- public protocol ----------------------------------------------------

    def answer(
        self, question: str, choices: Optional[Dict[str, str]] = None
    ) -> ChatTurnResult:
        """Answer a question with the real pipeline and adapt to the UI model."""
        from castlerag.eval.run_eval import run_question

        # Real MCQ calls pass ``choices``; free-form UI questions don't, so the
        # dummy "Option A".."Option D" we feed the pipeline (and the per-choice
        # priors it yields) are meaningless and must not drive the claim/support.
        is_mcq = choices is not None
        resolved = choices or {key: f"Option {key.upper()}" for key in _CHOICES}
        eval_q = EvalQuestion(
            question_id=_question_id(question),
            query=question,
            answers=resolved,
        )
        result = run_question(self.pipeline, self.cfg, eval_q)
        claim = self._synthesize_claim(
            result.prediction,
            result.support_priors,
            resolved,
            is_mcq=is_mcq,
            question=question,
            evidence_rows=result.evidence_rows,
        )
        moments = self._hits_to_moments(result.evidence_rows, claim.support) or [
            self._no_evidence_moment(claim.support)
        ]
        answer_text = result.prediction.raw_answer_text or (
            f"Predicted **{result.prediction.predicted_answer.upper()}** — "
            f"{resolved.get(result.prediction.predicted_answer, '')}."
        )
        return ChatTurnResult(
            answer_text=answer_text,
            route=result.hints.route,
            support_priors=result.support_priors,
            evidence=self._hits_to_evidence_refs(result.evidence_rows),
            predicted_choice=result.prediction.predicted_answer,
            is_placeholder=False,
            claim=claim,
            moments=moments,
        )

    def refine(
        self, claim: str, refined_query: str, iteration: int
    ) -> ChatTurnResult:
        """Re-run retrieval for the same claim with a sharper query.

        Support is recomputed from the fresh retrieval and is *not* guaranteed to
        climb monotonically — that honestly reflects what real retrieval returns.
        """
        result = self.answer(refined_query, choices=None)
        support = result.claim.support if result.claim else SupportLevel.PARTIAL
        result.claim = Claim(text=claim, support=support)
        result.moments = result.moments[:1]
        return result

    # -- review suggestions -------------------------------------------------

    def _chat_client(self) -> Any:
        """Build (once) and return the vLLM chat client used for suggestions."""
        if self._client is None:
            from castlerag.eval.run_eval import _build_vllm_chat_client

            self._client = _build_vllm_chat_client()
        return self._client

    def _gen_model(self) -> str:
        gen = getattr(self.cfg, "generation", None)
        return getattr(gen, "model", "Qwen/Qwen3-VL-8B-Instruct")

    def suggest_justification(
        self,
        claim: str,
        camera_id: str,
        verdict: str,
        evidence_text: Optional[str] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> str:
        """LLM-drafted per-camera justification; template fallback on failure."""
        from castlerag.generation.suggestions import suggest_justification_text

        try:
            text = suggest_justification_text(
                claim, camera_id, verdict, evidence_text, meta,
                self._chat_client(), model=self._gen_model(),
            )
            return text or compose_justification(
                claim, camera_id, verdict, evidence_text
            )
        except Exception as exc:  # never let the UI break on a model hiccup
            log.warning("suggest_justification fell back to template (%s)", exc)
            return compose_justification(claim, camera_id, verdict, evidence_text)

    def suggest_refined_query(
        self, claim: str, reviews: Dict[str, Dict[str, str]]
    ) -> str:
        """LLM-drafted refined query; template fallback on failure."""
        from castlerag.generation.suggestions import suggest_refined_query_text

        try:
            text = suggest_refined_query_text(
                claim, reviews, self._chat_client(), model=self._gen_model()
            )
            return text or compose_refined_query(claim, reviews)
        except Exception as exc:
            log.warning("suggest_refined_query fell back to template (%s)", exc)
            return compose_refined_query(claim, reviews)

    # -- adapters -----------------------------------------------------------

    def _synthesize_claim(
        self,
        prediction: Prediction,
        support_priors: Dict[str, float],
        choices: Dict[str, str],
        *,
        is_mcq: bool = True,
        question: str = "",
        evidence_rows: Optional[List[RetrievalHit]] = None,
    ) -> Claim:
        """Derive the claim under review and its support level.

        For a real MCQ (``is_mcq``) the claim is the predicted choice and support
        is thresholded on *that choice's* prior — not the max across choices, so
        support can't be borrowed from a different answer. For a free-form UI
        question the per-choice priors are meaningless (the choices are dummies),
        so the claim comes from the model's answer text / the question and support
        is gauged from retrieval evidence strength instead.
        """
        if is_mcq:
            choice = prediction.predicted_answer
            choice_text = (choices or {}).get(choice) or f"Option {choice.upper()}"
            text = f"The footage supports: {choice_text}"
            score = float(support_priors.get(choice, 0.0))
        else:
            text = (prediction.raw_answer_text or question or "").strip() or (
                "The retrieved footage"
            )
            rows = evidence_rows or []
            score = max((h.rerank_score or 0.0 for h in rows), default=0.0)
        if score >= _SUPPORTED_AT:
            support = SupportLevel.SUPPORTED
        elif score >= _PARTIAL_AT:
            support = SupportLevel.PARTIAL
        else:
            support = SupportLevel.UNSUPPORTED
        return Claim(text=text, support=support)

    def _no_evidence_moment(self, support: SupportLevel) -> EvidenceMoment:
        """Explicit placeholder moment when retrieval returns no timestamped hits.

        Keeps the investigation contract intact (callbacks always focus
        ``moments[0]``) while honestly signalling that nothing was retrieved:
        the synthetic camera tiles resolve no mirror embed and carry score 0.
        """
        cameras = self._pad_cameras([], "day1", 0, 0.0)
        cameras[0].is_best = True
        return EvidenceMoment(
            moment_id="m0",
            clock_label="--:--",
            place_label="No supporting footage found",
            camera_count=_FIXED_CAMERA_COUNT,
            aggregate_score=0.0,
            score_caption="no evidence",
            dot_color=_DOT_COLORS[support],
            cameras=cameras,
        )

    def _hits_to_moments(
        self, hits: List[RetrievalHit], support: SupportLevel
    ) -> List[EvidenceMoment]:
        """Cluster hits into synchronized 3-camera moments, ranked by score."""
        buckets: "OrderedDict[Tuple[str, int, int], List[RetrievalHit]]" = OrderedDict()
        for hit in hits:
            if not hit.camera_id or not hit.day:
                continue
            bucket = int(_seconds_within_hour(hit) // _BUCKET_SECONDS)
            key = (hit.day, _hour_of(hit), bucket)
            buckets.setdefault(key, []).append(hit)
        if not buckets:
            return []

        ranked_buckets = sorted(
            buckets.values(),
            key=lambda rows: max(h.score for h in rows),
            reverse=True,
        )[:_MAX_MOMENTS]

        score_mode: str = getattr(
            getattr(self.cfg, "ui", None), "score_mode", "rrf_normalized"
        )
        # Normalisation denominator for rrf_normalized mode: best RRF score across
        # all moments so the top moment always displays 1.0.
        max_rrf = max(
            (max(h.score for h in rows) for rows in ranked_buckets), default=1.0
        ) or 1.0

        moments: List[EvidenceMoment] = []
        for index, rows in enumerate(ranked_buckets):
            anchor = max(rows, key=lambda h: h.score)
            day = anchor.day or "day1"
            hour = _hour_of(anchor)
            start = _seconds_within_hour(anchor)

            # Best-scoring hit per distinct camera, highest first (kept whole so
            # we can surface its evidence text for grounded suggestions).
            by_camera: "OrderedDict[str, RetrievalHit]" = OrderedDict()
            for hit in sorted(rows, key=lambda h: h.score, reverse=True):
                cam = hit.camera_id
                if cam and cam not in by_camera:
                    by_camera[cam] = hit

            real = [
                CameraAngle(
                    camera_id=cam,
                    day=day,
                    hour=hour,
                    start_seconds=start,
                    match_score=round(_display_score(hit, score_mode, max_rrf), 4),
                    is_best=False,
                    evidence_text=_hit_evidence_text(hit),
                )
                for cam, hit in list(by_camera.items())[:_FIXED_CAMERA_COUNT]
            ]
            cameras = self._pad_cameras(real, day, hour, start)
            best = max(cameras, key=lambda c: c.match_score)
            best.is_best = True

            agg = round(_display_score(anchor, score_mode, max_rrf), 2)
            score_label = {
                "rrf_normalized": "rel",
                "cosine": "cos",
                "reranker": "rnk",
            }.get(score_mode, "match")
            minute = int(start // 60)
            moments.append(
                EvidenceMoment(
                    moment_id=f"m{index}",
                    clock_label=f"{hour:02d}:{minute:02d}",
                    place_label=anchor.room or "Scene",
                    camera_count=_FIXED_CAMERA_COUNT,
                    aggregate_score=agg,
                    score_caption=f"{score_label} {agg:.2f}",
                    dot_color=_DOT_COLORS[support],
                    cameras=cameras,
                )
            )
        return moments

    def _pad_cameras(
        self,
        real: List[CameraAngle],
        day: str,
        hour: int,
        start: float,
    ) -> List[CameraAngle]:
        """Pad to exactly ``_FIXED_CAMERA_COUNT`` cameras, deterministically.

        Padding pulls real ego-camera names from ``cfg.dataset.ego_cameras`` (so
        the YouTube mirror still resolves an embed at this (day, hour)), in roster
        order, skipping cameras already present. Padded angles carry score 0.0.
        Isolating this here keeps the future variable-count change to one site.
        """
        cameras = list(real[:_FIXED_CAMERA_COUNT])
        present = {cam.camera_id for cam in cameras}
        for name in self.ego_cameras:
            if len(cameras) >= _FIXED_CAMERA_COUNT:
                break
            if name in present:
                continue
            present.add(name)
            cameras.append(
                CameraAngle(
                    camera_id=name,
                    day=day,
                    hour=hour,
                    start_seconds=start,
                    match_score=0.0,
                    is_best=False,
                )
            )
        # Last resort: roster empty/too small AND too few real cameras. Pad with
        # DISTINCT synthetic placeholders (never a duplicate camera_id) so the UI
        # never renders the same camera twice. These won't resolve a mirror embed,
        # but neither would a duplicate — and distinct ids keep the tiles sane.
        slot = 0
        while len(cameras) < _FIXED_CAMERA_COUNT:
            slot += 1
            name = f"Camera {slot}"
            if name in present:
                continue
            present.add(name)
            cameras.append(
                CameraAngle(
                    camera_id=name,
                    day=day,
                    hour=hour,
                    start_seconds=start,
                    match_score=0.0,
                    is_best=False,
                )
            )
        return cameras

    def _hits_to_evidence_refs(
        self, hits: List[RetrievalHit]
    ) -> List[EvidenceRef]:
        """Project retrieval hits onto the legacy ``EvidenceRef`` contract."""
        refs: List[EvidenceRef] = []
        for hit in hits:
            if not hit.camera_id or not hit.day:
                continue
            hour = _hour_of(hit)
            start = _seconds_within_hour(hit)
            text = hit.transcript_text or hit.event_summary or hit.ocr_text or ""
            refs.append(
                EvidenceRef(
                    record_id=hit.record_id,
                    source_type=hit.source_type,
                    modality=hit.modality,
                    day=hit.day,
                    camera_id=hit.camera_id,
                    hour=hour,
                    start_seconds=start,
                    end_seconds=(
                        float(hit.end_seconds)
                        if hit.end_seconds is not None
                        else start + 30.0
                    ),
                    score=float(hit.score),
                    text=text,
                )
            )
        return refs


def _question_id(question: str) -> str:
    """Return a stable short id for a question string."""
    return "ui_" + hashlib.sha1(question.strip().lower().encode()).hexdigest()[:16]


def _hour_of(hit: RetrievalHit) -> int:
    """Return the within-day hour for a hit (from payload or derived from epoch)."""
    if hit.hour is not None:
        return int(hit.hour)
    if hit.absolute_start is not None:
        return int((hit.absolute_start // 1000 // 3600) % 24)
    return 0


def _seconds_within_hour(hit: RetrievalHit) -> float:
    """Return the second-offset within the hour (from payload or epoch)."""
    if hit.start_seconds is not None:
        return float(hit.start_seconds)
    if hit.absolute_start is not None:
        return float((hit.absolute_start // 1000) % 3600)
    return 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into the ``[low, high]`` range for display/scoring."""
    return max(low, min(high, float(value)))


def _display_score(
    hit: "RetrievalHit",
    mode: str,
    max_rrf: float,
) -> float:
    """Return the display score for a hit under the given score_mode.

    Falls back to rrf_normalized when the preferred field is unavailable
    (e.g. cosine before a dense search, reranker in offline/placeholder mode).
    """
    if mode == "cosine" and hit.raw_score is not None:
        return _clamp(hit.raw_score)
    if mode == "reranker" and hit.rerank_score is not None:
        return _clamp(hit.rerank_score)
    # rrf_normalized (default) or fallback
    return _clamp(hit.score / max_rrf if max_rrf > 0 else 0.0)


def _hit_evidence_text(hit: RetrievalHit, limit: int = 300) -> Optional[str]:
    """Best available evidence snippet for a hit (transcript/event/OCR)."""
    for text in (hit.transcript_text, hit.event_summary, hit.ocr_text):
        snippet = (text or "").strip()
        if snippet:
            return snippet[:limit]
    return None
