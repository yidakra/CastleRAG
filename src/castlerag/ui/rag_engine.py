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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

_MAX_MOMENTS = 3
_BUCKET_SECONDS = 60  # cluster width when grouping hits into one moment


@dataclass
class RagEngine:
    """Chat engine backed by the real CastleRAG retrieval/generation pipeline."""

    cfg: Any
    pipeline: Any
    ego_cameras: Tuple[str, ...] = ()
    is_live: bool = True
    # YouTube mirror, so synchronized-camera fills can prefer angles the UI can
    # actually embed (some cameras — e.g. Bao — are in the index but barely
    # mirrored, which otherwise yields "no footage" tiles). None -> no filtering.
    mirror: Any = None
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
        return cls(cfg=cfg, pipeline=pipeline, ego_cameras=ego, mirror=mirror)

    # -- public protocol ----------------------------------------------------

    def answer(
        self,
        question: str,
        choices: Optional[Dict[str, str]] = None,
        exclude_cameras: Sequence[str] = (),
        _refinement_context: Optional[str] = None,
    ) -> ChatTurnResult:
        """Answer a question with the real pipeline and adapt to the UI model."""
        from castlerag.eval.run_eval import run_question

        # Real MCQ calls pass ``choices``; free-form UI questions don't. For an
        # open question we feed BLANK choices (not "Option A".."Option D"): that
        # marks the EvalQuestion as free-form, so retrieval, reranking, and
        # generation all drop the four-option scaffolding instead of embedding
        # and scoring meaningless placeholders.
        is_mcq = choices is not None
        resolved = choices or {key: "" for key in _CHOICES}
        eval_q = EvalQuestion(
            question_id=_question_id(question),
            query=question,
            answers=resolved,
        )
        result = run_question(
            self.pipeline,
            self.cfg,
            eval_q,
            exclude_cameras=exclude_cameras,
            # Skip the MCQ generator for open questions — we answer them directly
            # below; running it would waste an LLM call and force a fake letter.
            generate_prediction=is_mcq,
        )
        # Open UI questions are answered directly; MCQ callers keep the raw graded
        # answer.
        if is_mcq:
            answer_text = result.prediction.raw_answer_text or (
                f"Predicted **{result.prediction.predicted_answer.upper()}** — "
                f"{resolved.get(result.prediction.predicted_answer, '')}."
            )
            freeform_answer = None
        else:
            answer_text = self._freeform_answer_text(
                eval_q, result, refinement_context=_refinement_context
            )
            freeform_answer = answer_text
        claim = self._synthesize_claim(
            result.prediction,
            result.support_priors,
            resolved,
            is_mcq=is_mcq,
            question=question,
            evidence_rows=result.evidence_rows,
            freeform_answer=freeform_answer,
            support_score=(
                None
                if is_mcq
                else self._freeform_support_score(
                    result.rerank_result, result.evidence_rows
                )
            ),
        )
        # Stamp the reranker's normalised relevance onto the displayed rows so
        # moments rank by reranked relevance (the evidence the answer is built on),
        # not raw RRF — otherwise the focused moment can be an unrelated, higher-RRF
        # bucket. Then query-score the synchronized angles via the same query vector.
        evidence_rows = self._stamp_rerank_scores(
            result.evidence_rows, result.rerank_result
        )
        query_vector = self._embed_query(question)
        moments = self._hits_to_moments(
            evidence_rows, claim.support, query_vector=query_vector
        ) or [self._no_evidence_moment(claim.support)]
        pipeline_stats = {
            "retrieved": len(result.retrieved),
            "reranked": len(result.evidence_rows),
            "candidates": len({
                (h.day, _hour_of(h), int(_seconds_within_hour(h) // _BUCKET_SECONDS))
                for h in result.evidence_rows
                if h.camera_id and h.day
            }),
            "displayed": sum(
                1 for m in moments if m.moment_id != "no-evidence"
            ),
        }
        return ChatTurnResult(
            answer_text=answer_text,
            route=result.hints.route,
            support_priors=result.support_priors,
            evidence=self._hits_to_evidence_refs(evidence_rows),
            # Free-form questions have no choice; the stub "a" is not a real pick.
            predicted_choice=(
                result.prediction.predicted_answer if is_mcq else None
            ),
            is_placeholder=False,
            claim=claim,
            moments=moments,
            pipeline_stats=pipeline_stats,
        )

    def refine(
        self,
        claim: str,
        refined_query: str,
        iteration: int,
        exclude_cameras: Sequence[str] = (),
        anchor: Optional[Tuple[Optional[str], Optional[int]]] = None,
        reviews: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> ChatTurnResult:
        """Run a real retrieval/generation pass for the refined query.

        Rejected cameras are hard-excluded from retrieval. ``reviews`` carries
        the human reviewer's per-camera verdicts and justifications; these are
        formatted into a context block appended to the generation prompt so the
        model grounds its answer in what the reviewer confirmed or rejected —
        not just in the raw retrieved evidence.

        ``anchor`` is accepted for backwards compatibility but no longer used.
        """
        review_context = _format_review_context(reviews) if reviews else None
        result = self.answer(
            refined_query,
            choices=None,
            exclude_cameras=exclude_cameras,
            _refinement_context=review_context,
        )
        if result.claim is None:
            result.claim = Claim(text=claim, support=SupportLevel.PARTIAL)
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

    def _freeform_answer_text(
        self,
        eval_q: EvalQuestion,
        result: Any,
        refinement_context: Optional[str] = None,
    ) -> str:
        """Direct open-question answer; cleaned MCQ text as a fallback."""
        from castlerag.generation.answer import (
            clean_answer_text,
            generate_freeform_answer,
        )

        try:
            text = generate_freeform_answer(
                eval_q,
                result.hints,
                result.evidence_rows,
                self._chat_client(),
                model=self._gen_model(),
                refinement_context=refinement_context,
            )
            if text:
                return text
        except Exception as exc:  # never break the UI on a model hiccup
            log.warning("free-form answer fell back to cleaned MCQ text (%s)", exc)
        cleaned = clean_answer_text(result.prediction.raw_answer_text or "")
        return cleaned or "No answer could be produced from the retrieved evidence."

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
        self,
        claim: str,
        reviews: Dict[str, Dict[str, str]],
        question: Optional[str] = None,
    ) -> str:
        """LLM-drafted refined query; template fallback on failure.

        ``question`` is the original user question; it anchors the refined query
        so the search is not biased back toward the (possibly wrong) prior answer
        carried in ``claim``.
        """
        from castlerag.generation.suggestions import suggest_refined_query_text
        from castlerag.ui.chat import strip_parentheticals

        try:
            text = suggest_refined_query_text(
                claim,
                reviews,
                self._chat_client(),
                question=question,
                model=self._gen_model(),
            )
            drafted = text or compose_refined_query(claim, reviews, question=question)
        except Exception as exc:
            log.warning("suggest_refined_query fell back to template (%s)", exc)
            drafted = compose_refined_query(claim, reviews, question=question)
        # Drop parenthetical asides so the editable query box reads as clean prose.
        return strip_parentheticals(drafted)

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
        freeform_answer: Optional[str] = None,
        support_score: Optional[float] = None,
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
            text = (
                freeform_answer or prediction.raw_answer_text or question or ""
            ).strip() or ("The retrieved footage")
            if support_score is not None:
                # Caller supplies the reranker's normalised evidence strength
                # (the rows handed to the UI carry no rerank_score themselves).
                score = support_score
            else:
                rows = evidence_rows or []
                score = max((h.rerank_score or 0.0 for h in rows), default=0.0)
        if score >= _SUPPORTED_AT:
            support = SupportLevel.SUPPORTED
        elif score >= _PARTIAL_AT:
            support = SupportLevel.PARTIAL
        else:
            support = SupportLevel.UNSUPPORTED
        return Claim(text=text, support=support)

    def _freeform_support_score(
        self,
        rerank_result: Any,
        evidence_rows: Optional[List[RetrievalHit]],
    ) -> float:
        """Evidence strength for an open question's claim, in [0, 1].

        The rows the UI displays carry no ``rerank_score`` (run_eval's flattening
        does not stamp it), so reading it off them always yielded 0.0 — every
        open answer was labelled UNSUPPORTED. Use the reranker's own normalised
        evidence scores instead; if the reranker kept nothing, fall back to the
        strongest dense cosine so a grounded dense-only answer isn't mislabelled.
        """
        reranked = getattr(rerank_result, "evidence_rows", None) or []
        scores = [r.rerank_score for r in reranked if r.rerank_score is not None]
        # Free-form questions have empty choice strings so the reranker scores all
        # packs 0; treat an all-zero result the same as no reranker scores so we
        # fall through to the cosine fallback rather than labelling every open
        # answer UNSUPPORTED.
        if scores and max(scores) > 0:
            return max(scores)
        cosines = [
            h.raw_score for h in (evidence_rows or []) if h.raw_score is not None
        ]
        return max(cosines) if cosines else 0.0

    def _stamp_rerank_scores(
        self, evidence_rows: List[RetrievalHit], rerank_result: Any
    ) -> List[RetrievalHit]:
        """Copy the reranker's normalised [0,1] scores onto the displayed rows.

        run_eval's evidence flattening drops ``rerank_score`` (it stays only on
        ``rerank_result.evidence_rows``), so moments would otherwise rank by raw
        RRF. Map by ``record_id`` and stamp; rows with no match keep what they had.
        """
        scores = {
            r.record_id: r.rerank_score
            for r in (getattr(rerank_result, "evidence_rows", None) or [])
            if r.rerank_score is not None
        }
        if not scores:
            return evidence_rows
        return [
            row.model_copy(update={"rerank_score": scores[row.record_id]})
            if row.record_id in scores and row.rerank_score is None
            else row
            for row in evidence_rows
        ]

    def _embed_query(self, text: str) -> Optional[List[float]]:
        """Embed the question for query-scoring synchronized angles; None if no
        embed client (offline/injected pipelines) or on failure."""
        embed = getattr(self.pipeline, "embed_client", None)
        if embed is None:
            return None
        try:
            vectors = embed.embed_texts([text])
            if vectors is not None and len(vectors):
                return list(vectors[0])
        except Exception as exc:  # never break a moment on an embed hiccup
            log.warning("query embedding for synchronized angles failed (%s)", exc)
        return None

    def _no_evidence_moment(self, support: SupportLevel) -> EvidenceMoment:
        """Explicit placeholder moment when retrieval returns no timestamped hits.

        Keeps the investigation contract intact (callbacks always focus
        ``moments[0]``) while honestly signalling that nothing was retrieved:
        the synthetic camera tiles resolve no mirror embed and carry score 0.
        """
        cameras = self._pad_cameras([], "day1", 0, 0.0, presence_score=0.0)
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
        self,
        hits: List[RetrievalHit],
        support: SupportLevel,
        query_vector: Optional[List[float]] = None,
    ) -> List[EvidenceMoment]:
        """Cluster hits into synchronized 3-camera moments, ranked by relevance.

        Buckets rank by reranked relevance (``rerank_score``) when available, so
        the focused moment is the evidence the answer is built on, not the bucket
        with the highest raw RRF. ``query_vector`` (when given) lets the
        synchronized-angle fill score the co-temporal cameras by relevance.
        """
        buckets: "OrderedDict[Tuple[str, int, int], List[RetrievalHit]]" = OrderedDict()
        for hit in hits:
            if not hit.camera_id or not hit.day:
                continue
            bucket = int(_seconds_within_hour(hit) // _BUCKET_SECONDS)
            key = (hit.day, _hour_of(hit), bucket)
            buckets.setdefault(key, []).append(hit)
        if not buckets:
            return []

        def _bucket_relevance(rows: List[RetrievalHit]) -> float:
            # Prefer the reranker's relevance; fall back to RRF when unstamped.
            return max(
                (h.rerank_score if h.rerank_score is not None else h.score)
                for h in rows
            )

        ranked_buckets = sorted(
            buckets.values(), key=_bucket_relevance, reverse=True
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

            # Score the primary cameras by the SAME metric the synchronized
            # angles use — query cosine (the hit's dense ``raw_score``) — so all
            # tiles in a moment are comparable. Mixing rerank-normalised (~1.0)
            # primaries with raw-cosine (~0.18) angles produced the "one full,
            # rest ~0" cliff; falling back to the display score only when a hit
            # carries no cosine (e.g. BM25-only).
            real = [
                CameraAngle(
                    camera_id=cam,
                    day=day,
                    hour=hour,
                    start_seconds=start,
                    match_score=round(
                        _clamp(hit.raw_score)
                        if hit.raw_score is not None
                        else _display_score(hit, score_mode, max_rrf),
                        4,
                    ),
                    is_best=False,
                    evidence_text=_hit_evidence_text(hit),
                )
                for cam, hit in list(by_camera.items())[:_FIXED_CAMERA_COUNT]
            ]
            cameras = self._fill_synchronized_cameras(
                real, day, hour, start, anchor.absolute_start, query_vector
            )
            # Normalise the tiles within the moment so the best angle reads 1.0 and
            # the others are comparable fractions of it (relative angle relevance),
            # instead of tiny absolute cosines next to a 1.0 primary.
            top_score = max((c.match_score for c in cameras), default=0.0)
            if top_score > 0:
                for cam_angle in cameras:
                    if cam_angle.match_score > 0:
                        cam_angle.match_score = round(
                            cam_angle.match_score / top_score, 4
                        )
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
                    camera_count=len(cameras),
                    aggregate_score=agg,
                    score_caption=f"{score_label} {agg:.2f}",
                    dot_color=_DOT_COLORS[support],
                    cameras=cameras,
                    absolute_start_ms=anchor.absolute_start,
                )
            )
        return moments

    def _fill_synchronized_cameras(
        self,
        real: List[CameraAngle],
        day: str,
        hour: int,
        start: float,
        abs_start_ms: Optional[int],
        query_vector: Optional[List[float]] = None,
    ) -> List[CameraAngle]:
        """Fill a moment's camera slots with the angles actually rolling then.

        The scored retrieval hits (``real``) lead; remaining slots are filled
        with the OTHER cameras rolling at this timestamp, fetched from the index.
        With a ``query_vector`` they are scored by relevance to the question (real
        match scores); without one they are ``is_context`` fillers (the "sync"
        tile). Only when no co-temporal data is available (offline / no Qdrant) do
        we fall back to the old ego-roster padding so the tiles still render.
        """
        cameras = list(real[:_FIXED_CAMERA_COUNT])
        present = {cam.camera_id for cam in cameras}
        if len(cameras) < _FIXED_CAMERA_COUNT:
            cotemporal = self._cotemporal_cameras(
                day, abs_start_ms, exclude=present, query_vector=query_vector
            )
            for cam in cotemporal:
                if len(cameras) >= _FIXED_CAMERA_COUNT:
                    break
                if cam.camera_id in present:
                    continue
                present.add(cam.camera_id)
                cameras.append(cam)
        if len(cameras) < _FIXED_CAMERA_COUNT:
            # No (or too few) co-temporal cameras from the index — keep the UI
            # contract with deterministic roster padding.
            cameras = self._pad_cameras(cameras, day, hour, start)
        return cameras

    def _cotemporal_cameras(
        self,
        day: Optional[str],
        abs_start_ms: Optional[int],
        *,
        exclude: Any = (),
        query_vector: Optional[List[float]] = None,
        window_ms: int = 45_000,
        limit: int = 128,
    ) -> List[CameraAngle]:
        """Cameras whose clips overlap ``abs_start_ms`` on ``day`` (from the index).

        The synchronized angles for a moment — every camera rolling at that
        timestamp. With ``query_vector`` they are ranked/scored by relevance to
        the question (a time-windowed dense search), so the most relevant angle
        can win "best" instead of being a blind ``0.00`` filler; without it they
        are unscored ``is_context`` ("sync") tiles. ``exclude`` drops named cameras
        (already shown, or rejected). Empty when there is no Qdrant handle
        (injected/offline pipelines) or the lookup fails.
        """
        client = getattr(self.pipeline, "qdrant_client", None)
        collection = getattr(self.pipeline, "collection_name", None)
        if client is None or collection is None or not day or abs_start_ms is None:
            return []
        start_ms = abs_start_ms - window_ms
        end_ms = abs_start_ms + window_ms
        exclude_ids = tuple(exclude) or None

        by_cam: "OrderedDict[str, CameraAngle]" = OrderedDict()
        if query_vector is not None:
            # Query-scored: a time-windowed dense search ranks the co-temporal
            # cameras by relevance to the question; Qdrant returns the cosine score.
            from castlerag.retrieval.search import _dense_search

            try:
                hits = _dense_search(
                    qdrant_client=client,
                    collection_name=collection,
                    query_vector=query_vector,
                    limit=limit,
                    source_type="main_clip",
                    modality="video",
                    day=day,
                    time_range_start_ms=start_ms,
                    time_range_end_ms=end_ms,
                    exclude_camera_ids=exclude_ids,
                )
            except Exception as exc:  # never break a moment on an index hiccup
                log.warning("co-temporal dense search failed (%s)", exc)
                hits = []
            for hit in hits:  # already ordered by score, highest first
                cam = hit.camera_id
                if not cam or cam in by_cam:
                    continue
                by_cam[cam] = CameraAngle(
                    camera_id=str(cam),
                    day=hit.day or day,
                    hour=_hour_of(hit),
                    start_seconds=_seconds_within_hour(hit),
                    match_score=round(_clamp(hit.score), 4),
                    is_best=False,
                    is_context=False,  # scored, not a blind sync filler
                    evidence_text=_hit_evidence_text(hit),
                )
        else:
            from castlerag.retrieval.filters import build_filter

            query_filter = build_filter(
                day=day,
                source_type="main_clip",
                time_range_start_ms=start_ms,
                time_range_end_ms=end_ms,
                exclude_camera_ids=exclude_ids,
            )
            try:
                points, _ = client.scroll(
                    collection_name=collection,
                    scroll_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:  # never break a moment on an index hiccup
                log.warning("co-temporal camera lookup failed (%s)", exc)
                return []
            for point in points:
                payload = dict(getattr(point, "payload", {}) or {})
                cam = payload.get("camera_id")
                if not cam or cam in by_cam:
                    continue
                by_cam[cam] = CameraAngle(
                    camera_id=str(cam),
                    day=str(payload.get("day") or day),
                    hour=int(payload.get("hour") or 0),
                    start_seconds=float(payload.get("start_seconds") or 0.0),
                    # Small presence floor: camera was physically at the scene
                    # even without a semantic score. After per-moment normalisation
                    # this shows as ~0.05-0.15 rather than a flat zero bar.
                    match_score=0.05,
                    is_best=False,
                    is_context=True,
                    evidence_text=(
                        payload.get("transcript_text")
                        or payload.get("event_summary")
                        or payload.get("ocr_text")
                        or None
                    ),
                )

        # Prefer angles the UI can actually embed (a camera in the index but not
        # mirrored at this (day, hour) — e.g. Bao — otherwise renders a "no
        # footage" tile), then by relevance. Stable sort keeps within-group order.
        cameras = list(by_cam.values())
        cameras.sort(
            key=lambda c: (
                not self._has_embed(c.day, c.camera_id, c.hour),
                -c.match_score,
            )
        )
        return cameras

    def _has_embed(self, day: str, camera: str, hour: int) -> bool:
        """True when the YouTube mirror can embed this (day, camera, hour).

        Returns True when there is no mirror (can't tell, so don't filter), so
        offline/injected engines keep their current behaviour.
        """
        mirror = self.mirror
        if mirror is None:
            return True
        try:
            return not mirror.is_placeholder(day, camera, int(hour))
        except Exception:  # never let a mirror lookup break a moment
            return True

    def _pad_cameras(
        self,
        real: List[CameraAngle],
        day: str,
        hour: int,
        start: float,
        presence_score: float = 0.05,
    ) -> List[CameraAngle]:
        """Pad to exactly ``_FIXED_CAMERA_COUNT`` cameras, deterministically.

        Padding pulls real ego-camera names from ``cfg.dataset.ego_cameras`` (so
        the YouTube mirror still resolves an embed at this (day, hour)), in roster
        order, skipping cameras already present. Padded angles carry score 0.0.
        Isolating this here keeps the future variable-count change to one site.
        """
        cameras = list(real[:_FIXED_CAMERA_COUNT])
        present = {cam.camera_id for cam in cameras}
        # Roster order, but cameras the mirror can embed at this (day, hour) first,
        # so padding doesn't burn a slot on a "no footage" angle. Stable sort keeps
        # the deterministic roster order within each group.
        roster = sorted(
            self.ego_cameras, key=lambda n: not self._has_embed(day, n, hour)
        )
        for name in roster:
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
                    match_score=presence_score,
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


def _format_review_context(
    reviews: Optional[Dict[str, Dict[str, str]]],
) -> Optional[str]:
    """Format per-camera review verdicts into a generation context block.

    Only includes cameras that have been reviewed (not pending). The result is
    appended after the evidence so the model knows which angles the human
    confirmed, rejected, or flagged as partially relevant.
    """
    if not reviews:
        return None
    _LABEL = {"confirmed": "CONFIRMED", "rejected": "REJECTED", "flagged": "FLAGGED"}
    lines: List[str] = []
    for camera_id, info in reviews.items():
        state = (info.get("state") or "pending").lower()
        if state == "pending":
            continue
        label = _LABEL.get(state, state.upper())
        just = (info.get("justification") or "").strip()
        line = f"- {camera_id}: {label}"
        if just:
            line += f' — "{just}"'
        lines.append(line)
    if not lines:
        return None
    return "\n".join(lines)


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
    # A rerank_score of exactly 0.0 means the reranker gave up (e.g. free-form
    # questions have empty choice strings); fall through to rrf_normalized so the
    # displayed score reflects real retrieval relevance instead of a forced zero.
    if mode == "reranker" and hit.rerank_score is not None and hit.rerank_score > 0:
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
