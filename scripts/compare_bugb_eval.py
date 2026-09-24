#!/usr/bin/env python
"""Score a day-1 smoke run against the issue #50 zero-evidence question list.

Reads the smoke-test output dir (predictions.json + evidence_traces.jsonl) and
the question CSV, then prints, for each #50 target question: whether evidence
was retrieved (is_supported / priors), correctness, the cameras in the final
evidence, and whether the CSV anchor camera (e.g. "Kitchen Day 1 12:16") or any
fixed room camera made it in. Pass --baseline-dir to diff against an earlier
run. Read-only; no GPU.

    python scripts/compare_bugb_eval.py \\
        --questions data/smoke_day1.csv \\
        --run-dir /scratch-shared/$USER/castle_outputs/smoke_test \\
        [--baseline-dir /scratch-shared/$USER/castle_outputs/smoke_test_pre_bugb]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

from castlerag.eval.io import load_predictions, load_questions

FIXED = ("Kitchen", "Living1", "Living2", "Meeting", "Reading")

# The 8 all-zero-support questions listed in issue #50 (matched by prefix).
ISSUE_50_TARGETS = (
    "How many duck sculptures are on the top shelf",
    "How much did Cathal spend on the four buzzers",
    "In which cupboard did Werner store the liquid measuring cup",
    "What brand is the fridge in the kitchen",
    "What family name is on the coat of arms in the reading area",
    "What is on the back of Werner's t-shirt on the first day",
    "What medium did Allie use for her painting of a Christmas tree",
    "What organisation's logo is on Werner's apron",
)


def _anchors(csv_path: Path) -> Dict[str, str]:
    """Question text -> anchor camera name ('' when the anchor names none)."""
    out: Dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            q = (row.get("Question") or "").strip()
            first = ((row.get("Anchor") or "").split() or [""])[0]
            out[q] = "" if first.lower() == "day" else first
    return out


def _load_run(run_dir: Path):
    """Return (predictions, traces, ids whose support is unknown).

    Compact submission-format predictions ({"qid": "a"}) carry no
    ``is_supported`` field, so ``Prediction`` would default them to supported;
    those ids are reported separately instead of trusting the default.
    """
    ppath = run_dir / "predictions.json"
    preds = load_predictions(ppath)
    raw = json.loads(ppath.read_text())
    unknown = {
        qid for qid, val in raw.items()
        if not (isinstance(val, dict) and "is_supported" in val)
    }
    traces = {}
    tpath = run_dir / "evidence_traces.jsonl"
    if tpath.exists():
        for line in tpath.read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                traces[t["question_id"]] = t
    return preds, traces, unknown


def _row(qid: str, q, preds, traces, anchor: str, unknown=()) -> Optional[dict]:
    p = preds.get(qid)
    if p is None:
        return None
    cams = traces.get(qid, {}).get("top_evidence_cameras") or []
    return {
        "supported": None if qid in unknown else bool(p.is_supported),
        "correct": q.ground_truth is not None and p.predicted_answer == q.ground_truth,
        "cams": cams,
        "fixed": [c for c in cams if c in FIXED],
        "anchor_hit": bool(anchor) and anchor in cams,
    }


def _support_tag(supported: Optional[bool]) -> str:
    return "????" if supported is None else ("EVID" if supported else "ZERO")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", type=Path, default=Path("data/smoke_day1.csv"))
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--baseline-dir", type=Path, default=None)
    args = ap.parse_args()

    questions = load_questions(args.questions)
    anchors = _anchors(args.questions) if args.questions.suffix == ".csv" else {}
    preds, traces, unknown = _load_run(args.run_dir)
    base = _load_run(args.baseline_dir) if args.baseline_dir else None

    graded = [qid for qid in preds if qid in questions]
    n_correct = sum(
        preds[q].predicted_answer == questions[q].ground_truth for q in graded
    )
    n_zero = sum(not preds[q].is_supported for q in graded if q not in unknown)
    n_unknown = sum(q in unknown for q in graded)
    n_fixed = sum(
        any(c in FIXED for c in traces.get(q, {}).get("top_evidence_cameras") or [])
        for q in graded
    )
    # Score against every loaded question so a partial run (crashed job,
    # missing predictions) can't look better than it is; coverage is separate.
    n_total = len(questions)
    print(f"run: {args.run_dir}")
    print(f"  accuracy          : {n_correct}/{n_total}  (missing predictions "
          "count as wrong)")
    print(f"  coverage          : {len(graded)}/{n_total} questions have a prediction")
    # Same denominator as accuracy: a missing prediction surfaced no evidence
    # (so it adds to n_zero) and no room camera (so it adds nothing to
    # n_fixed, whose denominator is already n_total). Predictions with unknown
    # support (compact format) are left out of the zero-evidence rate.
    n_zero += n_total - len(graded)
    if n_unknown:
        print(f"  WARN: {n_unknown} predictions carry no is_supported field "
              "(compact format); excluded from zero-evidence")
    print(f"  zero-evidence     : {n_zero}/{n_total - n_unknown}  (missing "
          "predictions count; issue #50 baseline 8/40, post-#53 ~11/40)")
    print(
        f"  fixed-cam evidence: {n_fixed}/{n_total} questions cite "
        ">=1 room camera"
    )
    print()
    print("issue #50 target questions:")
    for prefix in ISSUE_50_TARGETS:
        match = [(qid, q) for qid, q in questions.items() if q.query.startswith(prefix)]
        if not match:
            print(f"  ?? not in question file: {prefix}")
            continue
        qid, q = match[0]
        anchor = anchors.get(q.query, "")
        r = _row(qid, q, preds, traces, anchor, unknown)
        if r is None:
            print(f"  -- not run: {prefix}")
            continue
        line = (
            f"  {_support_tag(r['supported'])} "
            f"{'OK ' if r['correct'] else 'BAD'} anchor={anchor or '-':<8} "
            f"anchor_in_evidence={'Y' if r['anchor_hit'] else 'n'} "
            f"fixed={','.join(r['fixed']) or '-':<16} {prefix[:48]}"
        )
        if base is not None:
            b = _row(qid, q, base[0], base[1], anchor, base[2])
            if b is not None:
                line += (
                    f"   [baseline {_support_tag(b['supported'])}"
                    f"/{'OK' if b['correct'] else 'BAD'}]"
                )
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
