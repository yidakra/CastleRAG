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
    preds = load_predictions(run_dir / "predictions.json")
    traces = {}
    tpath = run_dir / "evidence_traces.jsonl"
    if tpath.exists():
        for line in tpath.read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                traces[t["question_id"]] = t
    return preds, traces


def _row(qid: str, q, preds, traces, anchor: str) -> Optional[dict]:
    p = preds.get(qid)
    if p is None:
        return None
    cams = traces.get(qid, {}).get("top_evidence_cameras") or []
    return {
        "supported": bool(p.is_supported),
        "correct": q.ground_truth is not None and p.predicted_answer == q.ground_truth,
        "cams": cams,
        "fixed": [c for c in cams if c in FIXED],
        "anchor_hit": bool(anchor) and anchor in cams,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", type=Path, default=Path("data/smoke_day1.csv"))
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--baseline-dir", type=Path, default=None)
    args = ap.parse_args()

    questions = load_questions(args.questions)
    anchors = _anchors(args.questions) if args.questions.suffix == ".csv" else {}
    preds, traces = _load_run(args.run_dir)
    base = _load_run(args.baseline_dir) if args.baseline_dir else None

    graded = [qid for qid in preds if qid in questions]
    n_correct = sum(
        preds[q].predicted_answer == questions[q].ground_truth for q in graded
    )
    n_zero = sum(not preds[q].is_supported for q in graded)
    n_fixed = sum(
        any(c in FIXED for c in traces.get(q, {}).get("top_evidence_cameras") or [])
        for q in graded
    )
    print(f"run: {args.run_dir}")
    print(f"  accuracy          : {n_correct}/{len(graded)}")
    print(f"  zero-evidence     : {n_zero}/{len(graded)}  (issue #50 baseline 8/40; "
          f"post-#53 ~11/40)")
    print(
        f"  fixed-cam evidence: {n_fixed}/{len(graded)} questions cite "
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
        r = _row(qid, q, preds, traces, anchor)
        if r is None:
            print(f"  -- not run: {prefix}")
            continue
        line = (
            f"  {'EVID' if r['supported'] else 'ZERO'} "
            f"{'OK ' if r['correct'] else 'BAD'} anchor={anchor or '-':<8} "
            f"anchor_in_evidence={'Y' if r['anchor_hit'] else 'n'} "
            f"fixed={','.join(r['fixed']) or '-':<16} {prefix[:48]}"
        )
        if base is not None:
            b = _row(qid, q, base[0], base[1], anchor)
            if b is not None:
                line += (
                    f"   [baseline {'EVID' if b['supported'] else 'ZERO'}"
                    f"/{'OK' if b['correct'] else 'BAD'}]"
                )
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
