"""Full benchmark loop and accuracy computation (issue #10).

Per question (SPEC §7.4):
  1. load question and options
  2. route and extract hints
  3. retrieve route-aware candidates
  4. rerank candidate evidence packs
  5. generate final choice
  6. save prediction plus evidence trace

Outputs:
  outputs/predictions.json
  outputs/evidence_traces.jsonl
  outputs/submissions.json
  outputs/metrics.json   (only when ground truth exists)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from castlerag.schemas import EvalQuestion


def run_eval(
    questions: Dict[str, EvalQuestion],
    config_path: Optional[Path] = None,
    answers_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> None:
    """Run the full prediction loop and write all output files."""
    raise NotImplementedError("Implemented in issue #10")
