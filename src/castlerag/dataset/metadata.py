"""Hourly sensor CSV loaders from main/{day}/{camera}/metadata/.

The exact column semantics of these CSVs must be validated against the
raw CASTLE files before use (see SPEC §9.4).  This module provides stubs
and a generic loader that returns the raw DataFrame for now.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

# Intentionally no pandas import at module level — not guaranteed in all envs.


def load_metadata_csv(path: Path) -> List[Dict[str, Any]]:
    """Load a single metadata CSV and return rows as a list of dicts."""
    import csv
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_hour_metadata(paths: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Load all metadata CSVs for a given hour, keyed by sensor name.

    Sensor name is derived from the filename pattern: {HH}.{sensor}.csv
    """
    result: Dict[str, List[Dict[str, Any]]] = {}
    for p in paths:
        parts = p.stem.split(".", 1)
        sensor = parts[1] if len(parts) == 2 else p.stem
        result[sensor] = load_metadata_csv(p)
    return result
