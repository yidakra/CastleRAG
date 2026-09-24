#!/usr/bin/env python
"""Count Qdrant points per camera for one day — the Bug B (#50) ingest check.

Prints exact point counts per ``camera_type`` and per ``camera_id`` x
``source_type`` for the collection, then lists which of the expected fixed
room cameras are MISSING (zero ``main_clip`` points). Read-only.

    python scripts/qdrant_camera_counts.py --day day1
    python scripts/qdrant_camera_counts.py --day day1 --missing-only  # e.g. "Reading"

Exit code 0 always, unless --fail-if-missing is set and a fixed camera has no
``main_clip`` points (exit 3) — handy as a gate in a SLURM job.
"""

from __future__ import annotations

import argparse
import sys

FIXED = ["Kitchen", "Living1", "Living2", "Meeting", "Reading"]
SOURCES = ["main_clip", "main_event_summary", "transcript_window"]


def _count(client, collection: str, **match: str) -> int:
    from qdrant_client.http import models as qm

    must = [
        qm.FieldCondition(key=k, match=qm.MatchValue(value=v))
        for k, v in match.items()
    ]
    res = client.count(
        collection_name=collection,
        count_filter=qm.Filter(must=must) if must else None,
        exact=True,
    )
    return int(res.count)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--collection", default="castle_multimodal_v1")
    ap.add_argument("--day", default="day1")
    ap.add_argument("--cameras", nargs="*", default=FIXED, help="fixed cams to check")
    ap.add_argument("--missing-only", action="store_true",
                    help="print only the space-separated missing fixed cameras")
    ap.add_argument("--fail-if-missing", action="store_true")
    args = ap.parse_args()

    from qdrant_client import QdrantClient

    client = QdrantClient(host=args.host, port=args.port, timeout=120)
    c = args.collection
    missing = [
        cam for cam in args.cameras
        if _count(client, c, day=args.day, camera_id=cam, source_type="main_clip") == 0
    ]
    if args.missing_only:
        print(" ".join(missing))
        return 3 if (missing and args.fail_if_missing) else 0

    total = _count(client, c)
    print(f"collection {c}: {total} points total")
    for ctype in ("ego", "fixed"):
        n = _count(client, c, day=args.day, camera_type=ctype)
        print(f"  {args.day} camera_type={ctype:<5} {n:>7}")

    # Per-camera breakdown (camera ids discovered from a payload facet if the
    # server supports it; otherwise ego roster is not known here, so fixed only).
    cams = list(args.cameras)
    try:
        facet = client.facet(collection_name=c, key="camera_id", limit=64, exact=True)
        cams = sorted({str(h.value) for h in facet.hits} | set(cams))
    except Exception:  # older qdrant-client / server without facet API
        pass
    header = "  " + "camera".ljust(10) + "".join(s.rjust(20) for s in SOURCES)
    print(header)
    for cam in cams:
        row = [
            _count(client, c, day=args.day, camera_id=cam, source_type=s)
            for s in SOURCES
        ]
        tag = "  (fixed)" if cam in FIXED else ""
        print("  " + cam.ljust(10) + "".join(str(n).rjust(20) for n in row) + tag)
    # Payload sanity for fixed points: room must be set, participant_id absent.
    for cam in args.cameras:
        n_room = _count(client, c, day=args.day, camera_id=cam, room=cam)
        n_all = _count(client, c, day=args.day, camera_id=cam)
        if n_all and n_room != n_all:
            print(f"  WARN {cam}: only {n_room}/{n_all} points carry room={cam!r}")
    print(f"missing fixed cameras (0 main_clip points): {missing or 'none'}")
    return 3 if (missing and args.fail_if_missing) else 0


if __name__ == "__main__":
    sys.exit(main())
