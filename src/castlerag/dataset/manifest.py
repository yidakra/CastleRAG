"""Normalized manifest generation for CASTLE dataset discovery (SPEC §2.1, §2.8).

Two manifests are produced:
- Main manifest: one row per camera-hour for main/day{1..4}/{camera}/ assets,
  including .novideo hours (missing_video=True) so downstream jobs can skip
  them intentionally rather than discover them at runtime.
- Auxiliary manifest: one row per file under auxiliary/{gaze,heartrate,photo,
  thermal,video}/, with best-effort participant and timestamp extraction.

Manifests are written as JSONL (one JSON object per line), versioned via the
`version` field on each row.  The output is deterministic given a fixed dataset
root: paths are sorted before iteration and rows carry no wall-clock timestamps.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from castlerag.dataset.layout import (
    build_camera_registry,
    hour_transcript_path,
    hour_video_path,
    is_novideo,
)
from castlerag.schemas import AuxSourceType, CameraType

# ---------------------------------------------------------------------------
# Manifest row models
# ---------------------------------------------------------------------------

_MANIFEST_VERSION = "0.1.0"


class MainHourRow(BaseModel):
    """One row in the main-hour manifest."""

    day: str
    camera_id: str
    camera_type: CameraType
    participant_id: Optional[str]
    hour: int
    video_path: str
    transcript_path: Optional[str]
    metadata_dir: str
    missing_video: bool
    version: str = _MANIFEST_VERSION


class AuxAssetRow(BaseModel):
    """One row in the auxiliary-asset manifest."""

    source_type: AuxSourceType
    participant_id: Optional[str]
    path: str
    day_hint: Optional[str] = None
    timestamp_hint: Optional[str] = None
    version: str = _MANIFEST_VERSION


# ---------------------------------------------------------------------------
# Timestamp hint extraction
# ---------------------------------------------------------------------------

# Ordered from most-specific to least-specific so the first match wins.
_TS_PATTERNS = [
    re.compile(r"(\d{4}[_-]\d{2}[_-]\d{2}[T_-]\d{2}[_:-]\d{2}[_:-]\d{2})"),
    re.compile(r"(\d{8}[T_]\d{6})"),
    re.compile(r"(\d{4}[_-]\d{2}[_-]\d{2})"),
]


def _extract_timestamp_hint(path: Path) -> Optional[str]:
    """Return the first timestamp-like substring found in the filename stem."""
    for pat in _TS_PATTERNS:
        m = pat.search(path.stem)
        if m:
            return m.group(1)
    return None


def _participant_from_stem(stem: str) -> Optional[str]:
    """Best-effort participant extraction from a bare filename stem.

    Used for gaze files where the participant may be encoded in the name
    (e.g. "Allie_gaze" → "Allie").  Returns None when the leading token
    contains digits or is empty.
    """
    token = re.split(r"[_\-.]", stem)[0]
    if token and token.isalpha():
        return token
    return None


# ---------------------------------------------------------------------------
# Main manifest discovery
# ---------------------------------------------------------------------------


def discover_main_manifest(
    root: Path,
    ego_cameras: List[str],
    exo_cameras: List[str],
    days: List[int],
    hours: List[int],
    camera_scope: str = "ego",
    version: str = _MANIFEST_VERSION,
) -> List[MainHourRow]:
    """Return a MainHourRow for every expected camera-hour slot in scope.

    Rows are emitted for:
    - Hours where a .mp4 file is present (missing_video=False).
    - Hours where a .novideo marker exists (missing_video=True).
    - Hours where neither exists are excluded (camera was not recording).

    camera_scope="ego" restricts to the 10 ego participants (TAHAKOM baseline).
    Pass "all" to include the 5 fixed exo cameras as well.
    """
    registry = build_camera_registry(ego_cameras, exo_cameras)
    in_scope = list(ego_cameras)
    if camera_scope == "all":
        in_scope = in_scope + list(exo_cameras)

    rows: List[MainHourRow] = []
    for day_num in days:
        day_str = f"day{day_num}"
        for camera_id in in_scope:
            cam = registry[camera_id]
            for hour in hours:
                novideo = is_novideo(root, day_str, camera_id, hour)
                video_path = hour_video_path(root, day_str, camera_id, hour)
                if not novideo and not video_path.exists():
                    continue
                tx_path = hour_transcript_path(root, day_str, camera_id, hour)
                meta_dir = root / "main" / day_str / camera_id / "metadata"
                rows.append(
                    MainHourRow(
                        day=day_str,
                        camera_id=camera_id,
                        camera_type=cam.camera_type,
                        participant_id=cam.participant_id,
                        hour=hour,
                        video_path=str(video_path),
                        transcript_path=str(tx_path) if tx_path.exists() else None,
                        metadata_dir=str(meta_dir),
                        missing_video=novideo,
                        version=version,
                    )
                )
    return rows


# ---------------------------------------------------------------------------
# Auxiliary manifest discovery
# ---------------------------------------------------------------------------


def discover_aux_manifest(
    root: Path,
    version: str = _MANIFEST_VERSION,
) -> List[AuxAssetRow]:
    """Return an AuxAssetRow for every file under root/auxiliary/.

    Layout:
      auxiliary/heartrate/{participant}/{files}
      auxiliary/gaze/{files.csv}
      auxiliary/photo/{participant}/{files}
      auxiliary/thermal/{files}
      auxiliary/video/{participant}/{files}

    Participant is derived from the immediate subdirectory name for heartrate,
    photo, and video.  For gaze, a best-effort extraction from the filename stem
    is attempted.  For thermal there is no participant association.
    """
    aux_root = root / "auxiliary"
    if not aux_root.exists():
        return []

    rows: List[AuxAssetRow] = []

    def _add(source_type: AuxSourceType, participant: Optional[str], f: Path) -> None:
        rows.append(
            AuxAssetRow(
                source_type=source_type,
                participant_id=participant,
                path=str(f),
                timestamp_hint=_extract_timestamp_hint(f),
                version=version,
            )
        )

    # heartrate/{participant}/{files}
    hr_root = aux_root / "heartrate"
    if hr_root.exists():
        for participant_dir in sorted(hr_root.iterdir()):
            if participant_dir.is_dir():
                for f in sorted(participant_dir.rglob("*")):
                    if f.is_file():
                        _add("aux_heartrate", participant_dir.name, f)

    # gaze/{files}
    gaze_root = aux_root / "gaze"
    if gaze_root.exists():
        for f in sorted(gaze_root.iterdir()):
            if f.is_file():
                _add("aux_gaze", _participant_from_stem(f.stem), f)

    # photo/{participant}/{files}
    photo_root = aux_root / "photo"
    if photo_root.exists():
        for participant_dir in sorted(photo_root.iterdir()):
            if participant_dir.is_dir():
                for f in sorted(participant_dir.rglob("*")):
                    if f.is_file():
                        _add("aux_photo", participant_dir.name, f)

    # thermal/{files}
    thermal_root = aux_root / "thermal"
    if thermal_root.exists():
        for f in sorted(thermal_root.rglob("*")):
            if f.is_file():
                _add("aux_thermal", None, f)

    # video/{participant}/{files}
    vid_root = aux_root / "video"
    if vid_root.exists():
        for participant_dir in sorted(vid_root.iterdir()):
            if participant_dir.is_dir():
                for f in sorted(participant_dir.rglob("*")):
                    if f.is_file():
                        _add("aux_video", participant_dir.name, f)

    return rows


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def write_manifest(rows: List[BaseModel], output_path: Path) -> Path:
    """Write manifest rows to a versioned JSONL file; return the path written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")
    return output_path


def read_main_manifest(path: Path) -> List[MainHourRow]:
    """Load a main-hour JSONL manifest."""
    rows: List[MainHourRow] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(MainHourRow.model_validate_json(line))
    return rows


def read_aux_manifest(path: Path) -> List[AuxAssetRow]:
    """Load an auxiliary-asset JSONL manifest."""
    rows: List[AuxAssetRow] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(AuxAssetRow.model_validate_json(line))
    return rows
