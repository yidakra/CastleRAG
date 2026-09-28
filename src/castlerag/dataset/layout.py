"""CASTLE dataset path discovery and camera metadata.

Layout:
  main/day{1..4}/{camera}/video/{HH}.mp4
  main/day{1..4}/{camera}/transcript/{HH}.json
  main/day{1..4}/{camera}/metadata/{HH}.*.csv

Camera types:
  ego   — participant-worn camera; camera_id == participant_id
  fixed — room-mounted camera; participant_id is None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional


@dataclass(frozen=True)
class CameraInfo:
    camera_id: str
    camera_type: str  # "ego" | "fixed"
    participant_id: Optional[str]  # None for fixed cameras
    room: Optional[str]  # None for ego cameras


def build_camera_registry(
    ego_cameras: List[str],
    exo_cameras: List[str],
) -> Dict[str, CameraInfo]:
    """Build a {camera_id: CameraInfo} registry from config lists."""
    overlap = set(ego_cameras) & set(exo_cameras)
    if overlap:
        raise ValueError(
            f"Camera IDs must be unique across ego and exo lists; "
            f"found in both: {sorted(overlap)}"
        )
    registry: Dict[str, CameraInfo] = {}
    for cam in ego_cameras:
        registry[cam] = CameraInfo(
            camera_id=cam,
            camera_type="ego",
            participant_id=cam,
            room=None,
        )
    for cam in exo_cameras:
        registry[cam] = CameraInfo(
            camera_id=cam,
            camera_type="fixed",
            participant_id=None,
            room=cam,
        )
    return registry


def scoped_cameras(
    ego_cameras: List[str],
    exo_cameras: List[str],
    camera_scope: str = "ego",
    only: Optional[List[str]] = None,
) -> List[str]:
    """Return the ordered camera ids in scope, optionally narrowed to ``only``.

    ``camera_scope="ego"`` keeps the ego roster; ``"all"`` appends the fixed
    room cameras. ``only`` restricts the result to a subset (e.g. the 5 fixed
    cameras for an additive ingest, or one camera per parallel worker). Names
    in ``only`` that are not in scope raise ``ValueError`` so a typo or a
    fixed camera requested under ``camera_scope="ego"`` fails fast instead of
    silently processing nothing.
    """
    in_scope = list(ego_cameras)
    if camera_scope == "all":
        in_scope = in_scope + list(exo_cameras)
    if not only:
        return in_scope
    unknown = [cam for cam in only if cam not in in_scope]
    if unknown:
        raise ValueError(
            f"Camera(s) {unknown} not in scope (camera_scope={camera_scope!r}, "
            f"in scope: {in_scope}). Fixed cameras need camera_scope='all'."
        )
    wanted = set(only)
    return [cam for cam in in_scope if cam in wanted]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def hour_video_path(root: Path, day: str, camera_id: str, hour: int) -> Path:
    """Return the expected MP4 path for one camera-hour recording."""
    return root / "main" / day / camera_id / "video" / f"{hour:02d}.mp4"


def hour_transcript_path(root: Path, day: str, camera_id: str, hour: int) -> Path:
    """Return the expected JSON transcript path for one camera-hour recording."""
    return root / "main" / day / camera_id / "transcript" / f"{hour:02d}.json"


def hour_metadata_paths(root: Path, day: str, camera_id: str, hour: int) -> List[Path]:
    """Return sorted paths of all metadata CSV files for one camera-hour."""
    meta_dir = root / "main" / day / camera_id / "metadata"
    if not meta_dir.exists():
        return []
    return sorted(meta_dir.glob(f"{hour:02d}.*.csv"))


def is_novideo(root: Path, day: str, camera_id: str, hour: int) -> bool:
    """Return True when the hour is marked as missing video."""
    novideo = root / "main" / day / camera_id / "video" / f"{hour:02d}.novideo"
    video = hour_video_path(root, day, camera_id, hour)
    return novideo.exists() and not video.exists()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass
class HourAsset:
    day: str
    camera_id: str
    camera_type: str
    participant_id: Optional[str]
    room: Optional[str]
    hour: int
    video_path: Path
    transcript_path: Optional[Path]
    metadata_paths: List[Path] = field(default_factory=list)
    missing_video: bool = False


def discover_hours(
    root: Path,
    ego_cameras: List[str],
    exo_cameras: List[str],
    days: List[int],
    hours: List[int],
    camera_scope: str = "ego",
    include_novideo: bool = False,
    cameras: Optional[List[str]] = None,
) -> Iterator[HourAsset]:
    """Yield one HourAsset for every available hour in scope.

    camera_scope="ego" (default) skips exo cameras entirely — this is the
    TAHAKOM-validated baseline that indexes only the 10 egocentric streams.
    Pass "all" to include fixed cameras as an extension.

    include_novideo=False (default) skips .novideo hours; set True to yield
    them with missing_video=True so manifest writers can record them explicitly.

    cameras (optional) narrows discovery to a subset of the in-scope cameras;
    see ``scoped_cameras``.
    """
    registry = build_camera_registry(ego_cameras, exo_cameras)

    in_scope = scoped_cameras(ego_cameras, exo_cameras, camera_scope, only=cameras)

    for day_num in days:
        day_str = f"day{day_num}"
        for camera_id in in_scope:
            cam = registry[camera_id]
            for hour in hours:
                novideo = is_novideo(root, day_str, camera_id, hour)
                if novideo:
                    if include_novideo:
                        video_path = hour_video_path(root, day_str, camera_id, hour)
                        tx_path = hour_transcript_path(root, day_str, camera_id, hour)
                        yield HourAsset(
                            day=day_str,
                            camera_id=camera_id,
                            camera_type=cam.camera_type,
                            participant_id=cam.participant_id,
                            room=cam.room,
                            hour=hour,
                            video_path=video_path,
                            transcript_path=tx_path if tx_path.exists() else None,
                            metadata_paths=hour_metadata_paths(
                                root, day_str, camera_id, hour
                            ),
                            missing_video=True,
                        )
                    continue
                video_path = hour_video_path(root, day_str, camera_id, hour)
                if not video_path.exists():
                    continue
                tx_path = hour_transcript_path(root, day_str, camera_id, hour)
                yield HourAsset(
                    day=day_str,
                    camera_id=camera_id,
                    camera_type=cam.camera_type,
                    participant_id=cam.participant_id,
                    room=cam.room,
                    hour=hour,
                    video_path=video_path,
                    transcript_path=tx_path if tx_path.exists() else None,
                    metadata_paths=hour_metadata_paths(root, day_str, camera_id, hour),
                    missing_video=False,
                )
