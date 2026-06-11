"""Normalization of auxiliary modalities: heartrate, gaze, photo, thermal, aux video."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterator, Optional

from castlerag.dataset.manifest import _extract_timestamp_hint
from castlerag.preprocess.media import get_video_duration
from castlerag.schemas import AuxRecord

log = logging.getLogger(__name__)


def _aux_record(
    source_type: str,
    participant: Optional[str],
    day: str,
    path: Path,
    absolute_start: int,
    absolute_end: int,
    summary_text: Optional[str] = None,
    has_reliable_timestamp: bool = True,
    version: str = "0.1.0",
) -> AuxRecord:
    """Build and return an AuxRecord with a stable content-hash clip_id."""
    id_material = "|".join(
        [
            source_type,
            participant or "",
            day,
            str(path),
            str(absolute_start),
            str(absolute_end),
        ]
    )
    stable_suffix = hashlib.sha1(id_material.encode()).hexdigest()[:12]
    return AuxRecord(
        clip_id=(
            f"{source_type}_{path.stem}_{absolute_start:013d}_{absolute_end:013d}_"
            f"{stable_suffix}"
        ),
        source_type=source_type,  # type: ignore[arg-type]
        modality=_modality_for(source_type),
        day=day,
        participant_id=participant,
        aux_owner=participant,
        asset_path=str(path),
        summary_text=summary_text,
        absolute_start=absolute_start,
        absolute_end=absolute_end,
        has_reliable_timestamp=has_reliable_timestamp,
        version=version,
    )


def _modality_for(source_type: str) -> str:
    """Return the modality string ('image', 'video', or 'text') for a source type."""
    if source_type in ("aux_photo", "aux_thermal"):
        return "image"
    if source_type == "aux_video":
        return "video"
    return "text"


def iter_heartrate_records(
    aux_root: Path, participant: str, day: str, version: str = "0.1.0"
) -> Iterator[AuxRecord]:
    """Yield 60-second heartrate summary records.

    Fields: bpm_mean, bpm_min, bpm_max, bpm_delta_prev embedded in raw_features.
    summary_text rendered as:
      "Heartrate for {participant} at {day} {HH:MM}-{HH:MM}: mean {bpm} bpm, ..."
    """
    raise NotImplementedError(
        "Implemented after auxiliary timestamp validation (SPEC §9.4)"
    )


def iter_gaze_records(
    aux_root: Path, participant: str, day: str, version: str = "0.1.0"
) -> Iterator[AuxRecord]:
    """Yield 10-second gaze summary records for intervals with data rows."""
    raise NotImplementedError(
        "Implemented after gaze column semantics are validated (SPEC §9.4)"
    )


def iter_photo_records(
    aux_root: Path,
    participant: str,
    day: str,
    version: str = "0.1.0",
) -> Iterator[AuxRecord]:
    """Yield one AuxRecord per photo file under aux_root/photo/{participant}/.

    Timestamp comes from EXIF when available; falls back to the filename
    timestamp hint.  Absolute times default to 0 when no timestamp can be
    derived — callers should set them during the indexing pass once the
    recording calendar is confirmed.
    """
    photo_dir = aux_root / "photo" / participant
    if not photo_dir.exists():
        return
    for f in sorted(photo_dir.rglob("*")):
        if not f.is_file():
            continue
        abs_start = _exif_unix_ms(f)
        if abs_start is None:
            ts_hint = _extract_timestamp_hint(f)
            abs_start = _parse_hint_to_ms(ts_hint) if ts_hint else 0
        reliable = abs_start != 0
        abs_end = abs_start + 1
        yield _aux_record(
            source_type="aux_photo",
            participant=participant,
            day=day,
            path=f,
            absolute_start=abs_start,
            absolute_end=abs_end,
            summary_text=f"Photo by {participant}",
            has_reliable_timestamp=reliable,
            version=version,
        )


def iter_thermal_records(
    aux_root: Path,
    day: str,
    version: str = "0.1.0",
) -> Iterator[AuxRecord]:
    """Yield one AuxRecord per thermal BMP image under aux_root/thermal/."""
    thermal_dir = aux_root / "thermal"
    if not thermal_dir.exists():
        return
    for f in sorted(thermal_dir.rglob("*")):
        if not f.is_file():
            continue
        abs_start = 0
        ts_hint = _extract_timestamp_hint(f)
        reliable = False
        if ts_hint:
            abs_start = _parse_hint_to_ms(ts_hint)
            reliable = abs_start != 0
        abs_end = abs_start + 1
        yield _aux_record(
            source_type="aux_thermal",
            participant=None,
            day=day,
            path=f,
            absolute_start=abs_start,
            absolute_end=abs_end,
            summary_text="Thermal frame",
            has_reliable_timestamp=reliable,
            version=version,
        )


def iter_aux_video_records(
    aux_root: Path,
    participant: str,
    day: str,
    version: str = "0.1.0",
) -> Iterator[AuxRecord]:
    """Yield AuxRecords for auxiliary video files under aux_root/video/{participant}/.

    Files ≤30 s → one record.  Longer files → re-windowed into 30 s clips.
    Duration is obtained via ffprobe; files where ffprobe fails are skipped.
    Absolute timestamps must be anchored from a filename timestamp hint. Files
    without a recoverable timestamp are skipped rather than emitting fake
    file-relative epoch values.
    """
    video_dir = aux_root / "video" / participant
    if not video_dir.exists():
        return
    for f in sorted(video_dir.rglob("*")):
        if not f.is_file():
            continue
        ts_hint = _extract_timestamp_hint(f)
        if not ts_hint:
            log.warning("Skipping aux video %s: no timestamp hint in filename", f)
            continue
        base_unix_ms = _parse_hint_to_ms(ts_hint)
        if base_unix_ms <= 0:
            log.warning(
                "Skipping aux video %s: could not parse timestamp hint %r", f, ts_hint
            )
            continue
        try:
            duration = get_video_duration(f)
        except Exception as exc:
            log.warning("Skipping aux video %s: duration probe failed (%s)", f, exc)
            continue
        if duration <= 30.0:
            yield _aux_record(
                source_type="aux_video",
                participant=participant,
                day=day,
                path=f,
                absolute_start=base_unix_ms,
                absolute_end=max(base_unix_ms + 1, base_unix_ms + int(duration * 1000)),
                version=version,
            )
        else:
            start = 0.0
            while start < duration:
                end = min(start + 30.0, duration)
                abs_start = base_unix_ms + int(start * 1000)
                abs_end = max(abs_start + 1, base_unix_ms + int(end * 1000))
                yield _aux_record(
                    source_type="aux_video",
                    participant=participant,
                    day=day,
                    path=f,
                    absolute_start=abs_start,
                    absolute_end=abs_end,
                    version=version,
                )
                start += 30.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _exif_unix_ms(path: Path) -> Optional[int]:
    """Return EXIF DateTimeOriginal as Unix ms, or None if not available."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except ImportError:
        return None
    try:
        img = Image.open(path)
        exif = img._getexif()  # type: ignore[attr-defined]
        if not exif:
            return None
        tag_map = {v: k for k, v in TAGS.items()}
        dto_tag = tag_map.get("DateTimeOriginal")
        dto = exif.get(dto_tag) if dto_tag else None
        if not dto:
            return None
        from datetime import datetime, timezone

        dt = datetime.strptime(str(dto), "%Y:%m:%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _parse_hint_to_ms(hint: str) -> int:
    """Best-effort conversion of a timestamp hint string to Unix ms.

    Supports: YYYY-MM-DD_HH-MM-SS, YYYYMMDD_HHMMSS, YYYY-MM-DD.
    Returns 0 on any parse failure.
    """
    import re
    from datetime import datetime, timezone

    patterns = [
        (
            r"(\d{4})[_-](\d{2})[_-](\d{2})[T_-](\d{2})[_:-](\d{2})[_:-](\d{2})",
            "%Y%m%d%H%M%S",
            6,
        ),
        (r"(\d{4})(\d{2})(\d{2})[T_](\d{2})(\d{2})(\d{2})", "%Y%m%d%H%M%S", 6),
        (r"(\d{4})[_-](\d{2})[_-](\d{2})", "%Y%m%d", 3),
    ]
    for pat, fmt, n in patterns:
        m = re.search(pat, hint)
        if m:
            joined = "".join(m.groups()[:n])
            try:
                dt = datetime.strptime(joined, fmt).replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)
            except ValueError:
                continue
    return 0
