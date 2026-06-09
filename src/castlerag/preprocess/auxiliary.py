"""Normalization of auxiliary modalities: heartrate, gaze, photo, thermal, aux video."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from castlerag.schemas import AuxRecord


def iter_heartrate_records(
    aux_root: Path, participant: str, day: str, version: str = "0.1.0"
) -> Iterator[AuxRecord]:
    """Yield 60-second heartrate summary records.

    Fields: bpm_mean, bpm_min, bpm_max, bpm_delta_prev embedded in raw_features.
    summary_text rendered as:
      "Heartrate for {participant} at {day} {HH:MM}-{HH:MM}: mean {bpm} bpm, ..."
    """
    raise NotImplementedError("Implemented after auxiliary timestamp validation (SPEC §9.4)")


def iter_gaze_records(
    aux_root: Path, participant: str, day: str, version: str = "0.1.0"
) -> Iterator[AuxRecord]:
    """Yield 10-second gaze summary records for intervals with data rows."""
    raise NotImplementedError("Implemented after gaze column semantics are validated (SPEC §9.4)")


def iter_photo_records(
    aux_root: Path, participant: str, day: str, version: str = "0.1.0"
) -> Iterator[AuxRecord]:
    """Yield one AuxRecord per photo.  EXIF timestamp preferred; filename fallback."""
    raise NotImplementedError("Implemented in issue #4")


def iter_thermal_records(
    aux_root: Path, day: str, version: str = "0.1.0"
) -> Iterator[AuxRecord]:
    """Yield one AuxRecord per thermal BMP image."""
    raise NotImplementedError("Implemented in issue #4")


def iter_aux_video_records(
    aux_root: Path, participant: str, day: str, version: str = "0.1.0"
) -> Iterator[AuxRecord]:
    """Yield AuxRecords for auxiliary video files.

    Files ≤30 s → one record.  Longer files → re-windowed into 30 s clips.
    """
    raise NotImplementedError("Implemented in issue #4")
