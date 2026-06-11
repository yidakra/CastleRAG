"""Tests for src/castlerag/dataset/metadata.py"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import pytest

from castlerag.dataset.metadata import load_hour_metadata, load_metadata_csv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: List[dict]) -> None:
    """Write a list of dicts to a CSV file."""
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# load_metadata_csv
# ---------------------------------------------------------------------------


def test_load_metadata_csv_normal(tmp_path: Path):
    p = tmp_path / "data.csv"
    _write_csv(p, [{"ts": "100", "value": "42"}, {"ts": "200", "value": "7"}])
    rows = load_metadata_csv(p)
    assert len(rows) == 2
    assert rows[0]["ts"] == "100"
    assert rows[0]["value"] == "42"
    assert rows[1]["ts"] == "200"
    assert rows[1]["value"] == "7"


def test_load_metadata_csv_returns_list_of_dicts(tmp_path: Path):
    p = tmp_path / "sensor.csv"
    _write_csv(p, [{"a": "1", "b": "2"}])
    rows = load_metadata_csv(p)
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict)


def test_load_metadata_csv_single_row(tmp_path: Path):
    p = tmp_path / "single.csv"
    _write_csv(p, [{"x": "hello"}])
    rows = load_metadata_csv(p)
    assert len(rows) == 1
    assert rows[0]["x"] == "hello"


def test_load_metadata_csv_multiple_columns(tmp_path: Path):
    p = tmp_path / "multi.csv"
    _write_csv(p, [{"col1": "v1", "col2": "v2", "col3": "v3"}])
    rows = load_metadata_csv(p)
    assert set(rows[0].keys()) == {"col1", "col2", "col3"}


# ---------------------------------------------------------------------------
# load_hour_metadata
# ---------------------------------------------------------------------------


def test_load_hour_metadata_multi_sensor(tmp_path: Path):
    imu_path = tmp_path / "08.imu.csv"
    gps_path = tmp_path / "08.gps.csv"
    _write_csv(imu_path, [{"ax": "0.1", "ay": "0.2"}])
    _write_csv(gps_path, [{"lat": "51.5", "lon": "-0.1"}])

    result = load_hour_metadata([imu_path, gps_path])

    assert set(result.keys()) == {"imu", "gps"}
    assert result["imu"][0]["ax"] == "0.1"
    assert result["gps"][0]["lat"] == "51.5"


def test_load_hour_metadata_single_sensor(tmp_path: Path):
    p = tmp_path / "09.heartrate.csv"
    _write_csv(p, [{"bpm": "72"}])
    result = load_hour_metadata([p])
    assert "heartrate" in result
    assert result["heartrate"][0]["bpm"] == "72"


def test_load_hour_metadata_sensor_key_is_suffix(tmp_path: Path):
    """Sensor key must be the part after the first dot, not the full stem."""
    p = tmp_path / "10.accel.csv"
    _write_csv(p, [{"val": "1"}])
    result = load_hour_metadata([p])
    assert list(result.keys()) == ["accel"]


def test_load_hour_metadata_empty_list(tmp_path: Path):
    result = load_hour_metadata([])
    assert result == {}


def test_load_hour_metadata_raises_on_bare_hour_filename(tmp_path: Path):
    """Files named like '08.csv' (no sensor suffix) must raise ValueError."""
    p = tmp_path / "08.csv"
    _write_csv(p, [{"x": "1"}])
    with pytest.raises(ValueError, match="08.csv"):
        load_hour_metadata([p])


def test_load_hour_metadata_raises_on_no_dot_stem(tmp_path: Path):
    """Files with stems containing no dot must raise ValueError."""
    p = tmp_path / "metadata.csv"
    _write_csv(p, [{"x": "1"}])
    with pytest.raises(ValueError, match="metadata.csv"):
        load_hour_metadata([p])


def test_load_hour_metadata_data_is_correct(tmp_path: Path):
    """Rows loaded via load_hour_metadata match what load_metadata_csv would return."""
    p = tmp_path / "08.temp.csv"
    _write_csv(p, [{"celsius": "36.6"}, {"celsius": "37.0"}])
    result = load_hour_metadata([p])
    assert len(result["temp"]) == 2
    assert result["temp"][1]["celsius"] == "37.0"
