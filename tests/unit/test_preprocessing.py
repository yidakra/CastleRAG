"""Tests for preprocessing modules and artifact contracts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from castlerag.dataset.transcripts import load_raw_segments, merge_into_windows
from castlerag.preprocess.event_compress import (
    _event_summary_id,
    compress_clips_to_event,
)
from castlerag.preprocess.windows import (
    VideoWindow,
    iter_windows,
    mark_placeholder_windows,
)
from castlerag.schemas import ClipRecord, TranscriptSegment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clip(
    n: int,
    start_ms: int,
    end_ms: int,
    caption: str = "test clip",
) -> ClipRecord:
    return ClipRecord(
        clip_id=f"clip_{n:04d}",
        parent_source_id="vid_0",
        day="day1",
        hour=8,
        camera_id="Allie",
        camera_type="ego",
        participant_id="Allie",
        start_seconds=float(n * 30),
        end_seconds=float(n * 30 + 30),
        absolute_start=start_ms,
        absolute_end=end_ms,
        source_video_path="/data/castle/main/day1/Allie/video/08.mp4",
        clip_caption=caption,
    )


def _four_clips() -> List[ClipRecord]:
    base = 1_672_531_200_000
    return [
        _make_clip(i, base + i * 30_000, base + i * 30_000 + 30_000) for i in range(4)
    ]


# ---------------------------------------------------------------------------
# iter_windows
# ---------------------------------------------------------------------------


def test_iter_windows_exact_60s():
    wins = list(iter_windows(Path("v.mp4"), "Allie", "day1", 8, 60.0))
    assert len(wins) == 2
    assert wins[0].start_seconds == 0.0
    assert wins[0].end_seconds == 30.0
    assert wins[1].start_seconds == 30.0
    assert wins[1].end_seconds == 60.0


def test_iter_windows_clip_indices():
    wins = list(iter_windows(Path("v.mp4"), "Allie", "day1", 8, 90.0))
    assert [w.clip_index for w in wins] == [0, 1, 2]


def test_iter_windows_partial_final_window():
    # 65 s → two full 30s windows + one 5s tail
    wins = list(iter_windows(Path("v.mp4"), "Allie", "day1", 8, 65.0))
    assert len(wins) == 3
    assert wins[-1].end_seconds == 65.0


def test_iter_windows_sub_second_tail_dropped():
    # 60.4 s → two full 30s windows, 0.4 s tail dropped
    wins = list(iter_windows(Path("v.mp4"), "Allie", "day1", 8, 60.4))
    assert len(wins) == 2


def test_iter_windows_zero_duration():
    wins = list(iter_windows(Path("v.mp4"), "Allie", "day1", 8, 0.0))
    assert wins == []


def test_iter_windows_short_video_under_30s():
    wins = list(iter_windows(Path("v.mp4"), "Allie", "day1", 8, 25.0))
    assert len(wins) == 1
    assert wins[0].end_seconds == 25.0


def test_iter_windows_metadata():
    wins = list(iter_windows(Path("/data/08.mp4"), "Bjorn", "day2", 14, 30.0))
    w = wins[0]
    assert w.camera_id == "Bjorn"
    assert w.day == "day2"
    assert w.hour == 14
    assert w.source_video_path == Path("/data/08.mp4")
    assert w.is_placeholder is False


def test_iter_windows_non_default_clip_seconds():
    wins = list(
        iter_windows(
            Path("v.mp4"), "Allie", "day1", 8, 60.0, clip_seconds=15, stride_seconds=15
        )
    )
    assert len(wins) == 4
    assert wins[0].end_seconds == 15.0


def test_iter_windows_rejects_zero_clip_seconds():
    with pytest.raises(ValueError, match="clip_seconds"):
        list(iter_windows(Path("v.mp4"), "Allie", "day1", 8, 60.0, clip_seconds=0))


def test_iter_windows_rejects_negative_stride():
    with pytest.raises(ValueError, match="stride_seconds"):
        list(iter_windows(Path("v.mp4"), "Allie", "day1", 8, 60.0, stride_seconds=-1))


# ---------------------------------------------------------------------------
# mark_placeholder_windows
# ---------------------------------------------------------------------------


def test_mark_placeholder_windows_invalid_threshold():
    with pytest.raises(ValueError, match="placeholder_threshold"):
        mark_placeholder_windows([], Path("/frames"), placeholder_threshold=1.5)


def test_mark_placeholder_windows_no_frames(tmp_path: Path):
    wins = [VideoWindow("Allie", "day1", 8, 0, 0.0, 30.0, Path("v.mp4"))]
    result = mark_placeholder_windows(wins, tmp_path / "frames")
    assert result[0].is_placeholder is False  # no frames → not marked


def test_mark_placeholder_windows_all_placeholder(tmp_path: Path):
    import numpy as np
    from PIL import Image

    clip_dir = tmp_path / "0"
    clip_dir.mkdir(parents=True)
    # Create uniform gray frames (placeholder)
    for i in range(5):
        img = Image.fromarray(np.full((10, 10), 128, dtype=np.uint8), mode="L")
        img.save(clip_dir / f"{i:04d}.jpg")

    wins = [VideoWindow("Allie", "day1", 8, 0, 0.0, 30.0, Path("v.mp4"))]
    result = mark_placeholder_windows(wins, tmp_path)
    assert result[0].is_placeholder is True


def test_mark_placeholder_windows_real_scene(tmp_path: Path):
    import numpy as np
    from PIL import Image

    clip_dir = tmp_path / "0"
    clip_dir.mkdir(parents=True)
    # Create high-variance frames (real scene)
    rng = np.random.default_rng(42)
    for i in range(5):
        arr = rng.integers(0, 256, (10, 10), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(clip_dir / f"{i:04d}.jpg")

    wins = [VideoWindow("Allie", "day1", 8, 0, 0.0, 30.0, Path("v.mp4"))]
    result = mark_placeholder_windows(wins, tmp_path)
    assert result[0].is_placeholder is False


# ---------------------------------------------------------------------------
# merge_into_windows (transcripts)
# ---------------------------------------------------------------------------


def _seg(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def test_merge_into_windows_single_segment():
    segs = [_seg(0.0, 5.0, "Hello world")]
    base = 1_672_531_200_000
    wins = merge_into_windows(segs, base, "Allie", "ego", "Allie", None, "day1", 8)
    assert len(wins) == 1
    assert wins[0].transcript_text == "Hello world"
    assert wins[0].has_speech is True
    assert wins[0].absolute_start == base
    assert wins[0].absolute_end == base + 5_000


def test_merge_into_windows_multiple_segs_within_cap():
    segs = [_seg(0.0, 5.0, "Hello"), _seg(5.0, 10.0, "world")]
    base = 1_672_531_200_000
    wins = merge_into_windows(segs, base, "Allie", "ego", "Allie", None, "day1", 8)
    assert len(wins) == 1
    assert "Hello" in wins[0].transcript_text
    assert "world" in wins[0].transcript_text


def test_merge_into_windows_splits_on_max_seconds():
    # first seg 0-14, second 14-16 → should split because adding second would exceed 15s
    segs = [_seg(0.0, 14.0, "A " * 10), _seg(14.0, 16.0, "B")]
    base = 1_672_531_200_000
    wins = merge_into_windows(
        segs, base, "Allie", "ego", "Allie", None, "day1", 8, max_seconds=15.0
    )
    assert len(wins) == 2


def test_merge_into_windows_splits_on_max_chars():
    long_text = "x" * 400
    segs = [_seg(0.0, 1.0, long_text), _seg(1.0, 2.0, long_text)]
    base = 1_672_531_200_000
    wins = merge_into_windows(
        segs, base, "Allie", "ego", "Allie", None, "day1", 8, max_chars=384
    )
    assert len(wins) == 2


def test_merge_into_windows_empty_segments():
    base = 1_672_531_200_000
    wins = merge_into_windows([], base, "Allie", "ego", "Allie", None, "day1", 8)
    assert wins == []


def test_merge_into_windows_preserves_metadata():
    segs = [_seg(0.0, 3.0, "Hi")]
    base = 1_672_531_200_000
    wins = merge_into_windows(segs, base, "Bjorn", "ego", "Bjorn", None, "day2", 14)
    w = wins[0]
    assert w.camera_id == "Bjorn"
    assert w.day == "day2"
    assert w.hour == 14
    assert w.participant_id == "Bjorn"


def test_merge_into_windows_window_id_deterministic():
    segs = [_seg(0.0, 5.0, "Test")]
    base = 1_672_531_200_000
    wins1 = merge_into_windows(segs, base, "Allie", "ego", "Allie", None, "day1", 8)
    wins2 = merge_into_windows(segs, base, "Allie", "ego", "Allie", None, "day1", 8)
    assert wins1[0].transcript_window_id == wins2[0].transcript_window_id


def test_merge_into_windows_absolute_timestamps():
    segs = [_seg(3600.0, 3610.0, "Late text")]  # 3600 s into hour
    base = 0
    wins = merge_into_windows(segs, base, "Allie", "ego", "Allie", None, "day1", 8)
    assert wins[0].absolute_start == 3_600_000
    assert wins[0].absolute_end == 3_610_000


def test_merge_into_windows_no_speech_segment():
    segs = [_seg(0.0, 5.0, "")]
    base = 0
    wins = merge_into_windows(segs, base, "Allie", "ego", "Allie", None, "day1", 8)
    assert len(wins) == 1
    assert wins[0].has_speech is False


def test_merge_into_windows_oversized_single_segment_emitted_alone():
    """A segment that alone exceeds max_seconds must be emitted as its own window."""
    segs = [_seg(0.0, 20.0, "Big"), _seg(20.0, 25.0, "Small")]
    base = 0
    wins = merge_into_windows(
        segs, base, "Allie", "ego", "Allie", None, "day1", 8, max_seconds=15.0
    )
    # oversized segment (20s > 15s) gets its own window; "Small" in a second window
    assert len(wins) == 2
    assert wins[0].transcript_text == "Big"
    assert wins[1].transcript_text == "Small"


def test_merge_into_windows_oversized_does_not_accumulate():
    """No further text should be added to a window after a single-segment overflow."""
    segs = [_seg(0.0, 20.0, "X" * 500), _seg(20.0, 21.0, "Y")]
    base = 0
    wins = merge_into_windows(
        segs, base, "Allie", "ego", "Allie", None, "day1", 8, max_chars=384
    )
    assert len(wins) == 2
    assert "Y" not in wins[0].transcript_text


# ---------------------------------------------------------------------------
# load_raw_segments
# ---------------------------------------------------------------------------


def test_load_raw_segments(tmp_path: Path):
    data = {
        "chunks": [
            {"timestamp": [0.0, 5.2], "text": "Hello"},
            {"timestamp": [5.2, 10.0], "text": "world"},
        ]
    }
    f = tmp_path / "08.json"
    f.write_text(json.dumps(data))
    segs = load_raw_segments(f)
    assert len(segs) == 2
    assert segs[0].text == "Hello"
    assert segs[1].start == pytest.approx(5.2)


def test_load_raw_segments_malformed_timestamp(tmp_path: Path):
    data = {"chunks": [{"timestamp": [0.0], "text": "bad"}]}
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Malformed timestamp"):
        load_raw_segments(f)


def test_load_raw_segments_empty(tmp_path: Path):
    f = tmp_path / "empty.json"
    f.write_text(json.dumps({"chunks": []}))
    assert load_raw_segments(f) == []


# ---------------------------------------------------------------------------
# media.py (subprocess-mocked)
# ---------------------------------------------------------------------------


def test_get_video_duration_calls_ffprobe():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="120.5\n", returncode=0)
        from castlerag.preprocess.media import get_video_duration

        dur = get_video_duration(Path("08.mp4"))
        assert dur == pytest.approx(120.5)
        args = mock_run.call_args[0][0]
        assert "ffprobe" in args


def test_extract_subclip_calls_ffmpeg(tmp_path: Path):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        from castlerag.preprocess.media import extract_subclip

        out = extract_subclip(Path("src.mp4"), tmp_path / "out.mp4", 0.0, 30.0)
        assert out == tmp_path / "out.mp4"
        args = mock_run.call_args[0][0]
        assert "ffmpeg" in args
        assert "-i" in args
        assert args.index("-i") < args.index("-ss")
        assert "-reset_timestamps" in args
        assert "-c:v" in args and "libx264" in args
        assert "-c:a" in args and "aac" in args


def test_extract_frames_1fps_calls_ffmpeg(tmp_path: Path):
    def fake_run(cmd, **kw):
        # create fake frame files so the glob finds something
        out_dir = Path(cmd[-1]).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, 4):
            (out_dir / f"{i:04d}.jpg").touch()
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        from castlerag.preprocess.media import extract_frames_1fps

        frames = extract_frames_1fps(Path("src.mp4"), tmp_path / "frames", 0.0, 30.0)
        assert len(frames) == 3
        assert all(str(f).endswith(".jpg") for f in frames)


def test_is_placeholder_frame_uniform(tmp_path: Path):
    import numpy as np
    from PIL import Image

    from castlerag.preprocess.media import is_placeholder_frame

    img = Image.fromarray(np.full((100, 100), 200, dtype=np.uint8), mode="L")
    p = tmp_path / "frame.jpg"
    img.save(p)
    assert is_placeholder_frame(p) is True


def test_is_placeholder_frame_real_scene(tmp_path: Path):
    import numpy as np
    from PIL import Image

    from castlerag.preprocess.media import is_placeholder_frame

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (100, 100), dtype=np.uint8)
    p = tmp_path / "real.jpg"
    Image.fromarray(arr, mode="L").save(p)
    assert is_placeholder_frame(p) is False


# ---------------------------------------------------------------------------
# event_compress
# ---------------------------------------------------------------------------


def test_event_summary_id_deterministic():
    ids = ["clip_0001", "clip_0002", "clip_0003", "clip_0004"]
    assert _event_summary_id(ids) == _event_summary_id(ids)


def test_event_summary_id_order_independent():
    ids = ["clip_0001", "clip_0002", "clip_0003", "clip_0004"]
    assert _event_summary_id(ids) == _event_summary_id(ids[::-1])


def test_compress_clips_rejects_wrong_count():
    with pytest.raises(ValueError, match="Expected 4 clips"):
        compress_clips_to_event(_four_clips()[:3], "model")


def test_compress_clips_without_vllm():
    clips = _four_clips()
    evt = compress_clips_to_event(clips, "model", vllm_base_url=None)
    assert evt.event_summary is None
    assert len(evt.member_clip_ids) == 4
    assert evt.absolute_start < evt.absolute_end


def test_compress_clips_metadata_from_first():
    clips = _four_clips()
    evt = compress_clips_to_event(clips, "model")
    assert evt.day == "day1"
    assert evt.camera_id == "Allie"
    assert evt.camera_type == "ego"


def test_compress_clips_time_span():
    clips = _four_clips()
    evt = compress_clips_to_event(clips, "model")
    assert evt.absolute_start == clips[0].absolute_start
    assert evt.absolute_end == clips[-1].absolute_end


def test_compress_clips_aggregates_ocr():
    base = 1_672_531_200_000
    clips = [
        _make_clip(i, base + i * 30_000, base + i * 30_000 + 30_000) for i in range(4)
    ]
    clips[0] = clips[0].model_copy(update={"ocr_text": "EXIT"})
    clips[2] = clips[2].model_copy(update={"ocr_text": "FIRE"})
    evt = compress_clips_to_event(clips, "model")
    assert evt.aggregated_ocr_text is not None
    assert "EXIT" in evt.aggregated_ocr_text
    assert "FIRE" in evt.aggregated_ocr_text


def test_compress_clips_calls_vllm(tmp_path: Path):
    clips = _four_clips()
    with patch(
        "castlerag.preprocess.event_compress._vllm_chat", return_value="Summary text"
    ) as mock_chat:
        evt = compress_clips_to_event(
            clips, "model", vllm_base_url="http://localhost:8000/v1"
        )
    assert evt.event_summary == "Summary text"
    mock_chat.assert_called_once()


def test_compress_clips_rejects_mixed_camera():
    clips = _four_clips()
    clips[2] = clips[2].model_copy(update={"camera_id": "Bjorn"})
    with pytest.raises(ValueError, match="mismatch"):
        compress_clips_to_event(clips, "model")


def test_compress_clips_rejects_overlapping():
    base = 1_672_531_200_000
    clips = [
        _make_clip(i, base + i * 30_000, base + i * 30_000 + 30_000) for i in range(4)
    ]
    # Make clip 1 start before clip 0 ends
    clips[1] = clips[1].model_copy(update={"absolute_start": base + 5_000})
    with pytest.raises(ValueError, match="adjacent"):
        compress_clips_to_event(clips, "model")


def test_compress_clips_rejects_gapped_sequence():
    base = 1_672_531_200_000
    clips = [
        _make_clip(i, base + i * 30_000, base + i * 30_000 + 30_000) for i in range(4)
    ]
    clips[1] = clips[1].model_copy(
        update={"absolute_start": base + 35_000, "absolute_end": base + 65_000}
    )
    with pytest.raises(ValueError, match="adjacent"):
        compress_clips_to_event(clips, "model")


# ---------------------------------------------------------------------------
# caption_ocr
# ---------------------------------------------------------------------------


def test_annotate_clip_requires_vllm_url():
    from castlerag.preprocess.caption_ocr import annotate_clip

    with pytest.raises(ValueError, match="vllm_base_url"):
        annotate_clip("clip_1", [], None, "model", vllm_base_url=None)


def test_annotate_clip_no_frames_returns_empty():
    from castlerag.preprocess.caption_ocr import annotate_clip

    ann = annotate_clip(
        "clip_1", [], None, "model", vllm_base_url="http://localhost:8000/v1"
    )
    assert ann.clip_caption is None
    assert ann.ocr_text is None
    assert ann.caption_confidence == 0.0


def test_annotate_clip_calls_vllm(tmp_path: Path):
    import numpy as np
    from PIL import Image

    from castlerag.preprocess.caption_ocr import annotate_clip

    # create a fake frame
    p = tmp_path / "0001.jpg"
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8), mode="L").save(p)

    with patch(
        "castlerag.preprocess.caption_ocr._vllm_chat",
        side_effect=["Caption text", "NONE"],
    ):
        ann = annotate_clip(
            "clip_1", [p], "Hello world", "model", "http://localhost:8000/v1"
        )
    assert ann.clip_caption == "Caption text"
    assert ann.ocr_text is None


# ---------------------------------------------------------------------------
# auxiliary
# ---------------------------------------------------------------------------


def test_iter_photo_records(tmp_path: Path):
    from castlerag.preprocess.auxiliary import iter_photo_records

    photo_dir = tmp_path / "photo" / "Allie"
    photo_dir.mkdir(parents=True)
    (photo_dir / "IMG_001.jpg").touch()
    (photo_dir / "IMG_002.jpg").touch()

    records = list(iter_photo_records(tmp_path, "Allie", "day1"))
    assert len(records) == 2
    assert all(r.source_type == "aux_photo" for r in records)
    assert all(r.participant_id == "Allie" for r in records)
    assert all(r.modality == "image" for r in records)


def test_iter_photo_records_missing_dir(tmp_path: Path):
    from castlerag.preprocess.auxiliary import iter_photo_records

    records = list(iter_photo_records(tmp_path, "Ghost", "day1"))
    assert records == []


def test_iter_thermal_records(tmp_path: Path):
    from castlerag.preprocess.auxiliary import iter_thermal_records

    thermal_dir = tmp_path / "thermal"
    thermal_dir.mkdir(parents=True)
    (thermal_dir / "frame_001.bmp").touch()

    records = list(iter_thermal_records(tmp_path, "day2"))
    assert len(records) == 1
    assert records[0].source_type == "aux_thermal"
    assert records[0].participant_id is None
    assert records[0].modality == "image"


def test_iter_thermal_records_missing_dir(tmp_path: Path):
    from castlerag.preprocess.auxiliary import iter_thermal_records

    records = list(iter_thermal_records(tmp_path, "day1"))
    assert records == []


def test_iter_aux_video_short_file(tmp_path: Path):
    from castlerag.preprocess.auxiliary import iter_aux_video_records

    video_dir = tmp_path / "video" / "Allie"
    video_dir.mkdir(parents=True)
    (video_dir / "2023-01-05_08-00-00_clip.mp4").touch()

    with patch("castlerag.preprocess.auxiliary.get_video_duration", return_value=20.0):
        records = list(iter_aux_video_records(tmp_path, "Allie", "day1"))
    assert len(records) == 1
    assert records[0].source_type == "aux_video"
    assert records[0].absolute_start > 0


def test_iter_aux_video_long_file_rewindowed(tmp_path: Path):
    from castlerag.preprocess.auxiliary import iter_aux_video_records

    video_dir = tmp_path / "video" / "Allie"
    video_dir.mkdir(parents=True)
    (video_dir / "2023-01-05_08-00-00_long.mp4").touch()

    with patch("castlerag.preprocess.auxiliary.get_video_duration", return_value=90.0):
        records = list(iter_aux_video_records(tmp_path, "Allie", "day1"))
    assert len(records) == 3  # 0-30, 30-60, 60-90
    assert records[1].absolute_start - records[0].absolute_start == 30_000


def test_iter_aux_video_skips_ffprobe_failure(tmp_path: Path):
    from castlerag.preprocess.auxiliary import iter_aux_video_records

    video_dir = tmp_path / "video" / "Allie"
    video_dir.mkdir(parents=True)
    (video_dir / "2023-01-05_08-00-00_bad.mp4").touch()

    with patch(
        "castlerag.preprocess.auxiliary.get_video_duration",
        side_effect=subprocess.CalledProcessError(1, "ffprobe"),
    ):
        records = list(iter_aux_video_records(tmp_path, "Allie", "day1"))
    assert records == []


def test_iter_aux_video_rewindowed_unique_clip_ids(tmp_path: Path):
    from castlerag.preprocess.auxiliary import iter_aux_video_records

    video_dir = tmp_path / "video" / "Allie"
    video_dir.mkdir(parents=True)
    (video_dir / "2023-01-05_08-00-00_long.mp4").touch()

    with patch("castlerag.preprocess.auxiliary.get_video_duration", return_value=90.0):
        records = list(iter_aux_video_records(tmp_path, "Allie", "day1"))
    clip_ids = [r.clip_id for r in records]
    assert len(clip_ids) == len(set(clip_ids)), (
        "re-windowed clips must have unique clip_ids"
    )


def test_iter_aux_video_skips_missing_timestamp_hint(tmp_path: Path):
    from castlerag.preprocess.auxiliary import iter_aux_video_records

    video_dir = tmp_path / "video" / "Allie"
    video_dir.mkdir(parents=True)
    (video_dir / "clip.mp4").touch()

    with patch("castlerag.preprocess.auxiliary.get_video_duration", return_value=20.0):
        records = list(iter_aux_video_records(tmp_path, "Allie", "day1"))
    assert records == []


def test_aux_record_ids_do_not_collide_for_same_basename(tmp_path: Path):
    from castlerag.preprocess.auxiliary import iter_photo_records

    allie_dir = tmp_path / "photo" / "Allie"
    bjorn_dir = tmp_path / "photo" / "Bjorn"
    allie_dir.mkdir(parents=True)
    bjorn_dir.mkdir(parents=True)
    (allie_dir / "IMG_0001.jpg").touch()
    (bjorn_dir / "IMG_0001.jpg").touch()

    with patch("castlerag.preprocess.auxiliary._exif_unix_ms", return_value=0):
        allie = list(iter_photo_records(tmp_path, "Allie", "day1"))
        bjorn = list(iter_photo_records(tmp_path, "Bjorn", "day1"))

    assert len(allie) == 1
    assert len(bjorn) == 1
    assert allie[0].clip_id != bjorn[0].clip_id


def test_iter_photo_records_uses_filename_hint(tmp_path: Path):
    from castlerag.preprocess.auxiliary import iter_photo_records

    photo_dir = tmp_path / "photo" / "Allie"
    photo_dir.mkdir(parents=True)
    # Filename with timestamp, no EXIF
    (photo_dir / "2023-01-05_08-00-00.jpg").touch()

    with patch("castlerag.preprocess.auxiliary._exif_unix_ms", return_value=None):
        records = list(iter_photo_records(tmp_path, "Allie", "day1"))
    assert len(records) == 1
    # Timestamp should be non-zero since it was parsed from the filename
    assert records[0].absolute_start > 0
