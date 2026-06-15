"""YouTube embed pipeline for the CASTLE dataset mirror.

The CASTLE 2024 dataset ships as multi-terabyte UHD video on HuggingFace, but the
official project also mirrors every stream on YouTube (one video per
``day / camera / hour``), which the CASTLE viewer embeds.  We reuse that mirror so
we never have to host HLS ourselves, seeking to the relevant offset with the
``?start=`` query parameter.

``youtube_mirror.csv`` holds the ``day,camera,hour,video_id`` mapping (666 rows),
generated from the CASTLE viewer's ``videos.json``
(github.com/CASTLE-Dataset/CASTLE-Dataset.github.io).  Edit the CSV to add or
correct rows — no code change required.  An unmapped triple (e.g. a camera/hour
the mirror skipped) falls back to ``placeholder_video_id``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

_DEFAULT_MAPPING_PATH = Path(__file__).parent / "youtube_mirror.csv"

# Big Buck Bunny (Blender Foundation, CC-BY 3.0) — an openly licensed, reliably
# embeddable fallback shown for triples the mirror doesn't cover.
PLACEHOLDER_VIDEO_ID = "aqz-KE-bpKQ"

# (day, camera, hour) — normalized: day lower-cased, camera verbatim, hour int.
MirrorKey = Tuple[str, str, int]


def _normalize_key(day: str, camera: str, hour: int) -> MirrorKey:
    """Return a normalized mapping key for a day / camera / hour triple."""
    return (str(day).strip().lower(), str(camera).strip(), int(hour))


@dataclass
class YouTubeMirror:
    """Resolve ``(day, camera, hour)`` evidence locations to YouTube embeds.

    ``mapping`` holds the known video ids; lookups for an unknown key fall back
    to ``placeholder_video_id`` so the embed always renders something while the
    mirror is still being populated.
    """

    mapping: Dict[MirrorKey, str] = field(default_factory=dict)
    placeholder_video_id: str = PLACEHOLDER_VIDEO_ID
    # Privacy-enhanced host, matching the CASTLE viewer's embeds.
    embed_host: str = "https://www.youtube-nocookie.com"
    watch_host: str = "https://www.youtube.com"

    @classmethod
    def from_csv(cls, path: Optional[Path] = None, **kwargs: object) -> "YouTubeMirror":
        """Build a mirror from a ``day,camera,hour,video_id`` CSV.

        A missing file yields an empty mapping (placeholder-only mode) rather
        than raising, so the UI still boots before the mirror sheet is created.
        """
        mapping: Dict[MirrorKey, str] = {}
        csv_path = Path(path) if path is not None else _DEFAULT_MAPPING_PATH
        if csv_path.exists():
            with csv_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    video_id = (row.get("video_id") or "").strip()
                    if not video_id:
                        continue
                    key = _normalize_key(
                        row["day"], row["camera"], int(row["hour"])
                    )
                    mapping[key] = video_id
        return cls(mapping=mapping, **kwargs)  # type: ignore[arg-type]

    def video_id(self, day: str, camera: str, hour: int) -> str:
        """Return the mapped video id, or the placeholder when unmapped."""
        return self.mapping.get(
            _normalize_key(day, camera, hour), self.placeholder_video_id
        )

    def is_placeholder(self, day: str, camera: str, hour: int) -> bool:
        """Return True when the triple still resolves to the placeholder video.

        Reflects the *resolved* id (not just mapping membership) so a seeded row
        that still carries the placeholder id reports as a placeholder, and flips
        to False the moment the CSV is edited to a real mirror upload.
        """
        return self.video_id(day, camera, hour) == self.placeholder_video_id

    def embed_url(
        self,
        day: str,
        camera: str,
        hour: int,
        start_seconds: float = 0.0,
        *,
        autoplay: bool = False,
    ) -> str:
        """Return an ``/embed`` iframe URL seeked to ``start_seconds``."""
        start = max(0, int(start_seconds))
        params = [f"start={start}", "rel=0"]
        if autoplay:
            params.append("autoplay=1")
        query = "&".join(params)
        return f"{self.embed_host}/embed/{self.video_id(day, camera, hour)}?{query}"

    def watch_url(
        self, day: str, camera: str, hour: int, start_seconds: float = 0.0
    ) -> str:
        """Return a public ``watch?v=`` URL seeked to ``start_seconds``."""
        start = max(0, int(start_seconds))
        vid = self.video_id(day, camera, hour)
        return f"{self.watch_host}/watch?v={vid}&t={start}s"

    def default_embed_url(self, start_seconds: float = 0.0) -> str:
        """Return an embed URL for the first mapped clip (UI initial state).

        Falls back to a placeholder embed when the mapping is empty.
        """
        if not self.mapping:
            return f"{self.embed_host}/embed/{self.placeholder_video_id}?rel=0"
        day, camera, hour = sorted(self.mapping)[0]
        return self.embed_url(day, camera, hour, start_seconds)
