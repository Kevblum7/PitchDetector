"""Unit tests for ffprobe-based video metadata extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.services import video_metadata
from backend.app.services.video_metadata import (
    FFprobeNotFoundError,
    parse_frame_rate,
    probe_video,
)


def test_parse_frame_rate_fraction() -> None:
    assert parse_frame_rate("30/1") == 30.0
    assert parse_frame_rate("30000/1001") == pytest.approx(29.97, abs=0.01)


def test_parse_frame_rate_plain_and_unknown() -> None:
    assert parse_frame_rate("25") == 25.0
    assert parse_frame_rate("0/0") == 0.0


def test_probe_missing_ffprobe_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_metadata.shutil, "which", lambda _name: None)
    with pytest.raises(FFprobeNotFoundError):
        probe_video(Path("does-not-matter.mp4"))


def test_probe_synthetic_video(synthetic_video: Path) -> None:
    meta = probe_video(synthetic_video)
    assert meta.width == 320
    assert meta.height == 240
    assert meta.fps == pytest.approx(30.0, abs=0.1)
    assert meta.frame_count >= 25  # ~30 frames for a 1s @ 30fps clip
    assert meta.duration_seconds == pytest.approx(1.0, abs=0.2)
