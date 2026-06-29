"""Unit tests for local video path validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.services.video_files import VideoFileError, validate_video_path


def test_valid_path_returns_resolved(tmp_path: Path) -> None:
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"not really a video")
    resolved = validate_video_path(str(f))
    assert resolved == f.resolve()


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(VideoFileError, match="does not exist"):
        validate_video_path(str(tmp_path / "nope.mp4"))


def test_unsupported_extension_rejected(tmp_path: Path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with pytest.raises(VideoFileError, match="unsupported video type"):
        validate_video_path(str(f))


def test_directory_rejected(tmp_path: Path) -> None:
    with pytest.raises(VideoFileError, match="not a regular file"):
        validate_video_path(str(tmp_path))


def test_empty_rejected() -> None:
    with pytest.raises(VideoFileError, match="must not be empty"):
        validate_video_path("   ")
