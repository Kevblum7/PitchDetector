"""Validation for registering local video files (CLAUDE.md §19, §21)."""

from __future__ import annotations

from pathlib import Path

from backend.app.core.config import ALLOWED_VIDEO_EXTENSIONS


class VideoFileError(ValueError):
    """Raised when a video path is missing, not a file, or an unsupported type."""


def validate_video_path(raw_path: str) -> Path:
    """Resolve and validate a user-supplied local video path.

    Returns the resolved absolute :class:`Path`.

    Raises:
        VideoFileError: the path does not exist, is not a regular file, or has
            an unsupported extension.
    """
    if not raw_path or not raw_path.strip():
        raise VideoFileError("file_path must not be empty")

    path = Path(raw_path).expanduser().resolve()

    if not path.exists():
        raise VideoFileError(f"file does not exist: {path}")
    if not path.is_file():
        raise VideoFileError(f"path is not a regular file: {path}")
    if path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
        raise VideoFileError(
            f"unsupported video type '{path.suffix}'; allowed extensions: {allowed}"
        )
    return path
