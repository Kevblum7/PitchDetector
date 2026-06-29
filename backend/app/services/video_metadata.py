"""Video metadata extraction via ``ffprobe`` (CLAUDE.md §9).

Uses ``subprocess.run`` with an argument list (never a shell string) and reports
clear, typed errors when ffprobe is missing or fails.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_FFPROBE_TIMEOUT_SECONDS = 30


class VideoMetadataError(RuntimeError):
    """Base error for metadata extraction failures."""


class FFprobeNotFoundError(VideoMetadataError):
    """Raised when the ffprobe executable is not on PATH."""


class VideoProbeError(VideoMetadataError):
    """Raised when ffprobe runs but cannot describe the file."""


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    duration_seconds: float
    frame_count: int


def parse_frame_rate(value: str) -> float:
    """Parse an ffprobe frame-rate string such as ``"30000/1001"``.

    Returns 0.0 for the ``"0/0"`` placeholder ffprobe uses when unknown.
    """
    value = value.strip()
    if "/" in value:
        num_str, den_str = value.split("/", 1)
        num = float(num_str)
        den = float(den_str)
        if den == 0:
            return 0.0
        return num / den
    return float(value)


def _select_video_stream(streams: list[dict]) -> dict:
    for stream in streams:
        if stream.get("codec_type") == "video":
            return stream
    raise VideoProbeError("no video stream found in file")


def probe_video(path: Path) -> VideoMetadata:
    """Extract width, height, fps, duration, and frame count from a video.

    Raises:
        FFprobeNotFoundError: ffprobe is not installed.
        VideoProbeError: the file could not be probed or lacks a video stream.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FFprobeNotFoundError(
            "ffprobe not found on PATH; install FFmpeg (e.g. `brew install ffmpeg`)"
        )

    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=_FFPROBE_TIMEOUT_SECONDS,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise VideoProbeError(f"ffprobe failed for {path}: {exc.stderr.strip()}") from exc
    except (subprocess.SubprocessError, OSError) as exc:
        raise VideoProbeError(f"could not run ffprobe for {path}: {exc}") from exc

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise VideoProbeError(f"could not parse ffprobe output for {path}: {exc}") from exc

    stream = _select_video_stream(payload.get("streams", []))
    fmt = payload.get("format", {})

    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, ValueError, TypeError) as exc:
        raise VideoProbeError(f"missing width/height for {path}") from exc

    fps = parse_frame_rate(stream.get("r_frame_rate", "0/0"))
    if fps <= 0:
        fps = parse_frame_rate(stream.get("avg_frame_rate", "0/0"))

    duration_str = stream.get("duration") or fmt.get("duration")
    duration_seconds = float(duration_str) if duration_str is not None else 0.0

    frame_count = _resolve_frame_count(stream, fps, duration_seconds)
    if frame_count <= 0:
        raise VideoProbeError(f"could not determine frame count for {path}")

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        duration_seconds=duration_seconds,
        frame_count=frame_count,
    )


def _resolve_frame_count(stream: dict, fps: float, duration_seconds: float) -> int:
    """Prefer ffprobe's reported frame count; fall back to duration * fps."""
    for key in ("nb_frames", "nb_read_frames"):
        raw = stream.get(key)
        if raw not in (None, "", "N/A"):
            try:
                return int(raw)
            except (ValueError, TypeError):
                pass
    if fps > 0 and duration_seconds > 0:
        return int(math.floor(duration_seconds * fps))
    return 0
