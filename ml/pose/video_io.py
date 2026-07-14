"""Frame-accurate video decode/encode through FFmpeg subprocess pipes.

OpenCV is deliberately not used: no working Intel-mac wheel exists in this
environment (see ``pyproject.toml``), and FFmpeg subprocess wrappers are the
CLAUDE.md-sanctioned path anyway. Raw RGB frames are streamed through pipes so
memory stays O(one frame) regardless of clip length.

Frame indexing matches ``ffprobe``'s frame numbering (0-based, decode order),
the same convention used by clip windows. Seeking uses an input-side ``-ss``
placed a quarter-frame *before* the target frame's timestamp: FFmpeg then
decodes from the previous keyframe and discards up to the target, which is
frame-accurate for constant-frame-rate video (the broadcast norm). The
synthetic-fixture tests assert exact frame indices.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import numpy as np


class VideoIOError(RuntimeError):
    """Base error for FFmpeg decode/encode failures."""


class FFmpegNotFoundError(VideoIOError):
    """Raised when the ffmpeg executable is not on PATH."""


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FFmpegNotFoundError(
            "ffmpeg not found on PATH; install FFmpeg (e.g. `brew install ffmpeg`)"
        )
    return ffmpeg


def iter_frames(
    path: Path,
    *,
    start_frame: int,
    end_frame: int,
    fps: float,
    width: int,
    height: int,
    out_width: int | None = None,
    out_height: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, rgb_frame)`` for frames ``start_frame..end_frame`` inclusive.

    Frames are ``(height, width, 3)`` uint8 RGB arrays in original-video
    resolution, unless ``out_width``/``out_height`` are given (both or
    neither), in which case FFmpeg downscales — used by the motion gate,
    which does not need full resolution. Raises :class:`VideoIOError` if the
    video ends before ``end_frame`` or FFmpeg fails.
    """
    if start_frame < 0 or end_frame < start_frame:
        raise VideoIOError(f"invalid frame range [{start_frame}, {end_frame}]")
    if fps <= 0 or width <= 0 or height <= 0:
        raise VideoIOError(f"invalid video geometry: fps={fps}, size={width}x{height}")
    if (out_width is None) != (out_height is None):
        raise VideoIOError("out_width and out_height must be given together")
    if out_width is not None and (out_width <= 0 or out_height is None or out_height <= 0):
        raise VideoIOError(f"invalid output size {out_width}x{out_height}")

    read_width = out_width if out_width is not None else width
    read_height = out_height if out_height is not None else height

    ffmpeg = _require_ffmpeg()
    frame_count = end_frame - start_frame + 1
    frame_bytes = read_width * read_height * 3

    cmd = [ffmpeg, "-v", "error", "-nostdin"]
    if start_frame > 0:
        # Aim a quarter-frame early so float rounding can never land us one
        # frame late; FFmpeg decodes forward from the prior keyframe.
        cmd += ["-ss", f"{(start_frame - 0.25) / fps:.6f}"]
    cmd += ["-i", str(path), "-frames:v", str(frame_count), "-fps_mode", "passthrough"]
    if out_width is not None:
        cmd += ["-vf", f"scale={read_width}:{read_height}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    try:
        for offset in range(frame_count):
            data = process.stdout.read(frame_bytes)
            if data is None or len(data) < frame_bytes:
                stderr = _drain_stderr(process)
                raise VideoIOError(
                    f"video ended after {offset} of {frame_count} requested frames "
                    f"(range [{start_frame}, {end_frame}] of {path}){stderr}"
                )
            frame = np.frombuffer(data, dtype=np.uint8).reshape(read_height, read_width, 3)
            yield start_frame + offset, frame
    finally:
        if process.stdout is not None:
            process.stdout.close()
        process.terminate()
        process.wait()


def _drain_stderr(process: subprocess.Popen) -> str:  # type: ignore[type-arg]
    if process.stderr is None:
        return ""
    text = process.stderr.read().decode(errors="replace").strip()
    return f": {text}" if text else ""


class VideoWriter:
    """Encode RGB frames to H.264 MP4 through an FFmpeg stdin pipe.

    Use as a context manager; frames must all match the declared size.
    Odd dimensions are padded by one pixel (yuv420p requires even sizes).
    """

    def __init__(self, path: Path, *, width: int, height: int, fps: float) -> None:
        if fps <= 0 or width <= 0 or height <= 0:
            raise VideoIOError(f"invalid writer geometry: fps={fps}, size={width}x{height}")
        ffmpeg = _require_ffmpeg()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.width = width
        self.height = height
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.6f}",
                "-i",
                "-",
                "-an",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_rgb: np.ndarray) -> None:
        if frame_rgb.shape != (self.height, self.width, 3) or frame_rgb.dtype != np.uint8:
            raise VideoIOError(
                f"expected uint8 frame of shape ({self.height}, {self.width}, 3), "
                f"got {frame_rgb.dtype} {frame_rgb.shape}"
            )
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(frame_rgb.tobytes())
        except BrokenPipeError as exc:
            raise VideoIOError(
                f"ffmpeg encoder exited early{_drain_stderr(self._process)}"
            ) from exc

    def close(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        returncode = self._process.wait()
        if returncode != 0:
            raise VideoIOError(
                f"ffmpeg encoding failed (exit {returncode}){_drain_stderr(self._process)}"
            )

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            # Abort without raising a second error that would mask the first.
            self._process.kill()
            self._process.wait()
            return
        self.close()
