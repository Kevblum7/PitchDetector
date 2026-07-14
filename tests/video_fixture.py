"""Builder for the index-encoded synthetic test video.

Frame ``i`` is a uniform gray image of value ``i * STEP``, encoded near-
losslessly, so tests can verify the FFmpeg reader returns exactly the frames
it was asked for — including after seeking. Kept out of ``conftest.py`` so
test modules can import the constants without importing conftest.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

INDEXED_VIDEO_FRAMES = 30
INDEXED_VIDEO_STEP = 8  # frame i is filled with gray value i * STEP
INDEXED_VIDEO_SIZE = (64, 48)  # (width, height)
INDEXED_VIDEO_FPS = 30.0


def build_indexed_video(out: Path) -> Path:
    width, height = INDEXED_VIDEO_SIZE
    raw = b"".join(
        np.full((height, width, 3), i * INDEXED_VIDEO_STEP, dtype=np.uint8).tobytes()
        for i in range(INDEXED_VIDEO_FRAMES)
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(int(INDEXED_VIDEO_FPS)),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-pix_fmt",
            "yuv444p",
            str(out),
        ],
        input=raw,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return out
