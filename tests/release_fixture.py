"""Synthetic 'pitch delivery' used by release-detection tests.

A bright blob (the fake pitcher, whose keypoints the blob estimator reports at
its centre) sits still, accelerates through a delivery, spikes hardest between
frames 57 and 58 (the 'release'), decelerates, and stops. Ground truth:
``TRUE_RELEASE_FRAME`` — the frame whose wrist displacement is largest.

Provided both as in-memory arrays (unit tests) and as a near-losslessly
encoded MP4 (integration tests).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

FRAME_WIDTH, FRAME_HEIGHT = 320, 240
BLOB_WIDTH, BLOB_HEIGHT = 40, 60
BLOB_Y = 90
N_FRAMES = 100
FPS = 30.0
TRUE_RELEASE_FRAME = 58

# Per-frame x displacement: still -> slow drive -> spike at 58 -> follow-through.
_DELTAS: dict[int, float] = {
    **{t: 2.0 for t in range(40, 55)},
    55: 5.0,
    56: 8.0,
    57: 12.0,
    58: 30.0,  # release: biggest single-frame movement
    59: 12.0,
    60: 6.0,
    **{t: 2.0 for t in range(61, 71)},
}


def blob_x_positions() -> list[int]:
    xs: list[int] = []
    x = 40.0
    for t in range(N_FRAMES):
        x += _DELTAS.get(t, 0.0)
        xs.append(int(round(x)))
    return xs


def make_delivery_frames() -> np.ndarray:
    """(N_FRAMES, H, W, 3) uint8: black frames with the moving white blob."""
    frames = np.zeros((N_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    for t, x in enumerate(blob_x_positions()):
        frames[t, BLOB_Y : BLOB_Y + BLOB_HEIGHT, x : x + BLOB_WIDTH] = 255
    return frames


def build_delivery_video(out: Path) -> Path:
    """Encode the delivery frames near-losslessly so blob edges stay crisp."""
    frames = make_delivery_frames()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            "-r",
            str(int(FPS)),
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
        input=frames.tobytes(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return out
