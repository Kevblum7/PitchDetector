"""Shot segmentation and center-field (CF) camera-angle check.

Savant clips usually open on a close-up/dugout/replay shot and then cut to the CF camera
for the pitch. We split the clip at hard cuts (HSV histogram correlation drop) and accept
the first shot whose layout looks like the CF pitching view: a lot of grass, and a dirt
mound in the bottom-centre of the frame. The mound location also anchors the pitcher crop.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Shot:
    start: int  # first frame index (inclusive)
    end: int  # last frame index (exclusive)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class CFLayout:
    is_cf: bool
    grass_frac: float  # grass share of the lower 60% of the frame
    mound_frac: float  # dirt share of the bottom-centre box
    mound_cx: float  # normalised x of the mound centroid (0..1)
    mound_top: float  # normalised y of the mound's upper edge


def read_video(path: Path, max_frames: int | None = None) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames: list[np.ndarray] = []
    while max_frames is None or len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames or fps <= 0:
        raise OSError(f"no frames/fps in {path}")
    return frames, fps


def _hist(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def find_shots(frames: list[np.ndarray], threshold: float, min_len: int = 3) -> list[Shot]:
    """Split at frames whose histogram correlation with the previous frame < threshold."""
    hists = [_hist(f) for f in frames]
    cuts = [0]
    for i in range(1, len(hists)):
        corr = cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_CORREL)
        if corr < threshold and i - cuts[-1] >= min_len:
            cuts.append(i)
    cuts.append(len(frames))
    return [Shot(a, b) for a, b in zip(cuts[:-1], cuts[1:], strict=False)]


def _masks(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, np.array([30, 50, 40]), np.array([90, 255, 255]))
    dirt = cv2.inRange(hsv, np.array([5, 50, 90]), np.array([25, 220, 255]))
    return grass > 0, dirt > 0


def cf_layout(frame: np.ndarray, min_grass: float = 0.30, min_mound: float = 0.12) -> CFLayout:
    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    grass, dirt = _masks(small)
    h, w = grass.shape
    grass_frac = float(grass[int(0.4 * h) :].mean())
    box = dirt[int(0.70 * h) :, int(0.15 * w) : int(0.85 * w)]
    mound_frac = float(box.mean())
    ys, xs = np.nonzero(dirt[int(0.6 * h) :])
    if len(xs) > 50:
        mound_cx = float(np.median(xs)) / w
        mound_top = (float(np.percentile(ys, 5)) + 0.6 * h) / h
    else:
        mound_cx, mound_top = 0.5, 1.0
    ok = grass_frac >= min_grass and mound_frac >= min_mound and 0.15 < mound_cx < 0.85
    return CFLayout(ok, grass_frac, mound_frac, mound_cx, mound_top)


def pick_cf_shot(
    frames: list[np.ndarray], shots: list[Shot], fps: float, min_seconds: float = 1.5
) -> tuple[Shot, CFLayout] | None:
    """First shot long enough to contain a delivery whose opening frames look CF.

    Only the opening ~0.5 s is probed: after the pitch the CF camera often pans with the ball.
    """
    for s in shots:
        if s.length < min_seconds * fps:
            continue
        probes = [cf_layout(frames[s.start + k]) for k in (2, min(int(0.5 * fps), s.length - 1))]
        if all(p.is_cf for p in probes):
            return s, probes[0]
    return None


def iter_range(frames: list[np.ndarray], shot: Shot) -> Iterator[tuple[int, np.ndarray]]:
    for i in range(shot.start, shot.end):
        yield i, frames[i]
