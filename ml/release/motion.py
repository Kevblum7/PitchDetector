"""Motion gate: find the high-motion delivery window inside a clip.

Cheap frame differencing over (typically downscaled) frames. This is stage 1
of automatic release-frame labeling (CLAUDE.md §9): it narrows the expensive
pose passes to the part of the clip where the pitch actually happens.

Pure over an ``(index, frame)`` iterable so it is unit-testable without video
files; the pipeline feeds it FFmpeg-downscaled frames.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


class MotionGateError(RuntimeError):
    """Raised when no usable motion window can be found."""


@dataclass(frozen=True)
class MotionWindow:
    """Contiguous frame range containing the dominant motion burst."""

    start_frame: int
    end_frame: int
    peak_frame: int
    peak_score: float


def find_motion_window(
    frames: Iterable[tuple[int, np.ndarray]],
    *,
    threshold_fraction: float = 0.3,
    margin_frames: int = 8,
) -> MotionWindow:
    """Locate the dominant motion burst in a frame sequence.

    Motion score per frame = mean absolute pixel difference to the previous
    frame. The window is the contiguous region around the peak where the score
    stays above ``threshold_fraction * peak``, expanded by ``margin_frames``
    on each side (clamped to the sequence).

    Raises:
        MotionGateError: fewer than 2 frames, or no motion at all.
    """
    if not 0 < threshold_fraction <= 1:
        raise MotionGateError(f"threshold_fraction must be in (0, 1], got {threshold_fraction}")

    indices: list[int] = []
    scores: list[float] = []
    previous: np.ndarray | None = None
    for index, frame in frames:
        current = frame.astype(np.int16)
        if previous is not None:
            indices.append(index)
            scores.append(float(np.abs(current - previous).mean()))
        previous = current

    if not scores:
        raise MotionGateError("need at least 2 frames to measure motion")

    peak_pos = int(np.argmax(scores))
    peak_score = scores[peak_pos]
    if peak_score <= 0:
        raise MotionGateError("no motion detected in the frame sequence")

    threshold = threshold_fraction * peak_score
    left = peak_pos
    while left > 0 and scores[left - 1] >= threshold:
        left -= 1
    right = peak_pos
    while right < len(scores) - 1 and scores[right + 1] >= threshold:
        right += 1

    first_index = indices[0]
    last_index = indices[-1]
    return MotionWindow(
        start_frame=max(first_index, indices[left] - margin_frames),
        end_frame=min(last_index, indices[right] + margin_frames),
        peak_frame=indices[peak_pos],
        peak_score=peak_score,
    )
