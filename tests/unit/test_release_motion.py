"""Unit tests for the motion gate (ml/release/motion.py)."""

from __future__ import annotations

import numpy as np
import pytest
from release_fixture import TRUE_RELEASE_FRAME, make_delivery_frames

from ml.release.motion import MotionGateError, find_motion_window


def as_stream(frames: np.ndarray) -> list[tuple[int, np.ndarray]]:
    return list(enumerate(frames))


class TestFindMotionWindow:
    def test_finds_delivery_window(self) -> None:
        window = find_motion_window(as_stream(make_delivery_frames()), margin_frames=8)
        # The delivery (movement) spans frames 40-70; the peak difference is
        # at the release spike. The window must contain the release and stay
        # inside the movement region plus margin.
        assert window.start_frame <= TRUE_RELEASE_FRAME <= window.end_frame
        assert window.start_frame >= 40 - 8 - 1
        assert window.end_frame <= 70 + 8 + 1
        assert abs(window.peak_frame - TRUE_RELEASE_FRAME) <= 1

    def test_window_clamped_to_sequence(self) -> None:
        frames = make_delivery_frames()[50:65]  # motion right up to the edges
        stream = [(50 + i, f) for i, f in enumerate(frames)]
        window = find_motion_window(stream, margin_frames=50)
        assert window.start_frame >= 50
        assert window.end_frame <= 64

    def test_static_video_raises(self) -> None:
        static = np.zeros((10, 24, 32, 3), dtype=np.uint8)
        with pytest.raises(MotionGateError, match="no motion"):
            find_motion_window(as_stream(static))

    def test_too_few_frames_raises(self) -> None:
        one = np.zeros((1, 24, 32, 3), dtype=np.uint8)
        with pytest.raises(MotionGateError, match="at least 2 frames"):
            find_motion_window(as_stream(one))

    def test_invalid_threshold_raises(self) -> None:
        frames = make_delivery_frames()[:10]
        with pytest.raises(MotionGateError, match="threshold_fraction"):
            find_motion_window(as_stream(frames), threshold_fraction=0.0)
