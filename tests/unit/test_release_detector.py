"""Unit tests for the kinematic release detector (ml/release/detector.py).

The blob fake estimator reports all keypoints (including wrists) at the blob
centre, so wrist speed == blob speed and the ground-truth release frame is the
frame with the largest single-frame displacement.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest
from pose_fakes import BlobPoseEstimator
from release_fixture import N_FRAMES, TRUE_RELEASE_FRAME, make_delivery_frames

from ml.release.detector import (
    ReleaseDetectionError,
    ReleaseDetectorConfig,
    detect_release_from_window,
)
from ml.release.motion import MotionWindow, find_motion_window


class CountingReader:
    """Frame reader over an in-memory array that records estimator workload."""

    def __init__(self, frames: np.ndarray) -> None:
        self.frames = frames
        self.frames_served = 0

    def __call__(self, start: int, end: int, step: int) -> Iterable[tuple[int, np.ndarray]]:
        for i in range(start, end + 1, step):
            self.frames_served += 1
            yield i, self.frames[i]


def detect(frames: np.ndarray, window: MotionWindow | None = None, **config_kwargs: object):  # type: ignore[no-untyped-def]
    if window is None:
        window = find_motion_window(list(enumerate(frames)))
    reader = CountingReader(frames)
    detection = detect_release_from_window(
        reader,
        window,
        video_start=0,
        video_end=len(frames) - 1,
        estimator=BlobPoseEstimator(),
        config=ReleaseDetectorConfig(**config_kwargs),  # type: ignore[arg-type]
    )
    return detection, reader


class TestDetectRelease:
    def test_finds_true_release_frame(self) -> None:
        detection, _ = detect(make_delivery_frames())
        assert detection.release_frame == TRUE_RELEASE_FRAME
        assert detection.wrist in ("left", "right")
        assert detection.peak_speed_px_per_frame == pytest.approx(30.0, abs=1.0)

    def test_clean_spike_is_high_confidence(self) -> None:
        detection, _ = detect(make_delivery_frames())
        assert detection.confidence >= 0.8
        assert detection.review_recommended is False

    def test_constant_motion_is_low_confidence(self) -> None:
        # Blob glides at a constant speed: no release-like spike exists, so
        # the detector must flag its answer for human review.
        frames = np.zeros((60, 240, 320, 3), dtype=np.uint8)
        for t in range(60):
            x = 30 + 4 * t
            frames[t, 90:150, x : x + 40] = 255
        detection, _ = detect(frames)
        assert detection.confidence < 0.6
        assert detection.review_recommended is True

    def test_coarse_to_fine_serves_fewer_frames_than_dense(self) -> None:
        frames = make_delivery_frames()
        window = find_motion_window(list(enumerate(frames)))
        _, reader = detect(frames, window=window)
        dense_cost = window.end_frame - window.start_frame + 1
        assert reader.frames_served < dense_cost

    def test_fine_range_is_clamped_to_video(self) -> None:
        frames = make_delivery_frames()[:62]  # video ends just after release
        detection, _ = detect(frames)
        assert detection.fine_range[1] <= len(frames) - 1
        assert detection.release_frame == TRUE_RELEASE_FRAME

    def test_no_person_raises(self) -> None:
        # Motion exists (noise burst) but nothing the estimator recognises.
        rng = np.random.default_rng(7)
        frames = (rng.random((30, 60, 80, 3)) * 120).astype(np.uint8)  # all < 200
        window = MotionWindow(start_frame=5, end_frame=25, peak_frame=15, peak_score=1.0)
        reader = CountingReader(frames)
        with pytest.raises(ReleaseDetectionError, match="detected in only"):
            detect_release_from_window(
                reader,
                window,
                video_start=0,
                video_end=29,
                estimator=BlobPoseEstimator(),
            )

    def test_detection_reports_provenance_metadata(self) -> None:
        detection, _ = detect(make_delivery_frames())
        payload = detection.as_dict()
        assert payload["detector_version"]
        assert payload["motion_window"][0] <= TRUE_RELEASE_FRAME <= payload["motion_window"][1]
        assert payload["frames_analyzed"] > 0
        assert 0.0 <= payload["confidence"] <= 1.0  # type: ignore[operator]

    def test_full_video_never_required(self) -> None:
        # Frames outside the motion window (plus refine radius) are never read:
        # the whole point of the gate on a CPU-only machine.
        frames = make_delivery_frames()
        window = find_motion_window(list(enumerate(frames)))
        _, reader = detect(frames, window=window)
        assert reader.frames_served < N_FRAMES
