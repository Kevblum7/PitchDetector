"""Unit tests for pitcher tracking (ml/pose/tracker.py).

Uses the blob fake estimator: frames are black with one bright rectangle, and
the fake genuinely finds it in the crop it receives — so these tests verify
the search-region cropping and crop → full-frame coordinate mapping, not just
scripted returns.
"""

from __future__ import annotations

import numpy as np
from pose_fakes import BlobPoseEstimator

from ml.pose.geometry import BoundingBox
from ml.pose.tracker import PitcherTracker, TrackerConfig

FRAME_W, FRAME_H = 320, 240


def make_frame(blob: BoundingBox | None) -> np.ndarray:
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    if blob is not None:
        x1, y1, x2, y2 = blob.to_xyxy_int()
        frame[y1:y2, x1:x2] = 255
    return frame


def make_tracker(initial: BoundingBox) -> PitcherTracker:
    return PitcherTracker(
        estimator=BlobPoseEstimator(),
        initial_box=initial,
        frame_width=FRAME_W,
        frame_height=FRAME_H,
        config=TrackerConfig(min_detection_score=0.5, min_association_iou=0.2),
    )


class TestTracking:
    def test_detection_reported_in_full_frame_coordinates(self) -> None:
        blob = BoundingBox(x=150, y=90, width=40, height=60)
        tracker = make_tracker(initial=blob)

        result = tracker.process(0, make_frame(blob))

        assert result.detected
        # The estimator only saw a crop; results must come back in full-frame
        # pixels (blob box recovered at its true location, +/- 1px cropping).
        assert abs(result.box.x - blob.x) <= 1
        assert abs(result.box.y - blob.y) <= 1
        assert abs(result.box.width - blob.width) <= 2
        assert abs(result.box.height - blob.height) <= 2
        assert result.keypoints is not None
        kx, ky = result.keypoints[0]
        bx, by = blob.center
        assert abs(kx - bx) <= 1 and abs(ky - by) <= 1
        # The search region never leaves the frame.
        assert result.search_region.x >= 0
        assert result.search_region.x2 <= FRAME_W

    def test_follows_moving_target(self) -> None:
        start = BoundingBox(x=100, y=80, width=40, height=60)
        tracker = make_tracker(initial=start)

        for i in range(10):
            blob = start.shift(3.0 * i, 1.0 * i)
            result = tracker.process(i, make_frame(blob))
            assert result.detected, f"lost target at frame {i}"
            assert abs(result.box.x - blob.x) <= 1
            assert abs(result.box.y - blob.y) <= 1
        assert tracker.consecutive_misses == 0

    def test_miss_carries_previous_box_forward(self) -> None:
        blob = BoundingBox(x=150, y=90, width=40, height=60)
        tracker = make_tracker(initial=blob)

        seen = tracker.process(0, make_frame(blob))
        vanished = tracker.process(1, make_frame(None))

        assert not vanished.detected
        assert vanished.keypoints is None
        assert vanished.box == seen.box  # carried forward, not invented
        assert tracker.consecutive_misses == 1

    def test_reacquires_after_miss(self) -> None:
        blob = BoundingBox(x=150, y=90, width=40, height=60)
        tracker = make_tracker(initial=blob)

        tracker.process(0, make_frame(blob))
        tracker.process(1, make_frame(None))
        result = tracker.process(2, make_frame(blob))

        assert result.detected
        assert tracker.consecutive_misses == 0

    def test_low_score_detection_is_ignored(self) -> None:
        blob = BoundingBox(x=150, y=90, width=40, height=60)
        tracker = PitcherTracker(
            estimator=BlobPoseEstimator(),
            initial_box=blob,
            frame_width=FRAME_W,
            frame_height=FRAME_H,
            config=TrackerConfig(min_detection_score=0.99),  # above fake's 0.95
        )
        result = tracker.process(0, make_frame(blob))
        assert not result.detected

    def test_distant_detection_is_not_associated(self) -> None:
        # Blob far outside the search region around the initial box: either the
        # crop misses it entirely or IoU with the previous box is ~0.
        initial = BoundingBox(x=20, y=20, width=30, height=40)
        far_blob = BoundingBox(x=260, y=180, width=30, height=40)
        tracker = make_tracker(initial=initial)

        result = tracker.process(0, make_frame(far_blob))

        assert not result.detected
        assert abs(result.box.x - initial.x) <= 1
