"""Unit tests for the pose pipeline: leakage guards, timestamps, quality.

The leakage tests here are required by CLAUDE.md §10/§18: they must fail if a
frame at or after the protected release cutoff can ever reach pose extraction.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from pose_fakes import BlobPoseEstimator

from ml.pose.geometry import BoundingBox
from ml.pose.pipeline import (
    PoseLeakageError,
    extract_pose_frames,
    relative_timestamp_ms,
)
from ml.pose.tracker import TrackerConfig

FRAME_W, FRAME_H = 320, 240
FPS = 30.0
BLOB = BoundingBox(x=140, y=80, width=40, height=60)


def blob_frames(
    indices: list[int], *, hidden: set[int] | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    hidden = hidden or set()
    for i in indices:
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        if i not in hidden:
            x1, y1, x2, y2 = BLOB.shift(2.0 * i, 0).to_xyxy_int()
            frame[y1:y2, x1:x2] = 255
        yield i, frame


def run(
    indices: list[int],
    *,
    release_frame: int = 20,
    pre_release_end_frame: int = 17,
    hidden: set[int] | None = None,
):  # type: ignore[no-untyped-def]  # -> PoseExtractionResult
    return extract_pose_frames(
        blob_frames(indices, hidden=hidden),
        release_frame=release_frame,
        pre_release_end_frame=pre_release_end_frame,
        fps=FPS,
        initial_box=BLOB,
        frame_width=FRAME_W,
        frame_height=FRAME_H,
        estimator=BlobPoseEstimator(),
        config=TrackerConfig(smoothing_window=5),
    )


class TestLeakageGuards:
    def test_frame_at_release_is_rejected(self) -> None:
        with pytest.raises(PoseLeakageError, match="beyond the pre-release cutoff"):
            run([18, 19, 20], pre_release_end_frame=19, release_frame=20)

    def test_frame_past_cutoff_is_rejected_even_before_release(self) -> None:
        # Frame 18 is before release (20) but past the clip's protected cutoff
        # (17): still refused — the stored window is the contract.
        with pytest.raises(PoseLeakageError):
            run([16, 17, 18])

    def test_window_reaching_release_is_rejected_up_front(self) -> None:
        with pytest.raises(PoseLeakageError):
            run([0], pre_release_end_frame=20, release_frame=20)

    def test_full_pre_release_window_is_accepted(self) -> None:
        result = run(list(range(0, 18)))
        assert len(result.frames) == 18
        assert max(f.frame_index for f in result.frames) == 17


class TestTimestamps:
    def test_relative_timestamp_math(self) -> None:
        assert relative_timestamp_ms(17, 20, 30.0) == pytest.approx(-100.0)
        assert relative_timestamp_ms(20, 20, 30.0) == 0.0
        assert relative_timestamp_ms(0, 30, 30.0) == pytest.approx(-1000.0)

    def test_relative_timestamp_rejects_bad_fps(self) -> None:
        with pytest.raises(ValueError):
            relative_timestamp_ms(0, 10, 0.0)

    def test_pipeline_timestamps_are_all_negative(self) -> None:
        result = run(list(range(0, 18)))
        for frame in result.frames:
            assert frame.timestamp_ms_relative_to_release < 0
        last = result.frames[-1]
        assert last.timestamp_ms_relative_to_release == pytest.approx(-100.0)


class TestQuality:
    def test_clean_track_is_ok(self) -> None:
        result = run(list(range(0, 18)))
        q = result.quality
        assert q.tracking_ok
        assert q.issues == []
        assert q.total_frames == 18
        assert q.detected_frames == 18
        assert q.max_consecutive_misses == 0
        assert q.mean_detection_score == pytest.approx(0.95)
        assert q.mean_keypoint_score == pytest.approx(0.9)

    def test_brief_occlusion_is_tolerated_but_counted(self) -> None:
        result = run(list(range(0, 18)), hidden={5, 6})
        q = result.quality
        assert q.detected_frames == 16
        assert q.max_consecutive_misses == 2
        assert q.tracking_ok  # 16/18 ≈ 89% detected, above the 80% floor

    def test_mostly_missing_target_fails_quality(self) -> None:
        result = run(list(range(0, 18)), hidden=set(range(4, 18)))
        q = result.quality
        assert not q.tracking_ok
        assert any("detected in only" in issue for issue in q.issues)
        assert any("lost track" in issue for issue in q.issues)

    def test_empty_frame_source_raises(self) -> None:
        with pytest.raises(ValueError, match="no frames"):
            run([])


class TestPoseResults:
    def test_boxes_follow_target_and_keypoints_are_full_frame(self) -> None:
        result = run(list(range(0, 18)))
        for frame in result.frames:
            expected = BLOB.shift(2.0 * frame.frame_index, 0)
            ex, ey = expected.center
            bx, by = frame.box.center
            # Constant-velocity motion: centred smoothing is unbiased.
            assert abs(bx - ex) <= 1.5
            assert abs(by - ey) <= 1.5
            assert frame.keypoints is not None
            kx, ky = frame.keypoints[0]
            assert abs(kx - ex) <= 1.5 and abs(ky - ey) <= 1.5

    def test_search_region_is_recorded(self) -> None:
        result = run(list(range(0, 5)))
        for frame in result.frames:
            region = frame.search_region
            assert region.x >= 0 and region.y >= 0
            assert region.x2 <= FRAME_W and region.y2 <= FRAME_H
            assert region.area >= frame.box.area  # search covers the target
