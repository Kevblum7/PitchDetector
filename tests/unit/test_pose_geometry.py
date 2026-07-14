"""Unit tests for bounding-box geometry (ml/pose/geometry.py)."""

from __future__ import annotations

import pytest

from ml.pose.geometry import (
    BoundingBox,
    GeometryError,
    iou,
    max_center_jump,
    smooth_boxes,
)


class TestBoundingBox:
    def test_derived_properties(self) -> None:
        box = BoundingBox(x=10, y=20, width=30, height=40)
        assert box.x2 == 40
        assert box.y2 == 60
        assert box.center == (25, 40)
        assert box.area == 1200

    def test_rejects_degenerate_size(self) -> None:
        with pytest.raises(GeometryError):
            BoundingBox(x=0, y=0, width=0, height=10)
        with pytest.raises(GeometryError):
            BoundingBox(x=0, y=0, width=10, height=-1)

    def test_expand_preserves_center(self) -> None:
        box = BoundingBox(x=10, y=10, width=20, height=40)
        grown = box.expand(2.0)
        assert grown.center == box.center
        assert grown.width == 40
        assert grown.height == 80

    def test_expand_enforces_min_size(self) -> None:
        box = BoundingBox(x=0, y=0, width=10, height=10)
        grown = box.expand(1.0, min_size=50)
        assert grown.width == 50
        assert grown.height == 50
        assert grown.center == box.center

    def test_clamp_clips_to_frame(self) -> None:
        box = BoundingBox(x=-10, y=-5, width=50, height=30)
        clamped = box.clamp(100, 100)
        assert clamped.x == 0
        assert clamped.y == 0
        assert clamped.x2 == 40
        assert clamped.y2 == 25

    def test_clamp_keeps_box_inside_far_edge(self) -> None:
        box = BoundingBox(x=90, y=95, width=50, height=30)
        clamped = box.clamp(100, 100)
        assert clamped.x2 <= 100
        assert clamped.y2 <= 100
        assert clamped.width >= 1
        assert clamped.height >= 1

    def test_dict_round_trip(self) -> None:
        box = BoundingBox(x=1.5, y=2.5, width=3.5, height=4.5)
        assert BoundingBox.from_dict(box.as_dict()) == box


class TestIou:
    def test_identical_boxes(self) -> None:
        box = BoundingBox(x=0, y=0, width=10, height=10)
        assert iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes(self) -> None:
        a = BoundingBox(x=0, y=0, width=10, height=10)
        b = BoundingBox(x=20, y=20, width=10, height=10)
        assert iou(a, b) == 0.0

    def test_half_overlap(self) -> None:
        a = BoundingBox(x=0, y=0, width=10, height=10)
        b = BoundingBox(x=5, y=0, width=10, height=10)
        # intersection 50, union 150
        assert iou(a, b) == pytest.approx(1 / 3)


class TestSmoothBoxes:
    def test_window_must_be_odd(self) -> None:
        with pytest.raises(GeometryError):
            smooth_boxes([BoundingBox(x=0, y=0, width=1, height=1)], window=2)

    def test_window_one_is_identity(self) -> None:
        boxes = [BoundingBox(x=float(i), y=0, width=10, height=10) for i in range(5)]
        assert smooth_boxes(boxes, window=1) == boxes

    def test_constant_sequence_unchanged(self) -> None:
        boxes = [BoundingBox(x=5, y=5, width=10, height=10)] * 6
        assert smooth_boxes(boxes, window=5) == boxes

    def test_constant_velocity_is_unbiased(self) -> None:
        # Centred averaging of a linear trajectory reproduces it exactly,
        # including at the (symmetrically shrunk) edges.
        boxes = [BoundingBox(x=10 + 2.0 * i, y=0, width=10, height=10) for i in range(9)]
        smoothed = smooth_boxes(boxes, window=5)
        for original, result in zip(boxes, smoothed, strict=True):
            assert result.x == pytest.approx(original.x)

    def test_outlier_is_damped(self) -> None:
        boxes = [BoundingBox(x=10, y=10, width=10, height=10) for _ in range(7)]
        boxes[3] = BoundingBox(x=60, y=10, width=10, height=10)  # teleport
        smoothed = smooth_boxes(boxes, window=5)
        assert smoothed[3].x < 30  # 50px outlier averaged over 5 frames


class TestMaxCenterJump:
    def test_fewer_than_two_boxes(self) -> None:
        assert max_center_jump([]) == 0.0
        assert max_center_jump([BoundingBox(x=0, y=0, width=2, height=2)]) == 0.0

    def test_reports_largest_jump(self) -> None:
        boxes = [
            BoundingBox(x=0, y=0, width=10, height=10),
            BoundingBox(x=3, y=4, width=10, height=10),  # jump 5
            BoundingBox(x=3, y=4, width=10, height=10),  # jump 0
        ]
        assert max_center_jump(boxes) == pytest.approx(5.0)
