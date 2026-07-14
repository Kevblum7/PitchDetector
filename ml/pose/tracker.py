"""Pitcher tracking by per-frame detection + IoU association.

The user supplies the initial pitcher box (CLAUDE.md §9: "Detect the pitcher or
let the user define an initial box"). Each frame, the estimator runs on a
search region expanded around the previous box — this keeps CPU inference fast
(small crop instead of the full broadcast frame) and makes association robust:
the chosen detection is the one that overlaps the previous box the most.

Per-frame detection is preferred over a correlation tracker because it cannot
drift silently: a frame either has a person detection that overlaps the track,
or it is recorded as a miss and surfaces in the quality summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml.pose.estimator import PoseEstimator
from ml.pose.geometry import BoundingBox, iou


@dataclass(frozen=True)
class TrackerConfig:
    """Tunable tracking parameters (persisted with each pose job)."""

    min_detection_score: float = 0.5
    min_association_iou: float = 0.2
    search_expand_factor: float = 1.8
    min_search_size_px: float = 128.0
    max_consecutive_misses: int = 10
    smoothing_window: int = 5  # must be odd; see geometry.smooth_boxes

    def as_dict(self) -> dict[str, float | int]:
        return {
            "min_detection_score": self.min_detection_score,
            "min_association_iou": self.min_association_iou,
            "search_expand_factor": self.search_expand_factor,
            "min_search_size_px": self.min_search_size_px,
            "max_consecutive_misses": self.max_consecutive_misses,
            "smoothing_window": self.smoothing_window,
        }


@dataclass
class TrackedFrame:
    """Tracking + pose result for one frame, in original-video coordinates."""

    frame_index: int
    box: BoundingBox  # associated detection box, or carried-forward on a miss
    search_region: BoundingBox  # crop the estimator actually saw
    detected: bool
    detection_score: float | None = None
    keypoints: list[tuple[float, float]] | None = None
    keypoint_scores: list[float] | None = None


@dataclass
class PitcherTracker:
    """Stateful frame-by-frame tracker; feed frames in ascending order."""

    estimator: PoseEstimator
    initial_box: BoundingBox
    frame_width: int
    frame_height: int
    config: TrackerConfig = field(default_factory=TrackerConfig)

    def __post_init__(self) -> None:
        self._previous_box = self.initial_box.clamp(self.frame_width, self.frame_height)
        self.consecutive_misses = 0

    def process(self, frame_index: int, frame_rgb: np.ndarray) -> TrackedFrame:
        search = self._previous_box.expand(
            self.config.search_expand_factor, min_size=self.config.min_search_size_px
        ).clamp(self.frame_width, self.frame_height)

        x1, y1, x2, y2 = search.to_xyxy_int()
        crop = frame_rgb[y1:y2, x1:x2]
        detections = [
            det.shifted(float(x1), float(y1))
            for det in self.estimator.detect(crop)
            if det.score >= self.config.min_detection_score
        ]

        best = None
        best_iou = 0.0
        for det in detections:
            overlap = iou(self._previous_box, det.box)
            if overlap > best_iou:
                best, best_iou = det, overlap

        if best is not None and best_iou >= self.config.min_association_iou:
            self._previous_box = best.box.clamp(self.frame_width, self.frame_height)
            self.consecutive_misses = 0
            return TrackedFrame(
                frame_index=frame_index,
                box=self._previous_box,
                search_region=search,
                detected=True,
                detection_score=best.score,
                keypoints=best.keypoints,
                keypoint_scores=best.keypoint_scores,
            )

        # Miss: carry the previous box forward so the track (and the next
        # search region) survives brief occlusions, but record it honestly.
        self.consecutive_misses += 1
        return TrackedFrame(
            frame_index=frame_index,
            box=self._previous_box,
            search_region=search,
            detected=False,
        )
