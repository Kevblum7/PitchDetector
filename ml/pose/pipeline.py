"""Clip → pose-sequence extraction pipeline.

Ties together frame decoding, pitcher tracking, box smoothing, release-relative
timestamps, and tracking-quality checks.

**Leakage guard (CLAUDE.md §10):** the pipeline refuses to process any frame at
or beyond ``release_frame``, and additionally refuses frames beyond the clip's
``pre_release_end_frame`` — even if a caller or a buggy decoder hands one in.
This is enforced *inside* the frame loop, not just at the boundaries, and is
covered by explicit leakage tests.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ml.pose.estimator import PoseEstimator
from ml.pose.geometry import BoundingBox, max_center_jump, smooth_boxes
from ml.pose.tracker import PitcherTracker, TrackedFrame, TrackerConfig
from ml.pose.video_io import iter_frames

# A track is considered unusable when fewer than this fraction of frames have a
# real detection (the rest are carried-forward misses).
MIN_DETECTED_FRACTION = 0.8
# A frame-to-frame centre jump larger than this fraction of the mean box height
# is flagged as physically implausible.
MAX_CENTER_JUMP_BOX_FRACTION = 0.5


class PoseLeakageError(RuntimeError):
    """A frame at or after the protected release cutoff reached the pipeline."""


@dataclass(frozen=True)
class PoseFrameResult:
    """Pose for one frame, in original-video coordinates."""

    frame_index: int
    timestamp_ms_relative_to_release: float  # negative before release
    box: BoundingBox  # smoothed pitcher box
    search_region: BoundingBox  # crop the estimator saw (normalization record)
    detected: bool
    detection_score: float | None
    keypoints: list[tuple[float, float]] | None
    keypoint_scores: list[float] | None


@dataclass(frozen=True)
class QualitySummary:
    """Tracking-quality report for a clip (CLAUDE.md §9: reject bad tracking)."""

    total_frames: int
    detected_frames: int
    max_consecutive_misses: int
    mean_detection_score: float | None
    mean_keypoint_score: float | None
    max_center_jump_px: float
    tracking_ok: bool
    issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "total_frames": self.total_frames,
            "detected_frames": self.detected_frames,
            "max_consecutive_misses": self.max_consecutive_misses,
            "mean_detection_score": self.mean_detection_score,
            "mean_keypoint_score": self.mean_keypoint_score,
            "max_center_jump_px": self.max_center_jump_px,
            "tracking_ok": self.tracking_ok,
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class PoseExtractionResult:
    frames: list[PoseFrameResult]
    quality: QualitySummary


def relative_timestamp_ms(frame_index: int, release_frame: int, fps: float) -> float:
    """Milliseconds relative to release (negative = before release)."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return (frame_index - release_frame) / fps * 1000.0


def extract_pose_frames(
    frame_source: Iterable[tuple[int, np.ndarray]],
    *,
    release_frame: int,
    pre_release_end_frame: int,
    fps: float,
    initial_box: BoundingBox,
    frame_width: int,
    frame_height: int,
    estimator: PoseEstimator,
    config: TrackerConfig | None = None,
) -> PoseExtractionResult:
    """Track the pitcher and estimate pose over an already-decoded frame stream.

    ``frame_source`` yields ``(frame_index, rgb_frame)`` in ascending order.
    Raises :class:`PoseLeakageError` if any frame index reaches the release
    frame or exceeds the pre-release cutoff.
    """
    if pre_release_end_frame >= release_frame:
        raise PoseLeakageError(
            f"pre_release_end_frame {pre_release_end_frame} reaches or passes "
            f"release_frame {release_frame}"
        )
    config = config or TrackerConfig()
    tracker = PitcherTracker(
        estimator=estimator,
        initial_box=initial_box,
        frame_width=frame_width,
        frame_height=frame_height,
        config=config,
    )

    tracked: list[TrackedFrame] = []
    worst_miss_streak = 0
    for frame_index, frame_rgb in frame_source:
        if frame_index > pre_release_end_frame or frame_index >= release_frame:
            raise PoseLeakageError(
                f"frame {frame_index} is beyond the pre-release cutoff "
                f"{pre_release_end_frame} (release at {release_frame}); "
                "refusing to extract pose from a potentially leaking frame"
            )
        tracked.append(tracker.process(frame_index, frame_rgb))
        worst_miss_streak = max(worst_miss_streak, tracker.consecutive_misses)

    if not tracked:
        raise ValueError("frame source yielded no frames")

    smoothed = smooth_boxes([t.box for t in tracked], config.smoothing_window)
    quality = _summarize_quality(tracked, smoothed, worst_miss_streak, config)

    frames = [
        PoseFrameResult(
            frame_index=t.frame_index,
            timestamp_ms_relative_to_release=relative_timestamp_ms(
                t.frame_index, release_frame, fps
            ),
            box=box,
            search_region=t.search_region,
            detected=t.detected,
            detection_score=t.detection_score,
            keypoints=t.keypoints,
            keypoint_scores=t.keypoint_scores,
        )
        for t, box in zip(tracked, smoothed, strict=True)
    ]
    return PoseExtractionResult(frames=frames, quality=quality)


def extract_clip_pose(
    video_path: Path,
    *,
    start_frame: int,
    release_frame: int,
    pre_release_end_frame: int,
    fps: float,
    frame_width: int,
    frame_height: int,
    initial_box: BoundingBox,
    estimator: PoseEstimator,
    config: TrackerConfig | None = None,
) -> PoseExtractionResult:
    """Decode a clip's pre-release window from disk and extract pose.

    Only frames ``[start_frame, pre_release_end_frame]`` are ever decoded —
    the request to FFmpeg itself stops at the cutoff.
    """
    frame_source = iter_frames(
        video_path,
        start_frame=start_frame,
        end_frame=pre_release_end_frame,
        fps=fps,
        width=frame_width,
        height=frame_height,
    )
    return extract_pose_frames(
        frame_source,
        release_frame=release_frame,
        pre_release_end_frame=pre_release_end_frame,
        fps=fps,
        initial_box=initial_box,
        frame_width=frame_width,
        frame_height=frame_height,
        estimator=estimator,
        config=config,
    )


def _summarize_quality(
    tracked: list[TrackedFrame],
    smoothed: list[BoundingBox],
    worst_miss_streak: int,
    config: TrackerConfig,
) -> QualitySummary:
    total = len(tracked)
    detected = sum(1 for t in tracked if t.detected)
    detection_scores = [t.detection_score for t in tracked if t.detection_score is not None]
    keypoint_means = [
        sum(t.keypoint_scores) / len(t.keypoint_scores) for t in tracked if t.keypoint_scores
    ]
    jump = max_center_jump(smoothed)
    mean_height = sum(b.height for b in smoothed) / len(smoothed)

    issues: list[str] = []
    detected_fraction = detected / total
    if detected_fraction < MIN_DETECTED_FRACTION:
        issues.append(
            f"pitcher detected in only {detected}/{total} frames "
            f"({detected_fraction:.0%}, minimum {MIN_DETECTED_FRACTION:.0%})"
        )
    if worst_miss_streak > config.max_consecutive_misses:
        issues.append(
            f"lost track for {worst_miss_streak} consecutive frames "
            f"(limit {config.max_consecutive_misses})"
        )
    jump_limit = mean_height * MAX_CENTER_JUMP_BOX_FRACTION
    if jump > jump_limit:
        issues.append(
            f"largest frame-to-frame box jump {jump:.0f}px exceeds "
            f"{jump_limit:.0f}px ({MAX_CENTER_JUMP_BOX_FRACTION:.0%} of mean box height)"
        )

    return QualitySummary(
        total_frames=total,
        detected_frames=detected,
        max_consecutive_misses=worst_miss_streak,
        mean_detection_score=(
            sum(detection_scores) / len(detection_scores) if detection_scores else None
        ),
        mean_keypoint_score=(sum(keypoint_means) / len(keypoint_means) if keypoint_means else None),
        max_center_jump_px=jump,
        tracking_ok=not issues,
        issues=issues,
    )
