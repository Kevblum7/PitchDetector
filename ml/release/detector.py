"""Automatic release-frame detection from wrist kinematics (CLAUDE.md §9).

Stage 2 + 3 of the auto-labeler: inside the motion window, sample pose
coarsely to find the fastest wrist movement, then refine frame-by-frame around
that candidate. The throwing wrist reaches peak speed at ball release, so the
per-frame speed peak is the release candidate.

The confidence score is an explicit heuristic (peak prominence x detection
coverage x edge penalty), NOT a calibrated probability. Its only job is to
rank clips for the human review queue; the cross-reference validation against
manual ground truth is what earns trust at scale.

**Leakage note:** this module deliberately analyses release and post-release
frames — it is labeling tooling, never model input. Its output feeds the
protected cutoff, which is why provenance and validation are mandatory
(see CLAUDE.md §9).

Frame access goes through a ``FrameReader`` callback so tests can feed
synthetic arrays without video files.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from ml.pose.estimator import PoseEstimator
from ml.release.motion import MotionWindow

RELEASE_DETECTOR_VERSION = "0.1.0"

# COCO-17 indices (see ml.pose.estimator.COCO_KEYPOINT_NAMES).
_WRIST_INDICES = {"left": 9, "right": 10}

# (start_frame, end_frame, step) -> iterable of (frame_index, rgb_frame).
FrameReader = Callable[[int, int, int], Iterable[tuple[int, np.ndarray]]]


class ReleaseDetectionError(RuntimeError):
    """Raised when no release frame can be detected."""


@dataclass(frozen=True)
class ReleaseDetectorConfig:
    """Tunables, persisted with every detection for reproducibility."""

    motion_scale_width: int = 160
    motion_threshold_fraction: float = 0.3
    motion_margin_frames: int = 8
    coarse_step: int = 4
    refine_radius: int = 6
    min_detection_score: float = 0.5
    # Peak speed at this multiple of the window's median speed earns a full
    # prominence score.
    prominence_full_score: float = 4.0
    # Below this confidence a human should review the label.
    review_confidence_threshold: float = 0.6

    def as_dict(self) -> dict[str, float | int]:
        return {
            "motion_scale_width": self.motion_scale_width,
            "motion_threshold_fraction": self.motion_threshold_fraction,
            "motion_margin_frames": self.motion_margin_frames,
            "coarse_step": self.coarse_step,
            "refine_radius": self.refine_radius,
            "min_detection_score": self.min_detection_score,
            "prominence_full_score": self.prominence_full_score,
            "review_confidence_threshold": self.review_confidence_threshold,
        }


@dataclass(frozen=True)
class ReleaseDetection:
    """One detected release frame with the evidence behind it."""

    release_frame: int
    confidence: float  # heuristic, 0..1 — ranks review priority, not truth
    review_recommended: bool
    wrist: str  # "left" | "right"
    peak_speed_px_per_frame: float
    median_speed_px_per_frame: float
    motion_window: tuple[int, int]
    coarse_candidate_frame: int
    fine_range: tuple[int, int]
    frames_analyzed: int
    detector_version: str = RELEASE_DETECTOR_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "release_frame": self.release_frame,
            "confidence": self.confidence,
            "review_recommended": self.review_recommended,
            "wrist": self.wrist,
            "peak_speed_px_per_frame": self.peak_speed_px_per_frame,
            "median_speed_px_per_frame": self.median_speed_px_per_frame,
            "motion_window": list(self.motion_window),
            "coarse_candidate_frame": self.coarse_candidate_frame,
            "fine_range": list(self.fine_range),
            "frames_analyzed": self.frames_analyzed,
            "detector_version": self.detector_version,
        }


def detect_release_from_window(
    read_frames: FrameReader,
    window: MotionWindow,
    *,
    video_start: int,
    video_end: int,
    estimator: PoseEstimator,
    config: ReleaseDetectorConfig | None = None,
) -> ReleaseDetection:
    """Detect the release frame inside an already-located motion window."""
    config = config or ReleaseDetectorConfig()

    # --- Coarse pass: subsampled pose over the motion window ------------------
    coarse_positions = _wrist_positions(
        read_frames(window.start_frame, window.end_frame, config.coarse_step),
        estimator,
        config,
    )
    if len(coarse_positions) < 2:
        raise ReleaseDetectionError(
            f"pitcher detected in only {len(coarse_positions)} coarse frames in "
            f"motion window [{window.start_frame}, {window.end_frame}]; "
            "cannot measure wrist speed"
        )
    coarse_speeds = _wrist_speeds(coarse_positions)
    candidate, _wrist, _speed = _peak(coarse_speeds)

    # --- Fine pass: every frame around the coarse candidate -------------------
    fine_start = max(video_start, candidate - config.refine_radius)
    fine_end = min(video_end, candidate + config.refine_radius)
    fine_positions = _wrist_positions(read_frames(fine_start, fine_end, 1), estimator, config)
    if len(fine_positions) < 2:
        raise ReleaseDetectionError(
            f"pitcher detected in only {len(fine_positions)} frames in refine "
            f"range [{fine_start}, {fine_end}]"
        )
    fine_speeds = _wrist_speeds(fine_positions)
    release_frame, wrist, peak_speed = _peak(fine_speeds)

    # --- Heuristic confidence --------------------------------------------------
    wrist_speeds = [speed[wrist] for _, speed in fine_speeds if wrist in speed]
    median_speed = float(np.median(wrist_speeds)) if wrist_speeds else 0.0
    prominence = peak_speed / (median_speed + 1e-6)
    prominence_score = min(1.0, prominence / config.prominence_full_score)
    coverage = len(fine_positions) / (fine_end - fine_start + 1)
    speed_frames = [frame for frame, _ in fine_speeds]
    at_edge = release_frame in (speed_frames[0], speed_frames[-1])
    edge_penalty = 0.5 if at_edge else 1.0
    confidence = round(prominence_score * coverage * edge_penalty, 4)

    return ReleaseDetection(
        release_frame=release_frame,
        confidence=confidence,
        review_recommended=confidence < config.review_confidence_threshold,
        wrist=wrist,
        peak_speed_px_per_frame=round(peak_speed, 2),
        median_speed_px_per_frame=round(median_speed, 2),
        motion_window=(window.start_frame, window.end_frame),
        coarse_candidate_frame=candidate,
        fine_range=(fine_start, fine_end),
        frames_analyzed=len(coarse_positions) + len(fine_positions),
    )


def _wrist_positions(
    frames: Iterable[tuple[int, np.ndarray]],
    estimator: PoseEstimator,
    config: ReleaseDetectorConfig,
) -> list[tuple[int, dict[str, tuple[float, float]]]]:
    """Per sampled frame, both wrist positions of the best person detection.

    The auto-labeler runs on full frames (no manual box exists yet); in the
    MVP's centre-field view the pitcher is the dominant detection, so the
    highest-scoring person is used.
    """
    positions: list[tuple[int, dict[str, tuple[float, float]]]] = []
    for frame_index, frame in frames:
        detections = [d for d in estimator.detect(frame) if d.score >= config.min_detection_score]
        if not detections:
            continue
        best = max(detections, key=lambda d: d.score)
        wrists = {
            side: best.keypoints[kp_index]
            for side, kp_index in _WRIST_INDICES.items()
            if kp_index < len(best.keypoints)
        }
        if wrists:
            positions.append((frame_index, wrists))
    return positions


def _wrist_speeds(
    positions: list[tuple[int, dict[str, tuple[float, float]]]],
) -> list[tuple[int, dict[str, float]]]:
    """Per-wrist speed (px/frame) between consecutive detected samples.

    Speed is attributed to the *later* frame of each pair and normalized by
    the frame gap, so subsampled and dense passes are comparable.
    """
    speeds: list[tuple[int, dict[str, float]]] = []
    for (prev_frame, prev_wrists), (cur_frame, cur_wrists) in zip(
        positions, positions[1:], strict=False
    ):
        gap = cur_frame - prev_frame
        if gap <= 0:  # pragma: no cover - reader yields ascending frames
            continue
        frame_speeds: dict[str, float] = {}
        for side in _WRIST_INDICES:
            if side in prev_wrists and side in cur_wrists:
                (x0, y0), (x1, y1) = prev_wrists[side], cur_wrists[side]
                frame_speeds[side] = math.dist((x0, y0), (x1, y1)) / gap
        if frame_speeds:
            speeds.append((cur_frame, frame_speeds))
    return speeds


def _peak(speeds: list[tuple[int, dict[str, float]]]) -> tuple[int, str, float]:
    """(frame, wrist, speed) of the fastest wrist movement in the series."""
    if not speeds:
        raise ReleaseDetectionError("no wrist speed samples available")
    best_frame, best_wrist, best_speed = -1, "", -1.0
    for frame, frame_speeds in speeds:
        for wrist, speed in frame_speeds.items():
            # Strict > keeps the tie-break deterministic: earliest frame wins,
            # "left" before "right" within a frame (dict order is fixed).
            if speed > best_speed:
                best_frame, best_wrist, best_speed = frame, wrist, speed
    return best_frame, best_wrist, best_speed
