"""Video-file orchestration for automatic release-frame detection.

Wires the motion gate (downscaled decode) and the coarse-to-fine kinematic
detector (native-resolution decode) to a video on disk.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ml.pose.estimator import PoseEstimator
from ml.pose.video_io import iter_frames
from ml.release.detector import (
    ReleaseDetection,
    ReleaseDetectorConfig,
    detect_release_from_window,
)
from ml.release.motion import find_motion_window


def detect_release_in_video(
    video_path: Path,
    *,
    fps: float,
    width: int,
    height: int,
    frame_count: int,
    estimator: PoseEstimator,
    config: ReleaseDetectorConfig | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> ReleaseDetection:
    """Detect the release frame in ``video_path`` (optionally within a range).

    Raises :class:`ml.release.motion.MotionGateError` or
    :class:`ml.release.detector.ReleaseDetectionError` when the video does not
    contain a detectable delivery.
    """
    config = config or ReleaseDetectorConfig()
    last_frame = frame_count - 1 if end_frame is None else min(end_frame, frame_count - 1)

    # Stage 1: motion gate on downscaled frames (cheap full pass).
    scale_w = min(config.motion_scale_width, width)
    # Preserve aspect ratio; keep dimensions even for the decoder.
    scale_h = max(2, round(height * scale_w / width / 2) * 2)
    small_frames = iter_frames(
        video_path,
        start_frame=start_frame,
        end_frame=last_frame,
        fps=fps,
        width=width,
        height=height,
        out_width=scale_w,
        out_height=scale_h,
    )
    window = find_motion_window(
        small_frames,
        threshold_fraction=config.motion_threshold_fraction,
        margin_frames=config.motion_margin_frames,
    )

    # Stages 2-3: coarse-to-fine pose at native resolution.
    def read_frames(
        range_start: int, range_end: int, step: int
    ) -> Iterable[tuple[int, np.ndarray]]:
        for frame_index, frame in iter_frames(
            video_path,
            start_frame=range_start,
            end_frame=range_end,
            fps=fps,
            width=width,
            height=height,
        ):
            if (frame_index - range_start) % step == 0:
                yield frame_index, frame

    return detect_release_from_window(
        read_frames,
        window,
        video_start=start_frame,
        video_end=last_frame,
        estimator=estimator,
        config=config,
    )
