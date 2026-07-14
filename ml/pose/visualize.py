"""Render tracked pitcher box + pose skeleton onto an annotated MP4.

Drawing uses PIL (no OpenCV — see ``video_io``); encoding streams raw RGB
frames into FFmpeg. All stored coordinates are original-video pixels, so
drawing happens at native resolution with no transform.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ml.pose.estimator import SKELETON_EDGES
from ml.pose.pipeline import PoseFrameResult
from ml.pose.video_io import VideoWriter, iter_frames

_BOX_DETECTED = (46, 204, 113)  # green
_BOX_CARRIED = (230, 126, 34)  # orange: carried-forward (no detection)
_SKELETON = (52, 152, 219)  # blue
_KEYPOINT = (231, 76, 60)  # red
_TEXT = (255, 255, 255)
_TEXT_BG = (0, 0, 0)


def render_pose_overlay(
    video_path: Path,
    frames: Sequence[PoseFrameResult],
    output_path: Path,
    *,
    fps: float,
    width: int,
    height: int,
    min_keypoint_score: float | None = None,
) -> Path:
    """Write an annotated MP4 for the given pose frames and return its path.

    ``frames`` must be contiguous ascending frame indices (the pipeline output
    or its DB round-trip). Only those frames are decoded — the render can never
    show release or post-release footage for the clip.
    """
    if not frames:
        raise ValueError("no pose frames to render")
    indices = [f.frame_index for f in frames]
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("pose frames must be contiguous and ascending")

    by_index = {f.frame_index: f for f in frames}
    with VideoWriter(output_path, width=width, height=height, fps=fps) as writer:
        for frame_index, rgb in iter_frames(
            video_path,
            start_frame=indices[0],
            end_frame=indices[-1],
            fps=fps,
            width=width,
            height=height,
        ):
            annotated = _draw_frame(rgb, by_index[frame_index], min_keypoint_score)
            writer.write(annotated)
    return output_path


def _draw_frame(
    rgb: np.ndarray,
    pose: PoseFrameResult,
    min_keypoint_score: float | None,
) -> np.ndarray:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    x1, y1, x2, y2 = pose.box.to_xyxy_int()
    box_color = _BOX_DETECTED if pose.detected else _BOX_CARRIED
    draw.rectangle((x1, y1, x2, y2), outline=box_color, width=2)

    if pose.keypoints is not None:
        scores = pose.keypoint_scores or [0.0] * len(pose.keypoints)
        visible = [min_keypoint_score is None or score >= min_keypoint_score for score in scores]
        for a, b in SKELETON_EDGES:
            if a < len(pose.keypoints) and b < len(pose.keypoints) and visible[a] and visible[b]:
                draw.line((pose.keypoints[a], pose.keypoints[b]), fill=_SKELETON, width=2)
        for point, ok in zip(pose.keypoints, visible, strict=True):
            if ok:
                px, py = point
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=_KEYPOINT)

    label = (
        f"frame {pose.frame_index}  "
        f"{abs(pose.timestamp_ms_relative_to_release):.0f} ms before release"
    )
    if not pose.detected:
        label += "  [no detection]"
    text_anchor = (x1, max(0, y1 - 14))
    text_box = draw.textbbox(text_anchor, label)
    draw.rectangle(text_box, fill=_TEXT_BG)
    draw.text(text_anchor, label, fill=_TEXT)

    return np.asarray(image)
