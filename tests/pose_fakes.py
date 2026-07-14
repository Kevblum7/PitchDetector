"""Fake pose estimators for tests (no model weights, no network).

``BlobPoseEstimator`` genuinely *finds* a bright rectangle in whatever crop it
is given, so tracker tests exercise the real crop → full-frame coordinate
mapping instead of asserting against scripted outputs. ``CenteredPoseEstimator``
always reports a person filling the middle of the crop — enough for end-to-end
API flow tests on arbitrary video content.
"""

from __future__ import annotations

import numpy as np

from ml.pose.estimator import PersonDetection
from ml.pose.geometry import BoundingBox

_NUM_KEYPOINTS = 17


class BlobPoseEstimator:
    """Detects the bounding box of pixels brighter than 200 (red channel)."""

    @property
    def name(self) -> str:
        return "fake-blob"

    def detect(self, image_rgb: np.ndarray) -> list[PersonDetection]:
        mask = image_rgb[:, :, 0] > 200
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return []
        box = BoundingBox(
            x=float(xs.min()),
            y=float(ys.min()),
            width=float(xs.max() - xs.min() + 1),
            height=float(ys.max() - ys.min() + 1),
        )
        cx, cy = box.center
        return [
            PersonDetection(
                box=box,
                score=0.95,
                keypoints=[(cx, cy)] * _NUM_KEYPOINTS,
                keypoint_scores=[0.9] * _NUM_KEYPOINTS,
            )
        ]


class CenteredPoseEstimator:
    """Always reports one person occupying the central 60% of the crop."""

    @property
    def name(self) -> str:
        return "fake-centered"

    def detect(self, image_rgb: np.ndarray) -> list[PersonDetection]:
        h, w = image_rgb.shape[:2]
        bw, bh = w * 0.6, h * 0.6
        box = BoundingBox(x=(w - bw) / 2, y=(h - bh) / 2, width=bw, height=bh)
        cx, cy = box.center
        return [
            PersonDetection(
                box=box,
                score=0.9,
                keypoints=[(cx, cy)] * _NUM_KEYPOINTS,
                keypoint_scores=[0.8] * _NUM_KEYPOINTS,
            )
        ]
