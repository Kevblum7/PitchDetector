"""Person detection + pose estimation.

The production estimator is torchvision's Keypoint R-CNN. MediaPipe (the first
choice in CLAUDE.md §6) ships no Intel-mac wheel, so this is the documented
CPU-compatible fallback: it stays inside the torch 2.2.2 wheel family and
returns person boxes *and* 17 COCO keypoints from a single model, which lets
detection, tracking-by-association, and pose share one dependency.

Everything downstream depends only on the :class:`PoseEstimator` protocol, so
tests inject a fake and never load model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ml.pose.geometry import BoundingBox

# COCO-17 keypoint order used by torchvision's Keypoint R-CNN.
COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Skeleton edges (indices into COCO_KEYPOINT_NAMES) for visualization.
SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (5, 6),  # shoulders
    (5, 11),
    (6, 12),  # torso sides
    (11, 12),  # hips
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
    (0, 5),
    (0, 6),  # head to shoulders
)


@dataclass(frozen=True)
class PersonDetection:
    """One detected person, in the coordinates of the image that was analysed."""

    box: BoundingBox
    score: float
    keypoints: list[tuple[float, float]]  # 17 (x, y) points
    keypoint_scores: list[float]  # 17 confidence values

    def shifted(self, dx: float, dy: float) -> PersonDetection:
        """Translate the detection (used to map crop coordinates to full frame)."""
        return PersonDetection(
            box=self.box.shift(dx, dy),
            score=self.score,
            keypoints=[(x + dx, y + dy) for x, y in self.keypoints],
            keypoint_scores=list(self.keypoint_scores),
        )


class PoseEstimator(Protocol):
    """Detects people (box + keypoints) in an RGB uint8 image (H, W, 3)."""

    @property
    def name(self) -> str: ...

    def detect(self, image_rgb: np.ndarray) -> list[PersonDetection]: ...


class TorchvisionPoseEstimator:
    """Keypoint R-CNN (ResNet-50 FPN) running on CPU.

    Weights (~230 MB) are downloaded by torchvision to ``~/.cache/torch`` on
    first use; the model is loaded lazily so constructing the estimator stays
    cheap.
    """

    def __init__(self, min_score: float = 0.5) -> None:
        self.min_score = min_score
        self._model = None

    @property
    def name(self) -> str:
        return "torchvision-keypointrcnn_resnet50_fpn"

    def _load(self):  # type: ignore[no-untyped-def]  # torchvision model type is unwieldy
        if self._model is None:
            from torchvision.models.detection import (
                KeypointRCNN_ResNet50_FPN_Weights,
                keypointrcnn_resnet50_fpn,
            )

            model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
            model.eval()
            self._model = model
        return self._model

    def detect(self, image_rgb: np.ndarray) -> list[PersonDetection]:
        import torch

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) RGB image, got shape {image_rgb.shape}")

        model = self._load()
        tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float() / 255.0
        with torch.inference_mode():
            output = model([tensor])[0]

        detections: list[PersonDetection] = []
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        keypoints = output["keypoints"].cpu().numpy()  # (N, 17, 3)
        keypoint_scores = output["keypoints_scores"].cpu().numpy()  # (N, 17)

        for i in range(len(scores)):
            if labels[i] != 1:  # COCO label 1 == person
                continue
            if float(scores[i]) < self.min_score:
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                PersonDetection(
                    box=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
                    score=float(scores[i]),
                    keypoints=[(float(x), float(y)) for x, y, _v in keypoints[i]],
                    keypoint_scores=[float(s) for s in keypoint_scores[i]],
                )
            )
        return detections
