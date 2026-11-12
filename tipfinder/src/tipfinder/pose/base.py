"""Backend-agnostic pose types.

A backend maps one RGB image to a (J, 3) array of (x_px, y_px, confidence) in the image's
pixel coordinates for the canonical JOINTS below, or None if nobody was found. Swapping in
ViTPose means writing one class with the same two methods and mapping its COCO(-WholeBody)
keypoints onto JOINTS (fingers: WholeBody hand keypoints 5/9/17 -> thumb/index/pinky tips).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

JOINTS: tuple[str, ...] = (
    "nose",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
J = {name: i for i, name in enumerate(JOINTS)}

# Joints whose detection quality decides whether a clip is usable.
KEY_JOINTS = ("left_wrist", "right_wrist", "left_shoulder", "right_shoulder")


class PoseBackend(Protocol):
    name: str

    def reset(self) -> None:
        """Forget tracking state (call between clips)."""

    def process(self, rgb: np.ndarray) -> np.ndarray | None:
        """Return (len(JOINTS), 3) array [x_px, y_px, conf] for the dominant person."""


@dataclass
class PoseSequence:
    """Pose for a contiguous run of clip frames, in ORIGINAL clip pixel coordinates."""

    frame_idx: np.ndarray  # (T,) original clip frame indices
    kp: np.ndarray  # (T, J, 3); NaN rows where no valid pitcher pose
    fps: float
    width: int
    height: int
    backend: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frame_idx=self.frame_idx,
            kp=self.kp,
            meta=np.array([self.fps, self.width, self.height]),
            backend=np.array(self.backend),
        )

    @classmethod
    def load(cls, path: Path) -> PoseSequence:
        z = np.load(path)
        fps, w, h = z["meta"]
        return cls(z["frame_idx"], z["kp"], float(fps), int(w), int(h), str(z["backend"]))

    def good_joint_share(self, threshold: float, joints: tuple[str, ...] = KEY_JOINTS) -> float:
        """Share of frames in which every key joint has confidence >= threshold."""
        if len(self.kp) == 0:
            return 0.0
        conf = self.kp[:, [J[j] for j in joints], 2]
        return float(np.mean(np.all(np.nan_to_num(conf) >= threshold, axis=1)))
