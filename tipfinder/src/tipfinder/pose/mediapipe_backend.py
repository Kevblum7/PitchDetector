"""MediaPipe Pose (BlazePose, CPU) backend. Models ship inside the 0.10.x wheel."""

from __future__ import annotations

import numpy as np

from tipfinder.pose.base import JOINTS

_MP_INDEX = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}
_ORDER = [_MP_INDEX[j] for j in JOINTS]


class MediaPipeBackend:
    name = "mediapipe"

    def __init__(self, model_complexity: int = 1, static: bool = True) -> None:
        import mediapipe as mp  # heavy import kept local

        self._mp = mp
        self._complexity = model_complexity
        # static=True: detect independently on each (moving) crop; we do our own tracking.
        self._static = static
        self._pose = self._make()

    def _make(self):  # type: ignore[no-untyped-def]
        return self._mp.solutions.pose.Pose(
            static_image_mode=self._static,
            model_complexity=self._complexity,
            smooth_landmarks=not self._static,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def reset(self) -> None:
        self._pose.close()
        self._pose = self._make()

    def process(self, rgb: np.ndarray) -> np.ndarray | None:
        res = self._pose.process(rgb)
        if not res.pose_landmarks:
            return None
        h, w = rgb.shape[:2]
        lm = res.pose_landmarks.landmark
        return np.array([[lm[i].x * w, lm[i].y * h, lm[i].visibility] for i in _ORDER], dtype=float)


def make_backend(name: str, model_complexity: int = 1) -> MediaPipeBackend:
    if name == "mediapipe":
        return MediaPipeBackend(model_complexity)
    raise ValueError(f"unknown pose backend {name!r} (implement PoseBackend to add one)")
