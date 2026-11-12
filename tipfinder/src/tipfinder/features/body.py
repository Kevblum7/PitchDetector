"""Interpretable body-cue features from the pre-separation window.

All per-frame quantities are expressed in units of torso length (shoulder-mid to hip-mid,
median over the window) so they are comparable across camera zoom levels. None of them
depend on MediaPipe's left/right labels, which flip when the pitcher is seen from behind:
two-sided quantities are averaged over both sides, and line angles order points by image x.

Each quantity is summarised twice: the mean over the last 0.5 s before the guard cutoff
("_mean") and over the last 0.1 s ("_end", the instant the hands break). Both windows are
anchored to separation, never to the start of the broadcast shot.
"""

from __future__ import annotations

import numpy as np

from tipfinder.pose.base import J, PoseSequence
from tipfinder.window.release import ModelWindow, ReleaseWindow

END_FRAMES = 6  # "_end": last 0.1 s before the guard cutoff
MEAN_FRAMES = 30  # "_mean": last 0.5 s before the cutoff (anchored to separation, not to the
# broadcast cut, so window length cannot encode where the clip was cut relative to release)

# Plain-language descriptions used by the explanation layer (higher value -> phrase).
DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "hands_height": ("hands/glove set lower (farther below the shoulders)", "glove height"),
    "hands_depth": ("hands set farther toward the third-base side of the hips (camera right)", "hands position"),
    "hands_gap": ("hands held less tightly together (wider gap between wrists)", "hand gap"),
    "forearm_angle": ("forearms more vertical", "glove/forearm angle"),
    "elbow_spread": ("elbows flared wider apart", "elbow flare"),
    "finger_flare": ("fingers/hand landmarks more spread", "wrist/finger flare"),
    "wrist_bend": ("wrists more bent relative to the forearm", "wrist angle"),
    "head_x": ("head shifted toward the third-base side (camera right)", "head position"),
    "head_y": ("head held lower relative to the shoulders", "head height"),
    "head_tilt": ("head tilted more (ear line rotated clockwise)", "head tilt"),
    "shoulder_tilt": ("shoulder line tilted more (rotated clockwise)", "shoulder tilt"),
    "shoulder_width": ("shoulders appear wider (more square to the camera)", "shoulder rotation"),
    "torso_lean": ("torso leaning more toward the third-base side (camera right)", "torso lean"),
}


def _pt(kp: np.ndarray, name: str) -> np.ndarray:
    return kp[:, J[name], :2]


def _mid(kp: np.ndarray, a: str, b: str) -> np.ndarray:
    return (_pt(kp, a) + _pt(kp, b)) / 2


def _line_angle(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Angle (deg) of the segment between p and q, ordered left->right in the image."""
    left = np.where(p[:, :1] <= q[:, :1], p, q)
    right = np.where(p[:, :1] <= q[:, :1], q, p)
    d = right - left
    return np.degrees(np.arctan2(d[:, 1], d[:, 0]))


def _angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    cos = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def per_frame(kp: np.ndarray) -> dict[str, np.ndarray]:
    """kp: (T, J, 3) in pixels, rows with NaN allowed. Returns dict of (T,) arrays."""
    sh = _mid(kp, "left_shoulder", "right_shoulder")
    hip = _mid(kp, "left_hip", "right_hip")
    scale = np.nanmedian(np.linalg.norm(sh - hip, axis=1))
    hands = _mid(kp, "left_wrist", "right_wrist")
    nose = _pt(kp, "nose")

    def norm(x: np.ndarray) -> np.ndarray:
        return x / scale

    forearm = []
    wrist_bend = []
    flare = []
    for s in ("left", "right"):
        fa = _pt(kp, f"{s}_wrist") - _pt(kp, f"{s}_elbow")
        forearm.append(np.degrees(np.arctan2(np.abs(fa[:, 1]), np.abs(fa[:, 0]))))
        hand = _pt(kp, f"{s}_index") - _pt(kp, f"{s}_wrist")
        wrist_bend.append(_angle_between(fa, hand))
        flare.append(norm(np.linalg.norm(_pt(kp, f"{s}_index") - _pt(kp, f"{s}_pinky"), axis=1)))

    torso_vec = sh - hip
    return {
        "hands_height": norm(hands[:, 1] - sh[:, 1]),
        "hands_depth": norm(hands[:, 0] - hip[:, 0]),
        "hands_gap": norm(np.linalg.norm(_pt(kp, "left_wrist") - _pt(kp, "right_wrist"), axis=1)),
        "forearm_angle": np.mean(forearm, axis=0),
        "elbow_spread": norm(
            np.linalg.norm(_pt(kp, "left_elbow") - _pt(kp, "right_elbow"), axis=1)
        ),
        "finger_flare": np.mean(flare, axis=0),
        "wrist_bend": np.mean(wrist_bend, axis=0),
        "head_x": norm(nose[:, 0] - sh[:, 0]),
        "head_y": norm(nose[:, 1] - sh[:, 1]),
        "head_tilt": _line_angle(_pt(kp, "left_ear"), _pt(kp, "right_ear")),
        "shoulder_tilt": _line_angle(_pt(kp, "left_shoulder"), _pt(kp, "right_shoulder")),
        "shoulder_width": norm(
            np.abs(_pt(kp, "left_shoulder")[:, 0] - _pt(kp, "right_shoulder")[:, 0])
        ),
        "torso_lean": np.degrees(np.arctan2(torso_vec[:, 0], -torso_vec[:, 1])),
    }


def summarise(series: dict[str, np.ndarray], end_frames: int = END_FRAMES) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, x in series.items():
        valid = x[~np.isnan(x)]
        out[f"{name}_mean"] = float(np.mean(valid)) if len(valid) else np.nan
        tail = x[-end_frames:]
        tail = tail[~np.isnan(tail)]
        out[f"{name}_end"] = float(np.mean(tail)) if len(tail) else np.nan
    return out


def body_feature_names() -> list[str]:
    # set_censored / n_window_frames are deliberately excluded: they depend on where the
    # broadcast cut falls relative to release (see features_for_window).
    return [f"{n}_{s}" for n in DESCRIPTIONS for s in ("mean", "end")] + ["time_in_set"]


def features_for_window(
    frame_idx: np.ndarray, kp: np.ndarray, rw: ReleaseWindow, fps: float, frames_used: int
) -> dict[str, float]:
    feats = summarise(per_frame(kp[-MEAN_FRAMES:]))
    # Savant clips are cut at a fixed offset before RELEASE, so absolute frame positions
    # encode separation-to-release timing (post-separation information). Time in set is
    # therefore only used when both ends are observed pre-separation events; a set already
    # under way when the CF shot began gives NaN, never "separation - shot start".
    feats["time_in_set"] = np.nan if rw.set_censored else (rw.separation - rw.set_start) / fps
    feats["set_censored"] = float(rw.set_censored)  # metadata only, not a model feature
    feats["n_window_frames"] = float(frames_used)
    return feats


def build_clip_features(
    seq: PoseSequence, rw: ReleaseWindow, guard_frames: int
) -> dict[str, float]:
    """Real pipeline: features from guarded, strictly pre-separation frames only."""
    mw = ModelWindow.from_release(rw, guard_frames)
    idx, wkp = mw.select(seq)  # raises LeakageError on any post-cutoff frame
    return features_for_window(idx, wkp, rw, seq.fps, len(idx))


def build_leaky_features_for_control(
    frame_idx: np.ndarray, kp: np.ndarray, rw: ReleaseWindow, fps: float, extra_frames: int
) -> dict[str, float]:
    """LEAKAGE CONTROL ONLY. Deliberately includes `extra_frames` frames after separation
    (arm action, grip exposure). Never used by the real pipeline; see evaluation.controls."""
    m = (frame_idx >= rw.set_start) & (frame_idx < rw.separation + extra_frames)
    return features_for_window(frame_idx[m], kp[m], rw, fps, int(m.sum()))
