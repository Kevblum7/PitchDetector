"""Set-position and hand-separation detection, plus the anti-leakage guard.

Signal: distance between the two wrists divided by torso length (shoulder-mid to hip-mid).
Both quantities are invariant to MediaPipe's left/right labels, which flip when the
pitcher is seen from behind.

* "together": gap < together_ratio.
* separation = first frame after a together-run of >= min_set_frames where the gap leaves
  the together band AND, within 0.8 s, the hands are clearly apart (>= apart_ratio for
  `sustain_frames` frames) and the arm swing opens the gap to >= delivery_ratio. This
  rejects glove fidgets and identifies the actual delivery.
* set_start = first frame of that together-run.

The model window is [set_start, separation - guard_frames). Nothing at or after
`separation - guard_frames` may reach the feature builder; `ModelWindow.check()` enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tipfinder.config import WindowCfg
from tipfinder.pose.base import J, PoseSequence


class LeakageError(RuntimeError):
    """Raised when a frame at/after the protected cutoff would enter the model."""


@dataclass(frozen=True)
class ReleaseWindow:
    set_start: int  # clip frame index
    separation: int  # clip frame index of first frame hands leave the together band
    set_censored: bool  # True if hands were already together when the CF shot began


@dataclass(frozen=True)
class ModelWindow:
    """Frames the model may see: clip frames in [start, cutoff)."""

    start: int
    cutoff: int  # exclusive; == separation - guard_frames
    separation: int
    guard_frames: int

    @classmethod
    def from_release(cls, rw: ReleaseWindow, guard_frames: int) -> ModelWindow:
        if guard_frames < 0:
            raise ValueError("guard_frames must be >= 0")
        cutoff = rw.separation - guard_frames
        if cutoff <= rw.set_start:
            raise ValueError("window empty after guard")
        return cls(rw.set_start, cutoff, rw.separation, guard_frames)

    def check(self, frame_indices: np.ndarray) -> None:
        """Hard guard: every frame index given to a model must be < cutoff < separation."""
        if self.cutoff > self.separation - self.guard_frames:
            raise LeakageError("cutoff later than separation - guard")
        if len(frame_indices) and int(np.max(frame_indices)) >= self.cutoff:
            raise LeakageError(
                f"frame {int(np.max(frame_indices))} >= cutoff {self.cutoff} "
                f"(separation {self.separation})"
            )

    def select(self, seq: PoseSequence) -> tuple[np.ndarray, np.ndarray]:
        """Return (frame_idx, kp) restricted to the window, after the guard check."""
        m = (seq.frame_idx >= self.start) & (seq.frame_idx < self.cutoff)
        idx, kp = seq.frame_idx[m], seq.kp[m]
        self.check(idx)
        return idx, kp


def _mid(kp: np.ndarray, a: str, b: str) -> np.ndarray:
    return (kp[:, J[a], :2] + kp[:, J[b], :2]) / 2.0


def torso_length(kp: np.ndarray) -> np.ndarray:
    sh = _mid(kp, "left_shoulder", "right_shoulder")
    hip = _mid(kp, "left_hip", "right_hip")
    return np.linalg.norm(sh - hip, axis=1)


def _fill_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    x = x.copy()
    n = len(x)
    good = ~np.isnan(x)
    if good.sum() < 2:
        return x
    idx = np.arange(n)
    interp = np.interp(idx, idx[good], x[good])
    i = 0
    while i < n:
        if not good[i]:
            j = i
            while j < n and not good[j]:
                j += 1
            if i > 0 and j < n and j - i <= max_gap:
                x[i:j] = interp[i:j]
            i = j
        else:
            i += 1
    return x


def _median_smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    h = k // 2
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        w = x[max(0, i - h) : i + h + 1]
        w = w[~np.isnan(w)]
        if len(w):
            out[i] = np.median(w)
    return out


def wrist_gap_ratio(seq: PoseSequence, smooth: int = 5) -> np.ndarray:
    kp = seq.kp
    gap = np.linalg.norm(kp[:, J["left_wrist"], :2] - kp[:, J["right_wrist"], :2], axis=1)
    scale = np.nanmedian(torso_length(kp)) if np.any(~np.isnan(gap)) else np.nan
    ratio = gap / scale
    return _median_smooth(_fill_short_gaps(ratio, 5), smooth)


def hand_speed(seq: PoseSequence, scale: float | None = None) -> np.ndarray:
    """Per-frame speed of the hands' midpoint in torso lengths per frame (NaN if unknown).
    `scale` defaults to the median torso length over `seq`."""
    hands = (seq.kp[:, J["left_wrist"], :2] + seq.kp[:, J["right_wrist"], :2]) / 2
    if scale is None:
        scale = float(np.nanmedian(torso_length(seq.kp)))
    v = np.full(len(hands), np.nan)
    v[1:] = np.linalg.norm(np.diff(hands, axis=0), axis=1) / scale
    return v


def detect_release_window(
    seq: PoseSequence, cfg: WindowCfg, delivery_ratio: float = 1.2
) -> ReleaseWindow | None:
    if np.all(np.isnan(seq.kp[:, 0, 0])):
        return None
    g = wrist_gap_ratio(seq, cfg.smooth_frames)  # short dropouts (<= 5 frames) interpolated
    measured = ~np.isnan(g)
    # A longer dropout breaks the set: unmeasured frames are never "together", and the
    # crossing itself must be observed on both sides, so tracking loss cannot fake a break.
    together = measured & (np.nan_to_num(g, nan=np.inf) < cfg.together_ratio)
    speed = hand_speed(seq)
    n = len(g)
    look = max(int(round(0.8 * seq.fps)), cfg.sustain_frames + 1)
    for t in range(1, n):
        if not (together[t - 1] and measured[t] and not together[t]):
            continue
        run_start = t - 1
        while run_start > 0 and together[run_start - 1]:
            run_start -= 1
        if t - run_start < cfg.min_set_frames:
            continue
        # A set is HELD: the arm swing can bring the hands briefly together but fast.
        run_speed = speed[run_start + 1 : t]
        if np.isnan(run_speed).all() or np.nanmedian(run_speed) > cfg.max_set_speed:
            continue
        ahead = np.nan_to_num(g[t : t + look], nan=0.0)
        apart = ahead >= cfg.apart_ratio
        sustained = any(
            apart[k : k + cfg.sustain_frames].all()
            for k in range(max(0, len(apart) - cfg.sustain_frames + 1))
        )
        if sustained and ahead.max(initial=0.0) >= delivery_ratio:
            return ReleaseWindow(
                set_start=int(seq.frame_idx[run_start]),
                separation=int(seq.frame_idx[t]),
                # The set start is only "observed" if the frame before it was measured with
                # the hands apart; after a dropout or at the shot start it is censored.
                set_censored=run_start == 0 or not measured[run_start - 1],
            )
    return None
