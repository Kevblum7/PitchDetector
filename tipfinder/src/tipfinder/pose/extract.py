"""Clip -> CF shot -> pitcher crop -> pose sequence -> release window (stage 2)."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tipfinder.config import Config
from tipfinder.pose.base import JOINTS, J, PoseBackend, PoseSequence
from tipfinder.pose.mediapipe_backend import make_backend
from tipfinder.video.shots import find_shots, pick_cf_shot, read_video
from tipfinder.window.release import detect_release_window

log = logging.getLogger(__name__)


@dataclass
class ExtractResult:
    pitch_key: str
    status: str  # ok | no_cf_shot | no_pitcher | no_separation | error
    shot_start: int | None = None
    shot_end: int | None = None
    set_start: int | None = None
    separation: int | None = None
    set_censored: bool | None = None
    pose_quality: float | None = None  # share of window frames with good key joints
    detected_share: float | None = None  # share of shot frames with a valid pitcher pose
    fps: float | None = None
    message: str = ""


@dataclass(frozen=True)
class PitcherLock:
    cx: float  # hip-centre x (px)
    hip_y: float  # hip-centre y (px)
    body_h: float  # nose-to-ankle height (px)


def _body(kp: np.ndarray) -> tuple[float, float, float, bool]:
    """(hip_x, hip_y, body_height, fully_visible) for one pose."""
    hip = (kp[J["left_hip"], :2] + kp[J["right_hip"], :2]) / 2
    feet_y = max(kp[J["left_ankle"], 1], kp[J["right_ankle"], 1])
    body_h = feet_y - kp[J["nose"], 1]
    vis = kp[[J["nose"], J["left_hip"], J["right_hip"], J["left_ankle"], J["right_ankle"]], 2]
    return float(hip[0]), float(hip[1]), float(body_h), bool(np.all(vis >= 0.5))


def _detect(
    backend: PoseBackend, frame: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray | None:
    x0, y0, x1, y1 = box
    kp = backend.process(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
    if kp is None:
        return None
    kp[:, 0] += x0
    kp[:, 1] += y0
    return kp


def locate_pitcher(
    frames: list[np.ndarray],
    start: int,
    end: int,
    backend: PoseBackend,
    min_h: float = 0.18,
    max_h: float = 0.80,
) -> PitcherLock | None:
    """The pitcher is the largest fully visible standing person in the CF view (closest to
    the camera). Scan overlapping crops on a few opening frames and keep the biggest body.
    Close-up shots (batter cams) fail the size/visibility gate and return None."""
    h, w = frames[0].shape[:2]
    boxes = [
        (int(x * w), int(0.05 * h), int(min(1.0, x + 0.45) * w), h) for x in (0.0, 0.18, 0.36, 0.55)
    ]
    locks = []
    for i in sorted({start + 2, min(start + 12, end - 1), min(start + 24, end - 1)}):
        best = None
        for box in boxes:
            kp = _detect(backend, frames[i], box)
            if kp is None:
                continue
            hx, hy, bh, full = _body(kp)
            feet_in_frame = max(kp[J["left_ankle"], 1], kp[J["right_ankle"], 1]) < 0.995 * h
            if (
                full
                and feet_in_frame
                and min_h * h <= bh <= max_h * h
                and (best is None or bh > best.body_h)
            ):
                best = PitcherLock(hx, hy, bh)
        if best:
            locks.append(best)
    if len(locks) < 2:
        return None
    return PitcherLock(*np.median([[lk.cx, lk.hip_y, lk.body_h] for lk in locks], axis=0))


def run_pose(
    frames: list[np.ndarray],
    start: int,
    end: int,
    lock: PitcherLock,
    backend: PoseBackend,
    fps: float,
) -> PoseSequence:
    """Track the locked pitcher: square crop around his last position, per-frame detection,
    accept only detections continuous with the previous position and of similar size."""
    h, w = frames[0].shape[:2]
    kps = np.full((end - start, len(JOINTS), 3), np.nan)
    cx, hy, bh = lock.cx, lock.hip_y, lock.body_h
    for t, i in enumerate(range(start, end)):
        side = 2.0 * bh
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, hy - 0.62 * side))
        box = (x0, y0, int(min(w, x0 + side)), int(min(h, y0 + side)))
        kp = _detect(backend, frames[i], box)
        if kp is None:
            continue
        hx, hy2, _, _ = _body(kp)
        torso = np.linalg.norm(
            (kp[J["left_shoulder"], :2] + kp[J["right_shoulder"], :2]) / 2 - np.array([hx, hy2])
        )
        if abs(hx - cx) < 0.6 * bh and abs(hy2 - hy) < 0.4 * bh and 0.15 * bh < torso < 0.55 * bh:
            kps[t] = kp
            cx, hy = 0.5 * cx + 0.5 * hx, 0.5 * hy + 0.5 * hy2
    return PoseSequence(np.arange(start, end), kps, fps, w, h, backend.name)


def extract_clip(cfg: Config, row: pd.Series, backend: PoseBackend) -> ExtractResult:
    run = cfg.run_dir
    key = row.pitch_key
    frames, fps = read_video(run / row.clip_path, cfg.video.max_pose_frame)
    shots = find_shots(frames, cfg.video.shot_cut_threshold)
    picked = pick_cf_shot(frames, shots, fps)
    if picked is None:
        return ExtractResult(key, "no_cf_shot", fps=fps)
    shot, _layout = picked
    lock = locate_pitcher(frames, shot.start, shot.end, backend)
    if lock is None:
        return ExtractResult(
            key,
            "no_pitcher",
            shot.start,
            shot.end,
            fps=fps,
            message="no full-body pitcher of CF size (close-up or occluded shot)",
        )
    # Savant clips are cut at a fixed offset before the pitch: separation lands around
    # frames 145-175 at 60 fps. Tracking stops at cfg.video.max_pose_frame (default 300),
    # which still leaves >= 2 s of margin for the post-separation leakage control.
    end = min(shot.end, cfg.video.max_pose_frame)
    if end - shot.start < 30:
        return ExtractResult(
            key, "no_separation", shot.start, shot.end, fps=fps, message="CF shot starts too late"
        )
    seq = run_pose(frames, shot.start, end, lock, backend, fps)
    seq.save(run / "pose" / f"{key}.npz")
    detected = float(np.mean(~np.isnan(seq.kp[:, 0, 0])))
    res = ExtractResult(key, "ok", shot.start, shot.end, fps=fps, detected_share=detected)
    if detected < 0.3:
        res.status = "no_pitcher"
        return res
    rw = detect_release_window(seq, cfg.window)
    if rw is None:
        res.status = "no_separation"
        return res
    res.set_start, res.separation, res.set_censored = rw.set_start, rw.separation, rw.set_censored
    m = (seq.frame_idx >= rw.set_start) & (seq.frame_idx < rw.separation - cfg.window.guard_frames)
    window_seq = PoseSequence(seq.frame_idx[m], seq.kp[m], fps, seq.width, seq.height, seq.backend)
    # Informational here; the QC gate itself is applied when the feature table is built,
    # so the threshold can be changed without re-running pose estimation.
    res.pose_quality = window_seq.good_joint_share(cfg.pose.good_visibility)
    return res


_BACKEND: PoseBackend | None = None


def _init_worker(name: str, complexity: int) -> None:
    global _BACKEND
    _BACKEND = make_backend(name, complexity)


def _one(args: tuple[Config, dict]) -> dict:
    cfg, r = args
    assert _BACKEND is not None
    try:
        return asdict(extract_clip(cfg, pd.Series(r), _BACKEND))
    except Exception as exc:  # recorded per clip, never silently dropped
        return asdict(ExtractResult(r["pitch_key"], "error", message=repr(exc)))


def extract_all(cfg: Config, workers: int = 4, limit: int | None = None) -> pd.DataFrame:
    """Process every downloaded clip not yet in extract.csv; saves every 10 clips (resumable)."""
    man = pd.read_csv(cfg.run_dir / "manifest.csv")
    todo = man[man["status"] == "downloaded"]
    out_path = cfg.run_dir / "extract.csv"
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["pitch_key"])
    todo = todo[~todo["pitch_key"].isin(set(done["pitch_key"]))]
    if limit is not None:
        todo = todo.head(limit)
    results: list[dict] = []

    def save() -> pd.DataFrame:
        frames = [d for d in (done, pd.DataFrame(results)) if len(d)]
        allres = pd.concat(frames, ignore_index=True) if frames else done
        allres.to_csv(out_path, index=False)
        return allres

    if len(todo):
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(cfg.pose.backend, cfg.pose.model_complexity),
        ) as ex:
            for n, res in enumerate(ex.map(_one, [(cfg, r) for r in todo.to_dict("records")])):
                results.append(res)
                if n % 10 == 9:
                    save()
                    log.info("extract %d/%d", n + 1, len(todo))
    allres = save()
    summary = allres["status"].value_counts().to_dict()
    (cfg.run_dir / "extract_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("extract: %s", summary)
    return allres


def redetect_all(cfg: Config) -> pd.DataFrame:
    """Re-run set/separation detection on stored pose (no video decoding, no pose model).
    Use after changing WindowCfg; pose-level outcomes (no_cf_shot, no_pitcher) are kept."""
    path = cfg.run_dir / "extract.csv"
    ext = pd.read_csv(path)
    for i, r in ext.iterrows():
        pose = cfg.run_dir / "pose" / f"{r.pitch_key}.npz"
        if r["status"] not in ("ok", "no_separation") or not pose.exists():
            continue
        seq = PoseSequence.load(pose)
        rw = detect_release_window(seq, cfg.window)
        if rw is None:
            ext.loc[i, ["status", "set_start", "separation", "set_censored", "pose_quality"]] = [
                "no_separation", np.nan, np.nan, np.nan, np.nan
            ]
            continue
        m = (seq.frame_idx >= rw.set_start) & (
            seq.frame_idx < rw.separation - cfg.window.guard_frames
        )
        win = PoseSequence(seq.frame_idx[m], seq.kp[m], seq.fps, seq.width, seq.height, "")
        ext.loc[i, ["status", "set_start", "separation", "set_censored", "pose_quality"]] = [
            "ok", rw.set_start, rw.separation, rw.set_censored,
            win.good_joint_share(cfg.pose.good_visibility),
        ]
    ext.to_csv(path, index=False)
    log.info("redetect: %s", ext["status"].value_counts().to_dict())
    return ext


def reset_extract(cfg: Config) -> None:
    p = cfg.run_dir / "extract.csv"
    if p.exists():
        p.unlink()


__all__ = ["extract_all", "extract_clip", "ExtractResult", "Path"]
