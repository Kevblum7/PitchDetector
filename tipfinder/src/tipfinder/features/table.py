"""Assemble the modelling table: one row per usable pitch = context + body features."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tipfinder.config import Config
from tipfinder.features.body import (
    MEAN_FRAMES,
    build_clip_features,
    build_leaky_features_for_control,
)
from tipfinder.pose.base import J, PoseSequence
from tipfinder.window.release import ReleaseWindow, hand_speed, torso_length

log = logging.getLogger(__name__)

# Post-separation frames added in the leakage control: 40 frames (~0.67 s at 60 fps) covers
# arm swing, release (~25 frames after separation) and early follow-through.
LEAK_EXTRA_FRAMES = 40

META = ["pitch_key", "game_pk", "game_date", "pitch_type", "is_fastball", "home_team"]


def window_quality(seq: PoseSequence, rw: ReleaseWindow, cfg: Config) -> dict:
    """Label-blind pose QC on the frames the features actually use (last MEAN_FRAMES frames
    before the guard cutoff)."""
    m = (seq.frame_idx >= rw.set_start) & (seq.frame_idx < rw.separation - cfg.window.guard_frames)
    idx = np.where(m)[0][-MEAN_FRAMES:]
    win = PoseSequence(seq.frame_idx[idx], seq.kp[idx], seq.fps, seq.width, seq.height, "")
    share = win.good_joint_share(cfg.pose.good_visibility)
    # Torso scale from the window itself, so no post-separation frame influences QC.
    win_scale = float(np.nanmedian(torso_length(win.kp))) if len(idx) else float("nan")
    speed = hand_speed(seq, win_scale)[idx]
    med_speed = float(np.nanmedian(speed)) if np.any(~np.isnan(speed)) else float("nan")
    missing = float(np.mean(np.isnan(seq.kp[idx, 0, 0]))) if len(idx) else 1.0
    q = cfg.pose
    passed = (
        len(idx) >= q.min_window_frames
        and share >= q.min_good_frame_share
        and missing <= q.max_missing_share
        and med_speed <= q.max_hand_speed
    )
    return {
        "window_frames": int(len(idx)),
        "pose_quality": share,
        "hand_speed": med_speed,
        "missing_share": missing,
        "passed": bool(passed),
    }


def _camera_geometry(seq: PoseSequence, rw: ReleaseWindow) -> tuple[float, float, float]:
    """Median hip-centre (x/W, y/H) and torso length (/H) over the set: where the camera
    puts the pitcher. Label-blind."""
    m = (seq.frame_idx >= rw.set_start) & (seq.frame_idx < rw.separation)
    kp = seq.kp[m]
    hip = (kp[:, J["left_hip"], :2] + kp[:, J["right_hip"], :2]) / 2
    sh = (kp[:, J["left_shoulder"], :2] + kp[:, J["right_shoulder"], :2]) / 2
    torso = np.linalg.norm(sh - hip, axis=1)
    return (
        float(np.nanmedian(hip[:, 0])) / seq.width,
        float(np.nanmedian(hip[:, 1])) / seq.height,
        float(np.nanmedian(torso)) / seq.height,
    )


def camera_deviation(geo: pd.DataFrame, min_clips: int) -> pd.Series:
    """Per-clip deviation from its game's median camera geometry (NaN if the game has fewer
    than `min_clips` clips, i.e. cannot be judged)."""
    g = geo.groupby("game_pk")
    dev = np.sqrt(
        ((geo["hip_x"] - g["hip_x"].transform("median")) / 0.04) ** 2
        + ((geo["hip_y"] - g["hip_y"].transform("median")) / 0.04) ** 2
        + ((geo["torso_h"] - g["torso_h"].transform("median")) / 0.02) ** 2
    )
    return dev.where(g["hip_x"].transform("count") >= min_clips)


def build_table(cfg: Config, leaky: bool = False) -> pd.DataFrame:
    run = cfg.run_dir
    man = pd.read_csv(run / "manifest.csv")
    ext = pd.read_csv(run / "extract.csv")
    ok = ext[ext["status"] == "ok"]
    loaded = []
    geo_rows = []
    for r in ok.itertuples():
        rw = ReleaseWindow(int(r.set_start), int(r.separation), bool(r.set_censored))
        seq = PoseSequence.load(run / "pose" / f"{r.pitch_key}.npz")
        loaded.append((r, rw, seq))
        hx, hy, th = _camera_geometry(seq, rw)
        geo_rows.append(
            {
                "pitch_key": r.pitch_key,
                "game_pk": int(r.pitch_key.split("_")[0]),
                "hip_x": hx,
                "hip_y": hy,
                "torso_h": th,
            }
        )
    geo = pd.DataFrame(geo_rows, columns=["pitch_key", "game_pk", "hip_x", "hip_y", "torso_h"])
    cam_dev = dict(
        zip(
            geo["pitch_key"],
            camera_deviation(geo, cfg.pose.min_clips_for_camera_check),
            strict=True,
        )
    )
    rows = []
    qc_rows = []
    for r, rw, seq in loaded:
        qc = window_quality(seq, rw, cfg)
        dev = cam_dev.get(r.pitch_key, np.nan)
        qc["camera_dev"] = dev
        qc["camera_ok"] = bool(np.isnan(dev) or dev <= cfg.pose.max_camera_dev)
        qc["pose_ok"] = qc["passed"]
        qc["passed"] = bool(qc["pose_ok"] and qc["camera_ok"])
        qc_rows.append({"pitch_key": r.pitch_key, **qc})
        if not qc["passed"]:
            continue
        if leaky:
            feats = build_leaky_features_for_control(
                seq.frame_idx, seq.kp, rw, seq.fps, LEAK_EXTRA_FRAMES
            )
        else:
            feats = build_clip_features(seq, rw, cfg.window.guard_frames)
        feats["pitch_key"] = r.pitch_key
        feats["pose_quality"] = qc["pose_quality"]
        rows.append(feats)
    if not leaky:
        pd.DataFrame(qc_rows).to_csv(run / "pose_qc.csv", index=False)
    body = pd.DataFrame(rows)
    table = man.merge(body, on="pitch_key", how="inner")
    table = table.replace([np.inf, -np.inf], np.nan)
    name = "features_leaky.csv" if leaky else "features.csv"
    table.to_csv(run / name, index=False)
    log.info("%s: %d pitches, %d games", name, len(table), table["game_pk"].nunique())
    return table
