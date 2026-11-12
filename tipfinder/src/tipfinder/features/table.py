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
from tipfinder.pose.base import PoseSequence
from tipfinder.window.release import ReleaseWindow, hand_speed

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
    speed = hand_speed(seq)[idx]
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


def build_table(cfg: Config, leaky: bool = False) -> pd.DataFrame:
    run = cfg.run_dir
    man = pd.read_csv(run / "manifest.csv")
    ext = pd.read_csv(run / "extract.csv")
    ok = ext[ext["status"] == "ok"]
    rows = []
    qc_rows = []
    for r in ok.itertuples():
        rw = ReleaseWindow(int(r.set_start), int(r.separation), bool(r.set_censored))
        seq = PoseSequence.load(run / "pose" / f"{r.pitch_key}.npz")
        qc = window_quality(seq, rw, cfg)
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
