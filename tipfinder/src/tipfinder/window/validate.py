"""Compare detected set/separation frames with hand labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tipfinder.config import Config


def compare(
    labels: pd.DataFrame, extract: pd.DataFrame, guard_frames: int, tolerance: int = 4
) -> dict:
    """labels: pitch_key, expect (keep|exclude), separation (clip frame, blank if exclude).
    extract: pitch_key, status, separation."""
    m = labels.merge(
        extract[["pitch_key", "status", "separation"]],
        on="pitch_key",
        how="left",
        suffixes=("_label", "_det"),
    )
    keep = m[m["expect"] == "keep"]
    excl = m[m["expect"] == "exclude"]
    found = keep[keep["status"] == "ok"]
    err = (found["separation_det"] - found["separation_label"]).to_numpy(dtype=float)
    return {
        "n_labelled": int(len(m)),
        "n_keep_labels": int(len(keep)),
        "n_keep_detected": int(len(found)),
        "keep_rejected": keep.loc[keep["status"] != "ok", ["pitch_key", "status"]].to_dict(
            "records"
        ),
        "n_exclude_labels": int(len(excl)),
        "exclude_correctly_rejected": int((excl["status"] != "ok").sum()),
        "exclude_wrongly_kept": excl.loc[excl["status"] == "ok", "pitch_key"].tolist(),
        "separation_error_frames": {
            k: int(e) for k, e in zip(found["pitch_key"], err, strict=True)
        },
        "separation_mae_frames": float(np.mean(np.abs(err))) if len(err) else None,
        "separation_within_tolerance": float(np.mean(np.abs(err) <= tolerance))
        if len(err)
        else None,
        "tolerance_frames": tolerance,
        # Leakage-relevant: a detection LATER than the label lets post-separation frames
        # in unless the guard covers it. Model cutoff = detection - guard.
        "max_late_error_frames": int(err.max()) if len(err) else None,
        "guard_frames": guard_frames,
        "cutoff_after_true_separation": int(np.sum(err - guard_frames >= 0)) if len(err) else 0,
    }


def validate_against_labels(cfg: Config, labels_path: Path, tolerance: int = 4) -> dict:
    """Score the detector twice: on every clip where a window was found (raw detector), and
    on the clips that also pass pose QC, i.e. the ones that reach the model."""
    lab = pd.read_csv(labels_path)
    ext = pd.read_csv(cfg.run_dir / "extract.csv")
    out = {"detector_all": compare(lab, ext, cfg.window.guard_frames, tolerance)}
    qc_path = cfg.run_dir / "pose_qc.csv"
    if qc_path.exists():
        qc = pd.read_csv(qc_path)
        final = ext.copy()
        failed = set(qc.loc[~qc["passed"], "pitch_key"])
        final.loc[final["pitch_key"].isin(failed), "status"] = "low_pose_quality"
        out["after_pose_qc"] = compare(lab, final, cfg.window.guard_frames, tolerance)
    return out
