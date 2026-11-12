"""Sanity controls that must behave as expected for the main result to be trusted.

1. Shuffled-label control: shuffle pitch labels within each game and rerun the context vs.
   full comparison. Lift must collapse to ~0 (body features cannot predict noise). A
   positive lift here would mean the evaluation itself leaks.
2. Leakage (positive) control: rebuild body features with frames AFTER hand separation
   deliberately included. If the pose pipeline can see pitch-type information at all,
   lift should jump; the real pipeline must never see those frames (see window.release).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tipfinder.evaluation.lift import LiftResult, evaluate_lift, pointwise_logloss
from tipfinder.model.models import CONTEXT_FEATURES, game_folds, oof_predict


def shuffle_labels_within_game(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for _, idx in df.groupby("game_pk").indices.items():
        perm = rng.permutation(idx)
        for col in ("is_fastball", "pitch_type"):
            out.iloc[idx, out.columns.get_loc(col)] = df.iloc[perm][col].to_numpy()
    return out


def shuffled_label_control(
    df: pd.DataFrame, body_cols: list[str], kind: str, n_repeats: int, n_splits: int, seed: int
) -> dict:
    df = df.reset_index(drop=True)
    lifts = []
    folds = game_folds(df["game_pk"], n_splits, seed)
    for k in range(n_repeats):
        d = shuffle_labels_within_game(df, seed + 100 + k)
        y = d["is_fastball"].to_numpy()
        pc = oof_predict(d, CONTEXT_FEATURES, kind, folds, seed=seed)
        pf = oof_predict(d, CONTEXT_FEATURES + body_cols, kind, folds, seed=seed)
        lifts.append(float(pointwise_logloss(y, pc).mean() - pointwise_logloss(y, pf).mean()))
    arr = np.array(lifts)
    return {
        "n_repeats": n_repeats,
        "mean_lift": float(arr.mean()),
        "sd_lift": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "max_lift": float(arr.max()),
        "lifts": lifts,
    }


def leakage_control(
    leaky_df: pd.DataFrame,
    body_cols: list[str],
    kind: str,
    n_splits: int,
    n_bootstrap: int,
    seed: int,
) -> LiftResult:
    # No permutation test needed here; we only compare its lift with the real one.
    return evaluate_lift(
        leaky_df, body_cols, kind, "binary", n_splits, n_bootstrap, n_permutations=0, seed=seed
    )
