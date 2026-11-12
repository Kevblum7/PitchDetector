"""Per-pitcher models and game-grouped out-of-fold prediction.

Two feature sets are always compared:
  context = count, outs, inning, runners, batter hand, previous pitch, times-through-order,
            and count-specific pitch mix (fastball share by count, estimated on TRAINING
            games of each fold only).
  full    = context + body-cue features.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from tipfinder.data.statcast import CONTEXT_NUMERIC

CONTEXT_FEATURES = [*CONTEXT_NUMERIC, "count_mix"]
MODEL_KINDS = ("logreg", "gbm")


def make_model(kind: str, seed: int = 0) -> Pipeline:
    if kind == "logreg":
        clf = LogisticRegression(C=0.3, max_iter=5000)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)
    if kind == "gbm":
        clf = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=150,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        )
        return make_pipeline(SimpleImputer(strategy="median"), clf)
    raise ValueError(kind)


def game_folds(games: Sequence[int], n_splits: int, seed: int) -> dict[int, int]:
    """Assign each game (not pitch) to a fold. Seeded, balanced by game count."""
    uniq = np.array(sorted(set(int(g) for g in games)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    return {int(g): i % n_splits for i, g in enumerate(uniq)}


def add_count_mix(
    train: pd.DataFrame, test: pd.DataFrame, y_col: str, m: float = 10.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fastball share by count from TRAIN rows only, shrunk toward the train base rate."""
    base = train[y_col].mean()
    grp = train.groupby(["balls", "strikes"])[y_col].agg(["sum", "count"])
    rate = (grp["sum"] + m * base) / (grp["count"] + m)

    def lookup(df: pd.DataFrame) -> pd.Series:
        keys = pd.MultiIndex.from_arrays([df["balls"], df["strikes"]])
        return pd.Series(rate.reindex(keys).fillna(base).to_numpy(), index=df.index)

    tr, te = train.copy(), test.copy()
    tr["count_mix"], te["count_mix"] = lookup(tr), lookup(te)
    return tr, te


def oof_predict(
    df: pd.DataFrame,
    features: list[str],
    kind: str,
    folds: dict[int, int],
    y_col: str = "is_fastball",
    seed: int = 0,
) -> np.ndarray:
    """Out-of-fold P(class) for every row; each game is predicted by a model that never
    saw any pitch from that game. Returns (n,) for binary, (n, k) for multiclass."""
    fold = df["game_pk"].map(folds).to_numpy()
    classes = np.sort(df[y_col].unique())
    out = np.zeros((len(df), len(classes)))
    for f in np.unique(fold):
        tr_m, te_m = fold != f, fold == f
        tr, te = add_count_mix(df[tr_m], df[te_m], "is_fastball")
        assert not set(tr["game_pk"]) & set(te["game_pk"]), "game leaked across split"
        model = make_model(kind, seed)
        model.fit(tr[features], tr[y_col])
        p = model.predict_proba(te[features])
        cols = np.searchsorted(classes, model.classes_)
        out[np.ix_(np.where(te_m)[0], cols)] = p
    out = np.clip(out, 1e-6, 1 - 1e-6)
    if len(classes) == 2:
        return out[:, 1]
    return out / out.sum(axis=1, keepdims=True)
