"""Which body cues carry the signal, in scouting language.

Importance = held-out permutation importance: in each game-fold, the increase in test
log-loss when one body feature is shuffled (fit on training games only). Direction comes
from the raw data: standardised mean difference (offspeed minus fastball) with a game
bootstrap CI. Sentences are generated only from these numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tipfinder.evaluation.lift import pointwise_logloss
from tipfinder.features.body import DESCRIPTIONS
from tipfinder.model.models import CONTEXT_FEATURES, add_count_mix, game_folds, make_model


def heldout_permutation_importance(
    df: pd.DataFrame, body_cols: list[str], kind: str, n_splits: int, seed: int, n_repeats: int = 10
) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    folds = game_folds(df["game_pk"], n_splits, seed)
    fold = df["game_pk"].map(folds).to_numpy()
    cols = CONTEXT_FEATURES + body_cols
    rng = np.random.default_rng(seed)
    gains: dict[str, list[float]] = {c: [] for c in body_cols}
    for f in np.unique(fold):
        tr, te = add_count_mix(df[fold != f], df[fold == f], "is_fastball")
        model = make_model(kind, seed).fit(tr[cols], tr["is_fastball"])
        y = te["is_fastball"].to_numpy()
        base = pointwise_logloss(
            y, np.clip(model.predict_proba(te[cols])[:, 1], 1e-6, 1 - 1e-6)
        ).mean()
        for c in body_cols:
            for _ in range(n_repeats):
                te2 = te.copy()
                te2[c] = rng.permutation(te2[c].to_numpy())
                p = np.clip(model.predict_proba(te2[cols])[:, 1], 1e-6, 1 - 1e-6)
                gains[c].append(float(pointwise_logloss(y, p).mean() - base))
    return (
        pd.DataFrame(
            {
                "feature": list(gains),
                "importance": [np.mean(v) for v in gains.values()],
                "importance_sd": [np.std(v) for v in gains.values()],
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def effect_size(
    df: pd.DataFrame, col: str, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Cohen's d (offspeed minus fastball) with a game-bootstrap 95% CI."""

    def d_of(sub: pd.DataFrame) -> float:
        a = sub.loc[sub.is_fastball == 0, col].dropna()
        b = sub.loc[sub.is_fastball == 1, col].dropna()
        if len(a) < 3 or len(b) < 3:
            return np.nan
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        return float((a.mean() - b.mean()) / sd) if sd > 0 else np.nan

    rng = np.random.default_rng(seed)
    games = df["game_pk"].unique()
    groups = {g: d for g, d in df.groupby("game_pk")}
    boots = [
        d_of(pd.concat([groups[g] for g in rng.choice(games, len(games))])) for _ in range(n_boot)
    ]
    boots_arr = np.array([b for b in boots if not np.isnan(b)])
    return d_of(df), float(np.percentile(boots_arr, 2.5)), float(np.percentile(boots_arr, 97.5))


def describe(feature: str, d: float) -> str:
    base, _, summary = feature.rpartition("_")
    phrase, label = DESCRIPTIONS.get(base, (base, base))
    when = (
        "at the moment the hands break"
        if summary == "end"
        else "over the last half-second of the set"
    )
    if feature == "time_in_set":
        phrase, label, when = "a longer hold in the set", "time in set", ""
    if feature == "set_censored":
        return "set already under way when the camera cut in (a broadcast artefact, not a cue)"
    direction = "before offspeed pitches" if d > 0 else "before fastballs"
    return f"{label}: {phrase} {when} {direction}".replace("  ", " ").strip()


def scouting_notes(
    df: pd.DataFrame, importance: pd.DataFrame, tip_detected: bool, top_k: int = 3, seed: int = 0
) -> list[dict]:
    notes = []
    for row in importance.head(top_k).itertuples():
        if row.importance <= 0:
            continue
        d, lo, hi = effect_size(df, row.feature, seed=seed)
        consistent = (lo > 0) or (hi < 0)
        notes.append(
            {
                "feature": row.feature,
                "heldout_importance": row.importance,
                "effect_size_d": d,
                "effect_ci": (lo, hi),
                "ci_excludes_zero": bool(consistent),
                "sentence": describe(row.feature, d),
                "qualifier": (
                    "supported"
                    if tip_detected and consistent
                    else "exploratory only: overall lift not significant"
                    if not tip_detected
                    else "direction not stable across games"
                ),
            }
        )
    return notes
