"""Lift of body cues over the context baseline, with game-level uncertainty.

Primary metric: log-loss improvement (context minus full; > 0 means body cues help).
Also reported: AUC and accuracy for both models, and a majority-class reference.

* Bootstrap: resample GAMES with replacement over the out-of-fold predictions.
* Permutation test: shuffle the body-feature rows within each game (keeps each game's
  distribution, breaks the link to the pitch), retrain everything, recompute lift.
* Decision: "tip" only if permutation p < alpha AND the bootstrap CI lower bound > 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from tipfinder.model.models import CONTEXT_FEATURES, game_folds, oof_predict


def pointwise_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    if p.ndim == 1:
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    classes = np.sort(np.unique(y))
    idx = np.searchsorted(classes, y)
    return -np.log(p[np.arange(len(y)), idx])


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    if p.ndim == 1:
        return float(roc_auc_score(y, p))
    return float(
        roc_auc_score(y, p, multi_class="ovr", average="macro", labels=np.sort(np.unique(y)))
    )


def _acc(y: np.ndarray, p: np.ndarray) -> float:
    if p.ndim == 1:
        return float(accuracy_score(y, (p >= 0.5).astype(int)))
    return float(accuracy_score(y, np.sort(np.unique(y))[p.argmax(axis=1)]))


@dataclass
class LiftResult:
    model: str
    target: str
    n_pitches: int
    n_games: int
    class_balance: dict
    majority_logloss: float
    majority_accuracy: float
    context_logloss: float
    full_logloss: float
    context_auc: float
    full_auc: float
    context_accuracy: float
    full_accuracy: float
    lift_logloss: float
    lift_ci: tuple[float, float]
    lift_auc: float
    lift_auc_ci: tuple[float, float]
    perm_p_value: float | None
    n_permutations: int
    tip_detected: bool
    verdict: str
    null_lifts: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("null_lifts")
        return d


def bootstrap_games(
    games: np.ndarray, ll_ctx: np.ndarray, ll_full: np.ndarray, n: int, seed: int
) -> np.ndarray:
    """Distribution of mean log-loss lift when resampling whole games with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(games, return_inverse=True)
    sums = np.bincount(inv, weights=ll_ctx - ll_full)
    counts = np.bincount(inv)
    draws = rng.integers(0, len(uniq), size=(n, len(uniq)))
    return sums[draws].sum(axis=1) / counts[draws].sum(axis=1)


def _bootstrap_auc(
    games: np.ndarray, y: np.ndarray, pc: np.ndarray, pf: np.ndarray, n: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed + 1)
    uniq = np.unique(games)
    by_game = {g: np.where(games == g)[0] for g in uniq}
    vals = []
    for _ in range(min(n, 500)):
        idx = np.concatenate([by_game[g] for g in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(_auc(y[idx], pf[idx]) - _auc(y[idx], pc[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def permute_within_game(
    df: pd.DataFrame, cols: list[str], rng: np.random.Generator
) -> pd.DataFrame:
    out = df.copy()
    for _, idx in df.groupby("game_pk").indices.items():
        perm = rng.permutation(idx)
        out.iloc[idx, [df.columns.get_loc(c) for c in cols]] = df.iloc[perm][cols].to_numpy()
    return out


def evaluate_lift(
    df: pd.DataFrame,
    body_cols: list[str],
    kind: str = "logreg",
    target: str = "binary",
    n_splits: int = 5,
    n_bootstrap: int = 2000,
    n_permutations: int = 200,
    alpha: float = 0.05,
    seed: int = 11,
) -> LiftResult:
    y_col = "is_fastball" if target == "binary" else "pitch_type"
    df = df.reset_index(drop=True)
    y = df[y_col].to_numpy()
    games = df["game_pk"].to_numpy()
    folds = game_folds(games, n_splits, seed)
    full_cols = CONTEXT_FEATURES + body_cols

    p_ctx = oof_predict(df, CONTEXT_FEATURES, kind, folds, y_col, seed)
    p_full = oof_predict(df, full_cols, kind, folds, y_col, seed)
    ll_ctx, ll_full = pointwise_logloss(y, p_ctx), pointwise_logloss(y, p_full)
    lift = float(ll_ctx.mean() - ll_full.mean())
    boots = bootstrap_games(games, ll_ctx, ll_full, n_bootstrap, seed)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    # Majority-class reference (rate from the whole sample; generous to the baseline).
    vals, cnts = np.unique(y, return_counts=True)
    maj_p = cnts / cnts.sum()
    maj_ll = float(-np.sum(maj_p * np.log(maj_p)))

    rng = np.random.default_rng(seed + 7)
    null = []
    for _ in range(n_permutations):
        dperm = permute_within_game(df, body_cols, rng)
        pp = oof_predict(dperm, full_cols, kind, folds, y_col, seed)
        null.append(float(ll_ctx.mean() - pointwise_logloss(y, pp).mean()))
    p_value = (1 + sum(v >= lift for v in null)) / (1 + len(null)) if null else None

    detected = p_value is not None and p_value < alpha and ci[0] > 0
    verdict = (
        "potential tip: body cues add held-out predictive value beyond game context"
        if detected
        else "no detectable tip"
    )
    auc_c, auc_f = _auc(y, p_ctx), _auc(y, p_full)
    return LiftResult(
        model=kind,
        target=target,
        n_pitches=len(df),
        n_games=int(len(np.unique(games))),
        class_balance={str(k): int(v) for k, v in zip(vals, cnts, strict=True)},
        majority_logloss=maj_ll,
        majority_accuracy=float(maj_p.max()),
        context_logloss=float(ll_ctx.mean()),
        full_logloss=float(ll_full.mean()),
        context_auc=auc_c,
        full_auc=auc_f,
        context_accuracy=_acc(y, p_ctx),
        full_accuracy=_acc(y, p_full),
        lift_logloss=lift,
        lift_ci=ci,
        lift_auc=auc_f - auc_c,
        lift_auc_ci=_bootstrap_auc(games, y, p_ctx, p_full, n_bootstrap, seed),
        perm_p_value=p_value,
        n_permutations=len(null),
        tip_detected=bool(detected),
        verdict=verdict,
        null_lifts=null,
    )
