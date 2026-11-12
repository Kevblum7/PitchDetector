"""Split integrity, train-only statistics, and the controls, on the REAL feature table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tipfinder.evaluation.controls import shuffle_labels_within_game, shuffled_label_control
from tipfinder.evaluation.lift import bootstrap_games, permute_within_game
from tipfinder.model.models import CONTEXT_FEATURES, add_count_mix, game_folds, oof_predict
from tipfinder.pipeline import body_columns


def test_game_folds_never_split_a_game(project_features: pd.DataFrame) -> None:
    folds = game_folds(project_features["game_pk"], 5, seed=11)
    assert set(folds) == set(project_features["game_pk"].unique())
    per_game = project_features.assign(fold=project_features["game_pk"].map(folds))
    assert (per_game.groupby("game_pk")["fold"].nunique() == 1).all()
    assert len(set(folds.values())) == 5


def test_count_mix_uses_training_rows_only(project_features: pd.DataFrame) -> None:
    df = project_features
    folds = game_folds(df["game_pk"], 5, seed=11)
    test_m = df["game_pk"].map(folds) == 0
    _, te1 = add_count_mix(df[~test_m], df[test_m], "is_fastball")
    flipped = df.copy()
    flipped.loc[test_m, "is_fastball"] = 1 - flipped.loc[test_m, "is_fastball"]
    _, te2 = add_count_mix(flipped[~test_m], flipped[test_m], "is_fastball")
    assert np.allclose(te1["count_mix"], te2["count_mix"])


def test_oof_predictions_are_out_of_game(project_features: pd.DataFrame) -> None:
    """Changing the labels of one game must not change that game's own predictions."""
    df = project_features.reset_index(drop=True)
    folds = game_folds(df["game_pk"], 5, seed=11)
    g = df["game_pk"].iloc[0]
    p1 = oof_predict(df, CONTEXT_FEATURES, "logreg", folds)
    df2 = df.copy()
    df2.loc[df2["game_pk"] == g, "is_fastball"] = 1 - df2.loc[df2["game_pk"] == g, "is_fastball"]
    p2 = oof_predict(df2, CONTEXT_FEATURES, "logreg", folds)
    m = (df["game_pk"] == g).to_numpy()
    assert np.allclose(p1[m], p2[m])


def test_within_game_shuffles_preserve_game_composition(project_features: pd.DataFrame) -> None:
    df = project_features.reset_index(drop=True)
    s = shuffle_labels_within_game(df, seed=1)
    assert (
        s.groupby("game_pk")["is_fastball"].sum() == df.groupby("game_pk")["is_fastball"].sum()
    ).all()
    body = body_columns(df)
    p = permute_within_game(df, body, np.random.default_rng(0))
    for c in body[:3]:
        a = df.groupby("game_pk")[c].apply(lambda x: sorted(x.dropna()))
        b = p.groupby("game_pk")[c].apply(lambda x: sorted(x.dropna()))
        assert all(np.allclose(x, y) for x, y in zip(a, b, strict=True))


def test_bootstrap_resamples_whole_games() -> None:
    games = np.array([1, 1, 1, 2, 2, 3])
    ll_ctx = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.5])
    ll_full = np.zeros(6)
    boots = bootstrap_games(games, ll_ctx, ll_full, 500, seed=0)
    # Possible values are game-weighted means, never per-pitch mixtures.
    assert boots.min() >= 0 and boots.max() <= 1
    assert np.isclose(boots.mean(), (3 * 1 + 0 + 0.5) / 6, atol=0.1)


def test_shuffled_label_control_kills_lift(project_features: pd.DataFrame) -> None:
    res = shuffled_label_control(
        project_features, body_columns(project_features), "logreg", n_repeats=5, n_splits=5, seed=11
    )
    # Body features cannot predict within-game-shuffled labels beyond context.
    assert res["mean_lift"] < 0.01, res
