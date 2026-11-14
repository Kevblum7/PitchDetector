"""The most important tests: no frame at/after hand separation can reach the model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tipfinder.config import WindowCfg
from tipfinder.features.body import build_clip_features
from tipfinder.pose.base import PoseSequence
from tipfinder.window.release import (
    LeakageError,
    ModelWindow,
    ReleaseWindow,
    detect_release_window,
)

GUARD = 3


def _rw(row: pd.Series) -> ReleaseWindow:
    return ReleaseWindow(int(row.set_start), int(row.separation), bool(row.set_censored))


def _seq(cache_dir: Path, key: str) -> PoseSequence:
    return PoseSequence.load(cache_dir / "pose" / f"{key}.npz")


def test_window_select_never_reaches_separation(cache_dir: Path, ok_pitches: pd.DataFrame) -> None:
    for row in ok_pitches.itertuples():
        seq = _seq(cache_dir, row.pitch_key)
        mw = ModelWindow.from_release(_rw(pd.Series(row._asdict())), GUARD)
        idx, kp = mw.select(seq)
        assert len(idx) > 0
        assert idx.max() < row.separation - GUARD + 0  # strictly before the cutoff
        assert idx.max() < row.separation
        assert kp.shape[0] == len(idx)


def test_guard_raises_on_post_cutoff_frame(cache_dir: Path, ok_pitches: pd.DataFrame) -> None:
    row = ok_pitches.iloc[0]
    mw = ModelWindow.from_release(_rw(row), GUARD)
    with pytest.raises(LeakageError):
        mw.check(np.array([mw.cutoff - 1, int(row.separation)]))
    with pytest.raises(LeakageError):
        mw.check(np.array([mw.cutoff]))
    mw.check(np.array([mw.start, mw.cutoff - 1]))  # inside window is fine


def test_tampered_window_is_caught(ok_pitches: pd.DataFrame) -> None:
    row = ok_pitches.iloc[0]
    bad = ModelWindow(int(row.set_start), int(row.separation) + 5, int(row.separation), GUARD)
    with pytest.raises(LeakageError):
        bad.check(np.array([int(row.separation)]))


def test_features_identical_with_or_without_post_separation_frames(
    cache_dir: Path, ok_pitches: pd.DataFrame
) -> None:
    """Strongest guarantee: scramble every frame at/after the cutoff; real features must not
    change at all. If anything after separation leaked in, they would."""
    rng = np.random.default_rng(0)
    for row in ok_pitches.head(8).itertuples():
        seq = _seq(cache_dir, row.pitch_key)
        rw = _rw(pd.Series(row._asdict()))
        base = build_clip_features(seq, rw, GUARD)
        kp2 = seq.kp.copy()
        post = seq.frame_idx >= rw.separation - GUARD
        kp2[post] = rng.normal(500, 300, size=kp2[post].shape)
        scrambled = build_clip_features(
            PoseSequence(seq.frame_idx, kp2, seq.fps, seq.width, seq.height, seq.backend), rw, GUARD
        )
        for k, v in base.items():
            assert (np.isnan(v) and np.isnan(scrambled[k])) or v == scrambled[k], k


def test_detector_matches_stored_extraction(cache_dir: Path, ok_pitches: pd.DataFrame) -> None:
    for row in ok_pitches.itertuples():
        rw = detect_release_window(_seq(cache_dir, row.pitch_key), WindowCfg())
        assert rw is not None
        assert rw.separation == row.separation
        assert rw.set_start <= rw.separation - WindowCfg().min_set_frames


def test_detector_within_tolerance_of_hand_labels(
    cache_dir: Path, cached_extract: pd.DataFrame
) -> None:
    """Hand labels vs detector. The leakage-critical property is about clips that REACH THE
    MODEL (window found and pose QC passed): their cutoff (detection - guard) must come before
    the true separation. Any clip detected late must be stopped by QC."""
    labels_path, qc_path = cache_dir / "labels.csv", cache_dir / "pose_qc.csv"
    if not labels_path.exists() or not qc_path.exists():
        pytest.skip("no hand labels / QC in cache")
    lab = pd.read_csv(labels_path)
    qc = pd.read_csv(qc_path)[["pitch_key", "passed"]]
    m = lab[lab["expect"] == "keep"].merge(cached_extract, on="pitch_key", suffixes=("_l", "_d"))
    m = m[m["status"] == "ok"].merge(qc, on="pitch_key")
    if m.empty:
        pytest.skip("no labelled usable clips in cache")
    m["err"] = m["separation_d"] - m["separation_l"]
    used = m[m["passed"]]
    assert len(used) > 0
    assert ((used["err"] - GUARD) < 0).all(), used[["pitch_key", "err"]].to_dict("records")
    assert (used["err"].abs() <= 4).mean() >= 0.9
    late = m[(m["err"] - GUARD) >= 0]
    assert (~late["passed"]).all(), "a late detection slipped past pose QC"


def test_clips_labelled_exclude_never_reach_model(
    cache_dir: Path, cached_extract: pd.DataFrame
) -> None:
    labels_path, qc_path = cache_dir / "labels.csv", cache_dir / "pose_qc.csv"
    if not labels_path.exists() or not qc_path.exists():
        pytest.skip("no hand labels / QC in cache")
    lab = pd.read_csv(labels_path)
    excl = set(lab.loc[lab["expect"] == "exclude", "pitch_key"])
    qc = pd.read_csv(qc_path)
    reach = set(cached_extract.loc[cached_extract["status"] == "ok", "pitch_key"]) & set(
        qc.loc[qc["passed"], "pitch_key"]
    )
    assert not (excl & reach)


def test_features_do_not_encode_broadcast_cut_position(
    cache_dir: Path, ok_pitches: pd.DataFrame
) -> None:
    """Clips are cut at a fixed offset before RELEASE, so any feature that depends on where
    the CF shot starts can encode post-separation timing. For a censored set (hands already
    together when the shot began), time_in_set must be NaN and nothing in the model's
    feature list may change if the cut had come a few frames later."""
    from tipfinder.features.body import body_feature_names

    assert "set_censored" not in body_feature_names()
    censored = ok_pitches[ok_pitches["set_censored"]]
    if censored.empty:
        pytest.skip("no censored clips in cache")
    row = censored.iloc[0]
    seq = _seq(cache_dir, row.pitch_key)
    rw = _rw(row)
    feats = build_clip_features(seq, rw, GUARD)
    assert np.isnan(feats["time_in_set"])
    later = ReleaseWindow(rw.set_start + 4, rw.separation, True)
    feats_later = build_clip_features(seq, later, GUARD)
    assert np.isnan(feats_later["time_in_set"])


def test_set_after_tracking_dropout_counts_as_censored(
    cache_dir: Path, ok_pitches: pd.DataFrame
) -> None:
    """If the set run begins right after a tracking dropout, its start was not observed:
    it must be flagged censored (so time_in_set is NaN), not treated as a real set start."""
    for row in ok_pitches.itertuples():
        seq = _seq(cache_dir, row.pitch_key)
        rw = detect_release_window(seq, WindowCfg())
        if rw is None or rw.separation - rw.set_start < 30:
            continue
        kp = seq.kp.copy()
        t_sep = int(np.where(seq.frame_idx == rw.separation)[0][0])
        kp[t_sep - 25 : t_sep - 14] = np.nan  # 11-frame dropout inside the held set
        cut = PoseSequence(seq.frame_idx, kp, seq.fps, seq.width, seq.height, seq.backend)
        rw2 = detect_release_window(cut, WindowCfg())
        assert rw2 is not None and rw2.set_censored
        assert np.isnan(build_clip_features(cut, rw2, GUARD)["time_in_set"])
        return
    pytest.skip("no clip with a long enough set in cache")
