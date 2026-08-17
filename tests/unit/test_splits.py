"""Unit tests for game-grouped splitting and leakage guards (CLAUDE.md §18)."""

from __future__ import annotations

import pytest

from ml.datasets.records import ClipRecord
from ml.datasets.splits import (
    GroupedSplit,
    LeakageError,
    assert_label_not_in_feature_names,
    assert_no_duplicate_source_across_splits,
    assert_no_game_overlap,
    group_clips_by_game,
    split_by_game,
)


def _records(spec: dict[str, int], *, start_id: int = 1) -> list[ClipRecord]:
    """Build records from a ``{game_id: n_clips}`` spec, alternating pitch type."""
    records: list[ClipRecord] = []
    clip_id = start_id
    for game, n in spec.items():
        for i in range(n):
            records.append(
                ClipRecord(
                    clip_id=clip_id,
                    game_id=game,
                    pitch_type="slider" if i % 2 else "four_seam_fastball",
                )
            )
            clip_id += 1
    return records


def test_split_keeps_every_game_in_one_split() -> None:
    records = _records({"g1": 10, "g2": 10, "g3": 10, "g4": 10, "g5": 10})
    split = split_by_game(records, test_fraction=0.2, val_fraction=0.2, seed=0)
    # No game crosses splits (this is also asserted inside split_by_game).
    assert_no_game_overlap(split)
    all_games = split.train_games | split.val_games | split.test_games
    assert all_games == {"g1", "g2", "g3", "g4", "g5"}
    # Every clip is assigned exactly once.
    assigned = split.n_train + split.n_val + split.n_test
    assert assigned == len(records)


def test_split_is_deterministic_for_a_seed() -> None:
    records = _records({f"g{i}": 5 for i in range(8)})
    a = split_by_game(records, test_fraction=0.25, val_fraction=0.25, seed=42)
    b = split_by_game(records, test_fraction=0.25, val_fraction=0.25, seed=42)
    assert a == b


def test_split_input_order_does_not_change_result() -> None:
    records = _records({f"g{i}": 5 for i in range(8)})
    forward = split_by_game(records, test_fraction=0.25, val_fraction=0.25, seed=7)
    reversed_ = split_by_game(
        list(reversed(records)), test_fraction=0.25, val_fraction=0.25, seed=7
    )
    assert forward == reversed_


def test_split_roughly_hits_target_fractions() -> None:
    records = _records({f"g{i}": 10 for i in range(10)})  # 100 clips, 10 games
    split = split_by_game(records, test_fraction=0.2, val_fraction=0.2, seed=1)
    # Whole-game assignment means fractions are approximate; each game is 10
    # clips, so test/val land within one game of target.
    assert 10 <= split.n_test <= 30
    assert 10 <= split.n_val <= 30
    assert split.n_train >= 40


def test_empty_records_yields_empty_split() -> None:
    split = split_by_game([], test_fraction=0.2, val_fraction=0.2, seed=0)
    assert split == GroupedSplit((), (), (), frozenset(), frozenset(), frozenset())


def test_missing_game_id_is_rejected() -> None:
    records = [ClipRecord(clip_id=1, game_id="", pitch_type="slider")]
    with pytest.raises(ValueError, match="no game_id"):
        group_clips_by_game(records)
    with pytest.raises(ValueError, match="no game_id"):
        split_by_game(records, test_fraction=0.2, val_fraction=0.2, seed=0)


def test_duplicate_clip_id_is_rejected() -> None:
    records = [
        ClipRecord(clip_id=1, game_id="g1", pitch_type="slider"),
        ClipRecord(clip_id=1, game_id="g2", pitch_type="slider"),
    ]
    with pytest.raises(ValueError, match="duplicate clip_id"):
        split_by_game(records, test_fraction=0.2, val_fraction=0.2, seed=0)


@pytest.mark.parametrize(
    ("test_fraction", "val_fraction"),
    [(-0.1, 0.2), (1.0, 0.0), (0.6, 0.5)],
)
def test_invalid_fractions_rejected(test_fraction: float, val_fraction: float) -> None:
    records = _records({"g1": 5, "g2": 5})
    with pytest.raises(ValueError):
        split_by_game(records, test_fraction=test_fraction, val_fraction=val_fraction, seed=0)


def test_duplicate_source_checksum_across_splits_is_detected() -> None:
    # Hand-build a split where two clips share a source checksum but sit in
    # different splits — the guard must catch it.
    records = [
        ClipRecord(clip_id=1, game_id="g1", pitch_type="slider", source_checksum="abc"),
        ClipRecord(clip_id=2, game_id="g2", pitch_type="slider", source_checksum="abc"),
    ]
    leaking = GroupedSplit(
        train_clip_ids=(1,),
        val_clip_ids=(),
        test_clip_ids=(2,),
        train_games=frozenset({"g1"}),
        val_games=frozenset(),
        test_games=frozenset({"g2"}),
    )
    with pytest.raises(LeakageError, match="spans splits"):
        assert_no_duplicate_source_across_splits(records, leaking)


def test_shared_source_checksum_within_a_game_is_kept_together() -> None:
    # Two clips from the same source AND same game always share a split, so no
    # leakage is possible; split_by_game must succeed.
    records = [
        ClipRecord(clip_id=1, game_id="g1", pitch_type="slider", source_checksum="abc"),
        ClipRecord(clip_id=2, game_id="g1", pitch_type="four_seam_fastball", source_checksum="abc"),
        ClipRecord(clip_id=3, game_id="g2", pitch_type="slider", source_checksum="def"),
    ]
    split = split_by_game(records, test_fraction=0.34, val_fraction=0.0, seed=3)
    assert_no_duplicate_source_across_splits(records, split)


def test_assert_no_game_overlap_flags_overlap() -> None:
    bad = GroupedSplit(
        train_clip_ids=(1,),
        val_clip_ids=(),
        test_clip_ids=(2,),
        train_games=frozenset({"g1"}),
        val_games=frozenset(),
        test_games=frozenset({"g1"}),
    )
    with pytest.raises(LeakageError, match="both train and test"):
        assert_no_game_overlap(bad)


def test_label_in_feature_name_is_rejected() -> None:
    with pytest.raises(LeakageError, match="target label"):
        assert_label_not_in_feature_names(
            feature_names=["glove_y", "pitch_type_slider"],
            label_values=["slider", "four_seam_fastball"],
        )


def test_safe_feature_names_pass() -> None:
    # Should not raise.
    assert_label_not_in_feature_names(
        feature_names=["glove_wrist_y", "elbow_height", "shoulder_rotation"],
        label_values=["slider", "four_seam_fastball"],
    )
