"""Game-grouped train/validation/test splitting with leakage guards.

This module enforces the project's single most important rule (CLAUDE.md §10):

    Split by **game**, never by individual clip, and never let a game or a
    duplicate clip appear in more than one split.

Splitting is done at the game level: every clip from a given game lands in
exactly one split. A deterministic, seeded ordering makes the result
reproducible (CLAUDE.md §19). Before returning, :func:`split_by_game` re-checks
its own output with the same assertions the leakage tests use, so a leaking
split can never be handed back (belt-and-suspenders, matching
``services/clip_frames``).

Everything here is pure Python (stdlib only) so it can be unit-tested without a
database, NumPy, or scikit-learn.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ml.datasets.records import ClipRecord


class LeakageError(AssertionError):
    """Raised when a split would leak information across train/val/test.

    Subclasses :class:`AssertionError` so leakage checks read as invariants and
    so a bug that produces a leaking split fails loudly rather than silently.
    """


@dataclass(frozen=True)
class GroupedSplit:
    """A validated, leak-free assignment of clips to splits.

    Clip ids and game ids are stored as sorted tuples/frozensets so the result
    is deterministic and hashable-friendly for logging and config hashing.
    """

    train_clip_ids: tuple[int, ...]
    val_clip_ids: tuple[int, ...]
    test_clip_ids: tuple[int, ...]
    train_games: frozenset[str]
    val_games: frozenset[str]
    test_games: frozenset[str]

    @property
    def n_train(self) -> int:
        return len(self.train_clip_ids)

    @property
    def n_val(self) -> int:
        return len(self.val_clip_ids)

    @property
    def n_test(self) -> int:
        return len(self.test_clip_ids)


def group_clips_by_game(records: Iterable[ClipRecord]) -> dict[str, list[ClipRecord]]:
    """Group clip records by ``game_id``.

    Raises:
        ValueError: if any record has an empty ``game_id`` — an ungrouped clip
            cannot be assigned to a split without risking leakage, so this is a
            hard error rather than a silent bucket.
    """
    groups: dict[str, list[ClipRecord]] = defaultdict(list)
    missing: list[int] = []
    for record in records:
        if not record.game_id:
            missing.append(record.clip_id)
            continue
        groups[record.game_id].append(record)
    if missing:
        raise ValueError(
            "cannot split clips with no game_id (grouping is by game to prevent "
            f"leakage); offending clip_ids: {sorted(missing)}"
        )
    return dict(groups)


def split_by_game(
    records: Iterable[ClipRecord],
    *,
    test_fraction: float,
    val_fraction: float,
    seed: int,
) -> GroupedSplit:
    """Assign whole games to train/val/test, targeting the given clip fractions.

    Games are shuffled with a seeded RNG and greedily filled into the test split
    until it reaches ``test_fraction`` of all clips, then the validation split
    until ``val_fraction``; every remaining game goes to train. Assigning whole
    games (rather than trimming to hit the fraction exactly) is what keeps games
    from crossing splits, so the realised fractions are approximate.

    Args:
        records: The clips to split. Duplicate ``clip_id`` values are rejected.
        test_fraction: Target share of clips in the test split, in ``[0, 1)``.
        val_fraction: Target share of clips in the validation split, in
            ``[0, 1)``. ``test_fraction + val_fraction`` must be ``< 1``.
        seed: RNG seed for the deterministic game ordering.

    Returns:
        A :class:`GroupedSplit` that has passed all leakage assertions.

    Raises:
        ValueError: on invalid fractions, an empty ``game_id``, or a duplicate
            ``clip_id``.
        LeakageError: if the produced split somehow overlaps (should not happen;
            this is the invariant check).
    """
    _validate_fractions(test_fraction=test_fraction, val_fraction=val_fraction)

    records = list(records)
    _reject_duplicate_clip_ids(records)
    groups = group_clips_by_game(records)

    total_clips = len(records)
    if total_clips == 0:
        return GroupedSplit((), (), (), frozenset(), frozenset(), frozenset())

    # Deterministic ordering: sort games first so the input order can't change
    # the result, then shuffle with the seeded RNG.
    ordered_games = sorted(groups)
    random.Random(seed).shuffle(ordered_games)

    test_target = test_fraction * total_clips
    val_target = val_fraction * total_clips

    test_games: list[str] = []
    val_games: list[str] = []
    train_games: list[str] = []
    test_count = 0
    val_count = 0

    for game in ordered_games:
        n = len(groups[game])
        # Fill test first, then val, then everything else is train. Compare on
        # the pre-add count so a split stops once it has reached its target
        # rather than always overshooting by a whole game.
        if test_count < test_target:
            test_games.append(game)
            test_count += n
        elif val_count < val_target:
            val_games.append(game)
            val_count += n
        else:
            train_games.append(game)

    split = GroupedSplit(
        train_clip_ids=_clip_ids_for(groups, train_games),
        val_clip_ids=_clip_ids_for(groups, val_games),
        test_clip_ids=_clip_ids_for(groups, test_games),
        train_games=frozenset(train_games),
        val_games=frozenset(val_games),
        test_games=frozenset(test_games),
    )

    # Belt-and-suspenders: never return a leaking split.
    assert_no_game_overlap(split)
    assert_no_clip_overlap(split)
    assert_no_duplicate_source_across_splits(records, split)
    return split


# --- leakage assertions (CLAUDE.md §18 leakage tests) ------------------------


def assert_no_game_overlap(split: GroupedSplit) -> None:
    """Raise :class:`LeakageError` if any game appears in more than one split."""
    for a_name, a, b_name, b in (
        ("train", split.train_games, "val", split.val_games),
        ("train", split.train_games, "test", split.test_games),
        ("val", split.val_games, "test", split.test_games),
    ):
        shared = a & b
        if shared:
            raise LeakageError(
                f"game(s) {sorted(shared)} appear in both {a_name} and {b_name} splits"
            )


def assert_no_clip_overlap(split: GroupedSplit) -> None:
    """Raise :class:`LeakageError` if a clip id appears in more than one split."""
    train, val, test = (
        set(split.train_clip_ids),
        set(split.val_clip_ids),
        set(split.test_clip_ids),
    )
    for a_name, a, b_name, b in (
        ("train", train, "val", val),
        ("train", train, "test", test),
        ("val", val, "test", test),
    ):
        shared = a & b
        if shared:
            raise LeakageError(
                f"clip(s) {sorted(shared)} appear in both {a_name} and {b_name} splits"
            )


def assert_no_duplicate_source_across_splits(
    records: Sequence[ClipRecord],
    split: GroupedSplit,
) -> None:
    """Raise if clips sharing a ``source_checksum`` land in different splits.

    Near-duplicate footage (same source video) in two splits leaks information
    even when the games differ (CLAUDE.md §10). Records with no checksum are
    skipped, since duplication cannot be established for them here.
    """
    split_of: dict[int, str] = {}
    for name, ids in (
        ("train", split.train_clip_ids),
        ("val", split.val_clip_ids),
        ("test", split.test_clip_ids),
    ):
        for clip_id in ids:
            split_of[clip_id] = name

    by_checksum: dict[str, set[str]] = defaultdict(set)
    checksum_examples: dict[str, list[int]] = defaultdict(list)
    for record in records:
        if not record.source_checksum:
            continue
        assigned = split_of.get(record.clip_id)
        if assigned is None:
            continue
        by_checksum[record.source_checksum].add(assigned)
        checksum_examples[record.source_checksum].append(record.clip_id)

    for checksum, splits in by_checksum.items():
        if len(splits) > 1:
            raise LeakageError(
                f"source checksum {checksum[:12]}… (clips "
                f"{sorted(checksum_examples[checksum])}) spans splits {sorted(splits)}"
            )


def assert_label_not_in_feature_names(
    feature_names: Iterable[str],
    label_values: Iterable[str],
) -> None:
    """Raise if any target label leaks into a feature-column name (CLAUDE.md §18).

    Guards against the classic mistake of a feature such as ``is_slider`` or a
    column that embeds the pitch type. Matching is case-insensitive and
    substring-based so ``pitch_type_slider`` is caught, not just an exact match.
    """
    labels = [value.lower() for value in label_values if value]
    offending: list[str] = []
    for name in feature_names:
        lowered = name.lower()
        if any(label in lowered for label in labels):
            offending.append(name)
    if offending:
        raise LeakageError(
            f"feature name(s) {sorted(offending)} contain a target label; "
            "the model could read the answer directly"
        )


# --- internals ---------------------------------------------------------------


def _validate_fractions(*, test_fraction: float, val_fraction: float) -> None:
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if test_fraction + val_fraction >= 1.0:
        raise ValueError(
            "test_fraction + val_fraction must be < 1 so the train split is "
            f"non-empty, got {test_fraction} + {val_fraction}"
        )


def _reject_duplicate_clip_ids(records: Sequence[ClipRecord]) -> None:
    seen: set[int] = set()
    dupes: set[int] = set()
    for record in records:
        if record.clip_id in seen:
            dupes.add(record.clip_id)
        seen.add(record.clip_id)
    if dupes:
        raise ValueError(f"duplicate clip_id(s) in records: {sorted(dupes)}")


def _clip_ids_for(groups: Mapping[str, list[ClipRecord]], games: Iterable[str]) -> tuple[int, ...]:
    ids = [record.clip_id for game in games for record in groups[game]]
    return tuple(sorted(ids))
