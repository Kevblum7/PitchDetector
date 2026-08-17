"""Dataset audit: counts, class balance, duplicates, and sufficiency checks.

Backs the "Dataset audit" screen (CLAUDE.md §17) and the pre-flight a run must
pass before it is worth training (CLAUDE.md §14). The audit is descriptive and
conservative: it reports counts and raises *warnings*, it does not silently drop
data or make a training decision on its own.

Pure Python (stdlib only); no database or third-party dependency.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ml.datasets.records import ClipRecord


@dataclass(frozen=True)
class AuditThresholds:
    """Sufficiency thresholds (CLAUDE.md §14 starting points, not constants).

    These are the values a dataset should clear before a tell claim is
    meaningful. They are configurable so an experiment can record exactly which
    thresholds it used.
    """

    minimum_clips_per_class: int = 20
    minimum_games: int = 2
    # Max allowed ratio of the largest class to the smallest, above which the
    # dataset is flagged as imbalanced (majority-class baseline gets strong).
    maximum_class_imbalance_ratio: float = 3.0


@dataclass(frozen=True)
class DatasetAudit:
    """Descriptive summary of a labelled clip set."""

    total_clips: int
    n_games: int
    counts_by_pitch_type: dict[str, int]
    counts_by_game: dict[str, int]
    counts_by_game_and_pitch: dict[str, dict[str, int]]
    counts_by_camera_angle: dict[str, int]
    duplicate_source_groups: dict[str, list[int]]
    class_imbalance_ratio: float | None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_sufficient(self) -> bool:
        """True when the audit produced no warnings."""
        return not self.warnings


def audit_dataset(
    records: Iterable[ClipRecord],
    *,
    thresholds: AuditThresholds | None = None,
) -> DatasetAudit:
    """Compute counts and sufficiency warnings for a set of clips."""
    thresholds = thresholds or AuditThresholds()
    records = list(records)

    counts_by_pitch_type = _count(r.pitch_type for r in records)
    counts_by_game = _count(r.game_id for r in records if r.game_id)
    counts_by_camera_angle = _count(
        r.camera_angle for r in records if r.camera_angle is not None
    )
    counts_by_game_and_pitch = _count_game_and_pitch(records)
    duplicate_source_groups = _duplicate_source_groups(records)
    imbalance = _class_imbalance_ratio(counts_by_pitch_type)

    warnings = _build_warnings(
        records=records,
        counts_by_pitch_type=counts_by_pitch_type,
        counts_by_game=counts_by_game,
        duplicate_source_groups=duplicate_source_groups,
        imbalance=imbalance,
        thresholds=thresholds,
    )

    return DatasetAudit(
        total_clips=len(records),
        n_games=len(counts_by_game),
        counts_by_pitch_type=counts_by_pitch_type,
        counts_by_game=counts_by_game,
        counts_by_game_and_pitch=counts_by_game_and_pitch,
        counts_by_camera_angle=counts_by_camera_angle,
        duplicate_source_groups=duplicate_source_groups,
        class_imbalance_ratio=imbalance,
        warnings=warnings,
    )


# --- internals ---------------------------------------------------------------


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    # Sorted for deterministic output.
    return dict(sorted(counts.items()))


def _count_game_and_pitch(records: Sequence[ClipRecord]) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        if not record.game_id:
            continue
        nested[record.game_id][record.pitch_type] += 1
    return {game: dict(sorted(pitches.items())) for game, pitches in sorted(nested.items())}


def _duplicate_source_groups(records: Sequence[ClipRecord]) -> dict[str, list[int]]:
    """Return checksum -> sorted clip_ids for any source shared by >1 clip."""
    by_checksum: dict[str, list[int]] = defaultdict(list)
    for record in records:
        if record.source_checksum:
            by_checksum[record.source_checksum].append(record.clip_id)
    return {
        checksum: sorted(ids)
        for checksum, ids in sorted(by_checksum.items())
        if len(ids) > 1
    }


def _class_imbalance_ratio(counts_by_pitch_type: dict[str, int]) -> float | None:
    if not counts_by_pitch_type:
        return None
    values = counts_by_pitch_type.values()
    smallest = min(values)
    largest = max(values)
    if smallest == 0:
        return None
    return largest / smallest


def _build_warnings(
    *,
    records: Sequence[ClipRecord],
    counts_by_pitch_type: dict[str, int],
    counts_by_game: dict[str, int],
    duplicate_source_groups: dict[str, list[int]],
    imbalance: float | None,
    thresholds: AuditThresholds,
) -> list[str]:
    warnings: list[str] = []

    if not records:
        warnings.append("dataset is empty")
        return warnings

    for pitch_type, count in counts_by_pitch_type.items():
        if count < thresholds.minimum_clips_per_class:
            warnings.append(
                f"pitch type '{pitch_type}' has {count} clips "
                f"(< minimum {thresholds.minimum_clips_per_class})"
            )

    if len(counts_by_pitch_type) < 2:
        warnings.append("fewer than 2 pitch types present; nothing to compare")

    if len(counts_by_game) < thresholds.minimum_games:
        warnings.append(
            f"only {len(counts_by_game)} game(s) present "
            f"(< minimum {thresholds.minimum_games}); cannot form held-out game splits"
        )

    missing_game = [r.clip_id for r in records if not r.game_id]
    if missing_game:
        warnings.append(f"{len(missing_game)} clip(s) missing game_id: {sorted(missing_game)}")

    if duplicate_source_groups:
        n_dupe_clips = sum(len(ids) for ids in duplicate_source_groups.values())
        warnings.append(
            f"{len(duplicate_source_groups)} source video(s) reused across "
            f"{n_dupe_clips} clips; keep duplicates within a single split"
        )

    if imbalance is not None and imbalance > thresholds.maximum_class_imbalance_ratio:
        warnings.append(
            f"class imbalance ratio {imbalance:.2f} exceeds "
            f"{thresholds.maximum_class_imbalance_ratio:.2f}; the majority-class "
            "baseline will be strong"
        )

    return warnings
