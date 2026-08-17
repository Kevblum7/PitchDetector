"""Unit tests for the dataset audit (CLAUDE.md §17, §18)."""

from __future__ import annotations

from ml.datasets.audit import AuditThresholds, audit_dataset
from ml.datasets.records import ClipRecord


def _many(game: str, pitch_type: str, n: int, *, start_id: int) -> list[ClipRecord]:
    return [
        ClipRecord(clip_id=start_id + i, game_id=game, pitch_type=pitch_type)
        for i in range(n)
    ]


def test_counts_by_pitch_type_and_game() -> None:
    records = [
        ClipRecord(clip_id=1, game_id="g1", pitch_type="slider", camera_angle="cf"),
        ClipRecord(clip_id=2, game_id="g1", pitch_type="slider", camera_angle="cf"),
        ClipRecord(clip_id=3, game_id="g2", pitch_type="four_seam_fastball", camera_angle="cf"),
    ]
    audit = audit_dataset(records)
    assert audit.total_clips == 3
    assert audit.n_games == 2
    assert audit.counts_by_pitch_type == {"four_seam_fastball": 1, "slider": 2}
    assert audit.counts_by_game == {"g1": 2, "g2": 1}
    assert audit.counts_by_game_and_pitch["g1"] == {"slider": 2}
    assert audit.counts_by_camera_angle == {"cf": 3}


def test_sufficient_dataset_has_no_warnings() -> None:
    thresholds = AuditThresholds(minimum_clips_per_class=5, minimum_games=2)
    records = (
        _many("g1", "slider", 5, start_id=1)
        + _many("g2", "four_seam_fastball", 5, start_id=100)
    )
    audit = audit_dataset(records, thresholds=thresholds)
    assert audit.warnings == []
    assert audit.is_sufficient


def test_empty_dataset_warns() -> None:
    audit = audit_dataset([])
    assert audit.total_clips == 0
    assert "dataset is empty" in audit.warnings
    assert not audit.is_sufficient


def test_too_few_clips_per_class_warns() -> None:
    thresholds = AuditThresholds(minimum_clips_per_class=20, minimum_games=1)
    records = (
        _many("g1", "slider", 3, start_id=1)
        + _many("g2", "four_seam_fastball", 25, start_id=100)
    )
    audit = audit_dataset(records, thresholds=thresholds)
    assert any("slider" in w and "minimum" in w for w in audit.warnings)


def test_single_game_warns_about_held_out_splits() -> None:
    thresholds = AuditThresholds(minimum_clips_per_class=1, minimum_games=2)
    records = (
        _many("g1", "slider", 2, start_id=1)
        + _many("g1", "four_seam_fastball", 2, start_id=100)
    )
    audit = audit_dataset(records, thresholds=thresholds)
    assert audit.n_games == 1
    assert any("held-out game" in w for w in audit.warnings)


def test_duplicate_source_video_is_flagged() -> None:
    thresholds = AuditThresholds(minimum_clips_per_class=1, minimum_games=1)
    records = [
        ClipRecord(clip_id=1, game_id="g1", pitch_type="slider", source_checksum="abc"),
        ClipRecord(clip_id=2, game_id="g2", pitch_type="slider", source_checksum="abc"),
        ClipRecord(clip_id=3, game_id="g3", pitch_type="four_seam_fastball", source_checksum="xyz"),
    ]
    audit = audit_dataset(records, thresholds=thresholds)
    assert audit.duplicate_source_groups == {"abc": [1, 2]}
    assert any("reused across" in w for w in audit.warnings)


def test_class_imbalance_is_flagged() -> None:
    thresholds = AuditThresholds(
        minimum_clips_per_class=1, minimum_games=1, maximum_class_imbalance_ratio=3.0
    )
    records = (
        _many("g1", "slider", 2, start_id=1)
        + _many("g2", "four_seam_fastball", 20, start_id=100)
    )
    audit = audit_dataset(records, thresholds=thresholds)
    assert audit.class_imbalance_ratio == 10.0
    assert any("imbalance" in w for w in audit.warnings)
