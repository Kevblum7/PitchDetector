"""SQLModel table definitions (CLAUDE.md §8).

All time-dependent clip fields are stored in original-video frames and are
release-relative by construction (see ``services/clip_frames``).

Two fields extend the minimal spec for reproducibility:
- ``SourceVideo.frame_count`` — cached total frames, used to validate clip
  windows without re-probing the file.
- ``PitchClip.release_guard_frames`` — the guard actually applied, so a clip's
  leak-free window can be reconstructed exactly.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlmodel import Field, SQLModel

from backend.app.core.enums import (
    BatterSide,
    DeliveryType,
    Handedness,
    LabelSource,
    PitchType,
    QualityStatus,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Pitcher(SQLModel, table=True):
    __tablename__ = "pitcher"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    throws: Handedness
    notes: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class SourceVideo(SQLModel, table=True):
    __tablename__ = "source_video"

    id: int | None = Field(default=None, primary_key=True)
    pitcher_id: int = Field(foreign_key="pitcher.id", index=True)
    file_path: str
    source_name: str | None = None
    game_id: str | None = Field(default=None, index=True)
    game_date: date | None = None
    camera_angle: str | None = None
    fps: float
    width: int
    height: int
    duration_seconds: float
    frame_count: int
    checksum: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=_utcnow)


class PitchClip(SQLModel, table=True):
    __tablename__ = "pitch_clip"

    id: int | None = Field(default=None, primary_key=True)
    source_video_id: int = Field(foreign_key="source_video.id", index=True)
    pitcher_id: int = Field(foreign_key="pitcher.id", index=True)
    game_id: str | None = Field(default=None, index=True)

    start_frame: int
    end_frame: int
    release_frame: int
    pre_release_end_frame: int
    release_guard_frames: int

    pitch_type: PitchType
    pitch_family: str | None = None
    delivery_type: DeliveryType | None = None
    batter_side: BatterSide | None = None
    runners_on: bool | None = None
    camera_angle: str | None = None
    quality_status: QualityStatus = QualityStatus.PENDING
    label_source: LabelSource = LabelSource.MANUAL
    notes: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
