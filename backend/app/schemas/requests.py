"""Pydantic request bodies for the v1 API (typed API boundary, CLAUDE.md §19)."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from backend.app.core.enums import (
    BatterSide,
    DeliveryType,
    Handedness,
    LabelSource,
    PitchType,
    QualityStatus,
    ReleaseFrameSource,
)


class PitcherCreate(BaseModel):
    name: str = Field(min_length=1)
    throws: Handedness
    notes: str | None = None


class VideoRegister(BaseModel):
    """Register an existing local video file by path (no upload/copy)."""

    pitcher_id: int
    file_path: str = Field(min_length=1)
    source_name: str | None = None
    game_id: str | None = None
    game_date: date | None = None
    camera_angle: str | None = None


class InitialPitcherBox(BaseModel):
    """Manual initial pitcher box, in original-video pixels."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=8)
    height: int = Field(ge=8)


class ClipCreate(BaseModel):
    release_frame: int = Field(ge=0)
    pitch_type: PitchType
    start_frame: int = Field(default=0, ge=0)
    end_frame: int | None = Field(default=None, ge=0)
    # Defaults to the project-wide guard when omitted.
    release_guard_frames: int | None = Field(default=None, ge=1)
    # Provenance of release_frame; pass "auto" (+ confidence) when the value
    # comes from a release-detection job.
    release_frame_source: ReleaseFrameSource = ReleaseFrameSource.MANUAL
    release_frame_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    pitch_family: str | None = None
    delivery_type: DeliveryType | None = None
    batter_side: BatterSide | None = None
    runners_on: bool | None = None
    camera_angle: str | None = None
    quality_status: QualityStatus = QualityStatus.PENDING
    label_source: LabelSource = LabelSource.MANUAL
    notes: str | None = None
    initial_pitcher_box: InitialPitcherBox | None = None


class ClipUpdate(BaseModel):
    """Partial update. Frame-affecting fields trigger re-validation."""

    release_frame: int | None = Field(default=None, ge=0)
    start_frame: int | None = Field(default=None, ge=0)
    end_frame: int | None = Field(default=None, ge=0)
    release_guard_frames: int | None = Field(default=None, ge=1)
    # When release_frame is changed without an explicit source, it is treated
    # as a manual mark. Set to "auto_confirmed" to confirm an auto label.
    release_frame_source: ReleaseFrameSource | None = None
    release_frame_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    pitch_type: PitchType | None = None
    pitch_family: str | None = None
    delivery_type: DeliveryType | None = None
    batter_side: BatterSide | None = None
    runners_on: bool | None = None
    camera_angle: str | None = None
    quality_status: QualityStatus | None = None
    label_source: LabelSource | None = None
    notes: str | None = None
    initial_pitcher_box: InitialPitcherBox | None = None
