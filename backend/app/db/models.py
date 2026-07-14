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

from sqlmodel import JSON, Column, Field, SQLModel, UniqueConstraint

from backend.app.core.enums import (
    BatterSide,
    DeliveryType,
    Handedness,
    JobStatus,
    LabelSource,
    PitchType,
    QualityStatus,
    ReleaseFrameSource,
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
    # Release-frame provenance (CLAUDE.md §9): who set it, and — for the
    # automatic detector — how sure it was. Auto labels never silently
    # overwrite manual ones (enforced in services/release_detection).
    release_frame_source: ReleaseFrameSource = ReleaseFrameSource.MANUAL
    release_frame_confidence: float | None = None

    pitch_type: PitchType
    pitch_family: str | None = None
    delivery_type: DeliveryType | None = None
    batter_side: BatterSide | None = None
    runners_on: bool | None = None
    camera_angle: str | None = None
    quality_status: QualityStatus = QualityStatus.PENDING
    label_source: LabelSource = LabelSource.MANUAL
    notes: str | None = None

    # Manual initial pitcher box (original-video pixels), required before pose
    # extraction can run (CLAUDE.md §9: "let the user define an initial box").
    initial_box_x: int | None = None
    initial_box_y: int | None = None
    initial_box_width: int | None = None
    initial_box_height: int | None = None

    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def has_initial_box(self) -> bool:
        return None not in (
            self.initial_box_x,
            self.initial_box_y,
            self.initial_box_width,
            self.initial_box_height,
        )


class PoseFrame(SQLModel, table=True):
    """Pose for one frame of a clip's pre-release window (CLAUDE.md §8).

    All coordinates are original-video pixels; timestamps are release-relative
    (negative before release). ``pitcher_bbox`` is the smoothed tracked box;
    ``normalization_transform`` records the search-region crop the estimator
    actually saw, so inference can be reproduced exactly.
    """

    __tablename__ = "pose_frame"
    __table_args__ = (UniqueConstraint("clip_id", "frame_index", name="uq_pose_clip_frame"),)

    id: int | None = Field(default=None, primary_key=True)
    clip_id: int = Field(foreign_key="pitch_clip.id", index=True)
    frame_index: int
    timestamp_ms_relative_to_release: float
    detected: bool
    detection_score: float | None = None
    # 17 COCO keypoints as [[x, y], ...] or null when the frame had no detection.
    keypoints: list[list[float]] | None = Field(default=None, sa_column=Column(JSON))
    visibility_scores: list[float] | None = Field(default=None, sa_column=Column(JSON))
    pitcher_bbox: dict[str, float] = Field(sa_column=Column(JSON))
    normalization_transform: dict[str, float] = Field(sa_column=Column(JSON))


class ReleaseDetectionJob(SQLModel, table=True):
    """A background automatic release-frame detection run for one video.

    ``result`` holds the full :class:`ml.release.detector.ReleaseDetection`
    payload (candidate frame, confidence, motion window, detector version) so
    every auto label can be traced to the exact run that produced it.
    """

    __tablename__ = "release_detection_job"

    id: int | None = Field(default=None, primary_key=True)
    source_video_id: int = Field(foreign_key="source_video.id", index=True)
    status: JobStatus = JobStatus.QUEUED
    estimator_name: str
    config: dict[str, float | int] = Field(sa_column=Column(JSON))
    result: dict[str, object] | None = Field(default=None, sa_column=Column(JSON))
    error_message: str | None = None
    # Full traceback, persisted in development mode only.
    error_stack: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class PoseJob(SQLModel, table=True):
    """A local background pose-extraction run for one clip (CLAUDE.md §16).

    Stores the full tracker/estimator configuration and quality summary so any
    pose artifact can be traced to the exact settings that produced it.
    """

    __tablename__ = "pose_job"

    id: int | None = Field(default=None, primary_key=True)
    clip_id: int = Field(foreign_key="pitch_clip.id", index=True)
    status: JobStatus = JobStatus.QUEUED
    estimator_name: str
    config: dict[str, float | int] = Field(sa_column=Column(JSON))
    quality: dict[str, object] | None = Field(default=None, sa_column=Column(JSON))
    frames_written: int | None = None
    error_message: str | None = None
    # Full traceback, persisted in development mode only.
    error_stack: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
