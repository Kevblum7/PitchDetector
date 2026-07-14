"""Pose-extraction job execution and persistence.

The heavy lifting lives in ``ml.pose``; this service loads the clip + video,
runs the pipeline with an injected estimator, replaces the clip's stored
``PoseFrame`` rows, and keeps the ``PoseJob`` record honest through the
queued → running → completed/failed lifecycle (CLAUDE.md §16).
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session, col, delete, select

from backend.app.core.config import ENVIRONMENT, overlays_dir
from backend.app.core.enums import JobStatus
from backend.app.db.models import PitchClip, PoseFrame, PoseJob, SourceVideo
from ml.pose.estimator import PoseEstimator
from ml.pose.geometry import BoundingBox
from ml.pose.pipeline import (
    PoseExtractionResult,
    PoseFrameResult,
    extract_clip_pose,
)
from ml.pose.tracker import TrackerConfig
from ml.pose.visualize import render_pose_overlay

logger = logging.getLogger(__name__)


class PoseExtractionError(RuntimeError):
    """Raised when a clip is not ready for pose extraction."""


def clip_initial_box(clip: PitchClip) -> BoundingBox:
    """The clip's manual initial pitcher box, or a clear error if unset."""
    if not clip.has_initial_box:
        raise PoseExtractionError(
            f"clip {clip.id} has no initial pitcher box; set one via "
            "PATCH /api/v1/clips/{id} (initial_pitcher_box) before extracting pose"
        )
    assert clip.initial_box_x is not None  # narrowed by has_initial_box
    assert clip.initial_box_y is not None
    assert clip.initial_box_width is not None
    assert clip.initial_box_height is not None
    return BoundingBox(
        x=float(clip.initial_box_x),
        y=float(clip.initial_box_y),
        width=float(clip.initial_box_width),
        height=float(clip.initial_box_height),
    )


def extract_and_store_pose(
    session: Session,
    clip: PitchClip,
    video: SourceVideo,
    estimator: PoseEstimator,
    config: TrackerConfig | None = None,
) -> PoseExtractionResult:
    """Run the pose pipeline for a clip and replace its stored pose frames."""
    video_path = Path(video.file_path)
    if not video_path.is_file():
        raise PoseExtractionError(f"source video file is missing: {video_path}")

    result = extract_clip_pose(
        video_path,
        start_frame=clip.start_frame,
        release_frame=clip.release_frame,
        pre_release_end_frame=clip.pre_release_end_frame,
        fps=video.fps,
        frame_width=video.width,
        frame_height=video.height,
        initial_box=clip_initial_box(clip),
        estimator=estimator,
        config=config,
    )

    assert clip.id is not None
    # Replace any previous extraction for this clip atomically with the insert.
    session.exec(delete(PoseFrame).where(col(PoseFrame.clip_id) == clip.id))  # type: ignore[call-overload]
    for frame in result.frames:
        session.add(
            PoseFrame(
                clip_id=clip.id,
                frame_index=frame.frame_index,
                timestamp_ms_relative_to_release=frame.timestamp_ms_relative_to_release,
                detected=frame.detected,
                detection_score=frame.detection_score,
                keypoints=(
                    [[x, y] for x, y in frame.keypoints] if frame.keypoints is not None else None
                ),
                visibility_scores=frame.keypoint_scores,
                pitcher_bbox=frame.box.as_dict(),
                normalization_transform=frame.search_region.as_dict(),
            )
        )
    session.commit()
    return result


def run_pose_job(
    job_id: int,
    session_factory: Callable[[], Session],
    estimator: PoseEstimator,
    config: TrackerConfig | None = None,
) -> None:
    """Execute a queued pose job, recording status transitions and errors.

    Never raises: failures are persisted on the job (readable message always,
    stack trace in development mode) so the API can report them.
    """
    with session_factory() as session:
        job = session.get(PoseJob, job_id)
        if job is None:  # pragma: no cover - job created just before dispatch
            logger.error("pose job %s vanished before execution", job_id)
            return
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        session.add(job)
        session.commit()

        try:
            clip = session.get(PitchClip, job.clip_id)
            if clip is None:
                raise PoseExtractionError(f"clip {job.clip_id} not found")
            video = session.get(SourceVideo, clip.source_video_id)
            if video is None:
                raise PoseExtractionError(f"source video {clip.source_video_id} not found")

            result = extract_and_store_pose(session, clip, video, estimator, config)

            job.status = JobStatus.COMPLETED
            job.quality = result.quality.as_dict()
            job.frames_written = len(result.frames)
        except Exception as exc:
            logger.exception("pose job %s failed", job_id)
            job.status = JobStatus.FAILED
            job.error_message = str(exc)
            if ENVIRONMENT == "development":
                job.error_stack = traceback.format_exc()
        finally:
            job.finished_at = datetime.now(UTC)
            session.add(job)
            session.commit()


def load_pose_frames(session: Session, clip_id: int) -> list[PoseFrame]:
    """Stored pose frames for a clip, in frame order."""
    statement = (
        select(PoseFrame).where(PoseFrame.clip_id == clip_id).order_by(col(PoseFrame.frame_index))
    )
    return list(session.exec(statement).all())


def latest_pose_job(session: Session, clip_id: int) -> PoseJob | None:
    statement = (
        select(PoseJob).where(PoseJob.clip_id == clip_id).order_by(col(PoseJob.id).desc()).limit(1)
    )
    return session.exec(statement).first()


def render_clip_pose_overlay(
    session: Session,
    clip: PitchClip,
    video: SourceVideo,
    *,
    min_keypoint_score: float | None = None,
) -> Path:
    """Render the stored pose for a clip to an annotated MP4 and return its path."""
    assert clip.id is not None
    rows = load_pose_frames(session, clip.id)
    if not rows:
        raise PoseExtractionError(
            f"clip {clip.id} has no stored pose frames; run pose extraction first"
        )
    video_path = Path(video.file_path)
    if not video_path.is_file():
        raise PoseExtractionError(f"source video file is missing: {video_path}")

    frames = [
        PoseFrameResult(
            frame_index=row.frame_index,
            timestamp_ms_relative_to_release=row.timestamp_ms_relative_to_release,
            box=BoundingBox.from_dict(row.pitcher_bbox),
            search_region=BoundingBox.from_dict(row.normalization_transform),
            detected=row.detected,
            detection_score=row.detection_score,
            keypoints=([(x, y) for x, y in row.keypoints] if row.keypoints is not None else None),
            keypoint_scores=row.visibility_scores,
        )
        for row in rows
    ]
    output_path = overlays_dir() / f"clip_{clip.id}_pose.mp4"
    return render_pose_overlay(
        video_path,
        frames,
        output_path,
        fps=video.fps,
        width=video.width,
        height=video.height,
        min_keypoint_score=min_keypoint_score,
    )


# --- Estimator provider (FastAPI dependency; tests override it) ---------------

_estimator_singleton: PoseEstimator | None = None


def get_pose_estimator() -> PoseEstimator:
    """Process-wide estimator instance (model weights load lazily, once)."""
    global _estimator_singleton
    if _estimator_singleton is None:
        from ml.pose.estimator import TorchvisionPoseEstimator

        _estimator_singleton = TorchvisionPoseEstimator()
    return _estimator_singleton
