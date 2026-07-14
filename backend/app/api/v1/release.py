"""Automatic release-frame detection endpoints (CLAUDE.md §9, Milestone 3.5).

Detection is CPU-heavy (motion pass + ~20-30 pose inferences), so it runs as a
persisted background job per video, mirroring the pose-job pattern. The review
queue lists auto-labeled clips whose confidence is below the review threshold.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlmodel import Session

from backend.app.core.enums import JobStatus
from backend.app.db.models import PitchClip, ReleaseDetectionJob, SourceVideo
from backend.app.db.session import get_session, get_session_factory
from backend.app.services.pose_extraction import get_pose_estimator
from backend.app.services.release_detection import (
    clips_needing_release_review,
    latest_release_job,
    run_release_detection_job,
)
from ml.pose.estimator import PoseEstimator
from ml.release.detector import ReleaseDetectorConfig

router = APIRouter(prefix="/api/v1", tags=["release-detection"])


def _get_video(video_id: int, session: Session) -> SourceVideo:
    video = session.get(SourceVideo, video_id)
    if video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"video {video_id} not found")
    return video


@router.post(
    "/videos/{video_id}/release-detection",
    response_model=ReleaseDetectionJob,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_release_detection(
    video_id: int,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
    session_factory: Callable[[], Session] = Depends(get_session_factory),
    estimator: PoseEstimator = Depends(get_pose_estimator),
) -> ReleaseDetectionJob:
    _get_video(video_id, session)

    active = latest_release_job(session, video_id)
    if active is not None and active.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"release detection job {active.id} for video {video_id} is already {active.status}",
        )

    config = ReleaseDetectorConfig()
    job = ReleaseDetectionJob(
        source_video_id=video_id,
        estimator_name=estimator.name,
        config=config.as_dict(),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    assert job.id is not None
    background.add_task(run_release_detection_job, job.id, session_factory, estimator, config)
    return job


@router.get("/videos/{video_id}/release-detection", response_model=ReleaseDetectionJob)
def get_release_detection(
    video_id: int, session: Session = Depends(get_session)
) -> ReleaseDetectionJob:
    _get_video(video_id, session)
    job = latest_release_job(session, video_id)
    if job is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no release detection has been started for video {video_id}",
        )
    return job


@router.get("/release-review", response_model=list[PitchClip])
def get_release_review_queue(session: Session = Depends(get_session)) -> list[PitchClip]:
    """Auto-labeled clips whose confidence is below the review threshold."""
    return clips_needing_release_review(session)
