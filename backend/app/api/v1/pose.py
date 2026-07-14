"""Pose extraction endpoints — CLAUDE.md §16 (``POST /clips/{id}/pose``).

Extraction is CPU-heavy (~seconds per frame with the real estimator), so it
runs as a local background job with persisted queued/running/completed/failed
status. Rendering the overlay from already-stored pose frames is fast and runs
synchronously.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlmodel import Session

from backend.app.core.enums import JobStatus
from backend.app.db.models import PitchClip, PoseFrame, PoseJob, SourceVideo
from backend.app.db.session import get_session, get_session_factory
from backend.app.services.pose_extraction import (
    PoseExtractionError,
    get_pose_estimator,
    latest_pose_job,
    load_pose_frames,
    render_clip_pose_overlay,
    run_pose_job,
)
from ml.pose.estimator import PoseEstimator
from ml.pose.tracker import TrackerConfig

router = APIRouter(prefix="/api/v1/clips", tags=["pose"])


def _get_clip_and_video(clip_id: int, session: Session) -> tuple[PitchClip, SourceVideo]:
    clip = session.get(PitchClip, clip_id)
    if clip is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"clip {clip_id} not found")
    video = session.get(SourceVideo, clip.source_video_id)
    if video is None:  # pragma: no cover - orphaned clip should not occur
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"source video {clip.source_video_id} for clip {clip_id} is missing",
        )
    return clip, video


@router.post("/{clip_id}/pose", response_model=PoseJob, status_code=status.HTTP_202_ACCEPTED)
def start_pose_extraction(
    clip_id: int,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
    session_factory: Callable[[], Session] = Depends(get_session_factory),
    estimator: PoseEstimator = Depends(get_pose_estimator),
) -> PoseJob:
    clip, _video = _get_clip_and_video(clip_id, session)

    if not clip.has_initial_box:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"clip {clip_id} has no initial pitcher box; set initial_pitcher_box via "
            f"PATCH /api/v1/clips/{clip_id} first",
        )

    active = latest_pose_job(session, clip_id)
    if active is not None and active.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"pose job {active.id} for clip {clip_id} is already {active.status}",
        )

    config = TrackerConfig()
    job = PoseJob(
        clip_id=clip_id,
        estimator_name=estimator.name,
        config=config.as_dict(),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    assert job.id is not None
    background.add_task(run_pose_job, job.id, session_factory, estimator, config)
    return job


@router.get("/{clip_id}/pose", response_model=PoseJob)
def get_pose_status(clip_id: int, session: Session = Depends(get_session)) -> PoseJob:
    _get_clip_and_video(clip_id, session)
    job = latest_pose_job(session, clip_id)
    if job is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"no pose extraction has been started for clip {clip_id}"
        )
    return job


@router.get("/{clip_id}/pose/frames", response_model=list[PoseFrame])
def get_pose_frames(clip_id: int, session: Session = Depends(get_session)) -> list[PoseFrame]:
    _get_clip_and_video(clip_id, session)
    frames = load_pose_frames(session, clip_id)
    if not frames:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"clip {clip_id} has no stored pose frames; run pose extraction first",
        )
    return frames


@router.post("/{clip_id}/pose/render")
def render_pose(clip_id: int, session: Session = Depends(get_session)) -> dict[str, str]:
    clip, video = _get_clip_and_video(clip_id, session)
    try:
        output_path = render_clip_pose_overlay(session, clip, video)
    except PoseExtractionError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return {"output_path": str(output_path)}
