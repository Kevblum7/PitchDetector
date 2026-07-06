"""Source-video registration and clip creation — CLAUDE.md §16.

Videos are registered by local path (no upload/copy) so the original source
file is preserved in place. Duplicate content (same checksum) is detected and
the existing record is returned instead of creating a second row.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlmodel import Session, select

from backend.app.core.config import DEFAULT_RELEASE_GUARD_FRAMES
from backend.app.db.models import PitchClip, Pitcher, SourceVideo
from backend.app.db.session import get_session
from backend.app.schemas.requests import ClipCreate, VideoRegister
from backend.app.services.checksum import sha256_file
from backend.app.services.clip_frames import ClipFrameError, build_clip_window
from backend.app.services.video_files import VideoFileError, validate_video_path
from backend.app.services.video_metadata import (
    FFprobeNotFoundError,
    VideoProbeError,
    probe_video,
)

router = APIRouter(prefix="/api/v1/videos", tags=["videos"])


@router.post("", response_model=SourceVideo)
def register_video(
    body: VideoRegister,
    response: Response,
    session: Session = Depends(get_session),
) -> SourceVideo:
    pitcher = session.get(Pitcher, body.pitcher_id)
    if pitcher is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"pitcher {body.pitcher_id} not found")

    try:
        path = validate_video_path(body.file_path)
    except VideoFileError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    checksum = sha256_file(path)
    existing = session.exec(select(SourceVideo).where(SourceVideo.checksum == checksum)).first()
    if existing is not None:
        # Idempotent registration of identical content.
        response.status_code = status.HTTP_200_OK
        return existing

    try:
        meta = probe_video(path)
    except FFprobeNotFoundError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except VideoProbeError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    video = SourceVideo(
        pitcher_id=body.pitcher_id,
        file_path=str(path),
        source_name=body.source_name,
        game_id=body.game_id,
        game_date=body.game_date,
        camera_angle=body.camera_angle,
        fps=meta.fps,
        width=meta.width,
        height=meta.height,
        duration_seconds=meta.duration_seconds,
        frame_count=meta.frame_count,
        checksum=checksum,
    )
    session.add(video)
    session.commit()
    session.refresh(video)
    response.status_code = status.HTTP_201_CREATED
    return video


@router.get("/{video_id}", response_model=SourceVideo)
def get_video(video_id: int, session: Session = Depends(get_session)) -> SourceVideo:
    video = session.get(SourceVideo, video_id)
    if video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"video {video_id} not found")
    return video


@router.get("/{video_id}/clips", response_model=list[PitchClip])
def list_video_clips(video_id: int, session: Session = Depends(get_session)) -> list[PitchClip]:
    video = session.get(SourceVideo, video_id)
    if video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"video {video_id} not found")
    return list(session.exec(select(PitchClip).where(PitchClip.source_video_id == video_id)).all())


@router.post(
    "/{video_id}/clips",
    response_model=PitchClip,
    status_code=status.HTTP_201_CREATED,
)
def create_clip(
    video_id: int,
    body: ClipCreate,
    session: Session = Depends(get_session),
) -> PitchClip:
    video = session.get(SourceVideo, video_id)
    if video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"video {video_id} not found")

    guard = (
        body.release_guard_frames
        if body.release_guard_frames is not None
        else DEFAULT_RELEASE_GUARD_FRAMES
    )
    try:
        window = build_clip_window(
            start_frame=body.start_frame,
            release_frame=body.release_frame,
            release_guard_frames=guard,
            video_frame_count=video.frame_count,
            end_frame=body.end_frame,
        )
    except ClipFrameError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    clip = PitchClip(
        source_video_id=video_id,
        pitcher_id=video.pitcher_id,
        game_id=video.game_id,
        start_frame=window.start_frame,
        end_frame=window.end_frame,
        release_frame=window.release_frame,
        pre_release_end_frame=window.pre_release_end_frame,
        release_guard_frames=window.release_guard_frames,
        pitch_type=body.pitch_type,
        pitch_family=body.pitch_family,
        delivery_type=body.delivery_type,
        batter_side=body.batter_side,
        runners_on=body.runners_on,
        camera_angle=body.camera_angle if body.camera_angle is not None else video.camera_angle,
        quality_status=body.quality_status,
        label_source=body.label_source,
        notes=body.notes,
    )
    session.add(clip)
    session.commit()
    session.refresh(clip)
    return clip
