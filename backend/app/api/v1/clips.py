"""Clip read/update endpoints — CLAUDE.md §16.

Any change to a frame-defining field (start, release, guard, end) re-runs the
leakage-safe window builder, so a clip can never be patched into a state that
includes the release frame or later.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from backend.app.db.models import PitchClip, SourceVideo
from backend.app.db.session import get_session
from backend.app.schemas.requests import ClipUpdate
from backend.app.services.clip_frames import ClipFrameError, build_clip_window

router = APIRouter(prefix="/api/v1/clips", tags=["clips"])

_FRAME_FIELDS = {"release_frame", "start_frame", "end_frame", "release_guard_frames"}


@router.get("/{clip_id}", response_model=PitchClip)
def get_clip(clip_id: int, session: Session = Depends(get_session)) -> PitchClip:
    clip = session.get(PitchClip, clip_id)
    if clip is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"clip {clip_id} not found")
    return clip


@router.patch("/{clip_id}", response_model=PitchClip)
def update_clip(
    clip_id: int,
    body: ClipUpdate,
    session: Session = Depends(get_session),
) -> PitchClip:
    clip = session.get(PitchClip, clip_id)
    if clip is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"clip {clip_id} not found")

    updates = body.model_dump(exclude_unset=True)

    if _FRAME_FIELDS & updates.keys():
        video = session.get(SourceVideo, clip.source_video_id)
        if video is None:  # pragma: no cover - orphaned clip should not occur
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"source video {clip.source_video_id} for clip {clip_id} is missing",
            )
        start_frame = updates.get("start_frame", clip.start_frame)
        release_frame = updates.get("release_frame", clip.release_frame)
        guard = updates.get("release_guard_frames", clip.release_guard_frames)
        # end_frame: honour an explicit new value; otherwise let the builder
        # recompute the default cutoff rather than reuse a now-stale end_frame.
        end_frame = updates.get("end_frame") if "end_frame" in updates else None
        try:
            window = build_clip_window(
                start_frame=start_frame,
                release_frame=release_frame,
                release_guard_frames=guard,
                video_frame_count=video.frame_count,
                end_frame=end_frame,
            )
        except ClipFrameError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

        clip.start_frame = window.start_frame
        clip.end_frame = window.end_frame
        clip.release_frame = window.release_frame
        clip.release_guard_frames = window.release_guard_frames
        clip.pre_release_end_frame = window.pre_release_end_frame

    for field in updates.keys() - _FRAME_FIELDS:
        setattr(clip, field, updates[field])

    session.add(clip)
    session.commit()
    session.refresh(clip)
    return clip
