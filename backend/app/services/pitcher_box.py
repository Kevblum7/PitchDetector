"""Validation for the manual initial pitcher box."""

from __future__ import annotations

from backend.app.db.models import PitchClip, SourceVideo
from backend.app.schemas.requests import InitialPitcherBox


class PitcherBoxError(ValueError):
    """Raised when an initial pitcher box does not fit inside the video frame."""


def apply_initial_box(clip: PitchClip, box: InitialPitcherBox, video: SourceVideo) -> None:
    """Validate the box against the video geometry and store it on the clip."""
    if box.x + box.width > video.width or box.y + box.height > video.height:
        raise PitcherBoxError(
            f"initial pitcher box {box.x},{box.y} {box.width}x{box.height} extends beyond "
            f"the {video.width}x{video.height} video frame"
        )
    clip.initial_box_x = box.x
    clip.initial_box_y = box.y
    clip.initial_box_width = box.width
    clip.initial_box_height = box.height
