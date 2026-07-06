"""Frame-window math and leakage guards for pitch clips.

This module is the single source of truth for the rule from CLAUDE.md §10:
**no frame at or after ball release may enter the model window.**

Frame semantics for a clip (all integer, 0-based, in original-video frames):

- ``start_frame``           first frame of the clip.
- ``release_frame``         the marked/estimated ball-release frame.
- ``release_guard_frames``  safety margin dropped before release (>= 1).
- ``pre_release_end_frame`` = ``release_frame - release_guard_frames`` — the
  **last** frame allowed into the model window.
- ``end_frame``             last frame of the clip; must not extend past the
  pre-release cutoff, so the stored clip is leak-free by construction.

The model window is the inclusive range ``[start_frame, pre_release_end_frame]``.

All functions here are pure so they can be unit-tested without a database.
"""

from __future__ import annotations

from dataclasses import dataclass


class ClipFrameError(ValueError):
    """Raised when a requested clip window is invalid or would leak."""


@dataclass(frozen=True)
class ClipWindow:
    """A validated, leak-free clip window."""

    start_frame: int
    end_frame: int
    release_frame: int
    release_guard_frames: int
    pre_release_end_frame: int


def compute_pre_release_end_frame(release_frame: int, release_guard_frames: int) -> int:
    """Return the last frame allowed into the model window."""
    return release_frame - release_guard_frames


def build_clip_window(
    *,
    start_frame: int,
    release_frame: int,
    release_guard_frames: int,
    video_frame_count: int,
    end_frame: int | None = None,
) -> ClipWindow:
    """Validate frame indices and return a leak-free :class:`ClipWindow`.

    ``end_frame`` defaults to the pre-release cutoff (a clip that ends exactly
    at the last model-safe frame).

    Raises:
        ClipFrameError: if any index is out of range or the window would
            include the release frame or later.
    """
    if video_frame_count <= 0:
        raise ClipFrameError(f"video_frame_count must be positive, got {video_frame_count}")
    if release_guard_frames < 1:
        raise ClipFrameError(
            "release_guard_frames must be >= 1 so the release frame stays out of "
            f"the model window, got {release_guard_frames}"
        )
    if start_frame < 0:
        raise ClipFrameError(f"start_frame must be >= 0, got {start_frame}")
    if release_frame < 0 or release_frame >= video_frame_count:
        raise ClipFrameError(
            f"release_frame {release_frame} must be within the video [0, {video_frame_count - 1}]"
        )

    pre_release_end_frame = compute_pre_release_end_frame(release_frame, release_guard_frames)

    if pre_release_end_frame < start_frame:
        raise ClipFrameError(
            f"clip has no pre-release frames: start_frame={start_frame} but "
            f"pre_release_end_frame={pre_release_end_frame} "
            f"(release_frame={release_frame}, guard={release_guard_frames})"
        )

    if end_frame is None:
        end_frame = pre_release_end_frame

    if end_frame < start_frame:
        raise ClipFrameError(f"end_frame {end_frame} must be >= start_frame {start_frame}")
    if end_frame > pre_release_end_frame:
        raise ClipFrameError(
            f"end_frame {end_frame} extends past the pre-release cutoff "
            f"{pre_release_end_frame}; the clip would leak release or post-release frames"
        )

    window = ClipWindow(
        start_frame=start_frame,
        end_frame=end_frame,
        release_frame=release_frame,
        release_guard_frames=release_guard_frames,
        pre_release_end_frame=pre_release_end_frame,
    )
    # Belt-and-suspenders: never hand back a leaking window.
    assert_no_post_release_leak(window)
    return window


def assert_no_post_release_leak(window: ClipWindow) -> None:
    """Raise if any part of the model window reaches the release frame.

    This is the invariant leakage tests assert against.
    """
    if window.pre_release_end_frame >= window.release_frame:
        raise ClipFrameError(
            f"pre_release_end_frame {window.pre_release_end_frame} reaches or passes "
            f"release_frame {window.release_frame}"
        )
    if window.end_frame >= window.release_frame:
        raise ClipFrameError(
            f"end_frame {window.end_frame} reaches or passes release_frame {window.release_frame}"
        )
    if window.end_frame > window.pre_release_end_frame:
        raise ClipFrameError(
            f"end_frame {window.end_frame} passes pre_release_end_frame "
            f"{window.pre_release_end_frame}"
        )
