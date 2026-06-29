"""Unit tests for leakage-safe clip frame windowing."""

from __future__ import annotations

import pytest

from backend.app.services.clip_frames import (
    ClipFrameError,
    ClipWindow,
    assert_no_post_release_leak,
    build_clip_window,
    compute_pre_release_end_frame,
)


def test_compute_pre_release_end_frame() -> None:
    assert compute_pre_release_end_frame(release_frame=30, release_guard_frames=3) == 27


def test_build_window_defaults_end_to_cutoff() -> None:
    window = build_clip_window(
        start_frame=0,
        release_frame=30,
        release_guard_frames=3,
        video_frame_count=100,
    )
    assert window.pre_release_end_frame == 27
    assert window.end_frame == 27  # defaults to the pre-release cutoff
    # The whole window is strictly before release.
    assert window.end_frame < window.release_frame
    assert window.pre_release_end_frame < window.release_frame


def test_build_window_explicit_end_within_range() -> None:
    window = build_clip_window(
        start_frame=5,
        release_frame=30,
        release_guard_frames=3,
        video_frame_count=100,
        end_frame=20,
    )
    assert window.end_frame == 20


def test_guard_below_one_rejected() -> None:
    with pytest.raises(ClipFrameError, match="release_guard_frames"):
        build_clip_window(
            start_frame=0,
            release_frame=30,
            release_guard_frames=0,
            video_frame_count=100,
        )


def test_release_frame_out_of_range_rejected() -> None:
    with pytest.raises(ClipFrameError, match="within the video"):
        build_clip_window(
            start_frame=0,
            release_frame=100,
            release_guard_frames=3,
            video_frame_count=100,
        )


def test_start_after_cutoff_rejected() -> None:
    # pre-release cutoff is 27; a start at 28 leaves no pre-release frames.
    with pytest.raises(ClipFrameError, match="no pre-release frames"):
        build_clip_window(
            start_frame=28,
            release_frame=30,
            release_guard_frames=3,
            video_frame_count=100,
        )


def test_end_past_cutoff_rejected_prevents_leak() -> None:
    # end at 28 would include the guard/release region (cutoff is 27).
    with pytest.raises(ClipFrameError, match="leak"):
        build_clip_window(
            start_frame=0,
            release_frame=30,
            release_guard_frames=3,
            video_frame_count=100,
            end_frame=28,
        )


def test_end_before_start_rejected() -> None:
    with pytest.raises(ClipFrameError, match="end_frame"):
        build_clip_window(
            start_frame=10,
            release_frame=30,
            release_guard_frames=3,
            video_frame_count=100,
            end_frame=5,
        )


def test_assert_no_post_release_leak_catches_bad_window() -> None:
    # Hand-built window whose model cutoff sits exactly on release.
    leaking = ClipWindow(
        start_frame=0,
        end_frame=30,
        release_frame=30,
        release_guard_frames=0,
        pre_release_end_frame=30,
    )
    with pytest.raises(ClipFrameError):
        assert_no_post_release_leak(leaking)
