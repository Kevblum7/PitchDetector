"""Provenance rules for automatic release labels (CLAUDE.md §9).

The core invariant: an automatic label never silently overwrites a manual or
confirmed one.
"""

from __future__ import annotations

import pytest

from backend.app.core.enums import PitchType, ReleaseFrameSource
from backend.app.db.models import PitchClip
from backend.app.services.release_detection import ReleaseLabelError, apply_auto_release
from ml.release.detector import ReleaseDetection


def make_clip(source: ReleaseFrameSource) -> PitchClip:
    return PitchClip(
        source_video_id=1,
        pitcher_id=1,
        start_frame=0,
        end_frame=50,
        release_frame=55,
        pre_release_end_frame=52,
        release_guard_frames=3,
        release_frame_source=source,
        pitch_type=PitchType.SLIDER,
    )


def make_detection(frame: int = 58, confidence: float = 0.9) -> ReleaseDetection:
    return ReleaseDetection(
        release_frame=frame,
        confidence=confidence,
        review_recommended=confidence < 0.6,
        wrist="right",
        peak_speed_px_per_frame=30.0,
        median_speed_px_per_frame=2.0,
        motion_window=(40, 70),
        coarse_candidate_frame=60,
        fine_range=(54, 66),
        frames_analyzed=20,
    )


class TestApplyAutoRelease:
    def test_refuses_to_overwrite_manual_label(self) -> None:
        clip = make_clip(ReleaseFrameSource.MANUAL)
        with pytest.raises(ReleaseLabelError, match="refusing to overwrite"):
            apply_auto_release(clip, make_detection())
        assert clip.release_frame == 55  # untouched

    def test_refuses_to_overwrite_confirmed_label(self) -> None:
        clip = make_clip(ReleaseFrameSource.AUTO_CONFIRMED)
        with pytest.raises(ReleaseLabelError, match="refusing to overwrite"):
            apply_auto_release(clip, make_detection())

    def test_updates_unconfirmed_auto_label(self) -> None:
        clip = make_clip(ReleaseFrameSource.AUTO)
        apply_auto_release(clip, make_detection(frame=58, confidence=0.75))
        assert clip.release_frame == 58
        assert clip.release_frame_source == ReleaseFrameSource.AUTO
        assert clip.release_frame_confidence == 0.75
