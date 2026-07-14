"""Automatic release-frame labeling: job execution, guards, and validation.

Implements the CLAUDE.md §9 rules around the auto-labeler:

- every detection is persisted with its config + detector version;
- an automatic label never silently overwrites a manual (or confirmed) one;
- low-confidence labels land in a review queue;
- the detector must be validated against manually labeled clips
  (cross-reference gate) before being trusted at scale.
"""

from __future__ import annotations

import json
import logging
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session, col, select

from backend.app.core.config import ENVIRONMENT, artifacts_dir
from backend.app.core.enums import JobStatus, ReleaseFrameSource
from backend.app.db.models import PitchClip, ReleaseDetectionJob, SourceVideo
from ml.pose.estimator import PoseEstimator
from ml.release.detector import ReleaseDetection, ReleaseDetectorConfig
from ml.release.pipeline import detect_release_in_video

logger = logging.getLogger(__name__)


class ReleaseLabelError(RuntimeError):
    """Raised when an auto label would violate a provenance rule."""


def detect_release_for_video(
    video: SourceVideo,
    estimator: PoseEstimator,
    config: ReleaseDetectorConfig | None = None,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> ReleaseDetection:
    """Run the full detection pipeline for a registered video (or a range).

    Pass a frame range when the video contains more than one pitch — the
    detector finds *the* dominant delivery in the range it is given.
    """
    video_path = Path(video.file_path)
    if not video_path.is_file():
        raise ReleaseLabelError(f"source video file is missing: {video_path}")
    return detect_release_in_video(
        video_path,
        fps=video.fps,
        width=video.width,
        height=video.height,
        frame_count=video.frame_count,
        estimator=estimator,
        config=config,
        start_frame=start_frame,
        end_frame=end_frame,
    )


# When validating against a clip, search from the clip start to a margin past
# the manual release so the true release is inside the searched range.
_VALIDATION_POST_RELEASE_MARGIN_FRAMES = 45


def run_release_detection_job(
    job_id: int,
    session_factory: Callable[[], Session],
    estimator: PoseEstimator,
    config: ReleaseDetectorConfig | None = None,
) -> None:
    """Execute a queued detection job; failures are persisted, never raised."""
    config = config or ReleaseDetectorConfig()
    with session_factory() as session:
        job = session.get(ReleaseDetectionJob, job_id)
        if job is None:  # pragma: no cover - job created just before dispatch
            logger.error("release detection job %s vanished before execution", job_id)
            return
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        session.add(job)
        session.commit()

        try:
            video = session.get(SourceVideo, job.source_video_id)
            if video is None:
                raise ReleaseLabelError(f"source video {job.source_video_id} not found")
            detection = detect_release_for_video(video, estimator, config)
            job.status = JobStatus.COMPLETED
            job.result = detection.as_dict()
        except Exception as exc:
            logger.exception("release detection job %s failed", job_id)
            job.status = JobStatus.FAILED
            job.error_message = str(exc)
            if ENVIRONMENT == "development":
                job.error_stack = traceback.format_exc()
        finally:
            job.finished_at = datetime.now(UTC)
            session.add(job)
            session.commit()


def latest_release_job(session: Session, video_id: int) -> ReleaseDetectionJob | None:
    statement = (
        select(ReleaseDetectionJob)
        .where(ReleaseDetectionJob.source_video_id == video_id)
        .order_by(col(ReleaseDetectionJob.id).desc())
        .limit(1)
    )
    return session.exec(statement).first()


def apply_auto_release(clip: PitchClip, detection: ReleaseDetection) -> None:
    """Write an automatic release label onto a clip, respecting provenance.

    Raises:
        ReleaseLabelError: the clip already has a manual or confirmed release
            frame — automatic labels never silently overwrite those.

    Note: this only updates the release fields; the caller must rebuild the
    clip window (``build_clip_window``) so the leak-free cutoff stays valid.
    """
    if clip.release_frame_source in (
        ReleaseFrameSource.MANUAL,
        ReleaseFrameSource.AUTO_CONFIRMED,
    ):
        raise ReleaseLabelError(
            f"clip {clip.id} has a {clip.release_frame_source} release frame "
            f"({clip.release_frame}); refusing to overwrite it with an automatic label. "
            "Change it explicitly via PATCH if that is intended."
        )
    clip.release_frame = detection.release_frame
    clip.release_frame_source = ReleaseFrameSource.AUTO
    clip.release_frame_confidence = detection.confidence


def clips_needing_release_review(
    session: Session,
    *,
    confidence_threshold: float | None = None,
) -> list[PitchClip]:
    """Auto-labeled clips a human should look at (the review queue)."""
    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else ReleaseDetectorConfig().review_confidence_threshold
    )
    statement = select(PitchClip).where(PitchClip.release_frame_source == ReleaseFrameSource.AUTO)
    return [
        clip
        for clip in session.exec(statement).all()
        if clip.release_frame_confidence is None or clip.release_frame_confidence < threshold
    ]


def validate_release_detector(
    session: Session,
    estimator: PoseEstimator,
    config: ReleaseDetectorConfig | None = None,
    *,
    frame_tolerance: int = 1,
) -> dict[str, object]:
    """Cross-reference gate: run the detector against manually labeled clips.

    Returns a report with per-clip frame errors and the fraction within
    ``frame_tolerance``; also writes it to ``artifacts/reports/``. Detection
    failures are recorded per clip, not hidden.
    """
    config = config or ReleaseDetectorConfig()
    manual_clips = list(
        session.exec(
            select(PitchClip).where(
                PitchClip.release_frame_source.in_(  # type: ignore[attr-defined]
                    [ReleaseFrameSource.MANUAL, ReleaseFrameSource.AUTO_CONFIRMED]
                )
            )
        ).all()
    )

    results: list[dict[str, object]] = []
    errors_within_tolerance = 0
    compared = 0
    for clip in manual_clips:
        video = session.get(SourceVideo, clip.source_video_id)
        entry: dict[str, object] = {
            "clip_id": clip.id,
            "manual_release_frame": clip.release_frame,
        }
        if video is None:  # pragma: no cover - orphaned clip should not occur
            entry["error"] = f"source video {clip.source_video_id} missing"
            results.append(entry)
            continue
        try:
            detection = detect_release_for_video(
                video,
                estimator,
                config,
                start_frame=clip.start_frame,
                end_frame=clip.release_frame + _VALIDATION_POST_RELEASE_MARGIN_FRAMES,
            )
        except Exception as exc:  # detector failures are data, not crashes
            entry["error"] = str(exc)
            results.append(entry)
            continue
        frame_error = detection.release_frame - clip.release_frame
        compared += 1
        if abs(frame_error) <= frame_tolerance:
            errors_within_tolerance += 1
        entry.update(
            {
                "detected_release_frame": detection.release_frame,
                "frame_error": frame_error,
                "confidence": detection.confidence,
                "review_recommended": detection.review_recommended,
            }
        )
        results.append(entry)

    report: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "detector_config": config.as_dict(),
        "estimator_name": estimator.name,
        "frame_tolerance": frame_tolerance,
        "manual_clips_total": len(manual_clips),
        "compared": compared,
        "failed": len(manual_clips) - compared,
        "within_tolerance": errors_within_tolerance,
        "within_tolerance_fraction": (
            round(errors_within_tolerance / compared, 4) if compared else None
        ),
        "clips": results,
    }

    reports_dir = artifacts_dir() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"release_validation_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2))
    report["report_path"] = str(report_path)
    return report
