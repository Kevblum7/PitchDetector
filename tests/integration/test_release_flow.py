"""Integration tests: automatic release-frame detection over a real MP4.

Exercises the parts the in-memory unit tests skip: FFmpeg downscaled decode
for the motion gate, native-resolution coarse-to-fine reading, the background
job lifecycle, provenance on created clips, and the cross-reference gate.

The blob fake estimator is injected through the FastAPI dependency, so no
model weights or network are needed; the ``delivery_video`` fixture is a
moving white blob whose largest single-frame jump is the known release frame.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pose_fakes import BlobPoseEstimator
from release_fixture import TRUE_RELEASE_FRAME

from backend.app.core.enums import ReleaseFrameSource
from backend.app.db.models import PitchClip, SourceVideo
from backend.app.main import app
from backend.app.services.pose_extraction import get_pose_estimator
from backend.app.services.release_detection import (
    validate_release_detector,
)
from ml.release.detector import ReleaseDetectorConfig
from ml.release.pipeline import detect_release_in_video


@pytest.fixture
def release_client(client: TestClient) -> TestClient:
    app.dependency_overrides[get_pose_estimator] = BlobPoseEstimator
    return client


def register_video(client: TestClient, video_path: Path) -> int:
    pitcher = client.post("/api/v1/pitchers", json={"name": "Delivery Pitcher", "throws": "R"})
    assert pitcher.status_code == 201
    video = client.post(
        "/api/v1/videos",
        json={"pitcher_id": pitcher.json()["id"], "file_path": str(video_path)},
    )
    assert video.status_code == 201
    video_id: int = video.json()["id"]
    return video_id


class TestReleasePipelineOverVideo:
    def test_detects_release_frame_in_mp4(self, delivery_video: Path) -> None:
        from backend.app.services.video_metadata import probe_video

        meta = probe_video(delivery_video)
        detection = detect_release_in_video(
            delivery_video,
            fps=meta.fps,
            width=meta.width,
            height=meta.height,
            frame_count=meta.frame_count,
            estimator=BlobPoseEstimator(),
            config=ReleaseDetectorConfig(),
        )
        assert abs(detection.release_frame - TRUE_RELEASE_FRAME) <= 1
        assert detection.confidence >= 0.8
        assert detection.review_recommended is False


class TestReleaseDetectionApi:
    def test_job_flow(self, release_client: TestClient, delivery_video: Path) -> None:
        video_id = register_video(release_client, delivery_video)

        started = release_client.post(f"/api/v1/videos/{video_id}/release-detection")
        assert started.status_code == 202
        assert started.json()["estimator_name"] == "fake-blob"

        status = release_client.get(f"/api/v1/videos/{video_id}/release-detection")
        assert status.status_code == 200
        job = status.json()
        assert job["status"] == "completed", job.get("error_message")
        result = job["result"]
        assert abs(result["release_frame"] - TRUE_RELEASE_FRAME) <= 1
        assert result["detector_version"]

    def test_status_404_before_run(self, release_client: TestClient, delivery_video: Path) -> None:
        video_id = register_video(release_client, delivery_video)
        assert release_client.get(f"/api/v1/videos/{video_id}/release-detection").status_code == 404

    def test_unknown_video_is_404(self, release_client: TestClient) -> None:
        assert release_client.post("/api/v1/videos/999/release-detection").status_code == 404

    def test_missing_file_job_fails_readably(
        self, release_client: TestClient, delivery_video: Path, session: object
    ) -> None:
        from sqlmodel import Session

        video_id = register_video(release_client, delivery_video)
        assert isinstance(session, Session)
        video = session.get(SourceVideo, video_id)
        assert video is not None
        video.file_path = str(delivery_video.with_name("gone.mp4"))
        session.add(video)
        session.commit()

        assert (
            release_client.post(f"/api/v1/videos/{video_id}/release-detection").status_code == 202
        )
        job = release_client.get(f"/api/v1/videos/{video_id}/release-detection").json()
        assert job["status"] == "failed"
        assert "missing" in job["error_message"]


class TestReviewQueueAndValidation:
    def test_review_queue_lists_low_confidence_auto_clips(
        self, release_client: TestClient, delivery_video: Path, session: object
    ) -> None:
        from sqlmodel import Session

        assert isinstance(session, Session)
        video_id = register_video(release_client, delivery_video)
        video = session.get(SourceVideo, video_id)
        assert video is not None

        # One confident auto clip, one low-confidence auto clip, one manual.
        session.add_all(
            [
                _auto_clip(video, release_frame=58, confidence=0.95),
                _auto_clip(video, release_frame=40, confidence=0.30),
                _manual_clip(video, release_frame=58),
            ]
        )
        session.commit()

        queue = release_client.get("/api/v1/release-review").json()
        confidences = [c["release_frame_confidence"] for c in queue]
        assert len(queue) == 1  # only the low-confidence auto clip
        assert confidences == [0.30]

    def test_cross_reference_gate_reports_frame_error(
        self, delivery_video: Path, session: object
    ) -> None:
        from sqlmodel import Session

        assert isinstance(session, Session)
        # A manually labeled clip 2 frames off from the true release; the
        # detector should find the true frame and report the disagreement.
        pitcher_video = _seed_video(session, delivery_video)
        session.add(_manual_clip(pitcher_video, release_frame=TRUE_RELEASE_FRAME + 2))
        session.commit()

        report = validate_release_detector(session, BlobPoseEstimator(), frame_tolerance=1)
        assert report["manual_clips_total"] == 1
        assert report["compared"] == 1
        clip_result = report["clips"][0]  # type: ignore[index]
        assert clip_result["frame_error"] in (-2, -3, -1)
        assert Path(str(report["report_path"])).is_file()


def _seed_video(session: object, video_path: Path) -> SourceVideo:
    from sqlmodel import Session

    from backend.app.db.models import Pitcher
    from backend.app.services.checksum import sha256_file
    from backend.app.services.video_metadata import probe_video

    assert isinstance(session, Session)
    pitcher = Pitcher(name="V", throws="R")  # type: ignore[arg-type]
    session.add(pitcher)
    session.commit()
    session.refresh(pitcher)
    meta = probe_video(video_path)
    assert pitcher.id is not None
    video = SourceVideo(
        pitcher_id=pitcher.id,
        file_path=str(video_path),
        fps=meta.fps,
        width=meta.width,
        height=meta.height,
        duration_seconds=meta.duration_seconds,
        frame_count=meta.frame_count,
        checksum=sha256_file(video_path),
    )
    session.add(video)
    session.commit()
    session.refresh(video)
    return video


def _auto_clip(video: SourceVideo, *, release_frame: int, confidence: float) -> PitchClip:
    return _clip(video, release_frame, ReleaseFrameSource.AUTO, confidence)


def _manual_clip(video: SourceVideo, *, release_frame: int) -> PitchClip:
    return _clip(video, release_frame, ReleaseFrameSource.MANUAL, None)


def _clip(
    video: SourceVideo,
    release_frame: int,
    source: ReleaseFrameSource,
    confidence: float | None,
) -> PitchClip:
    from backend.app.core.enums import PitchType

    assert video.id is not None
    return PitchClip(
        source_video_id=video.id,
        pitcher_id=video.pitcher_id,
        start_frame=0,
        end_frame=release_frame - 3,
        release_frame=release_frame,
        pre_release_end_frame=release_frame - 3,
        release_guard_frames=3,
        release_frame_source=source,
        release_frame_confidence=confidence,
        pitch_type=PitchType.SLIDER,
    )
