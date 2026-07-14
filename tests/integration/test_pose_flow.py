"""Integration tests: clip → pose job → stored pose frames → rendered overlay.

Uses the synthetic ffmpeg fixture video and a fake estimator injected through
the FastAPI dependency (no model weights, no network). Background pose jobs
run synchronously inside TestClient's request cycle, so the job has finished
by the time the POST returns.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pose_fakes import CenteredPoseEstimator

from backend.app.main import app
from backend.app.services.pose_extraction import get_pose_estimator
from backend.app.services.video_metadata import probe_video

RELEASE_FRAME = 25
GUARD = 3
PRE_RELEASE_END = RELEASE_FRAME - GUARD  # 22
EXPECTED_FRAMES = PRE_RELEASE_END + 1  # start_frame 0 .. 22 inclusive
INITIAL_BOX = {"x": 120, "y": 60, "width": 80, "height": 120}


@pytest.fixture
def pose_client(client: TestClient) -> TestClient:
    """The standard test client with the pose estimator faked."""
    app.dependency_overrides[get_pose_estimator] = CenteredPoseEstimator
    return client  # overrides are cleared by the client fixture's teardown


def make_labeled_clip(client: TestClient, synthetic_video: Path, *, with_box: bool = True) -> int:
    pitcher = client.post("/api/v1/pitchers", json={"name": "Test Pitcher", "throws": "R"})
    assert pitcher.status_code == 201
    video = client.post(
        "/api/v1/videos",
        json={"pitcher_id": pitcher.json()["id"], "file_path": str(synthetic_video)},
    )
    assert video.status_code == 201
    body: dict[str, object] = {
        "start_frame": 0,
        "release_frame": RELEASE_FRAME,
        "release_guard_frames": GUARD,
        "pitch_type": "slider",
    }
    if with_box:
        body["initial_pitcher_box"] = INITIAL_BOX
    clip = client.post(f"/api/v1/videos/{video.json()['id']}/clips", json=body)
    assert clip.status_code == 201
    clip_id: int = clip.json()["id"]
    return clip_id


class TestPoseExtractionFlow:
    def test_full_flow(
        self,
        pose_client: TestClient,
        synthetic_video: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        clip_id = make_labeled_clip(pose_client, synthetic_video)

        # Start extraction; the background job runs before the response returns.
        started = pose_client.post(f"/api/v1/clips/{clip_id}/pose")
        assert started.status_code == 202
        assert started.json()["estimator_name"] == "fake-centered"

        status = pose_client.get(f"/api/v1/clips/{clip_id}/pose")
        assert status.status_code == 200
        job = status.json()
        assert job["status"] == "completed", job.get("error_message")
        assert job["frames_written"] == EXPECTED_FRAMES
        assert job["quality"]["tracking_ok"] is True
        assert job["quality"]["detected_frames"] == EXPECTED_FRAMES
        assert job["config"]["smoothing_window"] == 5  # reproducibility record

        frames = pose_client.get(f"/api/v1/clips/{clip_id}/pose/frames").json()
        assert len(frames) == EXPECTED_FRAMES
        indices = [f["frame_index"] for f in frames]
        assert indices == list(range(EXPECTED_FRAMES))
        assert max(indices) == PRE_RELEASE_END  # leakage: nothing at/after release
        for frame in frames:
            assert frame["timestamp_ms_relative_to_release"] < 0
            assert frame["detected"] is True
            assert len(frame["keypoints"]) == 17
            assert len(frame["visibility_scores"]) == 17
            assert frame["pitcher_bbox"]["width"] > 0

        # Render the overlay into a redirected artifacts dir.
        monkeypatch.setenv("PITCH_ARTIFACTS_DIR", str(tmp_path))
        rendered = pose_client.post(f"/api/v1/clips/{clip_id}/pose/render")
        assert rendered.status_code == 200
        output = Path(rendered.json()["output_path"])
        assert output.is_file()
        assert output.suffix == ".mp4"
        meta = probe_video(output)
        assert meta.frame_count == EXPECTED_FRAMES
        assert (meta.width, meta.height) == (320, 240)

    def test_rerun_replaces_frames_instead_of_duplicating(
        self, pose_client: TestClient, synthetic_video: Path
    ) -> None:
        clip_id = make_labeled_clip(pose_client, synthetic_video)
        assert pose_client.post(f"/api/v1/clips/{clip_id}/pose").status_code == 202
        assert pose_client.post(f"/api/v1/clips/{clip_id}/pose").status_code == 202

        frames = pose_client.get(f"/api/v1/clips/{clip_id}/pose/frames").json()
        assert len(frames) == EXPECTED_FRAMES  # not doubled

    def test_pose_requires_initial_box(
        self, pose_client: TestClient, synthetic_video: Path
    ) -> None:
        clip_id = make_labeled_clip(pose_client, synthetic_video, with_box=False)
        response = pose_client.post(f"/api/v1/clips/{clip_id}/pose")
        assert response.status_code == 400
        assert "initial pitcher box" in response.json()["detail"]

    def test_box_can_be_set_by_patch(self, pose_client: TestClient, synthetic_video: Path) -> None:
        clip_id = make_labeled_clip(pose_client, synthetic_video, with_box=False)
        patched = pose_client.patch(
            f"/api/v1/clips/{clip_id}", json={"initial_pitcher_box": INITIAL_BOX}
        )
        assert patched.status_code == 200
        assert patched.json()["initial_box_x"] == INITIAL_BOX["x"]
        assert pose_client.post(f"/api/v1/clips/{clip_id}/pose").status_code == 202
        job = pose_client.get(f"/api/v1/clips/{clip_id}/pose").json()
        assert job["status"] == "completed"

    def test_out_of_bounds_box_is_rejected(
        self, pose_client: TestClient, synthetic_video: Path
    ) -> None:
        clip_id = make_labeled_clip(pose_client, synthetic_video, with_box=False)
        response = pose_client.patch(
            f"/api/v1/clips/{clip_id}",
            json={"initial_pitcher_box": {"x": 280, "y": 60, "width": 80, "height": 120}},
        )
        assert response.status_code == 400
        assert "extends beyond" in response.json()["detail"]

    def test_status_and_frames_404_before_extraction(
        self, pose_client: TestClient, synthetic_video: Path
    ) -> None:
        clip_id = make_labeled_clip(pose_client, synthetic_video)
        assert pose_client.get(f"/api/v1/clips/{clip_id}/pose").status_code == 404
        assert pose_client.get(f"/api/v1/clips/{clip_id}/pose/frames").status_code == 404
        assert pose_client.post(f"/api/v1/clips/{clip_id}/pose/render").status_code == 400

    def test_unknown_clip_is_404(self, pose_client: TestClient) -> None:
        assert pose_client.post("/api/v1/clips/999/pose").status_code == 404
        assert pose_client.get("/api/v1/clips/999/pose").status_code == 404

    def test_failed_job_reports_readable_error(
        self, pose_client: TestClient, synthetic_video: Path, session: object
    ) -> None:
        clip_id = make_labeled_clip(pose_client, synthetic_video)
        # Sabotage: point the video at a missing file after registration.
        from sqlmodel import Session

        from backend.app.db.models import PitchClip, SourceVideo

        assert isinstance(session, Session)
        clip = session.get(PitchClip, clip_id)
        assert clip is not None
        video = session.get(SourceVideo, clip.source_video_id)
        assert video is not None
        video.file_path = str(synthetic_video.with_name("gone.mp4"))
        session.add(video)
        session.commit()

        assert pose_client.post(f"/api/v1/clips/{clip_id}/pose").status_code == 202
        job = pose_client.get(f"/api/v1/clips/{clip_id}/pose").json()
        assert job["status"] == "failed"
        assert "missing" in job["error_message"]
        assert job["error_stack"]  # development mode persists the stack trace
