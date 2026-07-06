"""Integration test: pitcher -> video -> clip -> reload -> update.

Covers Milestone 2 acceptance: a user can import a video, define a clip, mark
release, label the pitch type, and reload the saved clip.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def _create_pitcher(client: TestClient) -> int:
    resp = client.post(
        "/api/v1/pitchers",
        json={"name": "Test Pitcher", "throws": "R"},
    )
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


def _register_video(client: TestClient, pitcher_id: int, path: Path) -> dict:
    resp = client.post(
        "/api/v1/videos",
        json={
            "pitcher_id": pitcher_id,
            "file_path": str(path),
            "source_name": "unit-fixture",
            "game_id": "G1",
            "camera_angle": "center_field",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_full_flow(client: TestClient, synthetic_video: Path) -> None:
    pitcher_id = _create_pitcher(client)
    video = _register_video(client, pitcher_id, synthetic_video)

    # Metadata extracted from the real file.
    assert video["fps"] > 0
    assert video["frame_count"] >= 25
    assert video["width"] == 320
    assert len(video["checksum"]) == 64

    frame_count = int(video["frame_count"])
    release_frame = frame_count - 5
    guard = 3

    # Create a clip and mark release + pitch type.
    resp = client.post(
        f"/api/v1/videos/{video['id']}/clips",
        json={
            "start_frame": 0,
            "release_frame": release_frame,
            "release_guard_frames": guard,
            "pitch_type": "slider",
        },
    )
    assert resp.status_code == 201, resp.text
    clip = resp.json()
    clip_id = clip["id"]
    assert clip["pre_release_end_frame"] == release_frame - guard
    assert clip["pre_release_end_frame"] < clip["release_frame"]  # no leak
    assert clip["end_frame"] < clip["release_frame"]
    assert clip["pitcher_id"] == pitcher_id
    assert clip["game_id"] == "G1"
    assert clip["camera_angle"] == "center_field"  # inherited from the video

    # Reload the saved clip.
    reload = client.get(f"/api/v1/clips/{clip_id}")
    assert reload.status_code == 200
    assert reload.json()["release_frame"] == release_frame
    assert reload.json()["pitch_type"] == "slider"

    # Update release frame + label; window is recomputed.
    new_release = release_frame - 2
    patch = client.patch(
        f"/api/v1/clips/{clip_id}",
        json={"release_frame": new_release, "quality_status": "usable"},
    )
    assert patch.status_code == 200, patch.text
    updated = patch.json()
    assert updated["release_frame"] == new_release
    assert updated["pre_release_end_frame"] == new_release - guard
    assert updated["quality_status"] == "usable"

    # It persisted.
    again = client.get(f"/api/v1/clips/{clip_id}").json()
    assert again["release_frame"] == new_release
    assert again["quality_status"] == "usable"


def test_duplicate_video_is_deduplicated(client: TestClient, synthetic_video: Path) -> None:
    pitcher_id = _create_pitcher(client)
    first = _register_video(client, pitcher_id, synthetic_video)

    # Same content again -> 200 and the same record, not a second row.
    resp = client.post(
        "/api/v1/videos",
        json={"pitcher_id": pitcher_id, "file_path": str(synthetic_video)},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == first["id"]


def test_clip_creation_rejects_leaky_window(client: TestClient, synthetic_video: Path) -> None:
    pitcher_id = _create_pitcher(client)
    video = _register_video(client, pitcher_id, synthetic_video)
    frame_count = int(video["frame_count"])

    # end_frame past the pre-release cutoff must be rejected.
    resp = client.post(
        f"/api/v1/videos/{video['id']}/clips",
        json={
            "start_frame": 0,
            "release_frame": frame_count - 3,
            "release_guard_frames": 3,
            "end_frame": frame_count - 4,  # cutoff is frame_count-6
            "pitch_type": "four_seam_fastball",
        },
    )
    assert resp.status_code == 400
    assert "leak" in resp.json()["detail"]


def test_release_frame_beyond_video_rejected(client: TestClient, synthetic_video: Path) -> None:
    pitcher_id = _create_pitcher(client)
    video = _register_video(client, pitcher_id, synthetic_video)
    resp = client.post(
        f"/api/v1/videos/{video['id']}/clips",
        json={
            "release_frame": int(video["frame_count"]) + 100,
            "pitch_type": "slider",
        },
    )
    assert resp.status_code == 400
    assert "within the video" in resp.json()["detail"]


def test_register_video_unknown_pitcher(client: TestClient, synthetic_video: Path) -> None:
    resp = client.post(
        "/api/v1/videos",
        json={"pitcher_id": 9999, "file_path": str(synthetic_video)},
    )
    assert resp.status_code == 404
