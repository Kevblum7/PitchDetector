"""Unit tests for FFmpeg-piped video decode/encode (ml/pose/video_io.py).

Frame accuracy matters here: clip windows are defined in absolute frame
indices, so the reader returning the wrong frame after a seek would silently
corrupt release-relative timing. The ``indexed_video`` fixture encodes each
frame's index in its pixel values, letting these tests assert exact indices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from video_fixture import (
    INDEXED_VIDEO_FPS,
    INDEXED_VIDEO_FRAMES,
    INDEXED_VIDEO_SIZE,
    INDEXED_VIDEO_STEP,
)

from backend.app.services.video_metadata import probe_video
from ml.pose.video_io import VideoIOError, VideoWriter, iter_frames

WIDTH, HEIGHT = INDEXED_VIDEO_SIZE
# Encode/decode round trip (RGB -> yuv444 qp0 -> RGB) is not bit-exact.
TOLERANCE = 3.0


def read_range(path: Path, start: int, end: int) -> list[tuple[int, np.ndarray]]:
    return list(
        iter_frames(
            path,
            start_frame=start,
            end_frame=end,
            fps=INDEXED_VIDEO_FPS,
            width=WIDTH,
            height=HEIGHT,
        )
    )


def assert_frame_value(frame: np.ndarray, expected_index: int) -> None:
    measured = float(frame.mean())
    expected = expected_index * INDEXED_VIDEO_STEP
    assert abs(measured - expected) <= TOLERANCE, (
        f"expected frame {expected_index} (value ~{expected}), measured {measured:.1f} "
        f"(~frame {measured / INDEXED_VIDEO_STEP:.1f})"
    )


class TestIterFrames:
    def test_reads_full_video_in_order(self, indexed_video: Path) -> None:
        frames = read_range(indexed_video, 0, INDEXED_VIDEO_FRAMES - 1)
        assert [i for i, _ in frames] == list(range(INDEXED_VIDEO_FRAMES))
        for index, frame in frames:
            assert frame.shape == (HEIGHT, WIDTH, 3)
            assert_frame_value(frame, index)

    def test_seeked_read_is_frame_accurate(self, indexed_video: Path) -> None:
        # The critical property: after -ss seeking, the first frame returned is
        # exactly the requested start frame, not a neighbour.
        frames = read_range(indexed_video, 17, 22)
        assert [i for i, _ in frames] == [17, 18, 19, 20, 21, 22]
        for index, frame in frames:
            assert_frame_value(frame, index)

    def test_single_frame_read(self, indexed_video: Path) -> None:
        frames = read_range(indexed_video, 11, 11)
        assert len(frames) == 1
        assert_frame_value(frames[0][1], 11)

    def test_range_past_end_of_video_raises(self, indexed_video: Path) -> None:
        with pytest.raises(VideoIOError, match="video ended"):
            read_range(indexed_video, 25, INDEXED_VIDEO_FRAMES + 10)

    def test_invalid_range_raises(self, indexed_video: Path) -> None:
        with pytest.raises(VideoIOError):
            read_range(indexed_video, 5, 4)
        with pytest.raises(VideoIOError):
            read_range(indexed_video, -1, 4)

    def test_invalid_geometry_raises(self, indexed_video: Path) -> None:
        with pytest.raises(VideoIOError):
            list(
                iter_frames(
                    indexed_video, start_frame=0, end_frame=1, fps=0.0, width=WIDTH, height=HEIGHT
                )
            )


class TestVideoWriter:
    def test_round_trip(self, tmp_path: Path, indexed_video: Path) -> None:
        out = tmp_path / "written.mp4"
        with VideoWriter(out, width=WIDTH, height=HEIGHT, fps=30.0) as writer:
            for value in (40, 120, 200):
                writer.write(np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8))

        meta = probe_video(out)
        assert meta.width == WIDTH
        assert meta.height == HEIGHT
        assert meta.frame_count == 3

        decoded = read_range(out, 0, 2)
        for (_, frame), value in zip(decoded, (40, 120, 200), strict=True):
            # crf-compressed, so allow a wider band than the lossless fixture.
            assert abs(float(frame.mean()) - value) <= 8

    def test_rejects_wrong_frame_shape(self, tmp_path: Path) -> None:
        out = tmp_path / "bad.mp4"
        writer = VideoWriter(out, width=WIDTH, height=HEIGHT, fps=30.0)
        try:
            with pytest.raises(VideoIOError, match="expected uint8 frame"):
                writer.write(np.zeros((HEIGHT + 1, WIDTH, 3), dtype=np.uint8))
        finally:
            writer._process.kill()
            writer._process.wait()

    def test_pads_odd_dimensions(self, tmp_path: Path) -> None:
        out = tmp_path / "odd.mp4"
        with VideoWriter(out, width=63, height=47, fps=30.0) as writer:
            writer.write(np.zeros((47, 63, 3), dtype=np.uint8))
        meta = probe_video(out)
        assert meta.width == 64
        assert meta.height == 48
