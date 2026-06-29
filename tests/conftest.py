"""Shared pytest fixtures.

Tests run against an isolated in-memory SQLite database (never the project's
real ``data/`` DB) via a dependency override on ``get_session``.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from backend.app.db.session import get_session
from backend.app.main import app

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture
def engine() -> Iterator[Engine]:
    """A fresh in-memory database, shared across connections for one test."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(test_engine)
    try:
        yield test_engine
    finally:
        SQLModel.metadata.drop_all(test_engine)
        test_engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with Session(engine) as db_session:
        yield db_session


@pytest.fixture
def client(engine: Engine) -> Iterator[TestClient]:
    """TestClient with ``get_session`` bound to the in-memory test engine.

    Note: ``TestClient(app)`` is not used as a context manager, so the app
    lifespan (which would create the real ``data/`` DB) does not run.
    """

    def _override() -> Iterator[Session]:
        with Session(engine) as db_session:
            yield db_session

    app.dependency_overrides[get_session] = _override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny 30-frame synthetic MP4 (1s @ 30fps, 320x240) built with ffmpeg.

    Skips the requesting test if ffmpeg/ffprobe are unavailable, so the suite
    never depends on a private MLB clip (CLAUDE.md §18).
    """
    if not _FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg/ffprobe not available")
    out = tmp_path_factory.mktemp("video") / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x240:rate=30",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return out
