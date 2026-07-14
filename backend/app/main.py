"""FastAPI application entrypoint for the Pitch Tip Detector.

Milestone 1: health + environment diagnostics.
Milestone 2: pitcher/video/clip registration and labeling backed by SQLite.
Milestone 3: pitcher tracking + pose extraction + overlay rendering.
No training or frontend yet.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.v1 import clips, pitchers, pose, release, system, videos
from backend.app.core.config import APP_NAME, APP_TITLE, APP_VERSION
from backend.app.db.session import init_db


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Create tables on startup (idempotent). Tests manage their own schema.
    init_db()
    yield


app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan)

app.include_router(system.router)
app.include_router(pitchers.router)
app.include_router(videos.router)
app.include_router(clips.router)
app.include_router(pose.router)
app.include_router(release.router)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Liveness probe with basic service identity."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "ok",
    }
