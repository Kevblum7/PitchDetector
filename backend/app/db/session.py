"""Database engine and session management.

The engine is built from ``config.DATABASE_URL`` (override with
``PITCH_DATABASE_URL``). ``get_session`` is a FastAPI dependency; tests override
it with a session bound to an isolated engine.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from backend.app.core.config import DATABASE_URL

# Import models so their tables are registered on SQLModel.metadata before
# create_all runs. (Imported for the side effect of table registration.)
from backend.app.db import models as models  # noqa: F401


def _connect_args(url: str) -> dict[str, object]:
    # SQLite + a threaded server (uvicorn/TestClient) needs this.
    if url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


def make_engine(url: str = DATABASE_URL) -> Engine:
    """Create an engine (no filesystem side effects)."""
    return create_engine(url, echo=False, connect_args=_connect_args(url))


def _ensure_sqlite_dir(url: str) -> None:
    """Create the parent directory for a file-backed SQLite DB, if needed."""
    if url.startswith("sqlite:///") and ":memory:" not in url:
        db_path = Path(url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)


engine: Engine = make_engine()


def init_db(target_engine: Engine | None = None) -> None:
    """Create all tables. Safe to call repeatedly.

    Ensures the SQLite directory exists for the default engine (done here, at
    app startup, rather than at import so tests never touch the real data dir).
    """
    if target_engine is None:
        _ensure_sqlite_dir(DATABASE_URL)
    SQLModel.metadata.create_all(target_engine or engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a database session."""
    with Session(engine) as session:
        yield session
