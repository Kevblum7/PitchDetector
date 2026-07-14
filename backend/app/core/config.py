"""Application-level constants and settings.

Kept intentionally small for Milestone 1. Reads a couple of optional values
from the environment so deployment settings are not hard-coded, but avoids a
full settings framework until it is genuinely needed.
"""

from __future__ import annotations

import os
from importlib import metadata
from pathlib import Path

APP_NAME = "pitch-tip-detector"
APP_TITLE = "Pitch Tip Detector"

# --- Storage -----------------------------------------------------------------

# Project data lives under data/ (git-ignored). The SQLite database default
# sits alongside it. Override the DB with PITCH_DATABASE_URL (used by tests).
DATA_DIR = Path(os.getenv("PITCH_DATA_DIR", "data")).resolve()
DEFAULT_DB_PATH = DATA_DIR / "pitchdetector.db"
DATABASE_URL = os.getenv("PITCH_DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# Video files we are willing to register (lower-case suffixes).
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def artifacts_dir() -> Path:
    """Root for generated artifacts (git-ignored). Env-resolved per call so
    tests can redirect output without reimporting this module."""
    return Path(os.getenv("PITCH_ARTIFACTS_DIR", "artifacts")).resolve()


def overlays_dir() -> Path:
    """Where rendered overlay videos are written."""
    return artifacts_dir() / "overlays"


# --- Leakage / clip windowing (see CLAUDE.md §9-§10) -------------------------

# Frames of safety margin dropped before the release frame. Must be >= 1 so the
# release frame itself can never enter the model window.
DEFAULT_RELEASE_GUARD_FRAMES = 3
DEFAULT_PRE_RELEASE_WINDOW_MS = 1500


def get_version() -> str:
    """Return the installed package version, falling back to the source default.

    When the project is installed (``uv sync``) the version comes from package
    metadata. When it is not installed (e.g. running straight from a checkout)
    ``PackageNotFoundError`` is raised and we fall back to a source constant.
    """
    try:
        return metadata.version(APP_NAME)
    except metadata.PackageNotFoundError:
        return "0.1.0"


APP_VERSION = get_version()

# Optional environment overrides (see .env.example).
ENVIRONMENT = os.getenv("PITCH_ENV", "development")
