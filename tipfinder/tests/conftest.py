"""Fixtures backed ONLY by real Baseball Savant / Statcast data.

`tests/.cache` is built with `uv run tipfinder build-test-cache` after `fetch` + `extract`.
It is git-ignored (MLB video must not be committed). Tests that need it skip without it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

CACHE = Path(__file__).parent / ".cache"
PROJECT = Path(__file__).resolve().parents[1]


def _need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"real-data test cache missing ({path.name}); run `tipfinder build-test-cache`")
    return path


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    return _need(CACHE)


@pytest.fixture(scope="session")
def cached_manifest(cache_dir: Path) -> pd.DataFrame:
    return pd.read_csv(_need(cache_dir / "manifest.csv"))


@pytest.fixture(scope="session")
def cached_extract(cache_dir: Path) -> pd.DataFrame:
    return pd.read_csv(_need(cache_dir / "extract.csv"))


@pytest.fixture(scope="session")
def ok_pitches(cached_extract: pd.DataFrame) -> pd.DataFrame:
    ok = cached_extract[cached_extract["status"] == "ok"]
    if ok.empty:
        pytest.skip("no usable clips in cache")
    return ok


@pytest.fixture(scope="session")
def statcast_games(cache_dir: Path) -> pd.DataFrame:
    return pd.read_csv(_need(cache_dir / "statcast_games.csv"))


@pytest.fixture(scope="session")
def project_features() -> pd.DataFrame:
    """Full real feature table from the last `tipfinder train` run (integration tests)."""
    p = PROJECT / "data" / "zack_wheeler_2024" / "features.csv"
    return pd.read_csv(_need(p))
