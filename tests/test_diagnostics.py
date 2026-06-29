"""Tests for environment diagnostics (module, CLI helpers, and API endpoint)."""

from __future__ import annotations

import shutil

from fastapi.testclient import TestClient

from backend.app.core import diagnostics as diag_mod
from backend.app.core.diagnostics import (
    _detect_ffmpeg,
    _is_apple_silicon,
    _recommend_execution_mode,
    collect_diagnostics,
)


def test_collect_diagnostics_shape() -> None:
    diag = collect_diagnostics()
    assert diag.python_version
    assert diag.cpu_architecture
    # Intel and Apple Silicon are mutually exclusive.
    assert not (diag.is_intel and diag.is_apple_silicon)
    assert diag.recommended_execution_mode in {"cpu", "cuda", "mps"}


def test_diagnostics_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/system/diagnostics")
    assert response.status_code == 200

    body = response.json()
    for key in (
        "python_version",
        "cpu_architecture",
        "is_intel",
        "is_apple_silicon",
        "ffmpeg",
        "opencv",
        "torch",
        "recommended_execution_mode",
    ):
        assert key in body
    assert "available" in body["ffmpeg"]
    assert isinstance(body["torch"]["cuda_available"], bool)
    assert isinstance(body["torch"]["mps_available"], bool)


# --- CPU architecture detection ------------------------------------------------


def test_apple_silicon_detection_arm64(monkeypatch) -> None:
    monkeypatch.setattr(diag_mod.platform, "system", lambda: "Darwin")
    assert _is_apple_silicon("arm64", "arm") is True


def test_apple_silicon_detection_intel(monkeypatch) -> None:
    monkeypatch.setattr(diag_mod.platform, "system", lambda: "Darwin")
    assert _is_apple_silicon("x86_64", "i386") is False


def test_apple_silicon_detection_rosetta(monkeypatch) -> None:
    # Under Rosetta machine may read x86_64 but processor mentions Apple.
    monkeypatch.setattr(diag_mod.platform, "system", lambda: "Darwin")
    assert _is_apple_silicon("x86_64", "Apple M1") is True


def test_non_darwin_is_not_apple_silicon(monkeypatch) -> None:
    monkeypatch.setattr(diag_mod.platform, "system", lambda: "Linux")
    assert _is_apple_silicon("arm64", "arm") is False


def test_recommend_execution_mode() -> None:
    mode = _recommend_execution_mode
    assert mode(cuda_available=True, mps_available=True, is_apple_silicon=True) == "cuda"
    assert mode(cuda_available=False, mps_available=True, is_apple_silicon=True) == "mps"
    # MPS reported but not Apple Silicon -> fall back to CPU.
    assert mode(cuda_available=False, mps_available=True, is_apple_silicon=False) == "cpu"
    assert mode(cuda_available=False, mps_available=False, is_apple_silicon=False) == "cpu"


# --- FFmpeg availability handling ---------------------------------------------


def test_ffmpeg_missing_is_reported(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    info = _detect_ffmpeg()
    assert info.available is False
    assert info.version is None
    assert info.error is not None


def test_ffmpeg_present_parses_version(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class _Result:
        stdout = "ffmpeg version 7.0.1 Copyright (c) 2000-2024 the FFmpeg developers\n"

    monkeypatch.setattr(diag_mod.subprocess, "run", lambda *a, **k: _Result())
    info = _detect_ffmpeg()
    assert info.available is True
    assert info.version == "7.0.1"
    assert info.path == "/usr/bin/ffmpeg"


def test_ffmpeg_run_failure_is_handled(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    def _boom(*_a, **_k):
        raise OSError("cannot exec")

    monkeypatch.setattr(diag_mod.subprocess, "run", _boom)
    info = _detect_ffmpeg()
    # Located on PATH but could not be executed: available flag stays True,
    # error explains the failure rather than raising.
    assert info.available is True
    assert info.error is not None
    assert info.version is None
