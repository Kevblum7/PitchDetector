"""Environment diagnostics for the Pitch Tip Detector.

Reports the local runtime so a developer can confirm the machine matches the
project's Intel-Mac / CPU-first assumptions (see CLAUDE.md sections 2 and 27).

Every optional capability (FFmpeg, OpenCV, PyTorch) is probed defensively:
a missing tool is reported as unavailable rather than raising, so the
diagnostics endpoint and CLI always return a complete picture.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

# FFmpeg version output looks like: "ffmpeg version 7.0.1 Copyright ..."
_SUBPROCESS_TIMEOUT_SECONDS = 5


@dataclass
class FFmpegInfo:
    available: bool
    path: str | None = None
    version: str | None = None
    error: str | None = None


@dataclass
class LibraryInfo:
    """Version info for an optional Python library."""

    available: bool
    version: str | None = None
    error: str | None = None


@dataclass
class TorchInfo:
    available: bool
    version: str | None = None
    cuda_available: bool = False
    mps_available: bool = False
    error: str | None = None


@dataclass
class Diagnostics:
    python_version: str
    platform_system: str
    macos_version: str | None
    cpu_architecture: str
    processor: str
    is_apple_silicon: bool
    is_intel: bool
    ffmpeg: FFmpegInfo
    opencv: LibraryInfo
    torch: TorchInfo
    recommended_execution_mode: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _detect_ffmpeg() -> FFmpegInfo:
    """Locate ffmpeg and read its version without raising on failure."""
    path = shutil.which("ffmpeg")
    if path is None:
        return FFmpegInfo(available=False, error="ffmpeg not found on PATH")
    try:
        completed = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return FFmpegInfo(available=True, path=path, error=f"failed to run ffmpeg: {exc}")

    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    version = None
    parts = first_line.split()
    if len(parts) >= 3 and parts[0] == "ffmpeg" and parts[1] == "version":
        version = parts[2]
    return FFmpegInfo(available=True, path=path, version=version or first_line or None)


def _detect_opencv() -> LibraryInfo:
    try:
        import cv2  # noqa: PLC0415  (imported lazily so missing OpenCV degrades gracefully)
    except Exception as exc:  # pragma: no cover - defensive: import errors vary by platform
        return LibraryInfo(available=False, error=f"opencv import failed: {exc}")
    return LibraryInfo(available=True, version=getattr(cv2, "__version__", None))


def _detect_torch() -> TorchInfo:
    try:
        import torch  # noqa: PLC0415  (imported lazily so missing PyTorch degrades gracefully)
    except Exception as exc:  # pragma: no cover - defensive: import errors vary by platform
        return TorchInfo(available=False, error=f"torch import failed: {exc}")

    cuda_available = False
    mps_available = False
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - backend probing can raise on odd builds
        cuda_available = False
    try:
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend is not None and mps_backend.is_available())
    except Exception:  # pragma: no cover
        mps_available = False

    return TorchInfo(
        available=True,
        version=getattr(torch, "__version__", None),
        cuda_available=cuda_available,
        mps_available=mps_available,
    )


def _detect_macos_version() -> str | None:
    """Return the macOS product version (e.g. "15.7.4"), or None off macOS.

    ``platform.mac_ver()`` reports a legacy "10.16" under the
    ``SYSTEM_VERSION_COMPAT`` shim for Pythons built against an older SDK, so we
    prefer ``sw_vers -productVersion`` and fall back to ``platform.mac_ver()``.
    """
    if platform.system() != "Darwin":
        return None
    sw_vers = shutil.which("sw_vers")
    if sw_vers is not None:
        try:
            completed = subprocess.run(
                [sw_vers, "-productVersion"],
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT_SECONDS,
                check=True,
            )
            version = completed.stdout.strip()
            if version:
                return version
        except (subprocess.SubprocessError, OSError):
            pass
    release = platform.mac_ver()[0]
    return release or None


def _is_apple_silicon(machine: str, processor: str) -> bool:
    """Return True on Apple Silicon (arm64) Macs.

    ``platform.machine()`` reports ``arm64`` on native Apple Silicon. Under
    Rosetta 2 it can report ``x86_64``; the processor string is checked as a
    secondary signal for that case.
    """
    if platform.system() != "Darwin":
        return False
    if machine.lower() == "arm64":
        return True
    return "apple" in processor.lower()


def _recommend_execution_mode(
    *, cuda_available: bool, mps_available: bool, is_apple_silicon: bool
) -> str:
    if cuda_available:
        return "cuda"
    if mps_available and is_apple_silicon:
        return "mps"
    return "cpu"


def collect_diagnostics() -> Diagnostics:
    """Gather a full environment snapshot. Never raises for optional tools."""
    machine = platform.machine()
    processor = platform.processor() or ""
    is_apple = _is_apple_silicon(machine, processor)
    is_intel = platform.system() == "Darwin" and not is_apple and machine.lower() == "x86_64"

    ffmpeg = _detect_ffmpeg()
    opencv = _detect_opencv()
    torch_info = _detect_torch()

    execution_mode = _recommend_execution_mode(
        cuda_available=torch_info.cuda_available,
        mps_available=torch_info.mps_available,
        is_apple_silicon=is_apple,
    )

    notes: list[str] = []
    if not ffmpeg.available:
        notes.append(
            "FFmpeg is required for video ingestion; install it (e.g. `brew install ffmpeg`)."
        )
    if not opencv.available:
        notes.append(
            "OpenCV is unavailable; it is an optional extra (needed from Milestone 3). "
            "Install with `uv sync --extra video`."
        )
    if not torch_info.available:
        notes.append("PyTorch is unavailable; run `uv sync` to install project dependencies.")
    if execution_mode == "cpu":
        notes.append("Running in CPU mode. This matches the Intel-Mac MVP target.")

    return Diagnostics(
        python_version=platform.python_version(),
        platform_system=platform.system(),
        macos_version=_detect_macos_version(),
        cpu_architecture=machine,
        processor=processor,
        is_apple_silicon=is_apple,
        is_intel=is_intel,
        ffmpeg=ffmpeg,
        opencv=opencv,
        torch=torch_info,
        recommended_execution_mode=execution_mode,
        notes=notes,
    )
