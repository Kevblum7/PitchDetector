"""System / environment diagnostics endpoints (API v1)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend.app.core.diagnostics import collect_diagnostics

router = APIRouter(prefix="/api/v1/system", tags=["system"])


@router.get("/diagnostics")
def system_diagnostics() -> dict[str, Any]:
    """Return a full snapshot of the local runtime environment.

    Optional capabilities (FFmpeg, OpenCV, PyTorch) are reported as available
    or not; missing tools never cause this endpoint to fail.
    """
    return collect_diagnostics().to_dict()
