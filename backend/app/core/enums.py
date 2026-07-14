"""Controlled vocabularies for pitchers, videos, and clips.

String-valued enums so they persist as readable text in SQLite and serialize
cleanly over the API.
"""

from __future__ import annotations

from enum import StrEnum


class Handedness(StrEnum):
    LEFT = "L"
    RIGHT = "R"


class PitchType(StrEnum):
    FOUR_SEAM = "four_seam_fastball"
    SINKER = "sinker"
    SLIDER = "slider"
    CURVEBALL = "curveball"
    CHANGEUP = "changeup"
    CUTTER = "cutter"
    SPLITTER = "splitter"


class DeliveryType(StrEnum):
    WINDUP = "windup"
    STRETCH = "stretch"


class BatterSide(StrEnum):
    LEFT = "L"
    RIGHT = "R"


class QualityStatus(StrEnum):
    PENDING = "pending"
    USABLE = "usable"
    REJECTED = "rejected"


class LabelSource(StrEnum):
    MANUAL = "manual"
    STATCAST = "statcast"
    IMPORTED = "imported"


class ReleaseFrameSource(StrEnum):
    """Provenance of a clip's release frame (CLAUDE.md §9).

    ``auto`` labels must never silently overwrite ``manual`` or
    ``auto_confirmed`` ones.
    """

    MANUAL = "manual"
    AUTO = "auto"
    AUTO_CONFIRMED = "auto_confirmed"


class JobStatus(StrEnum):
    """Lifecycle of a long-running local background job (CLAUDE.md §16)."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
