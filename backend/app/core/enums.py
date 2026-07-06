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
