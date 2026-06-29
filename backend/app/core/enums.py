"""Controlled vocabularies for pitchers, videos, and clips.

String-valued enums so they persist as readable text in SQLite and serialize
cleanly over the API.
"""

from __future__ import annotations

from enum import Enum


class Handedness(str, Enum):
    LEFT = "L"
    RIGHT = "R"


class PitchType(str, Enum):
    FOUR_SEAM = "four_seam_fastball"
    SINKER = "sinker"
    SLIDER = "slider"
    CURVEBALL = "curveball"
    CHANGEUP = "changeup"
    CUTTER = "cutter"
    SPLITTER = "splitter"


class DeliveryType(str, Enum):
    WINDUP = "windup"
    STRETCH = "stretch"


class BatterSide(str, Enum):
    LEFT = "L"
    RIGHT = "R"


class QualityStatus(str, Enum):
    PENDING = "pending"
    USABLE = "usable"
    REJECTED = "rejected"


class LabelSource(str, Enum):
    MANUAL = "manual"
    STATCAST = "statcast"
    IMPORTED = "imported"
