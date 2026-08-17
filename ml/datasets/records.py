"""The minimal clip descriptor used by dataset assembly (CLAUDE.md §8, §10).

``ClipRecord`` is a leak-free projection of a :class:`~backend.app.db.models.PitchClip`:
it carries only the fields needed to build safe splits and audit a dataset, and
deliberately **excludes** anything derived from at/after ball release. It is a
plain dataclass so splitting and auditing can be unit-tested without a database
or any third-party dependency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClipRecord:
    """A single labelled clip, projected for splitting and auditing.

    Attributes:
        clip_id: Stable identifier of the clip (``PitchClip.id``).
        game_id: The game the clip comes from. Splitting groups by this value,
            so it must be present — a clip with no game cannot be assigned to a
            split without risking cross-split leakage.
        pitch_type: The label being predicted (e.g. ``"slider"``).
        source_checksum: Checksum of the *source video* the clip was cut from.
            Clips sharing a checksum are near-duplicate footage and must never
            be separated across splits (CLAUDE.md §10).
        camera_angle: Optional camera-angle tag, used only for audit reporting.
    """

    clip_id: int
    game_id: str
    pitch_type: str
    source_checksum: str | None = None
    camera_angle: str | None = None
