"""Bounding-box primitives for pitcher tracking (pure, unit-testable).

All coordinates are in **original-video pixels** with the origin at the top
left. Boxes are ``(x, y, width, height)`` floats; conversions for display or
model input happen at the edges (CLAUDE.md §13: overlay coordinates are stored
in original-video coordinates).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


class GeometryError(ValueError):
    """Raised for degenerate boxes or invalid geometry parameters."""


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise GeometryError(f"box must have positive size, got {self}")

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def expand(self, factor: float, *, min_size: float = 0.0) -> BoundingBox:
        """Scale the box about its centre by ``factor`` (>= sides of min_size)."""
        if factor <= 0:
            raise GeometryError(f"expand factor must be positive, got {factor}")
        new_w = max(self.width * factor, min_size)
        new_h = max(self.height * factor, min_size)
        cx, cy = self.center
        return BoundingBox(x=cx - new_w / 2.0, y=cy - new_h / 2.0, width=new_w, height=new_h)

    def clamp(self, frame_width: int, frame_height: int) -> BoundingBox:
        """Clip the box to the frame, preserving as much of it as possible."""
        x1 = min(max(self.x, 0.0), frame_width - 1.0)
        y1 = min(max(self.y, 0.0), frame_height - 1.0)
        x2 = min(max(self.x2, x1 + 1.0), float(frame_width))
        y2 = min(max(self.y2, y1 + 1.0), float(frame_height))
        return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def shift(self, dx: float, dy: float) -> BoundingBox:
        return BoundingBox(x=self.x + dx, y=self.y + dy, width=self.width, height=self.height)

    def to_xyxy_int(self) -> tuple[int, int, int, int]:
        """Integer pixel corners (x1, y1, x2, y2) for cropping/drawing."""
        return (int(round(self.x)), int(round(self.y)), int(round(self.x2)), int(round(self.y2)))

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> BoundingBox:
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            width=float(data["width"]),
            height=float(data["height"]),
        )


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection over union of two boxes; 0.0 when disjoint."""
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / (a.area + b.area - inter)


def smooth_boxes(boxes: Sequence[BoundingBox], window: int) -> list[BoundingBox]:
    """Centred moving average over box coordinates (offline smoothing).

    ``window`` must be odd and >= 1; at the ends of the sequence the window
    shrinks symmetrically, so the first and last boxes are averaged over fewer
    neighbours rather than padded.
    """
    if window < 1 or window % 2 == 0:
        raise GeometryError(f"smoothing window must be odd and >= 1, got {window}")
    if window == 1 or len(boxes) <= 1:
        return list(boxes)

    half = window // 2
    smoothed: list[BoundingBox] = []
    for i in range(len(boxes)):
        # Shrink the half-window near the edges so the average stays centred.
        reach = min(half, i, len(boxes) - 1 - i)
        neighbourhood = boxes[i - reach : i + reach + 1]
        n = len(neighbourhood)
        smoothed.append(
            BoundingBox(
                x=sum(b.x for b in neighbourhood) / n,
                y=sum(b.y for b in neighbourhood) / n,
                width=sum(b.width for b in neighbourhood) / n,
                height=sum(b.height for b in neighbourhood) / n,
            )
        )
    return smoothed


def max_center_jump(boxes: Sequence[BoundingBox]) -> float:
    """Largest frame-to-frame centre displacement in pixels (0.0 if < 2 boxes).

    Used as a tracking-quality signal: a physically plausible pitcher does not
    teleport between consecutive frames.
    """
    worst = 0.0
    for prev, cur in zip(boxes, boxes[1:], strict=False):
        (px, py), (cx, cy) = prev.center, cur.center
        jump = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        worst = max(worst, jump)
    return worst
