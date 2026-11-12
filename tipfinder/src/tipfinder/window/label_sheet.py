"""Contact sheets for hand-labelling set position and hand separation.

Each tile is a pitcher crop with its clip frame index burned in, so a human can read off
the first frame where the hands are together (set) and the first frame they part.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tipfinder.pose.base import J, PoseSequence
from tipfinder.video.shots import read_video


def pitcher_box(seq: PoseSequence, pad: float = 0.35) -> tuple[int, int, int, int]:
    pts = seq.kp[:, :, :2].reshape(-1, 2)
    pts = pts[~np.isnan(pts).any(axis=1)]
    x0, y0 = np.percentile(pts, 2, axis=0)
    x1, y1 = np.percentile(pts, 98, axis=0)
    w, h = x1 - x0, y1 - y0
    return (
        int(max(0, x0 - pad * w)),
        int(max(0, y0 - 0.1 * h)),
        int(min(seq.width, x1 + pad * w)),
        int(min(seq.height, y1 + 0.05 * h)),
    )


def make_sheet(
    clip: Path,
    pose: Path,
    out: Path,
    start: int,
    end: int,
    step: int = 2,
    cols: int = 8,
    draw_wrists: bool = False,
) -> Path:
    frames, _ = read_video(clip)
    seq = PoseSequence.load(pose)
    x0, y0, x1, y1 = pitcher_box(seq)
    tiles = []
    for i in range(start, min(end, len(frames)), step):
        tile = frames[i][y0:y1, x0:x1].copy()
        if draw_wrists:
            t = np.where(seq.frame_idx == i)[0]
            if len(t) and not np.isnan(seq.kp[t[0], 0, 0]):
                for j in ("left_wrist", "right_wrist"):
                    x, y = seq.kp[t[0], J[j], :2]
                    cv2.circle(tile, (int(x - x0), int(y - y0)), 4, (0, 255, 255), -1)
        tile = cv2.resize(tile, (200, int(200 * tile.shape[0] / tile.shape[1])))
        cv2.putText(tile, str(i), (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r : r + cols]) for r in range(0, len(tiles), cols)]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    return out
