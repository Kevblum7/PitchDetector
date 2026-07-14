"""Run pose extraction for a labeled clip from the command line.

Synchronous equivalent of ``POST /api/v1/clips/{id}/pose`` — useful for the
first real-model run (the ~230 MB Keypoint R-CNN weights download shows
progress here) and for scripted workflows.

Usage:
    uv run python scripts/estimate_pose.py --clip-id 1 [--render]
"""

from __future__ import annotations

import argparse
import sys

from sqlmodel import Session

from backend.app.db.models import PitchClip, SourceVideo
from backend.app.db.session import engine, init_db
from backend.app.services.pose_extraction import (
    PoseExtractionError,
    extract_and_store_pose,
    get_pose_estimator,
    render_clip_pose_overlay,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-id", type=int, required=True, help="PitchClip id to process")
    parser.add_argument(
        "--render",
        action="store_true",
        help="also render the annotated overlay MP4 to artifacts/overlays/",
    )
    args = parser.parse_args(argv)

    init_db()
    with Session(engine) as session:
        clip = session.get(PitchClip, args.clip_id)
        if clip is None:
            print(f"error: clip {args.clip_id} not found", file=sys.stderr)
            return 1
        video = session.get(SourceVideo, clip.source_video_id)
        if video is None:
            print(f"error: source video {clip.source_video_id} not found", file=sys.stderr)
            return 1

        n_frames = clip.pre_release_end_frame - clip.start_frame + 1
        print(
            f"clip {clip.id}: frames [{clip.start_frame}, {clip.pre_release_end_frame}] "
            f"({n_frames} pre-release frames, release at {clip.release_frame})"
        )
        print("extracting pose (first run downloads ~230 MB of model weights)...")

        try:
            result = extract_and_store_pose(session, clip, video, get_pose_estimator())
        except PoseExtractionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        q = result.quality
        print(
            f"done: {q.detected_frames}/{q.total_frames} frames with a detection, "
            f"max miss streak {q.max_consecutive_misses}, "
            f"max box jump {q.max_center_jump_px:.0f}px"
        )
        print(f"tracking_ok: {q.tracking_ok}")
        for issue in q.issues:
            print(f"  issue: {issue}")

        if args.render:
            output = render_clip_pose_overlay(session, clip, video)
            print(f"overlay written to {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
