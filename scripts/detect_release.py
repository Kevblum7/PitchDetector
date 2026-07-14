"""Batch automatic release-frame labeling (CLAUDE.md §9, Milestone 3.5).

Detects the release frame in registered videos and (optionally) creates
auto-labeled clips. Designed for the Film Room workflow: one pitch per file,
downloaded per pitch type, so a whole batch shares one label.

Usage:
    # Detect and print, one video
    uv run python scripts/detect_release.py --video-id 3

    # Detect + create an auto-labeled clip for one video
    uv run python scripts/detect_release.py --video-id 3 --create-clip --pitch-type slider

    # Batch: every video that has no clips yet
    uv run python scripts/detect_release.py --all-unclipped --create-clip --pitch-type slider

Automatic labels never overwrite manual ones; clips are only created for
videos, never modified. Low-confidence detections are flagged for review.
"""

from __future__ import annotations

import argparse
import sys

from sqlmodel import Session, col, select

from backend.app.core.config import DEFAULT_RELEASE_GUARD_FRAMES
from backend.app.core.enums import PitchType, ReleaseFrameSource
from backend.app.db.models import PitchClip, SourceVideo
from backend.app.db.session import engine, init_db
from backend.app.services.clip_frames import ClipFrameError, build_clip_window
from backend.app.services.pose_extraction import get_pose_estimator
from backend.app.services.release_detection import detect_release_for_video


def _target_videos(session: Session, args: argparse.Namespace) -> list[SourceVideo]:
    if args.video_id is not None:
        video = session.get(SourceVideo, args.video_id)
        if video is None:
            print(f"error: video {args.video_id} not found", file=sys.stderr)
            return []
        return [video]
    # --all-unclipped: videos with no clips yet.
    clipped_ids = set(session.exec(select(col(PitchClip.source_video_id)).distinct()).all())
    videos = list(session.exec(select(SourceVideo)).all())
    return [v for v in videos if v.id not in clipped_ids]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--video-id", type=int, help="detect for one SourceVideo id")
    target.add_argument(
        "--all-unclipped",
        action="store_true",
        help="detect for every video that has no clips yet",
    )
    parser.add_argument(
        "--create-clip",
        action="store_true",
        help="create an auto-labeled PitchClip from each successful detection",
    )
    parser.add_argument(
        "--pitch-type",
        type=PitchType,
        choices=list(PitchType),
        help="pitch label for created clips (required with --create-clip)",
    )
    args = parser.parse_args(argv)

    if args.create_clip and args.pitch_type is None:
        parser.error("--create-clip requires --pitch-type")

    init_db()
    estimator = get_pose_estimator()
    failures = 0
    with Session(engine) as session:
        videos = _target_videos(session, args)
        if not videos:
            print("no target videos found")
            return 1
        print(f"processing {len(videos)} video(s)...")

        for video in videos:
            print(f"\nvideo {video.id}: {video.file_path}")
            try:
                detection = detect_release_for_video(video, estimator)
            except Exception as exc:
                print(f"  detection FAILED: {exc}", file=sys.stderr)
                failures += 1
                continue

            flag = "  [NEEDS REVIEW]" if detection.review_recommended else ""
            print(
                f"  release frame {detection.release_frame} "
                f"(confidence {detection.confidence:.2f}, {detection.wrist} wrist, "
                f"peak {detection.peak_speed_px_per_frame:.1f} px/frame){flag}"
            )

            if not args.create_clip:
                continue
            try:
                window = build_clip_window(
                    start_frame=0,
                    release_frame=detection.release_frame,
                    release_guard_frames=DEFAULT_RELEASE_GUARD_FRAMES,
                    video_frame_count=video.frame_count,
                )
            except ClipFrameError as exc:
                print(f"  clip NOT created: {exc}", file=sys.stderr)
                failures += 1
                continue
            assert video.id is not None
            clip = PitchClip(
                source_video_id=video.id,
                pitcher_id=video.pitcher_id,
                game_id=video.game_id,
                start_frame=window.start_frame,
                end_frame=window.end_frame,
                release_frame=window.release_frame,
                pre_release_end_frame=window.pre_release_end_frame,
                release_guard_frames=window.release_guard_frames,
                release_frame_source=ReleaseFrameSource.AUTO,
                release_frame_confidence=detection.confidence,
                pitch_type=args.pitch_type,
                camera_angle=video.camera_angle,
            )
            session.add(clip)
            session.commit()
            session.refresh(clip)
            print(f"  created clip {clip.id} (pitch_type={args.pitch_type}, source=auto)")

    if failures:
        print(f"\ndone with {failures} failure(s)", file=sys.stderr)
        return 1
    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
