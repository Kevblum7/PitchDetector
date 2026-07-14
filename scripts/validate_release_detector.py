"""Cross-reference gate: validate the auto release detector against manual marks.

Runs the automatic detector on every clip whose release frame was set manually
(or auto-confirmed) and reports per-clip frame error. This report is the
evidence required before auto labels are trusted at scale (CLAUDE.md §9:
target >= 90% of validation clips within +/- 1 frame).

Usage:
    uv run python scripts/validate_release_detector.py [--tolerance 1]
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, cast

from sqlmodel import Session

from backend.app.db.session import engine, init_db
from backend.app.services.pose_extraction import get_pose_estimator
from backend.app.services.release_detection import validate_release_detector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tolerance",
        type=int,
        default=1,
        help="frame error treated as agreement (default: 1)",
    )
    args = parser.parse_args(argv)

    init_db()
    with Session(engine) as session:
        report = validate_release_detector(
            session, get_pose_estimator(), frame_tolerance=args.tolerance
        )

    total = report["manual_clips_total"]
    compared = report["compared"]
    if total == 0:
        print(
            "no manually labeled clips to validate against — mark some release "
            "frames manually first (the cross-reference gate needs ground truth)",
            file=sys.stderr,
        )
        return 1

    fraction = cast("float | None", report["within_tolerance_fraction"])
    print(f"manually labeled clips: {total}")
    print(f"detector ran successfully on: {compared} (failed: {report['failed']})")
    if compared and fraction is not None:
        print(
            f"within +/-{args.tolerance} frame(s): {report['within_tolerance']}/{compared} "
            f"({fraction:.0%})"
        )
    print("\nper-clip results:")
    for entry in cast("list[dict[str, Any]]", report["clips"]):
        clip_id = entry["clip_id"]
        if "error" in entry:
            print(f"  clip {clip_id}: FAILED — {entry['error']}")
            continue
        print(
            f"  clip {clip_id}: manual {entry['manual_release_frame']} vs "
            f"auto {entry['detected_release_frame']} "
            f"(error {entry['frame_error']:+d}, confidence {entry['confidence']:.2f})"
        )
    print(f"\nreport written to {report['report_path']}")

    passed = compared != 0 and fraction is not None and fraction >= 0.9
    print("cross-reference gate:", "PASSED (>= 90% within tolerance)" if passed else "NOT PASSED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
