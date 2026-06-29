"""Command-line environment diagnostics.

Usage:
    uv run python scripts/diagnostics.py          # human-readable
    uv run python scripts/diagnostics.py --json    # machine-readable
    uv run pitch-diagnostics                       # installed console script
"""

from __future__ import annotations

import argparse
import json
import sys

from backend.app.core.diagnostics import Diagnostics, collect_diagnostics


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _format_human(diag: Diagnostics) -> str:
    lines = [
        "Pitch Tip Detector — Environment Diagnostics",
        "=" * 44,
        f"Python version        : {diag.python_version}",
        f"Platform              : {diag.platform_system}",
        f"macOS version         : {diag.macos_version or 'n/a'}",
        f"CPU architecture      : {diag.cpu_architecture}",
        f"Processor             : {diag.processor or 'n/a'}",
        f"Apple Silicon         : {_yes_no(diag.is_apple_silicon)}",
        f"Intel                 : {_yes_no(diag.is_intel)}",
        "",
        f"FFmpeg available      : {_yes_no(diag.ffmpeg.available)}",
        f"FFmpeg version        : {diag.ffmpeg.version or 'n/a'}",
        f"FFmpeg path           : {diag.ffmpeg.path or 'n/a'}",
        f"OpenCV available      : {_yes_no(diag.opencv.available)}",
        f"OpenCV version        : {diag.opencv.version or 'n/a'}",
        f"PyTorch available     : {_yes_no(diag.torch.available)}",
        f"PyTorch version       : {diag.torch.version or 'n/a'}",
        f"CUDA available        : {_yes_no(diag.torch.cuda_available)}",
        f"MPS available         : {_yes_no(diag.torch.mps_available)}",
        "",
        f"Recommended execution : {diag.recommended_execution_mode}",
    ]
    if diag.notes:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"  - {note}" for note in diag.notes)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report the local runtime environment.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args(argv)

    diag = collect_diagnostics()
    if args.json:
        print(json.dumps(diag.to_dict(), indent=2))
    else:
        print(_format_human(diag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
