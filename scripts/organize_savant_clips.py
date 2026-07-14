"""Rename + file user-downloaded Film Room clips to their Statcast identity.

You download each clip yourself through MLB Film Room's own **Download** button
(the only sanctioned path — this project never fetches MLB video; see
CLAUDE.md §5, §21). This script then does the tedious part: it matches each
downloaded file to its pitch in a Savant manifest and renames it to
``<clip_uid>.mp4`` in the right ``clips/<pitch_type>/`` folder.

Matching strategy, per source file:
  1. a Film Room ``playId`` GUID found in the filename -> the manifest row with
     that ``play_id`` (most reliable; Film Room embeds it);
  2. otherwise a ``clip_uid`` substring in the filename.

Every accepted file is **ffprobe-validated as a real, decodable video** before
it is filed, so an HTML error/block page saved as ``.mp4`` (a real failure mode
here) is rejected instead of poisoning the dataset.

The manifest's ``video_file`` column is updated in place so ingestion knows
which pitches now have footage, and ``--report`` lists what is still missing
with its Film Room link.

Usage:
    uv run python scripts/organize_savant_clips.py \
        --manifest savant_data/devin_williams/metadata/devin_williams_2026_FF.csv \
        --source ~/Downloads \
        --clips-dir savant_data/devin_williams/clips/FF \
        [--move] [--dry-run] [--report]
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path

_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".m4v"}
_GUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def _is_real_video(path: Path) -> bool:
    """True if ffprobe can read a video stream (rejects HTML block pages)."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise SystemExit("ffprobe not found on PATH; install FFmpeg (brew install ffmpeg)")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and "video" in result.stdout


def _load_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{str(k): str(v) for k, v in record.items()} for record in reader]
        fields = list(reader.fieldnames or [])
    if "video_file" not in fields:
        fields.append("video_file")
        for row in rows:
            row.setdefault("video_file", "")
    return rows, fields


def _write_manifest(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _match_row(source: Path, by_playid: dict[str, dict[str, str]], rows: list[dict[str, str]]):  # type: ignore[no-untyped-def]
    stem = source.stem.lower()
    for guid in _GUID_RE.findall(source.name):
        if guid.lower() in by_playid:
            return by_playid[guid.lower()]
    for row in rows:
        if row["clip_uid"].lower() in stem:
            return row
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="folder with downloaded clips")
    parser.add_argument("--clips-dir", type=Path, required=True, help="destination clips/<PT>/")
    parser.add_argument("--move", action="store_true", help="move instead of copy")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", action="store_true", help="list still-missing pitches + links")
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest}")
    if not args.source.is_dir():
        raise SystemExit(f"source folder not found: {args.source}")

    rows, fields = _load_manifest(args.manifest)
    by_playid = {r["play_id"].lower(): r for r in rows if r.get("play_id")}
    by_uid = {r["clip_uid"]: r for r in rows}
    args.clips_dir.mkdir(parents=True, exist_ok=True)

    filed = rejected = unmatched = 0
    source_files = sorted(p for p in args.source.iterdir() if p.suffix.lower() in _VIDEO_SUFFIXES)
    for source in source_files:
        row = _match_row(source, by_playid, rows)
        if row is None:
            print(f"  ? no manifest match: {source.name}")
            unmatched += 1
            continue
        if not _is_real_video(source):
            print(f"  x not a real video (block page / corrupt?): {source.name}")
            rejected += 1
            continue
        target = args.clips_dir / f"{row['clip_uid']}.mp4"
        action = "move" if args.move else "copy"
        print(f"  ok {source.name} -> {target}  [{action}]")
        if not args.dry_run:
            if args.move:
                shutil.move(str(source), target)
            else:
                shutil.copy2(source, target)
            row["video_file"] = str(target)
        filed += 1

    if not args.dry_run:
        _write_manifest(args.manifest, rows, fields)

    have = sum(1 for r in by_uid.values() if r.get("video_file"))
    print(
        f"\nfiled {filed}, rejected {rejected}, unmatched-in-source {unmatched}. "
        f"manifest now has video for {have}/{len(rows)} pitches."
    )

    if args.report:
        missing = [r for r in rows if not r.get("video_file")]
        print(f"\nstill needed ({len(missing)}): open each link, Download, save into {args.source}")
        for row in missing[:40]:
            print(f"  {row['clip_uid']}  {row['game_date']}  {row.get('filmroom_url', '')}")
        if len(missing) > 40:
            print(f"  ... and {len(missing) - 40} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
