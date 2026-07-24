"""Download pitch clips from Baseball Savant for a single pitcher.

Why this exists
---------------
The Statcast Search *results page* (``/statcast_search``) renders each player's
pitches in a collapsed row. The individual clip URLs are only injected by
JavaScript after you expand a player (e.g. click "Devin Williams"), so a script
that fetches the raw HTML of that page finds **no video URLs**. That is the
"no urls" error.

This script avoids the rendered page entirely and uses the two stable data
endpoints instead:

1. ``/statcast_search/csv`` with ``type=details`` returns one row per pitch,
   including a ``play_id`` UUID. No "expand the player" click is involved.
2. ``/sporty-videos?playId=<play_id>`` is a small page whose *initial* HTML
   contains the ``<video><source src="...mp4">`` — the downloadable clip.

So the flow is: CSV -> list of ``play_id`` -> per-play mp4 URL -> download.

Usage
-----
Recommended: build your query on baseballsavant.mlb.com, copy the browser URL,
and pass it. The script rewrites it to the CSV endpoint automatically::

    uv run python scripts/download_savant_videos.py \\
        --search-url "https://baseballsavant.mlb.com/statcast_search?..." \\
        --out data/raw/devin_williams

Or specify filters directly (Devin Williams = 642207)::

    uv run python scripts/download_savant_videos.py \\
        --player-id 642207 --season 2024 --pitch-type SL \\
        --out data/raw/devin_williams

Add ``--dry-run`` to resolve URLs without downloading, or ``--limit N`` to cap
the number of clips while testing.

Legal / etiquette note
----------------------
This is a *local, manual* research helper, not application functionality
(CLAUDE.md lists automatic scraping as a non-goal). It requests one clip at a
time with a polite delay and identifies itself via User-Agent. Respect Baseball
Savant's terms and don't hammer the endpoints. Video is copyrighted MLB footage;
keep it local and document provenance.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger("savant_download")

SAVANT_HOST = "https://baseballsavant.mlb.com"
CSV_PATH = "/statcast_search/csv"
SEARCH_PATH = "/statcast_search"
VIDEO_PAGE = SAVANT_HOST + "/sporty-videos"

# A descriptive User-Agent; the default urllib UA is sometimes rejected.
USER_AGENT = "PitchTipDetector/0.1 (research; local dataset ingestion)"

# Matches the mp4 source in a sporty-videos page, e.g.
#   <source src="https://sporty-clips.mlb.com/....mp4" type="video/mp4">
_MP4_RE = re.compile(r'https?://[^\s"\'<>]+?\.mp4', re.IGNORECASE)


@dataclass
class PitchRow:
    """One pitch parsed from the Statcast details CSV (only fields we use)."""

    play_id: str
    game_date: str = ""
    game_pk: str = ""
    pitch_type: str = ""
    pitcher: str = ""
    des: str = ""
    video_url: str | None = None
    saved_path: str | None = None


@dataclass
class DownloadSummary:
    csv_url: str
    total_rows: int = 0
    with_play_id: int = 0
    resolved_video: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    failed: int = 0
    rows: list[PitchRow] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Pure helpers (no network) — these are what the unit tests exercise.
# --------------------------------------------------------------------------- #


def build_csv_url(
    *,
    search_url: str | None = None,
    player_id: str | int | None = None,
    season: str | int | None = None,
    pitch_type: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    player_type: str = "pitcher",
) -> str:
    """Return a Statcast ``/statcast_search/csv`` URL that yields per-pitch rows.

    If ``search_url`` is given (a URL copied from the browser), its query is
    preserved and only rewritten to the CSV endpoint with ``type=details`` and
    ``all=true`` forced on. Otherwise a query is built from the explicit filters.
    """
    if search_url:
        parts = urllib.parse.urlsplit(search_url)
        # Accept either the results page or an already-CSV URL; the results
        # page path is rewritten to the CSV endpoint.
        path = CSV_PATH if parts.path.rstrip("/") == SEARCH_PATH else parts.path
        query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        # Force the params that make the response a per-pitch CSV.
        query = [(k, v) for (k, v) in query if k not in {"type", "all"}]
        query.extend([("type", "details"), ("all", "true")])
        return urllib.parse.urlunsplit(
            (
                parts.scheme or "https",
                parts.netloc or "baseballsavant.mlb.com",
                path,
                urllib.parse.urlencode(query),
                "",
            )
        )

    if player_id is None:
        raise ValueError("Provide either --search-url or --player-id.")

    # Build a minimal working query from explicit filters.
    params: list[tuple[str, str]] = [
        ("all", "true"),
        ("player_type", player_type),
        ("pitchers_lookup[]", str(player_id)),
        ("type", "details"),
    ]
    if season is not None:
        # Savant expects the season list to end with a trailing pipe.
        params.append(("hfSea", f"{season}|"))
    if pitch_type:
        params.append(("hfPT", f"{pitch_type}|"))
    if start_date:
        params.append(("game_date_gt", start_date))
    if end_date:
        params.append(("game_date_lt", end_date))
    return f"{SAVANT_HOST}{CSV_PATH}?" + urllib.parse.urlencode(params)


def parse_pitch_rows(csv_text: str) -> list[PitchRow]:
    """Parse the details CSV into rows that carry a non-empty ``play_id``."""
    reader = csv.DictReader(io.StringIO(csv_text))
    rows: list[PitchRow] = []
    for raw in reader:
        play_id = (raw.get("play_id") or "").strip()
        if not play_id:
            continue
        rows.append(
            PitchRow(
                play_id=play_id,
                game_date=(raw.get("game_date") or "").strip(),
                game_pk=(raw.get("game_pk") or "").strip(),
                pitch_type=(raw.get("pitch_type") or "").strip(),
                pitcher=(raw.get("player_name") or "").strip(),
                des=(raw.get("des") or "").strip(),
            )
        )
    return rows


def extract_video_url(html: str) -> str | None:
    """Extract the mp4 clip URL from a sporty-videos page's HTML."""
    match = _MP4_RE.search(html)
    return match.group(0) if match else None


def video_page_url(play_id: str) -> str:
    """Return the sporty-videos page URL for a given play id."""
    return f"{VIDEO_PAGE}?playId={urllib.parse.quote(play_id)}"


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def clip_filename(row: PitchRow) -> str:
    """Deterministic, filesystem-safe filename for a pitch clip."""
    stem_parts = [p for p in (row.game_date, row.pitch_type, row.play_id) if p]
    stem = "_".join(stem_parts) or row.play_id
    stem = _UNSAFE.sub("-", stem).strip("-")
    return f"{stem}.mp4"


# --------------------------------------------------------------------------- #
# Network I/O — isolated so tests can avoid it.
# --------------------------------------------------------------------------- #


def _request(url: str, *, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted host)
        return resp.read()


def fetch_text(url: str, *, timeout: float = 30.0) -> str:
    return _request(url, timeout=timeout).decode("utf-8", errors="replace")


def download_to(url: str, dest: Path, *, timeout: float = 60.0) -> int:
    """Download ``url`` to ``dest`` atomically. Returns bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    data = _request(url, timeout=timeout)
    tmp.write_bytes(data)
    tmp.replace(dest)
    return len(data)


def resolve_and_download(
    rows: list[PitchRow],
    out_dir: Path,
    *,
    delay: float = 1.0,
    dry_run: bool = False,
    overwrite: bool = False,
) -> DownloadSummary:
    """Resolve each play's mp4 URL and download it, skipping ones already saved."""
    summary = DownloadSummary(csv_url="", rows=rows)
    summary.total_rows = len(rows)
    summary.with_play_id = len(rows)

    for i, row in enumerate(rows, start=1):
        dest = out_dir / clip_filename(row)
        if dest.exists() and not overwrite:
            row.saved_path = str(dest)
            summary.skipped_existing += 1
            logger.info("[%d/%d] skip (exists): %s", i, len(rows), dest.name)
            continue
        try:
            page = fetch_text(video_page_url(row.play_id))
            video = extract_video_url(page)
            if not video:
                summary.failed += 1
                logger.warning("[%d/%d] no mp4 for play %s", i, len(rows), row.play_id)
                continue
            row.video_url = video
            summary.resolved_video += 1
            if dry_run:
                logger.info("[%d/%d] resolved: %s", i, len(rows), video)
            else:
                size = download_to(video, dest)
                row.saved_path = str(dest)
                summary.downloaded += 1
                logger.info("[%d/%d] saved %s (%d bytes)", i, len(rows), dest.name, size)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            summary.failed += 1
            logger.warning("[%d/%d] failed play %s: %s", i, len(rows), row.play_id, exc)
        if delay > 0 and i < len(rows):
            time.sleep(delay)

    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Baseball Savant pitch clips for one pitcher.",
    )
    parser.add_argument(
        "--search-url",
        help="A statcast_search URL copied from the browser (rewritten to CSV).",
    )
    parser.add_argument("--player-id", help="MLBAM player id, e.g. 642207 (Devin Williams).")
    parser.add_argument("--season", help="Season year, e.g. 2024.")
    parser.add_argument("--pitch-type", help="Pitch type code, e.g. SL, FF, CH.")
    parser.add_argument("--start-date", help="Inclusive start date YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Inclusive end date YYYY-MM-DD.")
    parser.add_argument(
        "--out",
        default="data/raw/savant",
        help="Output directory for clips + manifest (default: data/raw/savant).",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N pitches.")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between clips (default: 1.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve mp4 URLs and write the manifest, but do not download video.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing clips.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)

    try:
        csv_url = build_csv_url(
            search_url=args.search_url,
            player_id=args.player_id,
            season=args.season,
            pitch_type=args.pitch_type,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    logger.info("Fetching pitch list: %s", csv_url)
    try:
        csv_text = fetch_text(csv_url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        logger.error("Failed to fetch CSV: %s", exc)
        return 1

    rows = parse_pitch_rows(csv_text)
    if not rows:
        logger.error(
            "CSV returned 0 pitches with a play_id. Check the query (player id / "
            "season / dates). If you passed --search-url, make sure it is a "
            "statcast_search URL from the browser."
        )
        return 1
    if args.limit:
        rows = rows[: args.limit]
    logger.info("Found %d pitches with play_id.", len(rows))

    out_dir = Path(args.out)
    summary = resolve_and_download(
        rows,
        out_dir,
        delay=args.delay,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    summary.csv_url = csv_url

    manifest = out_dir / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "csv_url": summary.csv_url,
                "total_rows": summary.total_rows,
                "resolved_video": summary.resolved_video,
                "downloaded": summary.downloaded,
                "skipped_existing": summary.skipped_existing,
                "failed": summary.failed,
                "clips": [asdict(r) for r in summary.rows],
            },
            indent=2,
        )
    )
    logger.info(
        "Done. resolved=%d downloaded=%d skipped=%d failed=%d -> %s",
        summary.resolved_video,
        summary.downloaded,
        summary.skipped_existing,
        summary.failed,
        manifest,
    )
    return 0 if summary.failed == 0 or summary.downloaded or summary.skipped_existing else 1


if __name__ == "__main__":
    sys.exit(main())
