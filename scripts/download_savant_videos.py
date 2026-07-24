"""Download pitch clips from Baseball Savant for a single pitcher.

Why this exists
---------------
The Statcast Search *results page* (``/statcast_search``) renders each player's
pitches in a collapsed row. The individual clip URLs are only injected by
JavaScript after you expand a player (e.g. click "Devin Williams"), so a script
that fetches the raw HTML of that page finds **no video URLs**. That is the
"no urls" error.

This script avoids the rendered page entirely and uses stable data endpoints:

1. ``/statcast_search/csv`` with ``type=details`` returns one row per pitch.
   It identifies each pitch by ``game_pk`` + ``at_bat_number`` + ``pitch_number``
   but — importantly — the current CSV export does **not** contain the
   ``play_id`` UUID that the video page needs.
2. ``/gf?game_pk=<pk>`` is the per-game feed (JSON). Every pitch in it carries a
   ``play_id`` UUID, keyed by ``ab_number`` + ``pitch_number``. We fetch this
   once per game and join it back to the CSV rows to recover each ``play_id``.
3. ``/sporty-videos?playId=<play_id>`` is a small page whose *initial* HTML
   contains the ``<video><source src="...mp4">`` — the downloadable clip.

So the flow is:
CSV -> (game_pk, at_bat, pitch) -> per-game feed play_id -> mp4 URL -> download.

Usage
-----
Recommended: build your query on baseballsavant.mlb.com, copy the browser URL,
and pass it. The script rewrites it to the CSV endpoint automatically::

    uv run python scripts/download_savant_videos.py \\
        --search-url "https://baseballsavant.mlb.com/statcast_search?..." \\
        --out data/raw/devin_williams

Or specify filters directly. Devin Williams = 642207; note his 2024 arsenal is
four-seam (FF) and changeup (CH), not slider::

    uv run python scripts/download_savant_videos.py \\
        --player-id 642207 --season 2024 --pitch-type CH \\
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
import html
import io
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger("savant_download")

SAVANT_HOST = "https://baseballsavant.mlb.com"
CSV_PATH = "/statcast_search/csv"
SEARCH_PATH = "/statcast_search"
VIDEO_PAGE = SAVANT_HOST + "/sporty-videos"
GAME_FEED = SAVANT_HOST + "/gf"

# A descriptive User-Agent; the default urllib UA is sometimes rejected.
USER_AGENT = "PitchTipDetector/0.1 (research; local dataset ingestion)"

# Matches the mp4 source in a sporty-videos page, e.g.
#   <source src="https://sporty-clips.mlb.com/....mp4" type="video/mp4">
_MP4_RE = re.compile(r'https?://[^\s"\'<>]+?\.mp4', re.IGNORECASE)


@dataclass
class PitchRow:
    """One pitch parsed from the Statcast details CSV (only fields we use).

    ``play_id`` is not present in the CSV; it is resolved later from the
    per-game feed via the (``at_bat_number``, ``pitch_number``) key.
    """

    game_pk: str = ""
    at_bat_number: str = ""
    pitch_number: str = ""
    game_date: str = ""
    pitch_type: str = ""
    pitcher: str = ""
    des: str = ""
    play_id: str | None = None
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
    """Parse the details CSV into rows keyed by game/at-bat/pitch.

    The CSV is served with a UTF-8 BOM, which otherwise corrupts the first
    column name (``pitch_type``); it is stripped here. Rows are kept only when
    they carry the full join key (``game_pk``, ``at_bat_number``,
    ``pitch_number``) needed to look up the ``play_id`` from the game feed.
    """
    csv_text = csv_text.lstrip("﻿")
    reader = csv.DictReader(io.StringIO(csv_text))
    rows: list[PitchRow] = []
    for raw in reader:
        game_pk = (raw.get("game_pk") or "").strip()
        at_bat = (raw.get("at_bat_number") or "").strip()
        pitch_number = (raw.get("pitch_number") or "").strip()
        if not (game_pk and at_bat and pitch_number):
            continue
        rows.append(
            PitchRow(
                game_pk=game_pk,
                at_bat_number=at_bat,
                pitch_number=pitch_number,
                game_date=(raw.get("game_date") or "").strip(),
                pitch_type=(raw.get("pitch_type") or "").strip(),
                pitcher=(raw.get("player_name") or "").strip(),
                des=(raw.get("des") or "").strip(),
            )
        )
    return rows


def parse_game_feed(feed_json: str) -> dict[tuple[str, str], str]:
    """Map (``ab_number``, ``pitch_number``) -> ``play_id`` from a ``/gf`` feed.

    The feed splits pitches into ``team_home`` / ``team_away`` (by batting team).
    ``ab_number`` is sequential across the whole game, so (ab, pitch) is unique.
    """
    data = json.loads(feed_json)
    mapping: dict[tuple[str, str], str] = {}
    for side in ("team_home", "team_away"):
        for pitch in data.get(side) or []:
            play_id = pitch.get("play_id")
            if not play_id:
                continue
            key = (str(pitch.get("ab_number")), str(pitch.get("pitch_number")))
            mapping[key] = play_id
    return mapping


def extract_video_url(page_html: str) -> str | None:
    """Extract the mp4 clip URL from a sporty-videos page's HTML.

    The URL is HTML-escaped in the page (its base64 token ends with ``==``,
    rendered as ``&#x3D;&#x3D;``), so entities are unescaped before returning.
    """
    match = _MP4_RE.search(page_html)
    return html.unescape(match.group(0)) if match else None


def video_page_url(play_id: str) -> str:
    """Return the sporty-videos page URL for a given play id."""
    return f"{VIDEO_PAGE}?playId={urllib.parse.quote(play_id)}"


def game_feed_url(game_pk: str) -> str:
    """Return the per-game feed URL for a given game_pk."""
    return f"{GAME_FEED}?game_pk={urllib.parse.quote(str(game_pk))}"


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def clip_filename(row: PitchRow) -> str:
    """Deterministic, filesystem-safe filename for a pitch clip."""
    ident = row.play_id or "-".join(
        p for p in (row.game_pk, row.at_bat_number, row.pitch_number) if p
    )
    stem_parts = [p for p in (row.game_date, row.pitch_type, ident) if p]
    stem = "_".join(stem_parts) or ident or "clip"
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


def resolve_play_ids(rows: list[PitchRow], *, delay: float = 0.5) -> int:
    """Populate ``row.play_id`` for each row using the per-game feed.

    Rows are grouped by ``game_pk`` so each game feed is fetched only once.
    Returns the number of rows for which a ``play_id`` was resolved.
    """
    by_game: dict[str, list[PitchRow]] = defaultdict(list)
    for row in rows:
        by_game[row.game_pk].append(row)

    games = list(by_game.items())
    for i, (game_pk, group) in enumerate(games, start=1):
        try:
            feed = parse_game_feed(fetch_text(game_feed_url(game_pk)))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            logger.warning("[game %d/%d] feed failed for %s: %s", i, len(games), game_pk, exc)
            continue
        resolved_here = 0
        for row in group:
            row.play_id = feed.get((row.at_bat_number, row.pitch_number))
            if row.play_id:
                resolved_here += 1
        logger.info(
            "[game %d/%d] %s: resolved %d/%d play_ids",
            i,
            len(games),
            game_pk,
            resolved_here,
            len(group),
        )
        if delay > 0 and i < len(games):
            time.sleep(delay)

    return sum(1 for row in rows if row.play_id)


def resolve_and_download(
    rows: list[PitchRow],
    out_dir: Path,
    *,
    delay: float = 1.0,
    dry_run: bool = False,
    overwrite: bool = False,
) -> DownloadSummary:
    """Resolve each play's mp4 URL and download it, skipping ones already saved.

    ``rows`` must already have ``play_id`` populated (see ``resolve_play_ids``).
    """
    summary = DownloadSummary(csv_url="", rows=rows)
    summary.total_rows = len(rows)
    summary.with_play_id = sum(1 for r in rows if r.play_id)

    for i, row in enumerate(rows, start=1):
        if not row.play_id:
            summary.failed += 1
            logger.warning(
                "[%d/%d] no play_id for game %s ab %s pitch %s",
                i,
                len(rows),
                row.game_pk,
                row.at_bat_number,
                row.pitch_number,
            )
            continue
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
            "CSV returned 0 pitches. Check the query (player id / season / dates). "
            "If you passed --search-url, make sure it is a statcast_search URL from "
            "the browser."
        )
        return 1
    if args.limit:
        rows = rows[: args.limit]
    logger.info("Found %d pitches. Resolving play_ids via per-game feed...", len(rows))

    resolved = resolve_play_ids(rows, delay=min(args.delay, 1.0))
    if resolved == 0:
        logger.error(
            "Resolved 0 play_ids from the game feed(s). The feed format may have "
            "changed, or the (at_bat_number, pitch_number) join failed."
        )
        return 1
    logger.info("Resolved %d/%d play_ids.", resolved, len(rows))

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
                "with_play_id": summary.with_play_id,
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
        "Done. play_ids=%d resolved=%d downloaded=%d skipped=%d failed=%d -> %s",
        summary.with_play_id,
        summary.resolved_video,
        summary.downloaded,
        summary.skipped_existing,
        summary.failed,
        manifest,
    )
    return 0 if summary.downloaded or summary.skipped_existing or summary.resolved_video else 1


if __name__ == "__main__":
    sys.exit(main())
