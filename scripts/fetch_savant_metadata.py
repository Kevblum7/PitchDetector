"""Fetch public Statcast pitch metadata for a pitcher and tie it to Film Room.

This tool fetches **metadata only** — factual, public Statcast pitch data plus
the Film Room ``playId`` for each pitch. It downloads **no video** and stores
no copyrighted footage. That boundary is deliberate (CLAUDE.md §5, §21):
MLB game video must be obtained by the user through Film Room's own download
UI. This script just makes that manual step painless by producing a per-pitch
manifest with a direct Film Room link and a stable ``clip_uid`` filename to
save each clip under.

Sources (both public research/data endpoints, widely used, no auth):
- Statcast search CSV:  baseballsavant.mlb.com/statcast_search/csv
- Play-by-play (playId): statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay

The Statcast search CSV has no ``playId``; it is recovered by joining on
``game_pk`` + ``at_bat_number`` + ``pitch_number`` against statsapi (where
``at_bat_number == atBatIndex + 1``).

Usage:
    uv run python scripts/fetch_savant_metadata.py \
        --player-id 642207 --season 2026 --name devin_williams \
        --pitch-types FF CH --out-dir savant_data
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

_SAVANT_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"
_STATSAPI_PBP = "https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay"
_FILMROOM_SEARCH = 'https://www.mlb.com/video/search?q=playid="{play_id}"'
_USER_AGENT = "pitch-tip-detector/metadata (research; contact via repo)"
_REQUEST_PAUSE_SECONDS = 0.4  # be polite to the public endpoints


def _get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()  # type: ignore[no-any-return]


def fetch_statcast_csv(player_id: int, season: int) -> list[dict[str, str]]:
    """Pull every pitch a pitcher threw in a season (all pitch types)."""
    params = {
        "all": "true",
        "hfSea": f"{season}|",
        "player_type": "pitcher",
        "pitchers_lookup[]": str(player_id),
        "type": "details",
    }
    url = f"{_SAVANT_CSV}?{urllib.parse.urlencode(params)}"
    text = _get(url).decode("utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise SystemExit(
            f"no Statcast rows for player {player_id} in {season} "
            "(wrong id/season, or no data yet?)"
        )
    return rows


def build_playid_map(game_pk: str) -> dict[tuple[int, int], str]:
    """Map (at_bat_number, pitch_number) -> Film Room playId for one game."""
    payload = json.loads(_get(_STATSAPI_PBP.format(game_pk=game_pk)))
    mapping: dict[tuple[int, int], str] = {}
    for play in payload.get("allPlays", []):
        at_bat_number = play["about"]["atBatIndex"] + 1  # Statcast is 1-based
        for event in play.get("playEvents", []):
            if event.get("isPitch") and event.get("playId") and event.get("pitchNumber"):
                mapping[(at_bat_number, int(event["pitchNumber"]))] = event["playId"]
    return mapping


def _clip_uid(row: dict[str, str]) -> str:
    return f"{row['game_pk']}_ab{row['at_bat_number']}_p{row['pitch_number']}"


# Columns carried into the enriched per-pitch manifest (kept small and
# label-focused; the raw CSV keeps everything).
_MANIFEST_COLUMNS = (
    "clip_uid",
    "play_id",
    "filmroom_url",
    "pitch_type",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "stand",
    "p_throws",
    "balls",
    "strikes",
    "release_speed",
    "description",
    "video_file",  # filled in by the ingest step once the clip is downloaded
)


def enrich(rows: list[dict[str, str]], pitch_type: str) -> list[dict[str, str]]:
    """Filter to one pitch type and attach playId + Film Room URL per pitch."""
    subset = [r for r in rows if r.get("pitch_type") == pitch_type]
    game_pks = sorted({r["game_pk"] for r in subset})
    play_maps: dict[str, dict[tuple[int, int], str]] = {}
    for i, game_pk in enumerate(game_pks, 1):
        print(f"    play-by-play {i}/{len(game_pks)} (game {game_pk})", file=sys.stderr)
        try:
            play_maps[game_pk] = build_playid_map(game_pk)
        except Exception as exc:  # a missing game is data, not a crash
            print(f"      warning: no play-by-play for {game_pk}: {exc}", file=sys.stderr)
            play_maps[game_pk] = {}
        time.sleep(_REQUEST_PAUSE_SECONDS)

    manifest: list[dict[str, str]] = []
    for row in subset:
        key = (int(row["at_bat_number"]), int(row["pitch_number"]))
        play_id = play_maps.get(row["game_pk"], {}).get(key, "")
        entry = {col: row.get(col, "") for col in _MANIFEST_COLUMNS}
        entry["clip_uid"] = _clip_uid(row)
        entry["play_id"] = play_id
        entry["filmroom_url"] = _FILMROOM_SEARCH.format(play_id=play_id) if play_id else ""
        entry["video_file"] = ""
        manifest.append(entry)
    manifest.sort(key=lambda e: (e["game_date"], e["clip_uid"]))
    return manifest


def _write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-id", type=int, required=True, help="MLBAM player id")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--name", required=True, help="slug for folders, e.g. devin_williams")
    parser.add_argument("--pitch-types", nargs="+", required=True, help="e.g. FF CH")
    parser.add_argument("--out-dir", default="savant_data")
    args = parser.parse_args(argv)

    base = Path(args.out_dir) / args.name
    meta_dir = base / "metadata"
    print(f"fetching {args.season} Statcast metadata for player {args.player_id}...")
    rows = fetch_statcast_csv(args.player_id, args.season)
    print(f"  {len(rows)} pitches across {len({r['game_pk'] for r in rows})} games")

    # Keep the full raw pull for provenance/reproducibility.
    raw_path = meta_dir / f"{args.name}_{args.season}_all.csv"
    _write_csv(raw_path, rows, list(rows[0].keys()))
    print(f"  wrote {raw_path}")

    for pitch_type in args.pitch_types:
        print(f"  enriching {pitch_type}...")
        manifest = enrich(rows, pitch_type)
        matched = sum(1 for e in manifest if e["play_id"])
        out = meta_dir / f"{args.name}_{args.season}_{pitch_type}.csv"
        _write_csv(out, manifest, list(_MANIFEST_COLUMNS))
        (base / "clips" / pitch_type).mkdir(parents=True, exist_ok=True)
        (base / "clips" / pitch_type / ".gitkeep").touch()
        print(
            f"  wrote {out}: {len(manifest)} {pitch_type} pitches, "
            f"{matched} with a Film Room link ({matched}/{len(manifest)})"
        )

    print("\ndone. Video is NOT downloaded — open each filmroom_url and use")
    print("Film Room's Download button, saving as clips/<pitch_type>/<clip_uid>.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
