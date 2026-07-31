from __future__ import annotations

import argparse
import html
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from pybaseball import statcast_pitcher

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PLAYER_ID = 642207  # Devin Williams
START_DATE = "2026-03-01"
END_DATE = date.today().isoformat()

# Baseball Savant pitch codes:
# FF = four-seam fastball
# CH = changeup
# SI = sinker
# FC = cutter
PITCH_TYPES = {"FF"}

# Directory naming per pitch-type selection, so a changeup run cannot
# overwrite the fastball run's pitches.csv / video_manifest.csv.
PITCH_TYPE_DIR_NAMES = {
    "FF": "fastballs",
    "CH": "changeups",
    "SI": "sinkers",
    "FC": "cutters",
}

OUTPUT_DIR = Path("devin_williams_2026_fastballs")
VIDEO_DIR = OUTPUT_DIR / "videos"
CSV_PATH = OUTPUT_DIR / "pitches.csv"
MANIFEST_PATH = OUTPUT_DIR / "video_manifest.csv"

REQUEST_DELAY_SECONDS = 0.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/140 Safari/537.36"
    )
}

# Savant server-renders the clip into a <video><source src="..."> element
# pointing at the sporty-clips CDN. When the play has no video, the page
# renders "No Video Found" instead and contains no such element.
CLIP_URL_PATTERN = re.compile(r'src="(https://sporty-clips\.mlb\.com/[^"]+?\.mp4)"')

DOWNLOAD_CHUNK_BYTES = 1 << 16


def _optional_int(value: object) -> int | None:
    """Coerce a possibly-missing Statcast cell to int, or None if unusable."""
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None


def safe_filename(value: object) -> str:
    """Convert a value into a filesystem-safe string."""
    text = str(value).strip()
    text = re.sub(r"[^\w.-]+", "_", text)
    return text.strip("_")


def download_pitch_data() -> pd.DataFrame:
    """Download Devin Williams's pitch-level Statcast data."""
    print(f"Downloading Statcast data from {START_DATE} through {END_DATE}...")

    pitches = statcast_pitcher(
        start_dt=START_DATE,
        end_dt=END_DATE,
        player_id=PLAYER_ID,
    )

    if pitches.empty:
        raise RuntimeError("No Statcast pitches were returned.")

    pitches = pitches[pitches["pitch_type"].isin(PITCH_TYPES)].copy()

    if pitches.empty:
        raise RuntimeError(f"No pitches matched pitch types: {sorted(PITCH_TYPES)}")

    pitches.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"],
        inplace=True,
    )

    pitches.to_csv(CSV_PATH, index=False)

    print(f"Found {len(pitches)} matching pitches.")
    print(f"Saved pitch data to {CSV_PATH}")

    return pitches


def get_game_feed(
    session: requests.Session,
    game_pk: int,
) -> dict:
    """Download the MLB Stats API live game feed."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

    response = session.get(
        url,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    return response.json()


@dataclass(frozen=True)
class PitchVideo:
    """Video identity and matchup context for one pitch."""

    play_id: str
    batter_id: int | None
    batter_name: str | None


def build_play_id_lookup(game_feed: dict) -> dict[tuple[int, int], PitchVideo]:
    """
    Build a lookup:

        (at_bat_number, pitch_number) -> PitchVideo
    """
    lookup: dict[tuple[int, int], PitchVideo] = {}

    all_plays = game_feed.get("liveData", {}).get("plays", {}).get("allPlays", [])

    for play in all_plays:
        about = play.get("about", {})
        at_bat_index = about.get("atBatIndex")

        if at_bat_index is None:
            continue

        # Statcast at_bat_number is normally one-based, while
        # Stats API atBatIndex is zero-based.
        at_bat_number = int(at_bat_index) + 1

        batter = play.get("matchup", {}).get("batter", {})
        batter_id = batter.get("id")

        for event in play.get("playEvents", []):
            # `isPitch` is a top-level field on the play event, not inside
            # `details`. Non-pitch events (actions, pickoffs, stepoffs)
            # carry no playId.
            if not event.get("isPitch"):
                continue

            pitch_number = event.get("pitchNumber")
            play_id = event.get("playId")

            if pitch_number is None or not play_id:
                continue

            lookup[(at_bat_number, int(pitch_number))] = PitchVideo(
                play_id=play_id,
                batter_id=int(batter_id) if batter_id is not None else None,
                batter_name=batter.get("fullName"),
            )

    return lookup


def add_play_ids(pitches: pd.DataFrame) -> pd.DataFrame:
    """Match each Statcast pitch to its Baseball Savant playId."""
    session = requests.Session()
    results: list[pd.DataFrame] = []

    unique_games = pitches["game_pk"].dropna().astype(int).unique()

    for game_index, game_pk in enumerate(unique_games, start=1):
        print(f"Reading game {game_index}/{len(unique_games)}: {game_pk}")

        game_pitches = pitches[pitches["game_pk"].astype(int) == game_pk].copy()

        try:
            game_feed = get_game_feed(session, game_pk)
            lookup = build_play_id_lookup(game_feed)

        except requests.RequestException as exc:
            print(f"Could not read game {game_pk}: {exc}")
            game_pitches["play_id"] = None
            results.append(game_pitches)
            continue

        if not lookup:
            print(
                f"Warning: game {game_pk} returned no pitch playIds. "
                "The feed schema may have changed."
            )

        play_ids: list[str | None] = []
        batter_names: list[str | None] = []
        batter_mismatches = 0

        for row in game_pitches.itertuples():
            key = (
                int(row.at_bat_number),
                int(row.pitch_number),
            )
            pitch_video = lookup.get(key)

            if pitch_video is None:
                play_ids.append(None)
                batter_names.append(None)
                continue

            play_ids.append(pitch_video.play_id)
            batter_names.append(pitch_video.batter_name)

            # The Statcast `batter` column is an MLBAM id. If it disagrees
            # with the feed, the at-bat/pitch keys are misaligned and the
            # play_id belongs to the wrong pitch.
            # pd.notna() does not narrow for the type checker, so convert
            # explicitly and treat an unparseable id as "cannot cross-check".
            statcast_batter = _optional_int(getattr(row, "batter", None))

            if (
                pitch_video.batter_id is not None
                and statcast_batter is not None
                and statcast_batter != pitch_video.batter_id
            ):
                batter_mismatches += 1

        if batter_mismatches:
            print(
                f"Warning: game {game_pk} had {batter_mismatches} pitches "
                "whose batter id disagreed with the feed. Play IDs for this "
                "game may be misaligned."
            )

        game_pitches["play_id"] = play_ids
        game_pitches["batter_name"] = batter_names
        results.append(game_pitches)

        time.sleep(REQUEST_DELAY_SECONDS)

    combined = pd.concat(results, ignore_index=True)

    matched = combined["play_id"].notna().sum()
    print(f"Matched {matched}/{len(combined)} pitches to play IDs.")

    combined.to_csv(CSV_PATH, index=False)

    return combined


def create_video_filename(row: pd.Series) -> str:
    """Create a descriptive filename for a pitch video."""
    game_date = safe_filename(row.get("game_date", "unknown_date"))
    game_pk = int(row["game_pk"])
    at_bat = int(row["at_bat_number"])
    pitch_number = int(row["pitch_number"])
    pitch_type = safe_filename(row.get("pitch_type", "pitch"))
    velocity = row.get("release_speed")

    if pd.notna(velocity):
        velocity_text = f"{float(velocity):.1f}mph"
    else:
        velocity_text = "unknown_speed"

    # `player_name` from statcast_pitcher is the pitcher, not the batter.
    # The batter name comes from the Stats API feed; fall back to the
    # MLBAM batter id when the feed did not supply one.
    batter_name = row.get("batter_name")

    if pd.isna(batter_name) or not str(batter_name).strip():
        batter_id = row.get("batter")
        batter_name = f"mlbam-{int(batter_id)}" if pd.notna(batter_id) else "batter"

    batter = safe_filename(batter_name)

    return (
        f"{game_date}"
        f"_game-{game_pk}"
        f"_ab-{at_bat}"
        f"_pitch-{pitch_number}"
        f"_{pitch_type}"
        f"_{velocity_text}"
        f"_{batter}"
    )


def resolve_clip_url(
    session: requests.Session,
    savant_url: str,
) -> str | None:
    """
    Resolve a Savant video page to its direct sporty-clips MP4 URL.

    Returns None when Savant has no video for the play, which is a normal
    outcome (spring training games in particular are often missing) and is
    not a download failure.
    """
    response = session.get(
        savant_url,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    match = CLIP_URL_PATTERN.search(response.text)

    if match is None:
        return None

    # The URL is HTML-escaped in the page source; base64-ish clip tokens
    # end in "&#x3D;&#x3D;" rather than "==".
    return html.unescape(match.group(1))


def download_clip(
    session: requests.Session,
    clip_url: str,
    output_path: Path,
) -> None:
    """Stream one MP4 to disk, writing through a temporary part file."""
    part_path = output_path.with_suffix(output_path.suffix + ".part")

    with session.get(
        clip_url,
        headers=HEADERS,
        timeout=60,
        stream=True,
    ) as response:
        response.raise_for_status()

        with part_path.open("wb") as handle:
            for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)

    part_path.replace(output_path)


def download_all_videos(pitches: pd.DataFrame) -> None:
    """Download every matched Savant pitch video."""
    session = requests.Session()
    manifest_rows: list[dict] = []

    matched_pitches = pitches[pitches["play_id"].notna()].copy()

    for video_index, (_, row) in enumerate(
        matched_pitches.iterrows(),
        start=1,
    ):
        play_id = str(row["play_id"])

        savant_url = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"

        filename = create_video_filename(row)
        output_path = VIDEO_DIR / f"{filename}.mp4"

        print()
        print(f"[{video_index}/{len(matched_pitches)}] {filename}")

        clip_url: str | None = None

        if output_path.exists():
            status = "already_present"
            print("Already downloaded; skipping.")

        else:
            try:
                clip_url = resolve_clip_url(session, savant_url)

                if clip_url is None:
                    status = "no_video_available"
                    print(f"Savant has no video for this play: {savant_url}")

                else:
                    download_clip(session, clip_url, output_path)
                    status = "downloaded"
                    print(f"Saved {output_path.name}")

            except requests.RequestException as exc:
                status = "download_failed"
                print(f"Download failed: {savant_url}: {exc}")

        manifest_rows.append(
            {
                "game_date": row.get("game_date"),
                "game_pk": row.get("game_pk"),
                "at_bat_number": row.get("at_bat_number"),
                "pitch_number": row.get("pitch_number"),
                "pitch_type": row.get("pitch_type"),
                "release_speed": row.get("release_speed"),
                "batter_name": row.get("batter_name"),
                "play_id": play_id,
                "savant_url": savant_url,
                "clip_url": clip_url,
                "status": status,
                "download_success": status in {"downloaded", "already_present"},
                "output_name": filename,
            }
        )

        pd.DataFrame(manifest_rows).to_csv(
            MANIFEST_PATH,
            index=False,
        )

        time.sleep(REQUEST_DELAY_SECONDS)

    counts: dict[str, int] = {}

    for manifest_row in manifest_rows:
        status_value = str(manifest_row["status"])
        counts[status_value] = counts.get(status_value, 0) + 1

    print()
    print(f"Processed {len(manifest_rows)} pitches:")

    for status_value in sorted(counts):
        print(f"  {status_value}: {counts[status_value]}")

    print(f"Videos: {VIDEO_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


def configure_output(pitch_types: set[str], output_dir: Path | None) -> None:
    """
    Point the module-level output paths at this run's pitch-type selection.

    The download functions read these as module globals; rebinding here keeps
    the change contained to argument parsing rather than threading a config
    object through every function.
    """
    global PITCH_TYPES, OUTPUT_DIR, VIDEO_DIR, CSV_PATH, MANIFEST_PATH

    PITCH_TYPES = pitch_types

    if output_dir is None:
        label = "_".join(
            PITCH_TYPE_DIR_NAMES.get(code, code.lower()) for code in sorted(pitch_types)
        )
        output_dir = Path(f"devin_williams_2026_{label}")

    OUTPUT_DIR = output_dir
    VIDEO_DIR = OUTPUT_DIR / "videos"
    CSV_PATH = OUTPUT_DIR / "pitches.csv"
    MANIFEST_PATH = OUTPUT_DIR / "video_manifest.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download Baseball Savant pitch videos for one pitcher.",
    )
    parser.add_argument(
        "--pitch-types",
        nargs="+",
        default=sorted(PITCH_TYPES),
        metavar="CODE",
        help="Savant pitch codes to download, e.g. FF or CH (default: FF)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: derived from --pitch-types)",
    )
    args = parser.parse_args(argv)

    configure_output(
        pitch_types={code.upper() for code in args.pitch_types},
        output_dir=args.output_dir,
    )

    print(f"Pitch types: {sorted(PITCH_TYPES)}")
    print(f"Output directory: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    pitches = download_pitch_data()
    pitches = add_play_ids(pitches)
    download_all_videos(pitches)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
