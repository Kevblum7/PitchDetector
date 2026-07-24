"""Unit tests for the Baseball Savant downloader (pure logic, no network)."""

from __future__ import annotations

import json
import urllib.parse

import pytest

from scripts.download_savant_videos import (
    PitchRow,
    build_csv_url,
    clip_filename,
    extract_video_url,
    game_feed_url,
    parse_game_feed,
    parse_pitch_rows,
    video_page_url,
)


def _query(url: str) -> dict[str, list[str]]:
    return urllib.parse.parse_qs(urllib.parse.urlsplit(url).query, keep_blank_values=True)


def test_build_csv_url_from_search_url_rewrites_endpoint() -> None:
    search = (
        "https://baseballsavant.mlb.com/statcast_search?"
        "player_type=pitcher&pitchers_lookup%5B%5D=642207&hfSea=2024%7C"
    )
    url = build_csv_url(search_url=search)
    parts = urllib.parse.urlsplit(url)
    assert parts.path == "/statcast_search/csv"
    q = _query(url)
    # Original filters preserved.
    assert q["pitchers_lookup[]"] == ["642207"]
    # CSV/per-pitch params forced on.
    assert q["type"] == ["details"]
    assert q["all"] == ["true"]


def test_build_csv_url_forces_type_details_over_existing() -> None:
    search = "https://baseballsavant.mlb.com/statcast_search?type=summary&all=false"
    url = build_csv_url(search_url=search)
    q = _query(url)
    assert q["type"] == ["details"]
    assert q["all"] == ["true"]


def test_build_csv_url_from_filters() -> None:
    url = build_csv_url(player_id=642207, season=2024, pitch_type="SL")
    assert url.startswith("https://baseballsavant.mlb.com/statcast_search/csv?")
    q = _query(url)
    assert q["pitchers_lookup[]"] == ["642207"]
    assert q["type"] == ["details"]
    assert q["hfSea"] == ["2024|"]
    assert q["hfPT"] == ["SL|"]
    assert q["player_type"] == ["pitcher"]


def test_build_csv_url_requires_a_source() -> None:
    with pytest.raises(ValueError):
        build_csv_url()


def test_parse_pitch_rows_keys_on_game_at_bat_pitch_and_skips_incomplete() -> None:
    csv_text = (
        "game_date,player_name,pitch_type,game_pk,at_bat_number,pitch_number,des\n"
        "2024-05-01,Williams Devin,SL,745001,12,3,Strike\n"
        "2024-05-01,Williams Devin,FF,745001,,4,Ball\n"  # no at_bat_number -> skipped
        "2024-05-02,Williams Devin,CH,745002,7,1,In play\n"
    )
    rows = parse_pitch_rows(csv_text)
    assert [(r.game_pk, r.at_bat_number, r.pitch_number) for r in rows] == [
        ("745001", "12", "3"),
        ("745002", "7", "1"),
    ]
    # play_id is not in the CSV; it is resolved later from the game feed.
    assert rows[0].play_id is None
    assert rows[0].pitch_type == "SL"
    assert rows[0].game_date == "2024-05-01"
    assert rows[0].pitcher == "Williams Devin"


def test_parse_pitch_rows_strips_utf8_bom() -> None:
    # Savant serves the CSV with a leading BOM, which otherwise mangles the
    # first column name and makes pitch_type unreadable.
    csv_text = "﻿pitch_type,game_date,game_pk,at_bat_number,pitch_number\nCH,2024-07-30,745003,5,2\n"
    rows = parse_pitch_rows(csv_text)
    assert len(rows) == 1
    assert rows[0].pitch_type == "CH"


def test_parse_pitch_rows_empty() -> None:
    assert parse_pitch_rows("game_pk,at_bat_number,pitch_number\n") == []


def test_parse_game_feed_maps_ab_and_pitch_to_play_id() -> None:
    feed = json.dumps(
        {
            "team_home": [
                {"ab_number": 1, "pitch_number": 1, "play_id": "home-1-1"},
                {"ab_number": 1, "pitch_number": 2, "play_id": None},  # skipped
            ],
            "team_away": [
                {"ab_number": 2, "pitch_number": 1, "play_id": "away-2-1"},
            ],
        }
    )
    mapping = parse_game_feed(feed)
    assert mapping[("1", "1")] == "home-1-1"
    assert mapping[("2", "1")] == "away-2-1"
    assert ("1", "2") not in mapping


def test_game_feed_url_includes_game_pk() -> None:
    assert game_feed_url(745001) == "https://baseballsavant.mlb.com/gf?game_pk=745001"


def test_extract_video_url_finds_mp4_source() -> None:
    html = (
        "<html><body><video controls>"
        '<source src="https://sporty-clips.mlb.com/abc123_1080.mp4" type="video/mp4">'
        "</video></body></html>"
    )
    assert extract_video_url(html) == "https://sporty-clips.mlb.com/abc123_1080.mp4"


def test_extract_video_url_none_when_absent() -> None:
    assert extract_video_url("<html><body>no video here</body></html>") is None


def test_extract_video_url_unescapes_html_entities() -> None:
    # Savant serves the mp4 token HTML-escaped; the trailing "==" padding
    # appears as "&#x3D;&#x3D;" and must be decoded to a usable URL.
    html_page = '<source src="https://sporty-clips.mlb.com/abcAw&#x3D;&#x3D;.mp4">'
    assert extract_video_url(html_page) == "https://sporty-clips.mlb.com/abcAw==.mp4"


def test_video_page_url_encodes_play_id() -> None:
    assert video_page_url("aaaa-1111") == (
        "https://baseballsavant.mlb.com/sporty-videos?playId=aaaa-1111"
    )


def test_clip_filename_is_deterministic_and_safe() -> None:
    row = PitchRow(play_id="aaaa/1111", game_date="2024-05-01", pitch_type="SL")
    name = clip_filename(row)
    assert name == "2024-05-01_SL_aaaa-1111.mp4"
    # No path separators or unsafe characters remain in the stem.
    assert "/" not in name[:-4]


def test_clip_filename_falls_back_to_play_id() -> None:
    row = PitchRow(play_id="xyz-9")
    assert clip_filename(row) == "xyz-9.mp4"
