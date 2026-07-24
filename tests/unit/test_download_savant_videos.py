"""Unit tests for the Baseball Savant downloader (pure logic, no network)."""

from __future__ import annotations

import urllib.parse

import pytest

from scripts.download_savant_videos import (
    PitchRow,
    build_csv_url,
    clip_filename,
    extract_video_url,
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


def test_parse_pitch_rows_extracts_play_ids_and_skips_blank() -> None:
    csv_text = (
        "game_date,player_name,pitch_type,game_pk,play_id,des\n"
        "2024-05-01,Williams Devin,SL,745001,aaaa-1111,Strike\n"
        "2024-05-01,Williams Devin,FF,745001,,Ball\n"  # no play_id -> skipped
        "2024-05-02,Williams Devin,CH,745002,bbbb-2222,In play\n"
    )
    rows = parse_pitch_rows(csv_text)
    assert [r.play_id for r in rows] == ["aaaa-1111", "bbbb-2222"]
    assert rows[0].pitch_type == "SL"
    assert rows[0].game_date == "2024-05-01"
    assert rows[0].pitcher == "Williams Devin"


def test_parse_pitch_rows_empty() -> None:
    assert parse_pitch_rows("game_date,play_id\n") == []


def test_extract_video_url_finds_mp4_source() -> None:
    html = (
        "<html><body><video controls>"
        '<source src="https://sporty-clips.mlb.com/abc123_1080.mp4" type="video/mp4">'
        "</video></body></html>"
    )
    assert extract_video_url(html) == "https://sporty-clips.mlb.com/abc123_1080.mp4"


def test_extract_video_url_none_when_absent() -> None:
    assert extract_video_url("<html><body>no video here</body></html>") is None


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
