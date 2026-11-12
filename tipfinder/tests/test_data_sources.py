"""Parsing of REAL cached Baseball Savant responses and Statcast rows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tipfinder.data.savant import feed_order, parse_game_feed, parse_video_page
from tipfinder.data.statcast import add_context


def test_game_feed_has_play_ids_for_pitcher(cache_dir: Path) -> None:
    game_pk, pitcher = map(int, (cache_dir / "gf_meta.txt").read_text().split(","))
    pitches = parse_game_feed(json.loads((cache_dir / "gf.json").read_text()), pitcher)
    assert len(pitches) > 50  # a start is ~90-110 pitches
    keys = {(p.ab_number, p.pitch_number) for p in pitches}
    assert len(keys) == len(pitches), "(ab_number, pitch_number) must be a unique join key"
    assert all(len(p.play_id) == 36 for p in pitches)


def test_game_feed_join_agrees_with_statcast(cache_dir: Path, statcast_games: pd.DataFrame) -> None:
    game_pk, pitcher = map(int, (cache_dir / "gf_meta.txt").read_text().split(","))
    feed = {
        (p.ab_number, p.pitch_number): p
        for p in parse_game_feed(json.loads((cache_dir / "gf.json").read_text()), pitcher)
    }
    sc = statcast_games[statcast_games["game_pk"] == game_pk]
    matched = 0
    for r in sc.itertuples():
        p = feed.get((r.at_bat_number, r.pitch_number))
        if p is not None:
            matched += 1
            assert p.pitch_type == r.pitch_type
    assert matched / len(sc) > 0.95


def test_video_page_gives_clip_and_echoed_feed(cache_dir: Path) -> None:
    play_id, feed, url = (cache_dir / "sporty_meta.txt").read_text().split(",")
    ref = parse_video_page((cache_dir / "sporty.html").read_text())
    assert ref is not None
    assert ref.feed == feed
    assert ref.url == url
    assert ref.url.startswith("https://sporty-clips.mlb.com/") and ref.url.endswith(".mp4")


def test_feed_order_prefers_pitcher_team_broadcast() -> None:
    assert feed_order("PHI", "PHI", "pitcher_team", True) == ["HOME", "AWAY"]
    assert feed_order("PHI", "NYM", "pitcher_team", True) == ["AWAY", "HOME"]
    assert feed_order("PHI", "NYM", "pitcher_team", False) == ["AWAY"]


def test_context_uses_only_past_pitches(statcast_games: pd.DataFrame) -> None:
    raw = statcast_games.drop(
        columns=[
            c
            for c in statcast_games.columns
            if c
            not in (
                "game_pk",
                "game_date",
                "home_team",
                "away_team",
                "game_type",
                "at_bat_number",
                "pitch_number",
                "pitch_type",
                "pitch_name",
                "release_speed",
                "balls",
                "strikes",
                "outs_when_up",
                "inning",
                "on_1b",
                "on_2b",
                "on_3b",
                "stand",
                "p_throws",
                "pitcher",
                "batter",
                "n_thruorder_pitcher",
            )
        ]
    )
    df = add_context(raw, ["FF", "SI", "FC"])
    for _, ab in df.groupby(["game_pk", "at_bat_number"]):
        ab = ab.sort_values("pitch_number")
        assert ab["prev_missing"].iloc[0] == 1
        assert (
            ab["prev_fastball"].iloc[1:].to_numpy() == ab["is_fastball"].iloc[:-1].to_numpy()
        ).all()
    # Shuffling the CURRENT label must not change any context column.
    df2 = add_context(
        raw.assign(pitch_type=raw["pitch_type"].sample(frac=1, random_state=0).to_numpy()),
        ["FF", "SI", "FC"],
    )
    for col in ("balls", "strikes", "runners_on", "batter_is_L", "outs_when_up"):
        assert (df[col].to_numpy() == df2[col].to_numpy()).all()


def test_label_never_in_feature_names() -> None:
    from tipfinder.features.body import body_feature_names
    from tipfinder.model.models import CONTEXT_FEATURES

    banned = ("pitch_type", "is_fastball", "pitch_name", "release_speed", "description", "events")
    for name in body_feature_names() + CONTEXT_FEATURES:
        assert not any(b in name for b in banned), name
