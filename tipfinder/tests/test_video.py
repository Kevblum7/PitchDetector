"""Shot segmentation / CF-angle selection on real cached clips."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tipfinder.video.shots import find_shots, pick_cf_shot, read_video


def test_shots_partition_clip(cache_dir: Path, cached_manifest: pd.DataFrame) -> None:
    r = cached_manifest.iloc[0]
    frames, fps = read_video(cache_dir / r.clip_path)
    assert 50 < fps < 70
    shots = find_shots(frames, 0.55)
    assert shots[0].start == 0 and shots[-1].end == len(frames)
    assert all(a.end == b.start for a, b in zip(shots, shots[1:], strict=False))


def test_cf_shot_matches_extraction(
    cache_dir: Path, cached_manifest: pd.DataFrame, cached_extract: pd.DataFrame
) -> None:
    ext = cached_extract.set_index("pitch_key")
    checked = 0
    for r in cached_manifest.itertuples():
        if r.pitch_key not in ext.index:
            continue
        e = ext.loc[r.pitch_key]
        frames, fps = read_video(cache_dir / r.clip_path)
        picked = pick_cf_shot(frames, find_shots(frames, 0.55), fps)
        if e["status"] == "no_cf_shot":
            assert picked is None
        elif pd.notna(e["shot_start"]):
            assert picked is not None and picked[0].start == int(e["shot_start"])
        checked += 1
        if checked >= 6:
            break
    assert checked > 0
