"""Build tests/.cache from REAL downloaded data (never synthetic).

Contents: a handful of real clips + their pose files and extract rows, the matching
Statcast rows, one real /gf game-feed response and one real /sporty-videos page. The cache
is git-ignored; tests that need it skip when it is absent.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from tipfinder.config import Config
from tipfinder.data.fetch import make_client
from tipfinder.data.savant import SAVANT

CACHE = Path(__file__).resolve().parents[2] / "tests" / ".cache"


def build(cfg: Config, n: int = 16) -> str:
    run = cfg.run_dir
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "clips").mkdir(exist_ok=True)
    (CACHE / "pose").mkdir(exist_ok=True)
    man = pd.read_csv(run / "manifest.csv")
    ext = pd.read_csv(run / "extract.csv")
    labels_path = Path(__file__).resolve().parents[2] / "labels" / f"{cfg.slug}_windows.csv"
    labelled = set(pd.read_csv(labels_path)["pitch_key"]) if labels_path.exists() else set()
    qc = pd.read_csv(run / "pose_qc.csv")
    ok = ext[ext["status"] == "ok"]
    pick = list(ok[ok["pitch_key"].isin(labelled)]["pitch_key"])[: n - 2]
    pick += list(ok[~ok["pitch_key"].isin(pick)]["pitch_key"])[: n - 2 - len(pick)]
    pick += list(ext[ext["status"] == "no_cf_shot"]["pitch_key"])[:2]  # negative examples
    sub = man[man["pitch_key"].isin(pick)].copy()
    for r in sub.itertuples():
        shutil.copy2(run / r.clip_path, CACHE / "clips" / Path(r.clip_path).name)
        pose = run / "pose" / f"{r.pitch_key}.npz"
        if pose.exists():
            shutil.copy2(pose, CACHE / "pose" / pose.name)
    sub["clip_path"] = sub["clip_path"].map(lambda p: f"clips/{Path(p).name}")
    sub.to_csv(CACHE / "manifest.csv", index=False)
    ext[ext["pitch_key"].isin(pick)].to_csv(CACHE / "extract.csv", index=False)
    sc = pd.read_csv(run / "statcast.csv")
    sc[sc["game_pk"].isin(sub["game_pk"].unique()[:2])].to_csv(
        CACHE / "statcast_games.csv", index=False
    )
    # Real HTTP responses, served from the client's on-disk cache when present.
    client = make_client(cfg)
    r0 = sub.iloc[0]
    (CACHE / "gf.json").write_text(client.get_text(f"{SAVANT}/gf?game_pk={int(r0.game_pk)}"))
    (CACHE / "gf_meta.txt").write_text(f"{int(r0.game_pk)},{cfg.pitcher.mlbam_id}")
    page = f"{SAVANT}/sporty-videos?playId={r0.play_id}&videoType={r0.feed}"
    (CACHE / "sporty.html").write_text(client.get_text(page))
    (CACHE / "sporty_meta.txt").write_text(f"{r0.play_id},{r0.feed},{r0.clip_url}")
    if labels_path.exists():
        shutil.copy2(labels_path, CACHE / "labels.csv")
    qc[qc["pitch_key"].isin(pick)].to_csv(CACHE / "pose_qc.csv", index=False)
    return f"cached {len(sub)} real pitches in {CACHE}"
