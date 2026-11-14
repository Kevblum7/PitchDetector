"""Typed run configuration loaded from YAML (pitcher, date range, paths, thresholds)."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PitcherCfg(BaseModel):
    name: str
    mlbam_id: int
    throws: str = Field(pattern="^[LR]$")
    team: str  # team abbreviation as it appears in Statcast home_team/away_team


class SampleCfg(BaseModel):
    pitches_per_game: int = 10
    max_pitches: int = 320
    seed: int = 7


class NetworkCfg(BaseModel):
    min_interval_s: float = 1.5
    user_agent: str = "tipfinder/0.1 (portfolio research; low-rate, cached)"


class VideoCfg(BaseModel):
    # "pitcher_team": the broadcast of the pitcher's own club (consistent crew/zoom).
    preferred_feed: str = "pitcher_team"
    try_other_feed: bool = True
    shot_cut_threshold: float = 0.55  # HSV-histogram correlation below this = cut
    max_pose_frame: int = 300  # pose is not run past this clip frame (see pose.extract)


class WindowCfg(BaseModel):
    guard_frames: int = 3  # frames dropped before detected separation
    together_ratio: float = 0.55  # wrist gap / torso length below this = hands together
    apart_ratio: float = 0.85  # above this (sustained) = hands separated
    sustain_frames: int = 4
    min_set_frames: int = 6
    onset_fraction: float = 0.5  # see window.release.refine_onset
    max_onset_backtrack: int = 8  # frames; caps how far the onset can move back
    max_set_speed: float = 0.15  # median hand speed (torso lengths/frame) during a held set
    smooth_frames: int = 5


class PoseCfg(BaseModel):
    backend: str = "mediapipe"
    model_complexity: int = 1
    # QC over the 30-frame feature window (label-blind; calibrated on the distributions in
    # docs/pose_qc.md). MediaPipe "visibility" also drops when a wrist is inside the glove,
    # so 0.5 rejected ~77% of clips; 0.3 plus a steadiness check is used instead.
    good_visibility: float = 0.3
    min_good_frame_share: float = 0.5
    max_hand_speed: float = 0.10  # median hands speed, torso lengths/frame (sets are still)
    max_missing_share: float = 0.25
    min_window_frames: int = 15


class EvalCfg(BaseModel):
    n_bootstrap: int = 2000
    n_permutations: int = 200
    alpha: float = 0.05
    n_splits: int = 5
    seed: int = 11


class Config(BaseModel):
    pitcher: PitcherCfg
    season: int
    start_date: date
    end_date: date
    fastball_types: list[str] = ["FF", "SI", "FC"]
    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")
    sample: SampleCfg = SampleCfg()
    network: NetworkCfg = NetworkCfg()
    video: VideoCfg = VideoCfg()
    window: WindowCfg = WindowCfg()
    pose: PoseCfg = PoseCfg()
    eval: EvalCfg = EvalCfg()

    @property
    def slug(self) -> str:
        return f"{self.pitcher.name.lower().replace(' ', '_')}_{self.season}"

    @property
    def run_dir(self) -> Path:
        return self.data_dir / self.slug

    def hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    cfg = Config.model_validate(raw)
    base = Path(path).resolve().parent.parent  # configs/ lives at project root
    if not cfg.data_dir.is_absolute():
        cfg.data_dir = base / cfg.data_dir
    if not cfg.cache_dir.is_absolute():
        cfg.cache_dir = base / cfg.cache_dir
    return cfg
