"""Plain-Python tools. Usable directly (CLI, notebooks, tests) and wrapped for the agent.

Every tool returns a JSON-serialisable dict. All numbers come from the deterministic
pipeline; the agent only reads them.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from tipfinder.config import load_config

PROJECT = Path(__file__).resolve().parents[3]


def create_pitcher_config(
    name: str, mlbam_id: int, throws: str, team: str, season: int, start_date: str, end_date: str
) -> dict:
    """Write configs/<name>_<season>.yaml for a new pitcher (defaults for everything else)."""
    if throws not in ("L", "R"):
        raise ValueError("throws must be L or R")
    slug = f"{name.lower().replace(' ', '_')}_{season}"
    if not slug.replace("_", "").isalnum():
        raise ValueError("pitcher name must be letters/digits/spaces")
    path = PROJECT / "configs" / f"{slug}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "pitcher": {
                    "name": name,
                    "mlbam_id": int(mlbam_id),
                    "throws": throws,
                    "team": team,
                },
                "season": int(season),
                "start_date": start_date,
                "end_date": end_date,
            },
            sort_keys=False,
        )
    )
    load_config(path)  # validate
    return {"config_path": str(path.relative_to(PROJECT))}


def fetch_pitcher_data(config_path: str, limit: int | None = None) -> dict:
    """Statcast pitches + Savant clips for the configured pitcher (cached, resumable)."""
    from tipfinder.data.fetch import fetch

    man = fetch(load_config(PROJECT / config_path), limit=limit)
    return {
        "clip_status": man["status"].value_counts().to_dict(),
        "n_games": int(man["game_pk"].nunique()),
    }


def extract_clips(config_path: str) -> dict:
    """CF-shot selection, pose estimation, set/separation detection, pose QC."""
    from tipfinder.pose.extract import extract_all

    res = extract_all(load_config(PROJECT / config_path))
    return {"extract_status": res["status"].value_counts().to_dict()}


def run_pipeline(config_path: str, n_permutations: int | None = None) -> dict:
    """Build features, fit models, run held-out evaluation and controls. Returns metrics."""
    from tipfinder.pipeline import evaluate, train

    cfg = load_config(PROJECT / config_path)
    train(cfg)
    evaluate(cfg, n_permutations=n_permutations)
    return get_metrics(config_path)


def get_metrics(config_path: str) -> dict:
    """Compact view of the latest results.json, for reporting."""
    from tipfinder.report import confidence_level

    cfg = load_config(PROJECT / config_path)
    res = json.loads((cfg.run_dir / "results.json").read_text())
    m = res["models"][res["primary_model"]]
    keep = (
        "n_pitches",
        "n_games",
        "class_balance",
        "majority_accuracy",
        "context_logloss",
        "full_logloss",
        "lift_logloss",
        "lift_ci",
        "context_auc",
        "full_auc",
        "lift_auc_ci",
        "context_accuracy",
        "full_accuracy",
        "perm_p_value",
        "n_permutations",
    )
    leak = res["controls"]["post_separation_leak"]
    return {
        "pitcher": res["pitcher"]["name"],
        "season": res["season"],
        "verdict": res["verdict"],
        "tip_detected": res["tip_detected"],
        "confidence": confidence_level(res),
        "primary_model": res["primary_model"],
        "primary": {k: m[k] for k in keep},
        "secondary_gbm_lift": {
            k: res["models"]["gbm"][k] for k in ("lift_logloss", "lift_ci", "perm_p_value")
        },
        "controls": {
            "shuffled_label_mean_lift": res["controls"]["shuffled_labels"]["mean_lift"],
            "post_separation_leak_lift": leak["lift_logloss"],
            "post_separation_leak_auc": leak["full_auc"],
        },
        "top_cues": res["scouting_notes"],
        "pose_quality": res["pose_quality"],
        "release_window_validation": res["release_window_validation"],
        "pitch_type_counts": res["pitch_type_counts"],
    }
