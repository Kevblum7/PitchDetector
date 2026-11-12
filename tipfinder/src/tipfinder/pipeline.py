"""End-to-end steps shared by the CLI and the agent tools. No LLM involved here."""

from __future__ import annotations

import json
import logging
import pickle
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tipfinder.config import Config
from tipfinder.evaluation.controls import leakage_control, shuffled_label_control
from tipfinder.evaluation.lift import evaluate_lift
from tipfinder.explain.importance import heldout_permutation_importance, scouting_notes
from tipfinder.features.body import body_feature_names
from tipfinder.features.table import build_table
from tipfinder.model.models import CONTEXT_FEATURES, MODEL_KINDS, add_count_mix, make_model
from tipfinder.window.validate import validate_against_labels

log = logging.getLogger(__name__)
PRIMARY_MODEL = "logreg"  # fixed in advance; the decision is made on this model only


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def body_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in body_feature_names() if c in df.columns]
    # Drop features missing in most clips (e.g. fingers never detected).
    return [c for c in cols if df[c].notna().mean() >= 0.5]


def train(cfg: Config) -> dict:
    """Build feature tables (real + leakage-control) and fit final models on all games."""
    real = build_table(cfg)
    build_table(cfg, leaky=True)
    body = body_columns(real)
    out = {}
    for kind in MODEL_KINDS:
        tr, _ = add_count_mix(real, real.head(0), "is_fastball")
        model = make_model(kind, cfg.eval.seed).fit(tr[CONTEXT_FEATURES + body], tr["is_fastball"])
        path = cfg.run_dir / "models" / f"{kind}_full.pkl"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(pickle.dumps({"model": model, "features": CONTEXT_FEATURES + body}))
        out[kind] = str(path)
    return {
        "n_pitches": len(real),
        "n_games": int(real["game_pk"].nunique()),
        "body_features": body,
        "models": out,
    }


def pose_quality_summary(cfg: Config) -> dict:
    """Clip funnel (download -> CF shot -> pitcher -> separation -> pose QC) and QC stats."""
    run = cfg.run_dir
    man = pd.read_csv(run / "manifest.csv")
    ext = pd.read_csv(run / "extract.csv")
    qc = pd.read_csv(run / "pose_qc.csv")
    status = ext.set_index("pitch_key")["status"].copy()
    failed_qc = qc.loc[~qc["passed"], "pitch_key"]
    status.loc[status.index.isin(failed_qc)] = "low_pose_quality"
    final = man[["pitch_key", "home_team"]].merge(
        status.rename("final").reset_index(), on="pitch_key", how="left"
    )
    final["final"] = final["final"].fillna("no_video")
    by_park = final.groupby("home_team")["final"].apply(lambda s: float((s == "ok").mean()))
    q = qc["pose_quality"]
    return {
        "pitches_sampled": int(len(man)),
        "funnel": final["final"].value_counts().to_dict(),
        "good_visibility_threshold": cfg.pose.good_visibility,
        "min_good_frame_share": cfg.pose.min_good_frame_share,
        "key_joint_good_share_median": float(q.median()) if len(q) else None,
        "key_joint_good_share_quartiles": [float(x) for x in np.percentile(q, [25, 75])]
        if len(q)
        else None,
        "usable_share_by_home_park": by_park.round(2).to_dict(),
    }


def evaluate(
    cfg: Config, target: str = "binary", n_permutations: int | None = None, n_shuffles: int = 20
) -> dict:
    run = cfg.run_dir
    real = pd.read_csv(run / "features.csv")
    leaky = pd.read_csv(run / "features_leaky.csv")
    body = body_columns(real)
    e = cfg.eval
    n_perm = e.n_permutations if n_permutations is None else n_permutations
    results: dict = {
        "pitcher": cfg.pitcher.model_dump(),
        "season": cfg.season,
        "config_hash": cfg.hash(),
        "code_revision": _git_rev(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "fastball_types": cfg.fastball_types,
        "context_features": CONTEXT_FEATURES,
        "body_features": body,
        "primary_model": PRIMARY_MODEL,
        "pitch_type_counts": real["pitch_type"].value_counts().to_dict(),
    }
    results["models"] = {}
    for kind in MODEL_KINDS:
        r = evaluate_lift(
            real, body, kind, target, e.n_splits, e.n_bootstrap, n_perm, e.alpha, e.seed
        )
        results["models"][kind] = r.to_dict()
    primary = results["models"][PRIMARY_MODEL]
    results["tip_detected"] = primary["tip_detected"]
    results["verdict"] = primary["verdict"]

    results["controls"] = {
        "shuffled_labels": shuffled_label_control(
            real, body, PRIMARY_MODEL, n_shuffles, e.n_splits, e.seed
        ),
        "post_separation_leak": leakage_control(
            leaky, body, PRIMARY_MODEL, e.n_splits, e.n_bootstrap, e.seed
        ).to_dict(),
    }
    imp = heldout_permutation_importance(real, body, PRIMARY_MODEL, e.n_splits, e.seed)
    results["importance"] = imp.to_dict("records")
    results["scouting_notes"] = scouting_notes(real, imp, primary["tip_detected"], seed=e.seed)
    results["pose_quality"] = pose_quality_summary(cfg)
    labels = Path(__file__).resolve().parents[2] / "labels" / f"{cfg.slug}_windows.csv"
    results["release_window_validation"] = (
        validate_against_labels(cfg, labels) if labels.exists() else None
    )
    name = "results.json" if target == "binary" else f"results_{target}.json"
    (run / name).write_text(json.dumps(results, indent=2, default=_json_default))
    return results


def _json_default(o: object) -> object:
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))
