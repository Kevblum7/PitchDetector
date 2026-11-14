"""Deterministic Markdown report from results.json (no LLM)."""

from __future__ import annotations


def _ci(c: list | tuple) -> str:
    return f"[{c[0]:+.4f}, {c[1]:+.4f}]"


def control_fired(res: dict) -> bool:
    """Did the post-separation leakage control show the pipeline CAN see pitch-type
    information? If not, a "no tip" verdict says little: the method was insensitive."""
    return bool(res["controls"]["post_separation_leak"]["lift_auc_ci"][0] > 0)


def verdict_line(res: dict) -> str:
    m = res["models"][res["primary_model"]]
    if m["tip_detected"]:
        return f"**Verdict: {m['verdict'].upper()}** (confidence: {confidence_level(res)})"
    if not control_fired(res):
        return (
            "**Verdict: NO DETECTABLE TIP — INCONCLUSIVE.** The positive control did not fire: "
            "even with post-separation frames included, pose features did not predict pitch type, "
            "so this pipeline could not have detected a tip for this pitcher."
        )
    return "**Verdict: NO DETECTABLE TIP** (positive control fired, so the method could see a signal had there been one)"


def confidence_level(res: dict) -> str:
    m = res["models"][res["primary_model"]]
    if not m["tip_detected"]:
        return "none"
    p = m["perm_p_value"]
    return "moderate" if p < 0.01 and m["lift_ci"][0] > 0.01 else "low"


def render_markdown(res: dict) -> str:
    m = res["models"][res["primary_model"]]
    g = res["models"].get("gbm")
    c = res["controls"]
    leak = c["post_separation_leak"]
    shuf = c["shuffled_labels"]
    pq = res["pose_quality"]
    rv = res.get("release_window_validation")
    lines = [
        f"# Tip report: {res['pitcher']['name']} ({res['season']})",
        "",
        verdict_line(res),
        "",
        f"Sample: {m['n_pitches']} pitches from {m['n_games']} games; class balance "
        f"{m['class_balance']} (1 = fastball: {', '.join(res['fastball_types'])}).",
        "",
        "## Does body language beat game context? (held-out games)",
        "",
        "| model | log loss (context) | log loss (context+body) | lift | 95% CI (games) | AUC ctx -> full | accuracy ctx -> full | perm. p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for kind, r in res["models"].items():
        p = "n/a" if r["perm_p_value"] is None else f"{r['perm_p_value']:.3f}"
        lines.append(
            f"| {kind}{' (primary)' if kind == res['primary_model'] else ''} | {r['context_logloss']:.4f} | "
            f"{r['full_logloss']:.4f} | {r['lift_logloss']:+.4f} | {_ci(r['lift_ci'])} | "
            f"{r['context_auc']:.3f} -> {r['full_auc']:.3f} | {r['context_accuracy']:.3f} -> {r['full_accuracy']:.3f} | {p} |"
        )
    lines += [
        "",
        "How to read this: **lift** is the held-out log-loss improvement from adding body cues "
        "(> 0 = better predictions than context alone); the CI resamples whole games. The "
        "**permutation p** asks a weaker question: do the real body features beat the same "
        "features shuffled within each game? A small p with a lift CI that includes 0 means "
        "some association exists but it does not make predictions better than game context. "
        "A tip is declared only when p < 0.05 AND the lift CI is entirely above 0 (fixed in advance).",
        "",
        f"Majority-class reference: log loss {m['majority_logloss']:.4f}, accuracy {m['majority_accuracy']:.3f}. "
        f"Context accuracy {m['context_accuracy']:.3f}; context+body {m['full_accuracy']:.3f}.",
        "",
        "## Controls",
        "",
        f"- Shuffled labels within game ({shuf['n_repeats']} repeats): mean lift {shuf['mean_lift']:+.4f} "
        f"(sd {shuf['sd_lift']:.4f}). Expected ~0 or negative.",
        f"- Post-separation frames deliberately included (positive control): lift {leak['lift_logloss']:+.4f} {_ci(leak['lift_ci'])}, "
        f"AUC {leak['context_auc']:.3f} -> {leak['full_auc']:.3f} (AUC lift CI {_ci(leak['lift_auc_ci'])}), accuracy "
        f"{leak['context_accuracy']:.3f} -> {leak['full_accuracy']:.3f}; real pipeline: lift {m['lift_logloss']:+.4f}, "
        f"AUC lift CI {_ci(m['lift_auc_ci'])}, accuracy {m['context_accuracy']:.3f} -> {m['full_accuracy']:.3f}.",
    ]
    if g:
        lines.append(
            f"- Secondary model (gradient boosting) lift {g['lift_logloss']:+.4f} {_ci(g['lift_ci'])}."
        )
    lines += ["", "## Strongest body cues (held-out permutation importance)", ""]
    if not res["scouting_notes"]:
        lines.append("No body feature improved held-out log loss when present.")
    for n in res["scouting_notes"]:
        lines.append(
            f"- {n['sentence']} (effect d = {n['effect_size_d']:+.2f}, game-bootstrap CI "
            f"[{n['effect_ci'][0]:+.2f}, {n['effect_ci'][1]:+.2f}]; {n['qualifier']})"
        )
    lines += [
        "",
        "## Data quality",
        "",
        f"- Clip funnel ({pq['pitches_sampled']} sampled pitches): {pq['funnel']}",
        f"- Key-joint (wrists, shoulders) confidence >= {pq['good_visibility_threshold']}: median share of "
        f"window frames {pq['key_joint_good_share_median']:.2f}; clips below {pq['min_good_frame_share']} dropped.",
        f"- Usable share by home park: {pq['usable_share_by_home_park']}",
    ]
    if rv:
        a, q = rv["detector_all"], rv.get("after_pose_qc", rv["detector_all"])
        lines.append(
            f"- Release window vs {a['n_labelled']} hand-labelled clips: raw detector found "
            f"{a['n_keep_detected']}/{a['n_keep_labels']} labelled separations (MAE {a['separation_mae_frames']:.1f} frames, "
            f"latest {a['max_late_error_frames']} frames late). On the {q['n_keep_detected']} labelled clips that pass pose QC "
            f"(i.e. reach the model): MAE {q['separation_mae_frames']:.1f} frames, latest {q['max_late_error_frames']} frames "
            f"late vs a {q['guard_frames']}-frame guard, cutoff after true separation in {q['cutoff_after_true_separation']} clips; "
            f"{q['exclude_correctly_rejected']}/{q['n_exclude_labels']} clips labelled 'exclude' rejected."
        )
    lines += [
        "",
        f"_config {res['config_hash']}, code {res['code_revision']}, generated {res['created_utc']}_",
    ]
    return "\n".join(lines) + "\n"
