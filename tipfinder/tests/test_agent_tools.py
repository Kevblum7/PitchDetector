"""Agent tools work without the LLM; the verdict guard cannot be talked around."""

from __future__ import annotations

import pytest

from tipfinder.agent.scout import enforce_verdict
from tipfinder.agent.tools import PROJECT, get_metrics
from tipfinder.report import render_markdown


def test_no_tip_verdict_is_forced() -> None:
    text = "**Verdict: tip found** The glove is higher before sliders."
    out = enforce_verdict(text, tip_detected=False)
    assert out.startswith("**Verdict: no detectable tip**")
    assert (
        enforce_verdict("**Verdict: no detectable tip** ...", False).count("no detectable tip") == 1
    )
    assert enforce_verdict("**Verdict: potential tip**", True) == "**Verdict: potential tip**"


def test_get_metrics_without_llm() -> None:
    results = PROJECT / "data" / "zack_wheeler_2024" / "results.json"
    if not results.exists():
        pytest.skip("run `tipfinder evaluate` first")
    m = get_metrics("configs/wheeler_2024.yaml")
    assert m["verdict"] in ("no detectable tip",) or m["verdict"].startswith("potential tip")
    assert m["tip_detected"] == (m["verdict"] != "no detectable tip")
    assert m["primary"]["n_pitches"] > 0 and m["primary"]["n_games"] > 1
    md = render_markdown(__import__("json").loads(results.read_text()))
    if not m["tip_detected"]:
        assert "NO DETECTABLE TIP" in md


def test_verdict_flags_insensitive_pipeline() -> None:
    from tipfinder.report import verdict_line

    def res(tip: bool, leak_lo: float) -> dict:
        m = {
            "tip_detected": tip,
            "verdict": "no detectable tip",
            "perm_p_value": 0.5,
            "lift_ci": [-1, 1],
        }
        return {
            "primary_model": "logreg",
            "models": {"logreg": m},
            "controls": {"post_separation_leak": {"lift_auc_ci": [leak_lo, 0.2]}},
        }

    assert "INCONCLUSIVE" in verdict_line(res(False, -0.05))
    assert "INCONCLUSIVE" not in verdict_line(res(False, 0.01))
    assert "NO DETECTABLE TIP" in verdict_line(res(False, 0.01))
