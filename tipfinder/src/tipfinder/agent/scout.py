"""Scouting-report agent (OpenAI Agents SDK). Reads OPENAI_API_KEY from the environment.

The agent can run the pipeline through tools but cannot change its numbers. A
deterministic post-check enforces the verdict: if the pipeline found no tip, the report
must say "no detectable tip" in plain words.
"""

from __future__ import annotations

import json
import os

from tipfinder.agent import tools

INSTRUCTIONS = """You are an advance scout's analytics assistant for an MLB club.
Use the tools to get results for the requested pitcher, then write a short scouting report
(<= 250 words, Markdown) with these sections: Verdict, The tell, Confidence, Sample, Caveats.

Rules:
- Every number you state must come from a tool result. Do not estimate or invent numbers.
- The verdict comes ONLY from the `verdict` / `tip_detected` fields. If tip_detected is
  false, the first line must be exactly "**Verdict: no detectable tip**", and any cues you
  mention must be labelled exploratory, not as a tell.
- If positive_control_fired is false, say plainly that the result is INCONCLUSIVE: the
  pipeline could not detect pitch type even from post-separation frames for this pitcher.
- Explain lift as improvement over a game-context baseline (count, runners, batter hand,
  previous pitch, pitch mix) on games the model never saw.
- Translate cues into plain scouting language (e.g. "glove sits higher before changeups").
- Caveats must cover: sample size, pose-estimation noise from broadcast video, camera
  angle differences across parks, and that correlation is not proof a hitter could use it.
- If results are missing, call run_pipeline first. Prefer get_metrics when results exist.
"""


def build_agent():  # type: ignore[no-untyped-def]
    from agents import Agent, function_tool

    @function_tool
    def get_metrics(config_path: str) -> str:
        """Return the latest evaluation metrics for a pitcher config (e.g. configs/wheeler_2024.yaml)."""
        return json.dumps(tools.get_metrics(config_path), default=str)

    @function_tool
    def run_pipeline(config_path: str) -> str:
        """Rebuild features, retrain and re-evaluate a pitcher (slow). Returns metrics."""
        return json.dumps(tools.run_pipeline(config_path), default=str)

    @function_tool
    def fetch_pitcher_data(config_path: str) -> str:
        """Download Statcast data and broadcast clips for a pitcher config (slow, cached)."""
        return json.dumps(tools.fetch_pitcher_data(config_path), default=str)

    @function_tool
    def extract_clips(config_path: str) -> str:
        """Run pose estimation and release-window detection on downloaded clips (slow)."""
        return json.dumps(tools.extract_clips(config_path), default=str)

    return Agent(
        name="tip-scout",
        instructions=INSTRUCTIONS,
        tools=[get_metrics, run_pipeline, fetch_pitcher_data, extract_clips],
        model=os.environ.get("TIPFINDER_AGENT_MODEL") or None,
    )


def enforce_verdict(text: str, tip_detected: bool) -> str:
    """Deterministic guard: a no-tip result can never be reported as a tip."""
    line = "**Verdict: no detectable tip**"
    if not tip_detected and "no detectable tip" not in text.lower():
        return f"{line}\n\n{text}"
    return text


def run_scout(config_path: str) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set; the deterministic report (report.md) is still available."
        )
    from agents import Runner

    result = Runner.run_sync(
        build_agent(),
        f"Write the scouting report for config {config_path}.",
        max_turns=8,
    )
    # Read the verdict AFTER the run: the agent may have re-run the pipeline via a tool.
    metrics = tools.get_metrics(config_path)
    return enforce_verdict(str(result.final_output), bool(metrics["tip_detected"]))
