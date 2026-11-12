"""Command line: fetch, extract, train, evaluate, report."""

from __future__ import annotations

import json
import logging

import typer

from tipfinder.config import load_config

app = typer.Typer(add_completion=False, help="Pitch-tipping detection for one pitcher.")
CFG = typer.Option("configs/wheeler_2024.yaml", "--config", "-c", help="Run config YAML.")


@app.callback()
def main() -> None:
    """tipfinder: does pre-separation body language beat game context?"""


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


@app.command()
def fetch(config: str = CFG, limit: int = typer.Option(None, help="Max clips this run.")) -> None:
    """Statcast pitches -> Savant play_ids -> broadcast clips (cached, resumable)."""
    _setup_logging()
    from tipfinder.data.fetch import fetch as run_fetch

    man = run_fetch(load_config(config), limit=limit)
    typer.echo(json.dumps(man["status"].value_counts().to_dict()))


@app.command()
def extract(
    config: str = CFG,
    workers: int = typer.Option(4, help="Parallel processes."),
    limit: int = typer.Option(None, help="Max clips this run."),
    redo: bool = typer.Option(False, help="Discard previous extraction results."),
    redetect: bool = typer.Option(False, help="Only re-run window detection on stored pose."),
) -> None:
    """CF shot -> pitcher pose -> set/separation window -> pose quality (resumable)."""
    _setup_logging()
    from tipfinder.pose.extract import extract_all, redetect_all, reset_extract

    cfg = load_config(config)
    if redetect:
        res = redetect_all(cfg)
        typer.echo(json.dumps(res["status"].value_counts().to_dict()))
        return
    if redo:
        reset_extract(cfg)
    res = extract_all(cfg, workers=workers, limit=limit)
    typer.echo(json.dumps(res["status"].value_counts().to_dict()))


@app.command()
def train(config: str = CFG) -> None:
    """Build feature tables (real + leakage control) and fit final models on all games."""
    _setup_logging()
    from tipfinder.pipeline import train as run_train

    typer.echo(json.dumps(run_train(load_config(config)), indent=2))


@app.command()
def evaluate(
    config: str = CFG,
    target: str = typer.Option("binary", help="binary (fastball vs offspeed) or multiclass."),
    permutations: int = typer.Option(None, help="Override number of permutations."),
) -> None:
    """Game-grouped CV: context vs context+body, bootstrap CI, permutation test, controls."""
    _setup_logging()
    from tipfinder.pipeline import evaluate as run_eval

    res = run_eval(load_config(config), target=target, n_permutations=permutations)
    typer.echo(f"{res['verdict']} | lift {res['models']['logreg']['lift_logloss']:+.4f}")


@app.command()
def report(
    config: str = CFG,
    agent: bool = typer.Option(False, help="Also have the LLM agent write a scouting report."),
) -> None:
    """Render results.json as Markdown (and optionally an agent-written scouting report)."""
    from tipfinder.report import render_markdown

    cfg = load_config(config)
    res = json.loads((cfg.run_dir / "results.json").read_text())
    md = render_markdown(res)
    out = cfg.run_dir / "report.md"
    out.write_text(md)
    typer.echo(md)
    if agent:
        from tipfinder.agent.scout import run_scout

        text = run_scout(config)
        (cfg.run_dir / "scouting_report.md").write_text(text)
        typer.echo(text)


@app.command("label-sheet")
def label_sheet(
    pitch_key: str,
    config: str = CFG,
    start: int = typer.Option(None, help="First frame (default: CF shot start)."),
    end: int = typer.Option(None, help="Last frame (default: start + 100)."),
    step: int = typer.Option(2),
    out: str = typer.Option(None, help="Output JPEG path."),
) -> None:
    """Render a frame-numbered pitcher contact sheet for hand-labelling set/separation."""
    from pathlib import Path

    import pandas as pd

    from tipfinder.window.label_sheet import make_sheet

    cfg = load_config(config)
    man = pd.read_csv(cfg.run_dir / "manifest.csv").set_index("pitch_key")
    ext = pd.read_csv(cfg.run_dir / "extract.csv").set_index("pitch_key")
    s = int(ext.loc[pitch_key, "shot_start"]) if start is None else start
    e = s + 100 if end is None else end
    dest = Path(out) if out else cfg.run_dir / "label_sheets" / f"{pitch_key}.jpg"
    make_sheet(
        cfg.run_dir / man.loc[pitch_key, "clip_path"],
        cfg.run_dir / "pose" / f"{pitch_key}.npz",
        dest,
        s,
        e,
        step,
    )
    typer.echo(str(dest))


@app.command("build-test-cache")
def build_test_cache(config: str = CFG, n: int = typer.Option(16)) -> None:
    """Copy a small sample of REAL clips/pose/responses into tests/.cache (git-ignored)."""
    from tipfinder.testcache import build

    typer.echo(build(load_config(config), n))


if __name__ == "__main__":
    app()
