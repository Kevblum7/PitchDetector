# Pitch Tip Detector

Local-first video analysis tool to investigate whether an MLB pitcher may be
**tipping pitches** before ball release. It is a research and coaching aid — it
surfaces *potential* visual tells with supporting evidence, not definitive
accusations. See [`CLAUDE.md`](./CLAUDE.md) for the full product spec.

> **Status: Milestone 1** — project foundation only. This milestone delivers a
> FastAPI app with a health check and an environment-diagnostics endpoint plus
> CLI. No ML, pose estimation, video processing, or frontend yet.

## Requirements

- **Python 3.11** (the project targets `>=3.11,<3.12`).
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.
- **FFmpeg** on your `PATH` (needed later for video ingestion; the diagnostics
  report whether it is available).

### Hardware notes (Intel Mac target)

The MVP targets a 2020 Intel MacBook Pro and runs **CPU-first**. It does not
require CUDA or Apple Silicon. Pins chosen for the Intel-Mac baseline:

- **PyTorch `2.2.2`** — the last release that ships `x86_64` macOS (Intel)
  wheels; newer PyTorch drops Intel-Mac support.
- **`numpy<2`** — has an Intel-mac wheel and matches torch 2.2.2's build.
- **OpenCV is an optional extra**, not a core dependency. There is no prebuilt
  OpenCV wheel for Intel macOS in this environment, so installing it compiles
  from source (slow, several minutes). It is not needed for the health +
  diagnostics foundation (Milestone 1) — the app imports `cv2` lazily and the
  diagnostics report it as unavailable when absent. Install it when you reach
  the video/pose work (Milestone 3):

  ```bash
  uv sync --extra video    # builds opencv-python-headless from source
  ```

On Apple Silicon or Linux these pins still install; adjust only if you
deliberately move off the Intel-Mac baseline.

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(Alternatively `brew install uv`.) Restart your shell or `source` your profile
so `uv` is on your `PATH`.

### 2. Create and sync the environment

```bash
uv sync
```

This creates `.venv/` and installs runtime + dev dependencies (pytest, ruff,
mypy, httpx).

## Running

### API server

```bash
uv run uvicorn backend.app.main:app --reload
```

Then:

- Health check: <http://127.0.0.1:8000/health>
- Diagnostics:  <http://127.0.0.1:8000/api/v1/system/diagnostics>
- Interactive docs: <http://127.0.0.1:8000/docs>

### Environment diagnostics (CLI)

```bash
uv run python scripts/diagnostics.py          # human-readable
uv run python scripts/diagnostics.py --json    # machine-readable JSON
```

## Development checks

```bash
uv run pytest                    # tests
uv run ruff check .              # lint
uv run ruff format --check .     # formatting (use `ruff format .` to fix)
uv run mypy backend              # type-check
```

## Project layout (Milestone 1)

```text
backend/app/
  main.py               # FastAPI app + /health
  core/config.py        # app name/version/env
  core/diagnostics.py   # environment probing (FFmpeg/OpenCV/PyTorch)
  api/v1/system.py      # GET /api/v1/system/diagnostics
scripts/diagnostics.py  # CLI wrapper for diagnostics
tests/                  # health + diagnostics + arch/ffmpeg unit tests
docs/progress.md        # implementation checklist
```
# PitchDetector
