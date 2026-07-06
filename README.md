# Pitch Tip Detector

Local-first video analysis tool to investigate whether an MLB pitcher may be
**tipping pitches** before ball release. It is a research and coaching aid — it
surfaces *potential* visual tells with supporting evidence, not definitive
accusations. See [`CLAUDE.md`](./CLAUDE.md) for the full product spec.

> **Status: Milestone 2** — health + diagnostics (M1) plus SQLite-backed
> pitcher/video/clip ingestion and manual labeling over a v1 API (M2). Clip
> windows are leakage-safe (no frame at/after release enters the model window).
> No ML, pose estimation, or frontend yet.

## Requirements

- **Python 3.11** (the project targets `>=3.11,<3.12`).
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.
- **FFmpeg** on your `PATH` — `ffprobe` is used to extract video metadata; the
  diagnostics report whether it is available.

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

On startup the app creates a SQLite database at `data/pitchdetector.db`
(override with `PITCH_DATABASE_URL`).

### Environment diagnostics (CLI)

```bash
uv run python scripts/diagnostics.py          # human-readable
uv run python scripts/diagnostics.py --json    # machine-readable JSON
```

### Ingestion & labeling API (Milestone 2)

Videos are **registered by local path** (the original file is preserved in
place — no upload/copy). All endpoints are under `/api/v1`:

| Method & path | Purpose |
|---|---|
| `POST /pitchers`, `GET /pitchers`, `GET /pitchers/{id}` | Manage pitchers (projects) |
| `POST /videos` | Register a local video by path; extracts ffprobe metadata + checksum (deduplicates identical content) |
| `GET /videos/{id}`, `GET /videos/{id}/clips` | Read a video / list its clips |
| `POST /videos/{id}/clips` | Create a clip: mark release frame, set the guard, label the pitch type |
| `GET /clips/{id}`, `PATCH /clips/{id}` | Read / update a clip (frame changes re-validate the leak-free window) |

**Leakage safety:** a clip stores `pre_release_end_frame = release_frame -
release_guard_frames` (guard ≥ 1). The model window is `[start_frame,
pre_release_end_frame]`; the clip's `end_frame` can never extend past that
cutoff, so no frame at or after release can enter the window.

Example:

```bash
curl -X POST localhost:8000/api/v1/pitchers \
  -H 'content-type: application/json' \
  -d '{"name":"Zack Wheeler","throws":"R"}'

curl -X POST localhost:8000/api/v1/videos \
  -H 'content-type: application/json' \
  -d '{"pitcher_id":1,"file_path":"/abs/path/game.mp4","game_id":"2024-06-01"}'

curl -X POST localhost:8000/api/v1/videos/1/clips \
  -H 'content-type: application/json' \
  -d '{"start_frame":0,"release_frame":120,"release_guard_frames":3,"pitch_type":"slider"}'
```

## Development checks

```bash
uv run pytest                    # tests
uv run ruff check .              # lint
uv run ruff format --check .     # formatting (use `ruff format .` to fix)
uv run mypy backend              # type-check
```

## Project layout

```text
backend/app/
  main.py                    # FastAPI app, router wiring, DB-init lifespan
  core/config.py             # app metadata, storage/DB config, leakage defaults
  core/diagnostics.py        # environment probing (FFmpeg/OpenCV/PyTorch)
  core/enums.py              # pitch type, handedness, quality status, ...
  db/models.py               # SQLModel tables: Pitcher, SourceVideo, PitchClip
  db/session.py              # engine + get_session dependency + init_db
  schemas/requests.py        # Pydantic request bodies (typed API boundary)
  services/
    clip_frames.py           # leakage-safe clip windowing (pure)
    checksum.py              # SHA-256 for duplicate detection
    video_metadata.py        # ffprobe wrapper
    video_files.py           # local video path validation
  api/v1/
    system.py                # GET /api/v1/system/diagnostics
    pitchers.py  videos.py  clips.py
scripts/diagnostics.py       # CLI wrapper for diagnostics
tests/
  unit/                      # clip frames, checksum, ffprobe, path validation
  integration/               # pitcher -> video -> clip end-to-end flow
  test_health.py test_diagnostics.py
docs/progress.md             # implementation checklist
```
