# Implementation Progress

Running checklist for the Pitch Tip Detector MVP. See `CLAUDE.md` §26 for the
full roadmap.

## Milestone 1 — Repository and local environment ✅

- [x] `pyproject.toml` configured for `uv`, Python 3.11.
- [x] Core dependencies: FastAPI, uvicorn, pydantic, numpy (<2), torch (2.2.2
      pinned for Intel-Mac wheels), pytest, httpx, ruff, mypy. OpenCV is an
      optional `video` extra (no Intel-mac wheel; builds from source) — not
      required for this milestone.
- [x] FastAPI app with `GET /health` (name, version, status).
- [x] Environment diagnostics module (`backend/app/core/diagnostics.py`):
      Python version, macOS version, CPU arch, Intel vs Apple Silicon, FFmpeg
      availability + version, OpenCV version, PyTorch version, CUDA/MPS
      availability, recommended execution mode.
- [x] Diagnostics exposed via CLI (`scripts/diagnostics.py`) and
      `GET /api/v1/system/diagnostics`.
- [x] Graceful handling when FFmpeg / OpenCV / PyTorch are unavailable.
- [x] Unit tests: health endpoint, diagnostics endpoint, CPU-arch detection,
      FFmpeg availability handling.
- [x] `.gitignore`, `.env.example`, `README.md`, `docs/progress.md`.
- [x] Verified: `uv sync`, `uv run pytest`, `uv run ruff check .`,
      `uv run ruff format --check .`, `uv run mypy backend`, and uvicorn boot.

### Verified environment (dev machine, 2026-06-29)

- Python 3.11.5, macOS 15.7.4, `x86_64` (Intel), CPU execution mode.
- FFmpeg 7.0.1 available; PyTorch 2.2.2 (CUDA no, MPS no); numpy 1.26.4.
- OpenCV reported unavailable by default (optional `video` extra not installed).

### Known limitations

- No git repository initialized yet (the workspace is not under version
  control); `.gitignore` is in place for when it is.
- OpenCV has no prebuilt Intel-mac wheel here; `uv sync --extra video` compiles
  it from source (slow, and was intermittently failing at the cmake install
  step). Deferred until Milestone 3.
- CUDA/MPS branches of the diagnostics are covered by unit tests but not
  exercised on this Intel-Mac hardware.

## Data ingestion helper — Baseball Savant downloader

- [x] `scripts/download_savant_videos.py` — local research helper to pull a
      single pitcher's clips from Baseball Savant.
- Fixes the "no urls" error: the `/statcast_search` *results page* renders
  players collapsed and injects clip links via JS only after you expand a
  player, so scraping its raw HTML finds nothing. The script instead uses the
  stable data endpoints — `/statcast_search/csv?type=details` (one row per
  pitch, with `play_id`) → `/sporty-videos?playId=<id>` (mp4 in initial HTML).
- Accepts a copied browser search URL (`--search-url`, auto-rewritten to CSV)
  or explicit filters (`--player-id/--season/--pitch-type/--dates`). Supports
  `--dry-run`, `--limit`, `--delay`, `--overwrite`; writes a `manifest.json`.
- Zero new dependencies (stdlib `urllib`/`csv`/`re`). Polite UA + delay.
- Unit tests cover URL building, CSV parsing, mp4 extraction, filenames
  (`tests/unit/test_download_savant_videos.py`). Network I/O is isolated.

## Milestone 2 — Video ingestion and manual labeling ✅ (backend)

- [x] Video import + `ffprobe` metadata extraction
      (`services/video_metadata.py`, `POST /api/v1/videos`).
- [x] Checksum-based duplicate detection; idempotent re-registration.
- [x] Clip creation with manual release-frame selection and leakage-safe
      windowing (`services/clip_frames.py`, `POST /api/v1/videos/{id}/clips`).
- [x] Pitch label fields + clip metadata persistence in SQLite
      (`db/models.py`, `PATCH /api/v1/clips/{id}`).
- [x] Pitcher (project) CRUD (`api/v1/pitchers.py`).
- Remaining for a full UI slice: frame-accurate browser preview (frontend,
  deferred with the rest of the React app).

## Milestone 4 (partial) — Safe splits and dataset audit ✅ (pure-Python core)

Implemented ahead of Milestone 3 because it is dependency-free (no OpenCV/torch)
and addresses the project's stated #1 risk — data leakage (CLAUDE.md §10).

- [x] `ml/datasets/records.py` — `ClipRecord`, a leak-free projection of a
      `PitchClip` carrying only fields safe for splitting/auditing.
- [x] `ml/datasets/splits.py` — game-grouped, seeded, reproducible
      train/val/test splitting. Whole games are assigned to one split so no
      game crosses splits. `split_by_game` re-runs the leakage assertions on
      its own output before returning.
- [x] Leakage guards (the §18 leakage tests assert against these):
      `assert_no_game_overlap`, `assert_no_clip_overlap`,
      `assert_no_duplicate_source_across_splits`,
      `assert_label_not_in_feature_names`.
- [x] `ml/datasets/audit.py` — counts by pitch type / game / (game, pitch) /
      camera angle, duplicate-source detection, class-imbalance ratio, and
      configurable sufficiency warnings (`AuditThresholds`, CLAUDE.md §14).
- [x] Unit tests: `tests/unit/test_splits.py`, `tests/unit/test_audit.py`.
- [x] `ml` added to `mypy` files and the hatch wheel packages.

### Environment note (2026-08-17, web session)

- This web session's network policy **blocks PyPI egress** (gateway returns 403
  to CONNECT for `pypi.org` / `files.pythonhosted.org`), so `uv sync`, `pytest`,
  `ruff`, and `mypy` cannot install/run here. The full suite was last verified
  on the dev Mac.
- The new `ml/datasets` code is pure stdlib. It was verified in-session by
  running an equivalent assertion script under the base interpreter:
  `PYTHONPATH=. .venv/bin/python <script>` → **26 checks passed**. All new
  files byte-compile and are within the 100-char line limit.
- To run the committed pytest suite (once PyPI is reachable):
  `uv sync && uv run pytest tests/unit/test_splits.py tests/unit/test_audit.py`.

## Next

- [ ] Milestone 3 — pitcher crop + pose (needs the optional `video`/OpenCV
      extra; blocked in web sessions without PyPI access).
- [ ] Wire the audit into an API endpoint / dataset-audit screen.
