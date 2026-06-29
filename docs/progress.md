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

## Next: Milestone 2 — Video ingestion and manual labeling

- [ ] Video import + `ffprobe` metadata extraction.
- [ ] Clip creation with manual release-frame selection.
- [ ] Pitch label form + clip metadata persistence (SQLite).
- [ ] Frame-accurate preview.
