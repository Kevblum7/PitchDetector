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

## Milestone 2 — Video ingestion and manual labeling (backend) ✅

- [x] SQLite persistence via SQLModel: `Pitcher`, `SourceVideo`, `PitchClip`
      tables (`backend/app/db/models.py`), all clip frame fields stored
      release-relative. Engine/session in `backend/app/db/session.py`; tables
      created on app startup via lifespan.
- [x] `ffprobe` metadata extraction (`services/video_metadata.py`): width,
      height, fps, duration, frame count — `subprocess.run` with an arg list,
      typed errors for missing/failed ffprobe.
- [x] Video registration by local path (no upload/copy; original preserved),
      with path validation (`services/video_files.py`) and SHA-256 checksum
      (`services/checksum.py`) for duplicate detection.
- [x] Leakage-safe clip windowing (`services/clip_frames.py`): `pre_release_end_
      frame = release_frame - guard`, guard >= 1, end never past the cutoff, so
      no frame at/after release enters the model window. Re-validated on PATCH.
- [x] API (v1): `POST/GET /pitchers`, `POST/GET /videos`, `GET /videos/{id}/clips`,
      `POST /videos/{id}/clips`, `GET/PATCH /clips/{id}`.
- [x] Tests (37 total): clip-frame math + leakage, checksum/dedupe, ffprobe
      parse + missing-ffprobe, path validation, and an end-to-end integration
      flow (pitcher → video → clip → reload → update) using a synthetic ffmpeg
      fixture (no private footage).
- [x] Verified: ruff, ruff format, `mypy backend`, pytest, and a live server
      smoke test (create pitcher → register video → create/reload/patch clip →
      leakage guard returns 400).

### Milestone 2 decisions / limitations

- **Videos are registered by path, not uploaded.** Fits the local-first, "video
  already acquired" model and avoids copying large files. Multipart upload can
  be added later.
- **"Project" maps to `Pitcher`**; endpoints live under `/api/v1/pitchers`.
- **Frame-accurate preview and the labeling UI are frontend work** — deferred
  with the rest of the React UI (Milestone 3+). The API fully supports the
  underlying operations.
- Two schema fields extend the minimal spec for reproducibility:
  `SourceVideo.frame_count` and `PitchClip.release_guard_frames`.

## Next: Milestone 3 — Pitcher crop and pose

- [ ] Manual initial pitcher box + tracking across pre-release frames.
- [ ] Pose extraction (CPU-compatible estimator) + `PoseFrame` persistence.
- [ ] Pose visualization + quality checks.
- [ ] Install OpenCV (`uv sync --extra video`) for frame decoding.
