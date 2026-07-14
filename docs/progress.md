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

## Milestone 3 — Pitcher crop and pose ✅

- [x] **Pose estimator: torchvision Keypoint R-CNN** (`ml/pose/estimator.py`).
      MediaPipe has **no Intel-mac wheel** (only linux/arm64-mac/windows), so
      this is the documented CPU-compatible fallback (CLAUDE.md §2). One model
      gives person detection + 17 COCO keypoints; weights (~230 MB) download to
      `~/.cache/torch` on first use. Verified: weights downloaded and CPU
      inference ran on this machine (0 detections on a noise frame, as
      expected).
- [x] **No OpenCV** (revised from the M2 plan). The 4.x sdist build still
      fails at the cmake install step, and 5.x wheels force numpy≥2 which
      breaks torch 2.2.2's numpy bridge. Instead: frame-accurate decode/encode
      through FFmpeg subprocess pipes (`ml/pose/video_io.py`; seek accuracy is
      asserted by tests against an index-encoded fixture video) and PIL for
      overlay drawing.
- [x] Manual initial pitcher box: `initial_pitcher_box` on clip create/PATCH,
      validated against video geometry, stored on `PitchClip`.
- [x] Tracking (`ml/pose/tracker.py`): per-frame detection in a search region
      expanded around the previous box + IoU association; misses are carried
      forward and *counted*, never invented; offline centred box smoothing.
- [x] Pipeline leakage guard (`ml/pose/pipeline.py`): refuses any frame at or
      after release *or* beyond `pre_release_end_frame`, checked inside the
      frame loop; FFmpeg is only ever asked for the pre-release range.
- [x] `PoseFrame` persistence (keypoints, visibility, smoothed bbox,
      search-region transform, release-relative timestamps) + `PoseJob`
      background jobs with queued/running/completed/failed status, stored
      tracker config, quality summary, and readable errors (stack persisted in
      development mode).
- [x] Quality checks: detection fraction, max consecutive misses, max box
      centre jump → `tracking_ok` + human-readable issues on the job.
- [x] Visualization: annotated MP4 (tracked box, skeleton, per-frame
      "N ms before release" label) via `POST /clips/{id}/pose/render` or
      `scripts/estimate_pose.py --clip-id N --render`; output under
      `artifacts/overlays/` (override root with `PITCH_ARTIFACTS_DIR`).
- [x] Tests: 90 passing (53 new) — geometry, tracker coordinate mapping via a
      blob-finding fake estimator, pipeline leakage/timestamps/quality, FFmpeg
      reader frame accuracy (index-encoded fixture) + writer round trip, and
      an end-to-end API flow (clip → job → frames → rendered MP4) with the
      estimator faked through dependency injection.
- [x] Verified: `uv run ruff check .`, `uv run ruff format --check .`,
      `uv run mypy backend ml scripts`, `uv run pytest` (90 passed), uvicorn
      boot with pose routes registered, real-model load + CPU inference.

### Milestone 3 decisions / limitations

- **Tracking = per-frame detection + association**, not a correlation tracker:
  it cannot drift silently — every frame either has a real detection or is an
  honest miss surfaced in the quality summary.
- **Real-pitcher validation pending**: the full pipeline is verified with fake
  estimators and synthetic video; the acceptance criterion (stable box +
  skeleton on real pitching footage) still needs a real clip, which the user
  must supply. Expect roughly 1–3 s/frame CPU inference on this machine.
- **Schema changed without migrations** (new tables + `initial_box_*` columns
  on `pitch_clip`). There is no real data yet: delete `data/pitchdetector.db`
  and let startup recreate it. Add a migration tool only when there is data
  worth migrating.
- `.env.example` could not be updated in this session (file access denied);
  `PITCH_ARTIFACTS_DIR` is documented in the README instead.
- Keypoint visibility scores are Keypoint R-CNN heatmap scores (unbounded),
  not probabilities — stored raw; downstream consumers must not treat them as
  calibrated.

## Milestone 3.5 — Automatic release-frame labeling ✅

Added to the roadmap 2026-07-07 (see CLAUDE.md §9 and §26) to reduce manual
labeling to review-and-confirm at dataset scale (200+ clips per pitch type).

- [x] Motion gate (`ml/release/motion.py`): frame differencing over
      FFmpeg-downscaled frames locates the delivery window; raises on static
      video.
- [x] Coarse-to-fine detector (`ml/release/detector.py`): subsampled pose over
      the motion window finds the fastest-wrist candidate, then a dense pass
      refines it — full-clip dense pose is too slow on the Intel-Mac CPU. Tests
      confirm it reads far fewer frames than a dense pass.
- [x] Wrist-kinematics release detection: throwing-wrist speed peaks at
      release; emits candidate frame + heuristic confidence (peak prominence x
      detection coverage x edge penalty) + `review_recommended`. Confidence is
      explicitly **not** a calibrated probability — it only ranks the review
      queue.
- [x] `release_frame_source` (`manual` / `auto` / `auto_confirmed`) +
      `release_frame_confidence` on `PitchClip`. `apply_auto_release` refuses
      to overwrite a manual/confirmed label; changing `release_frame` via PATCH
      without an explicit source reverts it to `manual` and clears confidence.
- [x] Cross-reference gate (`validate_release_detector`): runs the detector
      against manually labeled clips, writes a per-clip frame-error report to
      `artifacts/reports/`, and reports the fraction within tolerance
      (acceptance target: >= 90% within +/- 1 frame). Detector failures are
      recorded per clip, not hidden.
- [x] Review queue: `GET /api/v1/release-review` lists auto clips below the
      confidence threshold.
- [x] Background job + API: `POST/GET /api/v1/videos/{id}/release-detection`
      (persisted queued/running/completed/failed, stored config + detector
      version + full result).
- [x] Batch CLI (`scripts/detect_release.py`): detect for one video or all
      unclipped videos, optionally creating auto-labeled clips; validation CLI
      (`scripts/validate_release_detector.py`) prints the cross-reference gate
      result.
- [x] Tests (23 new, 113 total): motion gate, detector accuracy on a synthetic
      delivery with a known release frame, coarse-to-fine efficiency,
      low-confidence flagging, provenance no-overwrite rules, and an
      end-to-end API + real-MP4 flow with the estimator faked.
- [x] Verified: ruff, ruff format, `mypy backend ml scripts`, pytest (113).

### Milestone 3.5 decisions / limitations

- **Detector accuracy is unproven on real footage.** It is validated only on a
  synthetic moving-blob delivery; the >= 90%-within-1-frame acceptance bar can
  only be checked once you have manually labeled real clips to cross-reference
  against. That is the intended workflow: mark ~20-30 by hand, run the
  validation CLI, then trust the auto-labeler for the rest.
- **Labeling UI still deferred.** The manual subset and review-queue
  confirmation currently go through the API / CLIs. A Gradio labeling screen is
  the next practical build (needed to make manual marking painless).
- The detector uses the highest-scoring person detection (no manual box exists
  at label time); this relies on the MVP's single-pitcher centre-field
  assumption. Multi-person frames may need the box-first path later.

## Next: Milestone 4 — Dataset audit and safe splits

- [ ] Counts by pitch type and game; duplicate detection.
- [ ] Grouped train/validation/test split (by game) + leakage assertions.
- [ ] Dataset audit report.
