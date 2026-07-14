# Pitch Tip Detector

Local-first video analysis tool to investigate whether an MLB pitcher may be
**tipping pitches** before ball release. It is a research and coaching aid — it
surfaces *potential* visual tells with supporting evidence, not definitive
accusations. See [`CLAUDE.md`](./CLAUDE.md) for the full product spec.

> **Status: Milestone 3.5** — health + diagnostics (M1), SQLite-backed
> pitcher/video/clip ingestion and manual labeling (M2), pitcher tracking +
> pose extraction + annotated overlay rendering (M3), and automatic
> release-frame labeling with provenance + a cross-reference gate (M3.5). Clip
> windows are leakage-safe (no frame at/after release enters the model window),
> and the pose pipeline refuses to decode past the pre-release cutoff. No
> training or frontend yet.

## Requirements

- **Python 3.11** (the project targets `>=3.11,<3.12`).
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.
- **FFmpeg** on your `PATH` — `ffprobe` is used to extract video metadata; the
  diagnostics report whether it is available.

### Hardware notes (Intel Mac target)

The MVP targets a 2020 Intel MacBook Pro and runs **CPU-first**. It does not
require CUDA or Apple Silicon. Pins chosen for the Intel-Mac baseline:

- **PyTorch `2.2.2` + torchvision `0.17.2`** — the last releases that ship
  `x86_64` macOS (Intel) wheels; newer versions drop Intel-Mac support.
- **`numpy<2`** — has an Intel-mac wheel and matches torch 2.2.2's build.
- **Pose estimator: torchvision Keypoint R-CNN** (CPU). MediaPipe ships no
  Intel-mac wheel at all, so this is the documented fallback: one model
  provides both person detection and 17 COCO keypoints. Its pretrained weights
  (~230 MB) are downloaded to `~/.cache/torch` on first use.
- **No OpenCV.** No working Intel-mac wheel exists in this environment (4.x is
  sdist-only and its source build fails; 5.x requires numpy≥2, which breaks
  torch 2.2.2). All video decode/encode goes through FFmpeg subprocess pipes
  (`ml/pose/video_io.py`) and overlay drawing uses PIL — both wheel-installable
  everywhere. The diagnostics report `cv2` as unavailable; that is expected.

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
| `GET /clips/{id}`, `PATCH /clips/{id}` | Read / update a clip (frame changes re-validate the leak-free window; `initial_pitcher_box` sets the manual tracking box) |
| `POST /clips/{id}/pose` | Start pose extraction as a background job (requires an initial pitcher box) |
| `GET /clips/{id}/pose` | Latest pose job: queued/running/completed/failed + quality summary |
| `GET /clips/{id}/pose/frames` | Stored per-frame pose (keypoints, boxes, release-relative timestamps) |
| `POST /clips/{id}/pose/render` | Render the annotated overlay MP4 (box + skeleton) to `artifacts/overlays/` |
| `POST /videos/{id}/release-detection` | Auto-detect the release frame (background job) |
| `GET /videos/{id}/release-detection` | Latest release-detection job: status + detected frame + confidence |
| `GET /release-review` | Auto-labeled clips below the confidence threshold (the review queue) |

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

### Pose extraction & overlay (Milestone 3)

The user marks the pitcher once with an initial box; tracking then follows the
pitcher by per-frame person detection (torchvision Keypoint R-CNN, CPU) with
IoU association inside a search region, smooths the boxes, and stores 17 COCO
keypoints per frame with release-relative timestamps. **Only frames in
`[start_frame, pre_release_end_frame]` are ever decoded** — the pipeline
raises rather than touch a frame at or after release.

```bash
# 1. Set the initial pitcher box (original-video pixels)
curl -X PATCH localhost:8000/api/v1/clips/1 \
  -H 'content-type: application/json' \
  -d '{"initial_pitcher_box":{"x":800,"y":300,"width":220,"height":420}}'

# 2. Extract pose (background job; first run downloads ~230 MB of weights)
curl -X POST localhost:8000/api/v1/clips/1/pose
curl localhost:8000/api/v1/clips/1/pose          # poll status + quality
curl localhost:8000/api/v1/clips/1/pose/frames   # per-frame keypoints

# 3. Render the annotated MP4 (tracked box + skeleton)
curl -X POST localhost:8000/api/v1/clips/1/pose/render
```

Or from the command line (synchronous, with progress output):

```bash
uv run python scripts/estimate_pose.py --clip-id 1 --render
```

Overlays are written to `artifacts/overlays/` (override the artifacts root
with `PITCH_ARTIFACTS_DIR`). Each pose job stores its tracker configuration
and a tracking-quality summary (detection rate, miss streaks, box jumps) so
bad tracking is reported, not hidden.

### Automatic release-frame labeling (Milestone 3.5)

Marking the release frame by hand for hundreds of clips is the main labeling
cost, so the system detects it automatically: a cheap motion gate finds the
delivery window, then a coarse-to-fine pose pass locates the frame where the
throwing wrist reaches peak speed. Each detection carries a **confidence**
score and a **provenance** tag — and an automatic label never silently
overwrites one you set by hand.

The intended workflow is *validate small, then trust*:

```bash
# 1. Manually mark ~20-30 release frames (via PATCH /clips/{id}), then check
#    the detector against them. Writes a per-clip error report and prints
#    whether it clears the >= 90%-within-1-frame gate.
uv run python scripts/validate_release_detector.py --tolerance 1

# 2. Once it passes, batch-label the rest. Film Room clips are one pitch per
#    file, downloaded per pitch type, so a whole batch shares one label:
uv run python scripts/detect_release.py --all-unclipped --create-clip --pitch-type slider

# Low-confidence detections are flagged for review:
curl localhost:8000/api/v1/release-review
```

Or per video over the API (background job):

```bash
curl -X POST localhost:8000/api/v1/videos/1/release-detection
curl localhost:8000/api/v1/videos/1/release-detection   # status + result
```

**Provenance rule:** `release_frame_source` is `manual`, `auto`, or
`auto_confirmed`. Automatic labels only fill in clips that are unset or already
`auto`; confirming or hand-editing a frame locks it. Confidence ranks the
review queue — it is a heuristic, not a calibrated probability. The
cross-reference report against manual marks is what earns trust at scale.

## Development checks

```bash
uv run pytest                    # tests
uv run ruff check .              # lint
uv run ruff format --check .     # formatting (use `ruff format .` to fix)
uv run mypy backend ml scripts   # type-check
```

## Project layout

```text
backend/app/
  main.py                    # FastAPI app, router wiring, DB-init lifespan
  core/config.py             # app metadata, storage/DB config, leakage defaults
  core/diagnostics.py        # environment probing (FFmpeg/OpenCV/PyTorch)
  core/enums.py              # pitch type, handedness, quality status, ...
  db/models.py               # SQLModel tables: Pitcher, SourceVideo, PitchClip,
                             #   PoseFrame, PoseJob
  db/session.py              # engine + session dependencies + init_db
  schemas/requests.py        # Pydantic request bodies (typed API boundary)
  services/
    clip_frames.py           # leakage-safe clip windowing (pure)
    checksum.py              # SHA-256 for duplicate detection
    video_metadata.py        # ffprobe wrapper
    video_files.py           # local video path validation
    pitcher_box.py           # initial-box validation
    pose_extraction.py       # pose job execution + persistence + render
    release_detection.py     # release job, provenance guards, validation
  api/v1/
    system.py                # GET /api/v1/system/diagnostics
    pitchers.py  videos.py  clips.py  pose.py  release.py
ml/pose/
  geometry.py                # bounding boxes, IoU, smoothing (pure)
  estimator.py               # PoseEstimator protocol + Keypoint R-CNN impl
  video_io.py                # frame-accurate FFmpeg decode/encode pipes
  tracker.py                 # detection + IoU association tracking
  pipeline.py                # orchestration + leakage guard + quality checks
  visualize.py               # box + skeleton overlay rendering (PIL)
ml/release/
  motion.py                  # motion-gate delivery-window detection (pure)
  detector.py                # coarse-to-fine wrist-kinematics release detector
  pipeline.py                # video-file orchestration for the detector
scripts/
  diagnostics.py             # CLI wrapper for diagnostics
  estimate_pose.py           # CLI pose extraction (+ --render)
  detect_release.py          # CLI/batch automatic release labeling
  validate_release_detector.py  # cross-reference gate vs manual marks
tests/
  unit/                      # clip frames, checksum, ffprobe, path validation,
                             #   geometry, tracker, pipeline leakage, video I/O,
                             #   motion gate, release detector, provenance
  integration/               # video->clip; clip->pose->overlay; release flow
  test_health.py test_diagnostics.py
docs/progress.md             # implementation checklist
```
