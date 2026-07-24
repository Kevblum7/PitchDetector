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
  pitch) → `/gf?game_pk=<pk>` per-game feed (carries `play_id`) →
  `/sporty-videos?playId=<id>` (mp4 in initial HTML).
- Accepts a copied browser search URL (`--search-url`, auto-rewritten to CSV)
  or explicit filters (`--player-id/--season/--pitch-type/--dates`). Supports
  `--dry-run`, `--limit`, `--delay`, `--overwrite`; writes a `manifest.json`.
- Zero new dependencies (stdlib `urllib`/`csv`/`re`/`html`/`json`). Polite UA +
  delay.
- Unit tests cover URL building, CSV parsing, game-feed join, mp4 extraction,
  filenames (`tests/unit/test_download_savant_videos.py`). Network I/O isolated.

### Fixes verified against the live endpoint (2026-07-24)

The originally-merged version resolved 0 clips against current Savant. Three
issues were found and fixed:

- **No `play_id` in the CSV.** The details CSV no longer contains a `play_id`
  column, so every row was discarded. Now each pitch is keyed by
  `game_pk` + `at_bat_number` + `pitch_number`, and `play_id` is looked up
  from the `/gf?game_pk=<pk>` per-game feed (one fetch per game).
- **UTF-8 BOM** on the CSV corrupted the first column name (`pitch_type` read
  as empty); stripped in `parse_pitch_rows`.
- **HTML-escaped mp4 URL.** The clip token's `==` padding is served as
  `&#x3D;&#x3D;`; `extract_video_url` now unescapes entities.

Verified: `--player-id 642207 --season 2024 --pitch-type CH` resolves 5/5
`play_id`s and 5/5 mp4 URLs. The actual binary download of the clips is blocked
in the sandbox — the MLB video CDN host `sporty-clips.mlb.com` is denied by the
egress policy (403 on CONNECT). URL resolution and the manifest are unaffected;
the download step should succeed in an environment where that host is allowed.
(The `SL`/April-2024 empty results seen while testing are correct: Williams
threw no sliders in 2024 and was injured that April.)

## Next: Milestone 2 — Video ingestion and manual labeling

- [ ] Video import + `ffprobe` metadata extraction.
- [ ] Clip creation with manual release-frame selection.
- [ ] Pitch label form + clip metadata persistence (SQLite).
- [ ] Frame-accurate preview.
