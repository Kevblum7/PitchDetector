# CLAUDE.md — Pitch Tip Detection Application

## 1. Project Mission

Build a local-first video analysis application that detects whether an MLB pitcher may be tipping pitches before ball release, identifies the repeatable visual pattern associated with pitch type, and highlights the suspected tell directly on the video.

The application must do more than classify a pitch. It must provide evidence that a pre-release movement or position is:

1. Predictive of pitch type.
2. Repeatable across multiple pitches.
3. Visible before ball release.
4. Not caused by camera angle, game context, broadcast graphics, pitch trajectory, catcher movement, or another data leak.
5. Explainable through overlays, measurements, comparisons, and confidence scores.

The initial product is a research and coaching analysis tool, not a definitive claim that a pitcher is tipping pitches.

---

## 2. Developer Environment and Hardware Constraints

Primary development machine:

- 2020 13-inch MacBook Pro
- Intel processor
- macOS
- Limited local GPU acceleration
- Local development should remain lightweight and CPU-compatible

Design all workflows so that:

- Preprocessing, labeling, UI development, lightweight inference, and testing can run locally.
- Expensive model training can run on a cloud GPU.
- The MVP does not require CUDA.
- Apple Silicon-only dependencies are not required.
- Every dependency must support Intel macOS or have a documented fallback.
- Large video files and model artifacts are excluded from Git.

Prefer Python 3.11 unless a critical ML dependency requires another supported version.

Use `uv` for Python dependency and virtual-environment management unless the repository already uses another tool.

---

## 3. Core Product Requirements

The application must allow a user to:

1. Create or select a pitcher project.
2. Import labeled or unlabeled pitching video.
3. Trim a pitch clip to the pre-release window.
4. Mark or automatically estimate the release frame.
5. Assign pitch labels such as four-seam fastball, sinker, slider, curveball, changeup, cutter, or splitter.
6. Detect and track the pitcher.
7. Normalize the pitcher crop.
8. Estimate body pose across frames.
9. Train or run a model using only information available before release.
10. Show pitch-type probabilities.
11. Detect whether prediction performance is meaningfully better than a baseline.
12. Identify the most predictive region and time window.
13. Draw a tracked square or rectangle around the suspected tell.
14. Optionally display a heatmap and pose overlay.
15. Compare representative examples side by side.
16. Produce a human-readable explanation of the suspected tell.
17. Show supporting sample size, validation accuracy, confidence, and known limitations.
18. Export an annotated MP4 and a JSON analysis report.

Example output:

```text
Potential tell detected: Moderate confidence

Pitcher: Zack Wheeler
Compared pitches: Four-seam fastball vs slider
Pre-release test accuracy: 76%
Baseline accuracy: 57%
Most informative region: Glove and throwing wrist
Most informative interval: 380–170 ms before release

Observed pattern:
The glove is higher and closer to the torso before sliders in 71% of
validated slider clips, compared with 29% of four-seam fastballs.

Evidence:
- 128 four-seam fastballs
- 84 sliders
- Evaluated on games excluded from training
```

The UI must avoid declaring a definitive tip when evidence is weak.

---

## 4. MVP Scope

The first working MVP must intentionally be narrow.

### MVP assumptions

- One pitcher.
- One consistent center-field camera angle.
- Two pitch types.
- User-supplied or already acquired video.
- At least 50 usable examples per pitch type for pipeline testing.
- Preferably 200 or more examples per type for meaningful modeling.
- Clips end before ball release.
- Pitcher-focused crop.
- Pose-based model first.
- RGB video model is optional until the pose pipeline works.
- Local CPU inference.
- Cloud GPU training only when needed.

### MVP success criteria

The MVP is successful when it can:

1. Import and label clips.
2. store clip metadata;
3. Detect or manually set the release frame.
4. Extract pre-release frames.
5. Estimate and visualize pose.
6. Train a lightweight classifier.
7. Evaluate using game-level grouped splits.
8. Beat the majority-class baseline on held-out games.
9. Run region or feature ablation.
10. Highlight a suspected body region with a tracked square.
11. Generate an annotated MP4.
12. Explain why the system considers the pattern potentially meaningful.
13. Reproduce all results from a versioned configuration.

Do not begin with a large video transformer, a universal all-pitcher model, live broadcast ingestion, or automated web scraping.

---

## 5. Non-Goals for the First Version

Do not build these until the MVP is validated:

- Real-time live-game analysis.
- A universal model for every pitcher.
- Automatic ingestion from copyrighted broadcast services.
- Automatic MLB video scraping.
- Betting automation.
- A mobile application.
- Multi-camera fusion.
- Generative counterfactual video.
- Finger-grip recognition through an opaque glove.
- Claims of causal proof based only on attention heatmaps.
- Deployment requiring a dedicated GPU.
- Microservice architecture.

Favor a modular monolith.

---

## 6. Recommended Technology Stack

### Backend and ML

- Python 3.11
- FastAPI
- Pydantic
- SQLModel or SQLAlchemy
- SQLite for MVP
- OpenCV
- FFmpeg through subprocess wrappers
- NumPy
- pandas
- scikit-learn
- PyTorch
- MediaPipe Pose or another CPU-compatible pose estimator
- Albumentations only where appropriate
- pytest
- Ruff
- mypy

### Frontend

Use one of these approaches:

Preferred MVP:

- React
- TypeScript
- Vite
- Tailwind CSS
- HTML5 video
- Canvas or SVG overlay synchronized to video

Acceptable faster prototype:

- Gradio

Use React when implementing the complete annotation and comparison interface. Gradio is acceptable for proving the ML workflow first.

### Packaging and tooling

- `uv`
- npm
- Docker only for reproducible cloud training or deployment; do not require Docker for normal local development
- GitHub Actions for linting and tests
- MLflow or a lightweight local experiment registry after the first model works
- DVC only if dataset versioning becomes difficult; do not add it prematurely

---

## 7. Repository Structure

Use this structure unless there is a strong reason to change it:

```text
pitch-tip-detector/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env.example
├── configs/
│   ├── default.yaml
│   ├── pose_baseline.yaml
│   └── rgb_baseline.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
├── models/
│   └── .gitkeep
├── artifacts/
│   ├── reports/
│   ├── overlays/
│   └── experiments/
├── backend/
│   └── app/
│       ├── main.py
│       ├── api/
│       ├── core/
│       ├── db/
│       ├── schemas/
│       ├── services/
│       └── workers/
├── ml/
│   ├── datasets/
│   ├── preprocessing/
│   ├── pose/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   └── reporting/
├── frontend/
│   ├── package.json
│   └── src/
│       ├── api/
│       ├── components/
│       ├── features/
│       ├── pages/
│       └── types/
├── scripts/
│   ├── ingest_video.py
│   ├── extract_clips.py
│   ├── estimate_pose.py
│   ├── train_pose_model.py
│   ├── evaluate_model.py
│   └── render_overlay.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── docs/
    ├── architecture.md
    ├── data-contract.md
    ├── evaluation.md
    └── model-card.md
```

Never commit raw broadcast video, private datasets, credentials, or large model checkpoints.

---

## 8. Data Model

At minimum, store the following entities.

### Pitcher

```text
id
name
throws
notes
created_at
```

### SourceVideo

```text
id
pitcher_id
file_path
source_name
game_id
game_date
camera_angle
fps
width
height
duration_seconds
checksum
created_at
```

### PitchClip

```text
id
source_video_id
pitcher_id
game_id
start_frame
end_frame
release_frame
release_frame_source
release_frame_confidence
pre_release_end_frame
pitch_type
pitch_family
delivery_type
batter_side
runners_on
camera_angle
quality_status
label_source
notes
```

### PoseFrame

```text
clip_id
frame_index
timestamp_ms_relative_to_release
keypoints
visibility_scores
pitcher_bbox
normalization_transform
```

### Experiment

```text
id
config_hash
dataset_version
model_type
feature_set
split_strategy
random_seed
metrics
artifact_paths
created_at
```

### SuspectedTell

```text
experiment_id
pitcher_id
pitch_type_a
pitch_type_b
body_region
start_ms_before_release
end_ms_before_release
effect_description
effect_size
stability_score
occlusion_delta
confidence_level
supporting_clip_ids
limitations
```

All time-dependent features should be represented relative to the release frame.

---

## 9. Video Ingestion and Preprocessing Rules

### Video handling

- Read metadata with `ffprobe`.
- Decode or transcode using FFmpeg.
- Preserve the original source file.
- Store generated clips in a deterministic directory.
- Create checksums so duplicate files can be detected.
- Never silently overwrite source video.
- Log every transformation.

### Release-frame handling

Support:

1. Manual release-frame selection.
2. Assisted frame selection.
3. Automatic release-frame labeling (see below), validated against manual
   ground truth before bulk use.

#### Automatic release-frame labeling

Reduce manual labeling to review-and-confirm by detecting the release frame
automatically:

1. **Motion gate**: cheap frame differencing locates the high-motion delivery
   window inside the clip.
2. **Coarse-to-fine pose pass**: run pose estimation on subsampled frames in
   that window, then densely around the best candidate (full-clip dense pose
   is too slow on the target CPU).
3. **Kinematic detector**: the throwing wrist reaches peak speed and forward
   extension at release; detect that signature and emit a candidate release
   frame plus a confidence score.
4. **Cross-reference gate**: before auto labels are trusted at scale, they
   must be validated against a manually labeled subset (default target:
   at least 90% of validation clips within +/- 1 frame). Low-confidence
   detections go to a human review queue. An automatic label must never
   silently overwrite a manual one.

Store provenance for every release frame: source (`manual`, `auto`,
`auto_confirmed`), detector confidence, and detector version/config.

**Leakage note:** the labeler must analyze release and post-release frames to
find the release point. That is acceptable because it is labeling tooling,
not model input — but its *outputs* define the protected cutoff, so the
cross-reference validation and provenance records above are mandatory, and
the release guard frames still apply on top of any auto label.

For training, the default clip must end several frames before visible ball release. Make the safety margin configurable.

Example:

```yaml
release_guard_frames: 3
pre_release_window_ms: 1500
```

No frame at or after release may enter the model input.

### Pitcher tracking

- Detect the pitcher or let the user define an initial box.
- Track the pitcher across frames.
- Expand the box enough to include the full delivery.
- Smooth box motion.
- Store the box for every frame.
- Allow manual correction.
- Reject clips where tracking fails badly.

### Normalization

- Normalize pose coordinates relative to body scale and pitcher crop.
- Preserve temporal velocity and acceleration.
- Do not mirror left-handed pitchers without recording the transform.
- Separate or explicitly encode windup versus stretch.
- Do not mix materially different camera angles without a camera-angle feature and a proper validation design.

---

## 10. Preventing Data Leakage

Data leakage is the primary technical risk.

The model must not use:

- Frames at or after release.
- Ball trajectory.
- Batter swing.
- Catcher receiving motion.
- Pitch result.
- On-screen pitch labels.
- Scoreboard pitch-type graphics.
- Different clip lengths that accidentally encode labels.
- File names containing the pitch label as an input feature.
- Game-specific overlays or crops that correlate with one pitch type.
- Data from the same game in both training and test sets.
- Duplicate or near-duplicate clips across splits.
- Features created using statistics computed from the test set.

Always:

- Split by game or date, not by individual clip.
- Fit normalization only on training data.
- Report majority-class and simple heuristic baselines.
- Test performance on entirely held-out games.
- Audit label distribution by game.
- Run a background-only control.
- Run a catcher-only control when the catcher is visible.
- Run a randomized-label test.
- Run temporal truncation tests.
- Run crop sensitivity tests.

If a suspiciously high result appears, assume leakage until disproven.

---

## 11. Modeling Strategy

Implement models in this order.

### Stage 0: Baselines

- Majority class.
- Stratified random prediction.
- Logistic regression on metadata only.
- Simple pose-summary logistic regression.

A learned model is only interesting when it improves on these baselines.

### Stage 1: Pose feature model

Extract normalized features such as:

- Joint coordinates.
- Joint angles.
- Pairwise distances.
- Shoulder and hip rotation.
- Glove-hand position relative to torso.
- Throwing-elbow height.
- Head tilt.
- Knee lift.
- Body center movement.
- Temporal velocity.
- Temporal acceleration.
- Delivery timing.

Start with:

- Logistic regression.
- Random forest or gradient boosting.
- Temporal convolutional network.
- Small LSTM only if it improves performance.

Prefer interpretable models before complex models.

### Stage 2: Region-specific RGB models

Train small models on controlled crops:

- Head.
- Glove.
- Throwing hand and wrist.
- Torso.
- Hips.
- Lead leg.
- Full pitcher.

Compare region performance and ablation effects.

### Stage 3: Combined pose and RGB model

Fuse:

- Pose sequence embedding.
- RGB region embedding.
- Optional metadata that is safe and justified.

### Stage 4: Video transformer

Only consider a pretrained lightweight video transformer after:

- The split is trustworthy.
- The dataset is large enough.
- The pose baseline is established.
- Cloud GPU execution is configured.
- The explainability pipeline is already working.

Do not train a large transformer from scratch.

---

## 12. Explainability and Tell Localization

A highlighted square must be evidence-driven, not decorative.

Implement the following hierarchy.

### A. Pose-feature attribution

For interpretable models, identify the strongest pose features and map them to body regions.

Example:

```text
Feature: glove_wrist_y_relative_to_sternum
Interval: -420 ms to -210 ms
Direction: higher before sliders
Effect size: 0.74 standard deviations
```

Draw a tracked square around the glove and wrist during that interval.

### B. Region occlusion

Mask one body region at a time and rerun inference.

Report:

```text
Full clip slider probability: 0.84
Glove masked: 0.53
Head masked: 0.81
Lead leg masked: 0.79
```

The highlighted region should preferably have:

- A material confidence drop.
- Consistent results across clips.
- Stability on held-out games.

### C. Temporal occlusion

Mask short frame windows and measure the change.

Use this to identify when the signal becomes visible.

### D. Heatmaps

Grad-CAM or equivalent may be shown as supporting evidence, but never treat a heatmap alone as proof.

### E. Statistical group comparison

For the suspected feature:

- Compare distributions by pitch type.
- Report sample counts.
- Report means or medians.
- Report effect size.
- Use bootstrap confidence intervals.
- Correct for multiple comparisons when searching many features.
- Measure consistency across games.

### F. Human-readable explanation

Generate explanations only from computed evidence.

Good:

```text
Between 420 and 210 ms before release, the glove wrist was higher relative
to the sternum on sliders in four of five held-out games. Masking the glove
region reduced slider classification confidence by an average of 21 points.
```

Bad:

```text
The pitcher definitely tips every slider.
```

---

## 13. Highlight Overlay Specification

The annotated video must support:

- Bounding box around the suspected tell.
- Smooth tracked movement.
- Configurable padding.
- Label above the box.
- Confidence indicator.
- Highlight only during the informative time interval.
- Optional pose skeleton.
- Optional heatmap.
- Timeline marker.
- Side-by-side comparison mode.
- Frame stepping.
- Slow-motion playback.

Default label format:

```text
Potential tell: glove height
Model evidence: moderate
```

Do not show false precision. Prefer `low`, `moderate`, or `high` evidence bands plus the underlying metrics.

### Overlay rendering

Support two methods:

1. Browser overlay using Canvas or SVG.
2. Exported video rendered through OpenCV and encoded with FFmpeg.

Overlay coordinates must be stored in original-video coordinates and transformed for the display resolution.

---

## 14. Confidence and Decision Logic

Do not label a pattern as a suspected tell unless it passes configurable checks.

Suggested initial checks:

```yaml
tell_detection:
  minimum_test_clips_per_class: 20
  minimum_games_in_test: 2
  minimum_accuracy_over_baseline: 0.08
  minimum_balanced_accuracy: 0.60
  minimum_occlusion_probability_drop: 0.10
  minimum_game_stability_fraction: 0.60
  minimum_effect_size: 0.35
```

These are starting points, not scientifically validated constants.

Evidence level:

- **Insufficient**: not enough data or no meaningful improvement over baseline.
- **Low**: some predictive signal, but unstable or weak localization.
- **Moderate**: held-out predictive signal plus consistent region evidence.
- **High**: replicated across multiple held-out games, strong effect, strong ablation result, and robust controls.

The UI must separately show:

- Pitch classification confidence.
- Confidence that a repeatable tell exists.
- Confidence in the localized body region.

These are not the same quantity.

---

## 15. Evaluation Metrics

Always report:

- Sample count by class.
- Sample count by game.
- Class balance.
- Accuracy.
- Balanced accuracy.
- Precision, recall, and F1 by class.
- Confusion matrix.
- ROC AUC when appropriate.
- Log loss or Brier score.
- Calibration plot.
- Majority-class baseline.
- Performance by held-out game.
- Performance by delivery type.
- Performance by batter side when available.
- Performance at multiple pre-release cutoff times.
- Occlusion delta by region.
- Bootstrap confidence intervals.

Critical evaluation plot:

```text
Prediction performance versus milliseconds before release
```

This reveals how early the potential tell becomes detectable.

---

## 16. API Requirements

Provide versioned endpoints.

Suggested MVP endpoints:

```text
POST   /api/v1/projects
GET    /api/v1/projects
POST   /api/v1/videos
GET    /api/v1/videos/{video_id}
POST   /api/v1/videos/{video_id}/clips
PATCH  /api/v1/clips/{clip_id}
POST   /api/v1/clips/{clip_id}/pose
POST   /api/v1/experiments
GET    /api/v1/experiments/{experiment_id}
POST   /api/v1/experiments/{experiment_id}/evaluate
GET    /api/v1/experiments/{experiment_id}/tells
POST   /api/v1/tells/{tell_id}/render
GET    /api/v1/artifacts/{artifact_id}
```

Long-running tasks may initially run as local background jobs with persisted status. Avoid adding Celery or Redis unless needed.

Every job must expose:

```text
queued
running
completed
failed
```

Store a readable error message and stack trace in development mode.

---

## 17. Frontend Screens

Build these screens in order.

### 1. Project screen

- Pitcher information.
- Dataset summary.
- Camera angles.
- Pitch counts.
- Warning when class or game coverage is weak.

### 2. Clip labeling screen

- Video player.
- Frame-by-frame controls.
- Start frame.
- Release frame.
- Pitch type.
- Camera angle.
- Delivery type.
- Quality status.
- Pitcher bounding-box correction.
- Keyboard shortcuts.

### 3. Dataset audit screen

- Counts by pitch type and game.
- Missing metadata.
- Duplicate warnings.
- Camera-angle distribution.
- Train/validation/test split preview.

### 4. Experiment screen

- Configuration.
- Training status.
- Metrics.
- Confusion matrix.
- Performance-by-game table.
- Performance-before-release curve.

### 5. Tell analysis screen

- Annotated video.
- Highlight box.
- Pose overlay.
- Region occlusion results.
- Feature comparison.
- Side-by-side examples.
- Human-readable explanation.
- Evidence and limitations.
- Export controls.

---

## 18. Testing Requirements

### Unit tests

Test:

- Release-relative frame calculations.
- Prevention of post-release frame inclusion.
- Coordinate transforms.
- Pose normalization.
- Game-grouped split logic.
- Feature extraction.
- Occlusion masks.
- Bounding-box interpolation.
- Report generation.
- Configuration hashing.

### Integration tests

Test:

- Upload video to saved clip.
- Clip to pose artifact.
- Dataset to experiment.
- Experiment to evaluation report.
- Suspected tell to annotated MP4.
- API and database behavior.

### Leakage tests

Create explicit automated tests that fail when:

- A game appears in more than one split.
- Any model frame is at or after the protected release cutoff.
- Duplicate checksums cross splits.
- Target labels appear in exported feature-column names.
- Test-set statistics are used during fitting.

### Fixtures

Keep very small synthetic videos and pose sequences in the test fixtures. Do not depend on a private MLB clip for normal tests.

---

## 19. Coding Standards

- Use type hints for all public Python functions.
- Use Pydantic models at API boundaries.
- Keep functions focused and testable.
- Avoid global mutable state.
- Use dependency injection for storage and model services.
- Validate all file paths.
- Never construct shell commands from untrusted strings.
- Use `subprocess.run()` with an argument list and error handling.
- Add docstrings where behavior is not obvious.
- Prefer dataclasses or typed models over loose dictionaries.
- Use structured logging.
- Include a correlation ID for long-running analysis jobs.
- Make random seeds configurable.
- Save the full experiment configuration with results.
- Do not hide exceptions.
- Do not add dependencies without explaining why.
- Do not refactor unrelated code during a focused task.
- Do not create placeholder implementations presented as complete.

Before marking a task complete:

1. Run formatting.
2. Run linting.
3. Run type checks where configured.
4. Run relevant tests.
5. State exactly what was changed.
6. State any unverified assumptions.
7. Provide the exact command to exercise the feature.

---

## 20. Commands

Claude should maintain these commands as the project evolves.

Target developer workflow:

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy backend ml
uv run pytest
```

Backend:

```bash
uv run uvicorn backend.app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
npm run lint
npm test
```

Example ML workflow:

```bash
uv run python scripts/ingest_video.py --config configs/default.yaml
uv run python scripts/estimate_pose.py --config configs/pose_baseline.yaml
uv run python scripts/train_pose_model.py --config configs/pose_baseline.yaml
uv run python scripts/evaluate_model.py --config configs/pose_baseline.yaml
uv run python scripts/render_overlay.py --experiment-id EXPERIMENT_ID
```

Do not claim a command works until it has been run successfully or clearly mark it as proposed.

---

## 21. Security, Privacy, and Legal Constraints

- Do not implement unauthorized scraping or access controls bypasses.
- Do not store access tokens in source code.
- Use `.env` for local secrets and commit only `.env.example`.
- Validate uploaded video types and file sizes.
- Sanitize generated file names.
- Prevent path traversal.
- Treat video and metadata as potentially private.
- Document dataset provenance and permitted use.
- Do not imply ownership of third-party footage.
- Do not present model output as a factual accusation.
- Use language such as `potential tell`, `suspected visual signal`, and `requires human review`.

---

## 22. Claude Code Working Instructions

When working in this repository, Claude must:

1. Read this file before planning or editing.
2. Inspect existing code before proposing replacements.
3. Keep the MVP narrow.
4. Build one complete vertical slice at a time.
5. Prefer simple, inspectable solutions.
6. Make hardware-aware choices for an Intel Mac.
7. Ask for cloud GPU use only when local execution is genuinely impractical.
8. Never allow post-release information into training.
9. Treat explainability as a required product feature.
10. Treat data leakage testing as required, not optional.
11. Preserve reproducibility.
12. Update documentation and tests with behavioral changes.
13. Avoid adding MCP servers, agents, or skills unless they solve a real workflow problem.
14. Never state that a detected region proves causality.
15. Keep a running implementation checklist in `docs/progress.md`.

For every significant coding request:

- First inspect the relevant files.
- Write a concise implementation plan.
- Identify data-leakage risks.
- Implement the smallest useful increment.
- Add or update tests.
- Run the tests.
- Summarize changed files and remaining limitations.

When blocked by missing real data, create a synthetic test fixture and continue building the pipeline. Do not fabricate model-performance results.

---

## 23. Suggested Claude Code Subagents

Use subagents selectively. Do not create many agents at the beginning.

Recommended agents after initial repository setup:

### `video-pipeline-reviewer`

Responsibilities:

- Review FFmpeg and OpenCV transformations.
- Verify frame numbering and timestamps.
- Verify that post-release frames cannot leak into model input.
- Review coordinate transforms used by overlays.
- Check Intel macOS compatibility.

### `ml-evaluation-reviewer`

Responsibilities:

- Review grouped split logic.
- Audit leakage.
- Validate metrics and baselines.
- Review confidence claims.
- Check whether interpretations are supported by held-out evidence.

### `test-engineer`

Responsibilities:

- Add unit and integration tests.
- Create synthetic fixtures.
- Test failure cases.
- Verify reproducibility.

Do not delegate final architectural control to subagents. The main Claude Code session should integrate and verify their work.

---

## 24. MCP Guidance

MCP is not required for the core application.

Useful optional MCP integrations:

- GitHub MCP for issues, pull requests, and repository navigation.
- Filesystem access when not already provided by Claude Code.
- A database inspection tool later in development.
- Playwright MCP for browser-based end-to-end UI testing.

Do not add an MCP server merely because one exists.

Do not use MCP for:

- Core video processing.
- Model training.
- Pose estimation.
- Normal application database access.
- Replacing ordinary Python modules.
- Accessing copyrighted or restricted video sources.

The project should run normally without MCP.

---

## 25. Skills Guidance

Create reusable Claude Code skills only after a workflow repeats several times.

Potential future skills:

### `/audit-dataset`

- Summarize clips by game, pitch type, angle, and delivery.
- Detect duplicates and split leakage.
- Flag insufficient samples.
- Produce a dataset audit report.

### `/run-pose-baseline`

- Validate configuration.
- Extract pose.
- Train baseline.
- Evaluate held-out games.
- Generate report.

### `/review-tell-evidence`

- Inspect metrics.
- Inspect region occlusion.
- Inspect game stability.
- Check claim wording.
- Produce a model-review checklist.

### `/render-tell-overlay`

- Load a selected experiment.
- Identify the supported body region and interval.
- Render a tracked square.
- Export annotated video and JSON.

Do not create skills before the underlying scripts and tests work manually.

---

## 26. Initial Implementation Roadmap

### Milestone 1: Repository and local environment

Deliver:

- Repository structure.
- `pyproject.toml`.
- FastAPI health endpoint.
- SQLite connection.
- Basic React or Gradio interface.
- FFmpeg availability check.
- CI lint and unit test workflow.
- `docs/progress.md`.

Acceptance:

```bash
uv sync
uv run pytest
uv run uvicorn backend.app.main:app --reload
```

work on the Intel Mac.

### Milestone 2: Video ingestion and manual labeling

Deliver:

- Video import.
- Metadata extraction.
- Clip creation.
- Manual release-frame selector.
- Pitch label form.
- Clip metadata persistence.
- Frame-accurate preview.

Acceptance:

A user can import one game video, define a clip, mark release, label pitch type, and reload the saved clip.

### Milestone 3: Pitcher crop and pose

Deliver:

- Manual initial pitcher box.
- Tracking.
- Pose extraction.
- Pose visualization.
- Quality checks.
- Saved pose artifact.

Acceptance:

A selected clip displays a stable tracked pitcher box and skeleton across the pre-release sequence.

### Milestone 3.5: Automatic release-frame labeling

Deliver:

- Motion-gate delivery-window detection (frame differencing).
- Coarse-to-fine pose sampling within the delivery window.
- Wrist-kinematics release detector with a confidence score.
- Release-frame provenance fields (`release_frame_source`,
  `release_frame_confidence`) and detector version/config records.
- Cross-reference validation report: auto labels versus a manually labeled
  subset, with per-clip frame error.
- Review queue: low-confidence or disagreeing clips flagged for manual
  confirmation; auto labels never silently overwrite manual labels.
- Batch labeling CLI/endpoint to process many clips unattended.

Acceptance:

On a manually labeled validation set, at least 90% of automatic release
frames fall within +/- 1 frame of the manual mark, disagreements are surfaced
for review rather than silently accepted, and every auto-labeled clip records
source, confidence, and detector config.

Deliver:

- Counts by pitch and game.
- Duplicate detection.
- Grouped train/validation/test split.
- Leakage assertions.
- Dataset audit report.

Acceptance:

No game or duplicate clip can cross splits.

### Milestone 5: Pose baseline model

Deliver:

- Pose features.
- Baselines.
- Logistic regression classifier.
- Metrics.
- Calibration.
- Performance-by-game.
- Performance-versus-time-before-release.

Acceptance:

The experiment is reproducible and explicitly reports whether it beats baseline.

### Milestone 6: Tell localization

Deliver:

- Pose feature attribution.
- Body-region mapping.
- Region and temporal occlusion.
- Effect-size analysis.
- Evidence level.
- Supporting examples.

Acceptance:

A suspected tell is only produced when configured evidence thresholds pass.

### Milestone 7: Video highlight and report

Deliver:

- Smooth tracked highlight square.
- Time-limited overlay.
- Human-readable evidence.
- Annotated MP4.
- JSON report.
- Side-by-side comparison.

Acceptance:

The exported video clearly shows the suspected region only during the pre-release interval supported by the analysis.

### Milestone 8: RGB region model

Deliver only after pose MVP validation:

- Glove, head, torso, arm, and leg crops.
- Lightweight pretrained RGB encoder.
- Region comparison.
- Combined pose and RGB inference.
- Updated ablation report.

---

## 27. First Task for Claude Code

Begin by completing Milestone 1 only.

Claude should:

1. Inspect the current repository.
2. Create the proposed structure without unnecessary empty abstraction.
3. Configure `uv`, FastAPI, pytest, Ruff, and mypy.
4. Add a `/health` endpoint.
5. Add a small environment diagnostic that reports:
   - Python version.
   - macOS version.
   - CPU architecture.
   - FFmpeg availability.
   - OpenCV version.
   - PyTorch version and available acceleration backend.
6. Add unit tests for the health endpoint and diagnostics.
7. Add a minimal README with exact setup commands.
8. Add `docs/progress.md`.
9. Run all checks.
10. Stop after Milestone 1 and summarize the result.

Do not start model training or frontend complexity in the first task.

---

## 28. Definition of Done

A feature is done only when:

- It works on the intended Intel Mac environment or has a documented fallback.
- Its behavior is covered by tests.
- Errors are handled visibly.
- Inputs and outputs are typed.
- Relevant documentation is updated.
- The exact verification command is provided.
- No post-release leakage is introduced.
- Results are not overstated.
- Generated artifacts can be traced to a dataset version, config, code revision, and random seed.

The final product should help a knowledgeable human investigate a possible pitch tip. It must not disguise correlation, model attention, or weak evidence as certainty.
