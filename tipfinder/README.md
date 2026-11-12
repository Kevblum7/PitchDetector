# tipfinder: is this pitcher tipping his pitches?

A portfolio project for an Applied AI Engineer role. Given one MLB pitcher, it asks a narrow,
testable question:

> Do body cues visible **before the hands separate** in the set predict fastball vs. offspeed
> **better than game context already does**, on games the model has never seen?

A "tip" means real, held-out lift over the context baseline, nothing else. If the lift isn't
statistically distinguishable from zero, the tool says **"no detectable tip"**.

<!-- RESULTS -->

## Why the baseline and leakage matter

**The baseline.** A hitter doesn't need body language to guess fastball on 3-0. Count,
runners, batter hand, the previous pitch and the pitcher's pitch mix already predict a lot.
Any "tip" model that doesn't beat *that* is just rediscovering the scouting report. So every
result here is a **difference**: context+body vs. context only, same folds, same model class.
Pitch mix by count is estimated on training games only within each fold.

**Leakage.** Pose from broadcast video can easily "predict" pitch type for the wrong reason:

- *Frames after hand separation* show grip and arm action. Including even a few frames is the
  classic way to get a fake tip. The detector finds separation, a 3-frame guard is subtracted,
  and `ModelWindow.check()` raises `LeakageError` if any later frame reaches the feature code.
  A test scrambles every post-cutoff frame and asserts the features are bit-for-bit unchanged.
- *Same-game splits.* Pitches from one game share camera zoom, lighting, uniform and the
  pitcher's feel that day. Splitting by pitch lets a model memorise games. Every split here is
  by **game**, and the bootstrap resamples **games**.
- *Train/test statistics.* Pitch-mix rates are fitted on training games only (tested).
- *Broadcast graphics.* The scorebug shows the pitch type *after* the pitch. The model only sees
  pose keypoints from pre-separation frames, never pixels.

Two controls must behave before a result is believed:

1. **Shuffled labels within game** → lift must collapse to ~0.
2. **Post-separation frames deliberately included** → this is the positive control. It shows
   what leaked information does to the numbers, and the real pipeline is shown never to see
   those frames.

## Method

1. **Data** (`tipfinder fetch`). Statcast pitches via `pybaseball`. Statcast's CSV has no video
   id, so each pitch is joined to Savant's game feed (`/gf?game_pk=`) on
   `(at_bat_number, pitch_number)`, and the join is accepted only if pitch type and speed agree.
   The clip URL comes from `/sporty-videos?playId=…&videoType=HOME|AWAY` (the page echoes the
   served feed, which is verified). The pitcher's own club's broadcast is preferred. Requests go
   at most one per 1.5 s; responses are cached and downloads resume from `.part` files.
   Sampling: 10 random pitches from each start (seeded), so evaluation has many game groups.
2. **Release window** (`tipfinder extract`). Clips usually open on a close-up and then cut to the
   center-field camera. Shots are split on colour-histogram cuts. The first shot with a CF layout
   (grass plus a mound in the lower frame) is used. Inside it the pitcher is the largest fully
   visible person, which rejects batter close-ups. Pose is then tracked with continuity checks.
   Hand separation is the first measured frame where the wrist gap (in torso lengths) leaves the
   "together" band after a held set and goes on to open into a delivery. Tracking dropouts
   inherit the previous state, so they can never fake a separation.
3. **Features.** About a dozen interpretable quantities, each summarised over the set and over
   the last 6 frames before the break: hands/glove height, hands depth, wrist gap, forearm
   angle, elbow flare, finger spread, wrist bend, head position and tilt, shoulder tilt and
   width, torso lean, time in set. All are in torso-length units (camera-zoom invariant).
   None use MediaPipe's left/right labels, which flip when the pitcher is seen from behind.
4. **Models.** Per-pitcher logistic regression (primary, chosen in advance) and gradient
   boosting, binary fastball (FF/SI/FC) vs. offspeed. `--target multiclass` is available.
5. **Evaluation.** 5-fold cross-validation grouped by game, giving out-of-fold probabilities for
   every pitch. Lift = context log loss − (context+body) log loss. 95% CI from 2,000 game-level
   bootstrap resamples. Permutation test: body features shuffled within game, the whole
   pipeline retrained, 200 times. **Verdict = tip only if p < 0.05 and CI lower bound > 0.**
6. **Explanation.** Held-out permutation importance of each body feature, then the direction
   and size of the raw difference (Cohen's d, offspeed − fastball, game-bootstrap CI). Both are
   rendered as scouting sentences. When the overall lift is not significant, cues are labelled
   *exploratory*.
7. **Agent** (`tipfinder report --agent`). An OpenAI Agents SDK agent with tools that fetch,
   extract, run the pipeline and return metrics. It writes a short scouting report. It cannot
   change numbers, and a deterministic guard forces the "no detectable tip" verdict when the
   pipeline says so. All tools are plain Python functions usable without the agent.

## Running it

```bash
uv sync                                        # Python 3.12, lockfile committed
uv run tipfinder fetch                         # ~10 min at 1 req / 1.5 s; resumable
uv run tipfinder extract --workers 4           # pose + release window; resumable
uv run tipfinder train                         # feature tables + final models
uv run tipfinder evaluate                      # CV, bootstrap, permutation test, controls
uv run tipfinder report                        # Markdown report (no LLM)
OPENAI_API_KEY=... uv run tipfinder report --agent   # + agent-written scouting report

uv run tipfinder label-sheet 745557_19_3       # contact sheet for hand-labelling a clip
uv run tipfinder build-test-cache              # copy ~16 real clips into tests/.cache
uv run pytest
```

Another pitcher means another config: copy `configs/wheeler_2024.yaml` and change the name,
MLBAM id, team and dates (or call `agent.tools.create_pitcher_config`).

Nothing downloaded is committed: `data/`, `cache/`, `tests/.cache/` and `*.mp4` are git-ignored,
and the OpenAI key is read from the environment only.

<!-- LIMITATIONS -->
