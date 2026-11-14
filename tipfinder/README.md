# tipfinder: is this pitcher tipping his pitches?

A portfolio project for an Applied AI Engineer role. Given one MLB pitcher, it asks a narrow,
testable question:

> Do body cues visible **before the hands separate** in the set predict fastball vs. offspeed
> **better than game context already does**, on games the model has never seen?

A "tip" means real, held-out lift over the context baseline, nothing else. If the lift isn't
statistically distinguishable from zero, the tool says **"no detectable tip"**.

## Results (real Statcast + real Savant broadcast video, 2024)

Four NL East starters, one pipeline, identical settings (guard 8 frames, same QC). 10 random
pitches per start were sampled. All evaluation holds out whole games.

| | Zack Wheeler (PHI, R) | Aaron Nola (PHI, R) | Luis Severino (NYM, R) | Sean Manaea (NYM, L) |
|---|---|---|---|---|
| Usable pitches / starts | 145 / 26 | 199 / 28 | 53 / 18 | 40 / 14 |
| Log-loss lift (95% CI over games) | +0.039 [−0.053, +0.125] | −0.083 [−0.150, −0.013] | +0.016 [−0.106, +0.129] | −0.170 [−0.308, −0.016] |
| AUC, context → +body | 0.62 → 0.72 | 0.52 → 0.52 | 0.53 → 0.63 | 0.72 → 0.65 |
| Permutation p | 0.005 | 0.62 | 0.12 | 0.33 |
| Positive control fired? (post-separation frames raise AUC, CI > 0) | **yes** (0.62 → 0.78) | no | no | no |
| **Verdict** | **No detectable tip (borderline)** | Inconclusive | Inconclusive | Inconclusive |

**Wheeler is the only pitcher the method can speak about.** His positive control fires: once
post-separation frames are let in, AUC jumps from 0.62 to 0.78. Pre-separation body cues give
a permutation p of 0.005, and the AUC gain's CI ([+0.03, +0.18]) excludes zero. But the
pre-registered criterion is held-out *log-loss* lift with a game-bootstrap CI above zero, and
that CI still includes zero. So the verdict is **no detectable tip, borderline**: a weak,
consistent signal worth a scout's look on video and a full-season rerun, not a finding.

**Nola, Severino, Manaea: inconclusive.** For each, even grip and arm-action frames didn't make
pose features predictive, so the pipeline can't rule a tell in or out. Two different causes:
- *Nola*: the pose is measured well (median key-joint visibility 0.93), but from the CF camera
  his fastball and knuckle-curve arm actions look alike to MediaPipe keypoints.
- *Severino and Manaea*: both set their hands tight against the chest, so the CF camera,
  which films from behind, sees the hands hidden behind the torso. Pose QC (unchanged across
  pitchers, not loosened after the fact) keeps only 53 and 40 pitches. Manaea also loses 84
  clips with no usable CF shot.

**Release-window validation (54 hand-labelled real clips, 4 pitchers).** On every labelled clip
that reaches the model, the model's last frame comes before the true hand separation.

| | labels | reach model | separation error (detected − label) | cutoff after true separation |
|---|---|---|---|---|
| Wheeler | 20 | 11 | always early (−2 to −9) | 0 |
| Nola | 18 | 17 | 0 to −27 | 0 |
| Severino | 8 | 6 | −2 to +5 | 0 (guard 8) |
| **Manaea, blind holdout** (labelled after all tuning) | 8 | 8 | −5 to +1, MAE 1.75 | **0** |

**What running more pitchers found (all fixed, all four pitchers re-run):**
1. *Nola*: looser set and gradual break, so the fixed threshold fired late. Added `refine_onset`.
2. *Severino*: even slower break, so detections were up to 5 frames late. Onset threshold
   lowered and computed against a local set baseline; guard raised from 3 to 8 frames
   (0.13 s, so features stop slightly earlier before the break).
3. *Severino*: a side-angle clip passed the CF colour check. Added a label-blind
   **camera-consistency gate**: the CF camera is fixed within a game, so clips where the
   pitcher's position or size deviates from the game median are a different camera (wide,
   first-base, crowd). The threshold was set by eye on rendered frames, and it rejects 4–8
   clips per pitcher.
4. Onset refinement could leave a window shorter than the guard. That is now rejected, and a
   test covers it.

The Manaea labels were made only after all of this, as an untouched holdout.

**Data funnel** (320 sampled each): Wheeler 145 usable, Nola 199, Severino 53, Manaea 40.
Usable share by park ranges from 0% to 100% because of camera placement.

**Synthetic data.** None was used. The brief asked for results on synthetic and real data,
but also for real data only, and the second wins: every test and every number here comes from
real Baseball Savant video and Statcast data.

**Agent.** The OpenAI Agents SDK agent and its tools are implemented. The tools, the
deterministic report and the verdict guard are tested without an LLM. The agent's own written
report has **not** been run end-to-end in this environment because no OpenAI key was provided.
Run `OPENAI_API_KEY=... uv run tipfinder report --agent` to produce it.


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

## Limitations (read before trusting anything)

- **Small sample.** 40–199 pitches / 14–28 games per pitcher. Effects of the size a hitter could exploit could
  still be missed: the lift CI spans roughly ±0.08 log loss. The next step is the full season
  (~3,100 pitches, about 1,500 usable at this funnel rate), which is just `sample.pitches_per_game`.
- **Camera angle differs across parks.** "Center field" means a different height, offset and
  zoom in every park. Oracle Park, T-Mobile Park and Busch Stadium are visibly offset. Features
  are normalised by torso length but not by viewpoint. Some cues (lean, "third-base side") are
  partly defined by the camera. Within-game permutation guards against park-level confounding,
  not against angle-dependent noise.
- **Pose noise.** MediaPipe on 720p broadcast frames, with the pitcher about 250–380 px tall
  and seen from behind. Wrists in the glove are often occluded. Finger landmarks are near noise,
  so "finger flare" is effectively unmeasurable here. The detector interface allows swapping in
  ViTPose (`pose/base.py`), which is the obvious upgrade.
- **CF-shot detection is heuristic** (grass/mound colour layout plus a full-body size gate).
  One batter close-up passed it and was caught only by pose QC.
- **Windup and stretch are mixed.** They're separated only through the runners-on context
  features.
- **Broadcast timing.** Savant clips are cut at a fixed offset before release, so absolute frame
  positions encode separation-to-release time, which is post-separation information. Features
  are anchored to separation for this reason, and "time in set" is used only when the set start
  is actually observed. Most sets are already under way when the CF shot begins, so this
  feature is mostly missing.
- **Four pitchers, one season.** Nothing generalises to other pitchers without rerunning, and
  as Nola shows, the method can be blind for some deliveries. Always check the positive control.
- **Hand-labelled validation has a single annotator** (labels were made by the AI assistant
  that built the pipeline, from 60 fps contact sheets, ±1 frame). Worth a second human pass.
- **Correlation, not proof.** Even a significant lift would show the cue is *predictive* in
  held-out games, not that a hitter could see and use it in real time.

