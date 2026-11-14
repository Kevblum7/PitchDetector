# tipfinder: is this pitcher tipping his pitches?

A portfolio project for an Applied AI Engineer role. Given one MLB pitcher, it asks a narrow,
testable question:

> Do body cues visible **before the hands separate** in the set predict fastball vs. offspeed
> **better than game context already does**, on games the model has never seen?

A "tip" means real, held-out lift over the context baseline, nothing else. If the lift isn't
statistically distinguishable from zero, the tool says **"no detectable tip"**.

## Result: Zack Wheeler, 2024 (real Statcast + real Savant broadcast video)

**Verdict: no detectable tip.** Pre-separation body cues did not improve held-out predictions
of fastball vs. offspeed beyond what game context already gives.

| | context only | context + body | lift (log loss) | 95% CI over games |
|---|---|---|---|---|
| Logistic regression (primary, fixed in advance) | 0.611 | 0.621 | **−0.010** | [−0.089, +0.060] |
| Gradient boosting | 0.681 | 0.695 | −0.014 | [−0.091, +0.065] |

Sample: 156 usable pitches from 26 starts (108 fastballs / 48 offspeed), held out by game.
For reference, always guessing fastball gives log loss 0.617 and accuracy 69%.

**What the controls show**

| run | AUC (context → +body) | accuracy | log-loss lift |
|---|---|---|---|
| Real pipeline (frames before hand separation only) | 0.62 → 0.67 (lift CI [−0.02, +0.12]) | 68% → 71% | −0.010 |
| **Leakage control:** 40 post-separation frames deliberately included | 0.62 → **0.76** (lift CI [+0.05, +0.22]) | 68% → 72% | **+0.065** |
| Shuffled labels within game (20 repeats) | — | — | −0.082 (overfitting cost of noise) |

- The **leakage control** behaves as it should. Once grip and arm-action frames are allowed in,
  AUC jumps and its CI clears zero. The pipeline *can* pick up pitch-type information when it
  is there, and the real pipeline is shown, by test, never to see those frames.
- The **permutation test is p = 0.01**, while the lift CI includes zero. The real body features
  hurt the model much less than within-game-shuffled ones do (−0.010 vs. about −0.08), so some
  weak association with pitch type exists. It isn't enough to beat game context once the
  model pays for 26 extra features on 156 pitches. By the rule fixed in advance (p < 0.05
  **and** lift CI above 0), that is not a tip.
- The most important cue, exploratory only: **torso lean at the moment the hands break**.
  Wheeler's torso leans slightly more toward the third-base side before offspeed pitches
  (Cohen's d = +0.48, game-bootstrap CI [+0.28, +0.64]). It's consistent across games but small,
  and it doesn't translate into better predictions here. It is a lead for a scout to look at
  on video, not a finding.

**Release-window check (20 hand-labelled real clips).** On the 12 labelled clips that reach the
model, detected separation is within 2 frames of the label (MAE 0.6 frames at 60 fps). None is
late enough to beat the 3-frame guard, so the model cutoff precedes the true separation in every
case. All 3 clips labelled "should be excluded" (a batter close-up, a CF shot starting
mid-delivery, one starting as the hands part) are rejected. The raw detector did produce
one +16-frame late detection on a clip with noisy pose; pose QC removes it, and a test asserts
that any late detection fails QC.

**Data funnel.** 320 sampled pitches (10 per start, 32 starts) went in and 156 were usable:
20 had no clip on Savant, 35 had no center-field shot, 20 had no full-body pitcher in it, 31 had
no observable hand break, and 58 failed pose QC. Usable share varies a lot by park (SF, SEA,
NYM, MIL, BOS: 0%; CIN, LAA, AZ, STL: 80–100%), largely because of camera placement.

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

- **Small sample.** 156 pitches / 26 games. Effects of the size a hitter could exploit could
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
- **One pitcher, one season.** Nothing here generalises to other pitchers without rerunning.
- **Correlation, not proof.** Even a significant lift would show the cue is *predictive* in
  held-out games, not that a hitter could see and use it in real time.

