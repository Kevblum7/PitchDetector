# tipfinder: is this pitcher tipping his pitches?

A portfolio project for an Applied AI Engineer role. Given one MLB pitcher, it asks a narrow,
testable question:

> Do body cues visible **before the hands separate** in the set predict fastball vs. offspeed
> **better than game context already does**, on games the model has never seen?

A "tip" means real, held-out lift over the context baseline, nothing else. If the lift isn't
statistically distinguishable from zero, the tool says **"no detectable tip"**.

## Results (real Statcast + real Savant broadcast video, 2024)

Two Phillies starters, same pipeline and settings. 10 random pitches per start were sampled;
all evaluation holds out whole games.

| | Zack Wheeler | Aaron Nola |
|---|---|---|
| Usable pitches / starts | 154 / 26 | 204 / 28 |
| Fastball share (FF/SI/FC) | 69% | 67% |
| Log-loss lift, logistic regression (95% CI over games) | −0.004 [−0.082, +0.078] | −0.092 [−0.151, −0.036] |
| AUC, context → context+body | 0.64 → 0.68 | 0.51 → 0.46 |
| Permutation p | 0.025 | 0.56 |
| Shuffled-label control (mean lift) | −0.080 | −0.082 |
| **Positive control:** AUC with post-separation frames (CI on AUC lift) | 0.64 → **0.73** [+0.01, +0.16] | 0.51 → 0.53 [−0.08, +0.10] |
| **Verdict** | **No detectable tip** | **No detectable tip, inconclusive** |

**Wheeler: no detectable tip.** His body cues before the hands break did not improve held-out
predictions beyond game context. The permutation p of 0.025 says the real features beat
within-game-shuffled ones, so some weak association exists, but the lift CI straddles zero.
By the rule fixed in advance (p < 0.05 **and** lift CI above 0), that is not a tip. The
positive control *fired*: once post-separation frames are let in, AUC rises and its CI clears
zero. So the method could see pitch-type information here if it were there before the break.

**Nola: inconclusive.** Body cues made predictions *worse* (the model pays for 26 noisy features).
More importantly, the positive control did **not** fire: even with grip and arm-action frames
included, pose features did not predict his pitch type. From the CF camera, MediaPipe
keypoints don't separate Nola's fastball and knuckle-curve arm actions. A pipeline that
can't see a known leak can't rule out a subtle tell, so the report and the agent label this
result inconclusive rather than "no tip". Game context also barely predicts Nola (AUC 0.51),
consistent with his reputation for throwing the curve in any count.

**Release-window check (38 hand-labelled real clips).**

| | labelled clips | reaching the model | detected separation vs label | cutoff after true separation |
|---|---|---|---|---|
| Wheeler | 20 (17 keep, 3 exclude) | 12 | −2 to −9 frames (always early) | 0 |
| Nola, development set | 10 | 9 | 0 to −27 frames | 0 |
| **Nola, blind holdout** (labelled after the fix below) | 8 | 8 | −2 to −25 frames | **0** |

Running the second pitcher exposed a real leakage bug. Nola holds his hands looser in the set
(resting wrist gap about 0.3 torso lengths vs. Wheeler's 0.1) and breaks gradually, so the
fixed-threshold detector fired 2–5 frames late. That left 4 of 9 labelled clips with
post-separation frames inside the model window. The fix (`refine_onset`) measures each clip's
own set baseline and walks back to the start of the break, at most 8 frames. It can only move
separation earlier. A fresh blind holdout labelled afterwards confirms no clip is late. Both
pitchers were re-run with the fix; the numbers above are post-fix.

**Data funnel.** Wheeler: 320 sampled → 154 usable (no Savant clip 20, no CF shot 35, no
full-body pitcher 20, no hand break found 31, pose QC 60). Nola: 320 → 204 (30 / 34 / 15 / 10 / 27).
Usable share varies a lot by park (Wheeler at SF/SEA/NYM/MIL/BOS: 0%; Nola at MIL and PIT:
0–10%), mostly because of camera placement.

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

- **Small sample.** 154–204 pitches / 26–28 games per pitcher. Effects of the size a hitter could exploit could
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
- **Two pitchers, one season.** Nothing generalises to other pitchers without rerunning, and
  as Nola shows, the method can be blind for some deliveries. Always check the positive control.
- **Hand-labelled validation has a single annotator** (labels were made by the AI assistant
  that built the pipeline, from 60 fps contact sheets, ±1 frame). Worth a second human pass.
- **Correlation, not proof.** Even a significant lift would show the cue is *predictive* in
  held-out games, not that a hitter could see and use it in real time.

