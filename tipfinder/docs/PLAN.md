# Plan and layout

## Question

For one pitcher, do body cues visible **before the hands separate** predict fastball vs.
offspeed better than game context already does, on games the model never saw?

## Stages (each tested before the next)

| # | Stage | Module | Check |
|---|-------|--------|-------|
| 1 | Statcast pitches, Savant play_id, broadcast clip (cached, rate-limited, resumable) | `data/` | join validated by pitch type + speed; parsing tests on real responses |
| 2 | CF shot → pitcher lock → pose → set/separation window | `video/`, `pose/`, `window/` | 20 hand-labelled clips; guard tests |
| 3 | Interpretable body features, normalised by torso length | `features/` | features unchanged when post-separation frames are scrambled |
| 4 | Logistic regression (primary) and gradient boosting, context vs. context+body | `model/` | out-of-game predictions test |
| 5 | Game-grouped CV, game bootstrap CI, permutation test, controls | `evaluation/` | shuffled-label control ≈ 0; post-separation leak control |
| 6 | Held-out permutation importance → scouting sentences | `explain/` | sentences generated only from numbers |
| 7 | OpenAI Agents SDK scouting report on top of plain tools | `agent/` | verdict guard test; tools run without LLM |

## Layout

```
tipfinder/
├── configs/wheeler_2024.yaml     pitcher, dates, paths, thresholds
├── labels/                       hand labels (small CSV, committed)
├── src/tipfinder/
│   ├── cli.py                    fetch | extract | train | evaluate | report | label-sheet | build-test-cache
│   ├── config.py                 typed config (pydantic)
│   ├── pipeline.py               train/evaluate orchestration (shared by CLI and agent)
│   ├── data/                     http.py (polite client), statcast.py, savant.py, fetch.py
│   ├── video/shots.py            shot cuts + CF-layout check
│   ├── pose/                     base.py (backend protocol), mediapipe_backend.py, extract.py
│   ├── window/                   release.py (detector + LeakageError guard), validate.py, label_sheet.py
│   ├── features/                 body.py (features), table.py (modelling table)
│   ├── model/models.py           models, game folds, train-only pitch mix
│   ├── evaluation/               lift.py (bootstrap, permutation), controls.py
│   ├── explain/importance.py     permutation importance → plain language
│   ├── agent/                    tools.py (plain functions), scout.py (Agents SDK)
│   └── report.py                 deterministic Markdown report
├── tests/                        real-data tests; tests/.cache is git-ignored
├── data/  cache/                 downloads and intermediate files (git-ignored)
└── docs/
```

## Decisions fixed in advance

- Primary model: logistic regression. Primary metric: log-loss lift. GBM is reported but
  does not drive the verdict (avoids picking the better of two by chance).
- Verdict "potential tip" requires permutation p < 0.05 **and** game-bootstrap 95% CI lower
  bound > 0. Anything else prints "no detectable tip".
- Fastball family = FF, SI, FC.
- Pose QC threshold calibrated on the visibility distribution only (label-blind) before
  any model was run.
