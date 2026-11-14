# Pose QC calibration (label-blind)

Measured on the 214 clips with a detected release window (Zack Wheeler 2024), over the
30-frame feature window before the guard cutoff. No pitch-type information was used.

| metric | p10 | p25 | median | p75 | p90 |
|---|---|---|---|---|---|
| share of frames, all key joints visibility >= 0.3 | 0.33 | 0.57 | 0.77 | 0.87 | 0.97 |
| share of frames, all key joints visibility >= 0.5 | 0.03 | 0.13 | 0.27 | 0.50 | 0.63 |
| median hands speed (torso lengths / frame) | 0.024 | 0.028 | 0.035 | 0.052 | 0.082 |

MediaPipe "visibility" estimates occlusion, and in the set the wrists really are inside or
behind the glove, so a 0.5 threshold rejected about 77% of clips with good keypoints. Chosen gate:
visibility >= 0.3 in >= 50% of frames, median hand speed <= 0.10 (a held set is still; noisy
tracks jump), <= 25% untracked frames, >= 15 window frames. Result: 156 of 214 pass. On the
hand-labelled clips this gate removes the one late detection (+16 frames, noisy pose) and the
batter close-up, and keeps all clean labelled clips.
