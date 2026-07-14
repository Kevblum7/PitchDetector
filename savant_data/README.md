# Savant dataset (metadata only)

This folder holds **public Statcast pitch metadata** and a per-pitch download
manifest. It contains **no video** and never should — MLB game video is
copyrighted broadcast content and must be obtained through MLB Film Room's own
download UI, by hand (see `CLAUDE.md` §5, §21). The `*.mp4` files you save into
`clips/` are git-ignored so raw footage is never committed.

## Provenance

- **Statcast pitch data**: `baseballsavant.mlb.com/statcast_search` (public
  research data — factual pitch measurements and labels).
- **Film Room `playId`**: `statsapi.mlb.com` play-by-play, joined on
  `game_pk` + `at_bat_number` + `pitch_number`.
- Pulled with `scripts/fetch_savant_metadata.py` (metadata only; downloads no
  video).

Permitted use: personal research and coaching analysis. This project does not
own or redistribute MLB footage and makes no factual accusation of pitch
tipping — outputs are *potential* tells requiring human review.

## Layout

```
savant_data/<pitcher>/
  metadata/
    <pitcher>_<season>_all.csv     # full raw Statcast pull (every pitch type)
    <pitcher>_<season>_FF.csv      # four-seam manifest (+ playId, Film Room URL)
    <pitcher>_<season>_CH.csv      # changeup manifest
  clips/
    FF/  CH/                        # you save downloaded clips here (git-ignored)
```

Each manifest row has a stable **`clip_uid`** = `<game_pk>_ab<at_bat>_p<pitch>`
and a **`filmroom_url`**.

## Workflow

1. Open each `filmroom_url` from a pitch-type manifest, and use Film Room's
   **Download** button.
2. Save the clip as `clips/<PITCH_TYPE>/<clip_uid>.mp4` (matching the manifest
   row). A downloaded clip's `clip_uid` filename is how the importer ties the
   video to its Statcast labels — no guessing from broadcast filenames.
3. Ingest: `scripts/import_savant.py` (next milestone) reads a manifest +
   `clips/` folder and registers each video with its Statcast labels
   (`pitch_type`, `game_pk`→`game_id`, `game_date`, `batter_side`, …,
   `label_source=statcast`), leaving the release frame for the auto-detector.

You do not need to download all of them — a few dozen per pitch type is enough
to validate the pipeline before scaling up.
