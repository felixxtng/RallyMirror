# Shuttle temporal filtering

`ShuttleDetector.predict_candidates()` runs the same three-RGB-frame network and
full-frame letterbox transform as before. It returns every connected component
with at least three heatmap pixels, sorted by peak score. Centroids are confidence
weighted and mapped into original-image pixels; `area` is in heatmap pixels.
`predict()` still returns only `x`, `y`, `score` for the highest scoring component,
or `None`. Model, preprocessing, weights and labels are unchanged.

`TemporalTracker` ranks candidates causally using heatmap score minus soft motion
and persistent-location costs. Velocity uses the last two accepted positions;
distance to both the extrapolated position and last position allows reversals
and deceleration. The motion scale grows with speed and time since observation.
There is no hard movement cutoff. Low quality means missing, without extrapolated
output. After three misses the motion cost is removed and stale velocity is reset
on reacquisition. Every inference searches the entire frame. Trails break at gaps.

Hotspot evidence includes all candidates, including rejected ones. Nearby repeated
locations accumulate observations in a rolling window. A grace period protects
brief stationary appearances; the penalty ramps up afterward and evidence expires.
This is a heuristic: a long apex, stationary real shuttle, slow camera pan or a
later shuttle crossing a persistent hotspot can be suppressed. Fast shots or
camera cuts can still cause misses, and isolated background peaks can still pass.
No optical flow or camera compensation is implemented.

## Defaults and configuration

All configuration is in `TrackerConfig`; optional `--tracker-config config.json`
accepts a JSON object overriding any fields. Defaults were chosen before inspecting
the filtered rally29 metrics and were not swept against its labels.

| Field | Default | Meaning |
|---|---:|---|
| motion_radius | 0.06 | Base soft distance scale, fraction of image diagonal |
| motion_penalty | 0.4 | Maximum subtracted motion cost |
| minimum_quality | 0.6 | Minimum score after costs |
| reacquire_misses | 3 | Missed frames before global motion reset |
| hotspot_radius | 0.004 | Location matching radius, fraction of image diagonal |
| hotspot_grace_seconds | 0.27 | Observations required before penalizing (at least two) |
| hotspot_window_seconds | 2.0 | Rolling observation expiry |
| hotspot_penalty | 0.65 | Maximum subtracted persistence cost |

Motion scale adds 1.5 times recent displacement per elapsed frame. A hotspot's
penalty reaches its maximum after twice the grace count within the window.
Hotspots keep fixed anchor locations (no drift) and at most 256 locations are
retained, prioritizing repeated/recent evidence. Time is converted using actual
FPS. Tracking uses small Python lists/deques and introduces no dependencies.
Candidate extraction and neural inference remain the main compute costs.

## Running

Use the repository environment (`source .venv/bin/activate`) or substitute
`.venv/bin/python` for `python`. Both inference commands require fresh output paths;
choose a new suffix if already run. The filtered path below was produced during
implementation; use `rally29_run02` to run it again.

```bash
python -m src.shuttle.demo data/shuttle/videos/rally29.mp4 \
  --weights outputs/shuttle_v2/best.pt \
  --output outputs/shuttle_v2_baseline/rally29 --device auto

python -m src.shuttle.demo data/shuttle/videos/rally29.mp4 \
  --weights outputs/shuttle_v2/best.pt \
  --output outputs/shuttle_v2_filtered/rally29_run02 --temporal-filter --device auto

python -m src.shuttle.evaluate \
  --labels data/shuttle/clips/rally29/labels.csv \
  --video data/shuttle/videos/rally29.mp4 \
  --predictions outputs/shuttle_v2_test/rally29/shuttle_tracking.jsonl \
                outputs/shuttle_v2_filtered/rally29/shuttle_tracking.jsonl \
  --confirm-original-coordinates --tolerance 20 --jump-threshold 200

python -m unittest discover -s tests -v
```

Outputs retain `shuttle_demo.mp4`, `shuttle_tracking.jsonl`, and all existing row
fields. Filtered rows add `raw_detection`, `candidates`, `tracking_reason`, and,
for accepted points, quality and cost fields. `score` remains the original heatmap
score. Missing rows have null coordinates and score. A new `shuttle_metadata.json`
records dimensions, frame origin, coordinate system, FPS, video and tracker config.

## Evaluation rules and alignment

A TP requires visible ground truth and distance <= tolerance. A misplaced visible
prediction is both FP and FN. Invisible detections are FP; invisible misses are TN.
Unknown labels are skipped, including for acceptance and jump metrics. Jumps count
only accepted detections on adjacent labelled frame indices; gaps are never bridged.
Mean/median errors describe TPs only, so they must be interpreted with recall.
Acceptance is the percentage of labelled frames with a detection. Undefined ratios
and errors are null. Pixel tolerance and jump threshold are shared for all files.

Evaluation checks exact zero-based prediction coverage, duplicate indices, image
bounds, video/label dimensions, frame count, FPS, offsets and optional timestamps.
Sidecars must declare original pixels and the matching frame range. Legacy JSONL
cannot prove its own coordinate scale or source identity: the confirmation flag
requires manual verification of provenance. Video/label identity beyond this
metadata should also be checked visually. Nonzero preparation offsets require the
exact cropped video and matching local metadata; ambiguous offsets are rejected.
