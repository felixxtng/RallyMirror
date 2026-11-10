# Offline shuttle trajectory refinement

The opt-in `--offline-refine` mode performs candidate collection, whole-video
hotspot analysis and exact Viterbi sequence selection, then decodes the source
video again to render the selected path. It preserves the existing baseline and
`--temporal-filter` modes. It does not train or change ShuttleNet.

## Algorithm

1. Reuse `ShuttleDetector.predict_candidates()` on three consecutive RGB frames.
   Endpoint frames repeat their neighbour as before. Every heatmap component is
   searched across the full image; original-pixel coordinates, score and area
   retain the existing transform and conventions. Only candidate lists are kept,
   not decoded frames or heatmaps.
2. Cluster candidate locations using a spatial grid and fixed anchors within
   `hotspot_radius × image_diagonal`. Deduplicate observations from a frame.
   Count distinct episodes separated by more than `hotspot_episode_gap_seconds`,
   total time span, and longest uninterrupted occupancy. Recurrences accumulate
   over the entire video, even across long absences. Locations and strength come
   entirely from candidate evidence; no advertisement positions are hardcoded.
3. Apply a soft penalty to locations with at least three episodes spanning at
   least one second, or continuous occupancy exceeding one second. A short apex
   alone creates no penalty. Recurrence strength is
   `min(1, (episodes - 1) / hotspot_min_episodes)` once its episode/span requirements
   hold. Continuous strength ramps from zero at one stationary interval to one at
   twice that interval. Use the larger strength.
4. Discount the hotspot cost when candidate evidence on both immediately adjacent
   frames supports movement through the location. Neighbour displacement must
   exceed twice the cluster radius and be within the support search radius. The
   incoming/outgoing vectors must have cosine >= -0.5 and speed ratio <= 4. This
   geometric support is a heuristic, not a ground-truth visibility test. It does
   not require the neighbours to be on the final selected path.
5. Each Viterbi state is a candidate or missing. Candidate reward is
   `score - confidence_reference - hotspot_penalty × strength × support_factor`.
   Missing has zero reward. Detection-to-detection transitions subtract
   `motion_penalty × (1 - exp(-0.5 × (distance / motion_scale)^2))`, where
   `motion_scale = motion_radius × diagonal × 30 / fps`. Entering/leaving missing
   costs `switch_cost`; starting and ending a detection segment also pay this
   cost. Missing-to-missing is free. The all-missing trajectory is always valid.
   Backtracking finds the globally best first-order path using future and past
   evidence. Exact ties prefer missing, then the earlier candidate in input order.
6. Emit only real candidate coordinates or nulls, never interpolated or frozen
   coordinates. Missing states release spatial memory, allowing full-frame
   reacquisition on the next frame. There is no displacement cutoff. Rendering
   uses the refined path, and trails break across every missing frame.

This first-order model intentionally avoids a constant-velocity/ballistic
assumption that could over-penalize smashes, reversals or camera movement.
Its bounded motion cost also allows undesirable jumps. It has no explicit
acceleration state, camera compensation or memory through missing states.

## Parameters

Use `--offline-config path.json` to override any `OfflineConfig` fields.
Defaults were fixed before evaluating the offline rally29 metrics. No parameter
sweep or label-driven default tuning was performed. Synthetic tests cover the
algorithm without using rally29 labels.

| Parameter | Default | Meaning |
|---|---:|---|
| confidence_reference | 0.5 | Detection reward reference; score is not a calibrated probability |
| switch_cost | 0.12 | Cost of entering/leaving a detected segment |
| motion_radius | 0.07 | Diagonal fraction per 1/30 second for soft displacement scale |
| motion_penalty | 0.35 | Saturating motion cost |
| hotspot_radius | 0.003 | Diagonal fraction for fixed-anchor clustering |
| hotspot_episode_gap_seconds | 0.25 | Gap separating recurring-location episodes |
| hotspot_min_episodes | 3 | Minimum episodes for recurrence evidence |
| hotspot_min_span_seconds | 1.0 | Required span for recurrence evidence |
| hotspot_stationary_seconds | 1.0 | Continuous occupancy grace interval |
| hotspot_penalty | 0.65 | Maximum hotspot cost before support discount |
| support_radius | 0.12 | Diagonal fraction per 1/30 second for moving-neighbour search |
| hotspot_support_discount | 0.8 | Fraction of hotspot cost removed by moving support |

## Commands

From the repository root, run fresh model inference and optionally generate the
three-way diagnostic video. Choose another output path if it already exists.

```bash
.venv/bin/python -m src.shuttle.demo data/shuttle/videos/rally29.mp4 \
  --weights outputs/shuttle_v2/best.pt \
  --output outputs/shuttle_v2_offline/rally29_new \
  --offline-refine --comparison-video --device auto
```

To reproduce the evaluated run without network inference, reuse the previously
saved full candidate lists (the command used for the reviewed artifacts):

```bash
.venv/bin/python -m src.shuttle.demo data/shuttle/videos/rally29.mp4 \
  --weights outputs/shuttle_v2/best.pt \
  --output outputs/shuttle_v2_offline/rally29_review \
  --offline-refine --comparison-video \
  --candidate-cache outputs/shuttle_v2_filtered/rally29/shuttle_tracking.jsonl \
  --device auto
```

That output directory now exists: use a new suffix when rerunning. Cache mode
skips model construction and ignores the weights/device arguments. It requires
all candidates for every frame and a sibling `shuttle_metadata.json`. It validates
video name, dimensions, count, FPS, origin, timestamps and coordinate system using
the existing alignment utility; the tracker also validates candidate bounds and
finite scores. Cached inputs must come from the intended video and checkpoint:
metadata cannot establish source identity cryptographically. The older temporal
sidecar lacks a candidate threshold; compatibility assumes its known CLI default
0.5 and records this assumption. New offline sidecars record the threshold.
The requested threshold must match the cache threshold.

Evaluate all three with identical rules:

```bash
.venv/bin/python -m src.shuttle.evaluate \
  --labels data/shuttle/clips/rally29/labels.csv \
  --video data/shuttle/videos/rally29.mp4 \
  --predictions outputs/shuttle_v2_test/rally29/shuttle_tracking.jsonl \
                outputs/shuttle_v2_filtered/rally29/shuttle_tracking.jsonl \
                outputs/shuttle_v2_offline/rally29_review/shuttle_tracking.jsonl \
  --tolerance 20 --jump-threshold 200 --confirm-original-coordinates
```

The legacy baseline's original-coordinate provenance was established in the
previous evaluation and reconfirmed against these identical cached candidates.
Unknown labels remain excluded; a mislocalized visible prediction counts as
both FP and FN. Jumps count adjacent detected labelled frames, without bridging
gaps. Acceptance uses labelled frames. See the existing evaluator documentation
for alignment and denominator details.

View the generated videos on macOS:

```bash
open outputs/shuttle_v2_offline/rally29_review/shuttle_demo.mp4
open outputs/shuttle_v2_offline/rally29_review/shuttle_comparison.mp4
```

## Outputs and diagnostics

`shuttle_tracking.jsonl` preserves `frame`, `time_seconds`, `status`, `x`, `y`,
`score`. Scores remain heatmap scores. Extra fields include raw detections, all
candidates, offline candidate/cluster IDs, selected hotspot cost, moving support
and selection reason. Missing frames have null coordinates and score. The sidecar
records original-pixel conventions, source, threshold and config.
`shuttle_offline_diagnostics.json` records the objective value and automatically
identified hotspot centers, observation/episode counts, span and strength.

`shuttle_demo.mp4` has original dimensions/FPS and uses refined detections.
Optional `shuttle_comparison.mp4` presents original V2 (orange), default temporal V2
(yellow) and offline V2 (green), with markers, one-second trails and detection
status. Labels sit outside the source image to keep advertisement regions visible.
All panels use the same candidates; default temporal decisions were checked to
match the saved temporal output on every rally29 frame. A custom temporal config
is not applied to the comparison panel. Existing outputs are refused, never
silently replaced. Interrupted runs require a fresh output path.

## Cost, tests and limits

DP work is `sum((K[t-1]+1) × (K[t]+1))` and backpointer/candidate storage is linear
in candidate count plus frame count. Grid-based clustering is local. The support
check additionally considers neighbouring candidate pairs for hotspot candidates
(worst case cubic in candidates/frame). No candidates are truncated. For rally29,
there are only 567 candidates, maximum six/frame; three refinement-only runs took
approximately 2.3–3.5 milliseconds for all 594 frames. This excludes inference,
video I/O and rendering and is not an M2 end-to-end performance claim. No new
packages were added. Video memory is bounded to decoded working frames; candidate
and path memory grow with uploaded video length.

```bash
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

All 20 tests pass, including synthetic stationary/intermittent hotspots, moving
through hotspots, competing candidates, smashes/reversals, future evidence,
occlusion/recovery, empty input, determinism, input validation, an exhaustive
small-sequence check of the Viterbi objective, cached/fresh-inference integration,
rendering of missing coordinates, output protection and prior compatibility tests.

Whole-video stationary evidence can suppress a genuine shuttle that revisits a
location. Incorrect moving support can protect a background peak. The optimizer
can accept a locally unfavorable background candidate to connect two otherwise
rewarding trajectory segments. Missing states can allow distant reacquisition
after only one missed frame. Scores and penalties are uncalibrated. Camera pans,
cuts and unusually long stationary shuttle intervals need further evaluation.
Do not claim this is a globally optimal physical trajectory: it optimizes the
specified first-order heuristic objective exactly.

Results and observed failure frames are in
[the exploratory offline evaluation](shuttle_offline_rally29_results.md).
Match identities remain incomplete; preserve all existing metadata/splits and
use [the match mapping audit](shuttle_match_mapping.md) before any future formal
benchmark or retraining. No random rally-level split was introduced.
