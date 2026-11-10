# Multi-rally shuttle evaluation

Run date: 2026-11-10 (Asia/Singapore). Frozen V2 checkpoint and tracker defaults; exploratory evaluation of rally18, rally24, rally35 and rally44. Rally29 is shown separately and excluded from every four-rally aggregate.

## Findings

- Original V2 pooled precision 71.30%, recall 71.04%, F1 71.17%; TP=2360, FP=950, FN=962.
- Temporal pooled precision 85.98%, recall 69.24%, F1 76.71%; TP=2300, FP=375, FN=1022.
- Offline pooled precision 85.79%, recall 71.61%, F1 78.06%; TP=2379, FP=394, FN=943.

F1 winners: rally18: Offline; rally24: Offline; rally35: Offline; rally44: Offline.
Offline versus Original V2 by F1: better on rally18, rally24, rally35, rally44; tied on none; worse on none. It consistently improves on all four observed rallies.
Offline versus Temporal by F1: better on rally18, rally24, rally35, rally44; tied on none; worse on none. It consistently improves on all four observed rallies.

The consistency is specific to F1. Offline recall exceeds temporal on every rally, but offline has 19 more pooled FPs (394 versus 375), 79 fewer FNs (943 versus 1,022), and more jumps (46 versus 15; original has 341). Its pooled invisible-frame FP rate is 17.65% (170/963), compared with 15.99% temporal (154/963) and 38.53% original (371/963). Offline's invisible-frame FP rate is worse than temporal on rally18, rally35 and rally44, and better on rally24. It is therefore not uniformly better at background suppression or trajectory stability than temporal filtering.

## Input and alignment checks

| Rally | Video resolution | FPS | Decoded/label rows | Visible | Invisible | Unknown (excluded) | Scored frames |
|---|---|---:|---:|---:|---:|---:|---:|
| rally18 | 1280×720 | 30 | 1199 | 978 | 221 | 0 | 1199 |
| rally24 | 1280×720 | 30 | 1642 | 1117 | 522 | 3 | 1639 |
| rally35 | 1920×1080 | 30 | 682 | 631 | 51 | 0 | 682 |
| rally44 | 1918×1058 | 30 | 769 | 596 | 169 | 4 | 765 |

Total: 4,292 decoded frames; 4,285 scored frames (3,322 visible, 963 invisible), seven unknown. Unknown indices: rally24 822, 1327, 1328; rally44 504–507. Unknown visibility is never treated as invisible.

All four video/metadata resolutions, FPS values, zero start offsets and frame counts agree. Labels have unique indices and explicitly cover the whole clip. Predictions cover exactly the zero-based video range with matching timestamps and finite coordinates inside original-image bounds. The existing evaluator validates each exported mode before scoring.

Five prepared JPEGs per rally (first, quarter, middle, three-quarter and last) were compared against the corresponding decoded/letterboxed source frames: mean absolute pixel differences range from 1.129 to 1.745 on the 0–255 scale. Large discrepancies would abort scoring. Ground-truth overlays were also inspected at rally18:577, rally24:750, rally35:332 and rally44:382; no obvious coordinate-scale discrepancy was observed. The preview is `outputs/shuttle_multi_rally_eval_20261110/alignment_previews.jpg`.

Missing files, frame-count mismatches and failed inference: none.

## Frozen methodology

The current CLI and evaluation code were inspected first. A batch orchestration utility reuses the existing detector, trackers and evaluator. It runs ShuttleNet once per frame and supplies the identical full-frame candidate list to every mode: Original V2 chooses the highest-score component using the `predict()` convention; temporal uses a fresh `TemporalTracker`; offline uses `OfflineTracker.refine()` on the whole rally. Endpoint three-frame handling and original-pixel coordinate transforms are unchanged. No tracking algorithm or parameter was modified. No small ROI, retraining, threshold sweep, label-driven tuning or match reassignment was performed.

Heatmap threshold 0.5; localisation tolerance 20 original pixels; jump threshold strictly greater than 200 original pixels. A mislocalized visible prediction contributes one FP and one FN. True-positive error summaries include only predictions within tolerance. Unknown labels are skipped for detection, jump and acceptance metrics. Jumps count consecutive accepted labelled frames, never across missing/unknown gaps. Acceptance is accepted detections divided by scored frames. Undefined ratios would be reported as undefined/null.

All four runs used CPU because auto device selection found MPS unavailable in this environment. These are inference/evaluation runs, not a performance benchmark. Outputs are JSONL predictions, candidate caches and diagnostics; annotated videos were not generated for this metrics-only batch.

Configuration equality with the saved rally29 temporal/offline sidecars is asserted before inference. The exact frozen values are:

```json
{
  "heatmap_threshold": 0.5,
  "localisation_tolerance_pixels": 20,
  "jump_threshold_pixels": 200,
  "temporal": {
    "motion_radius": 0.06,
    "motion_penalty": 0.4,
    "minimum_quality": 0.6,
    "reacquire_misses": 3,
    "hotspot_radius": 0.004,
    "hotspot_grace_seconds": 0.27,
    "hotspot_window_seconds": 2.0,
    "hotspot_penalty": 0.65
  },
  "offline": {
    "confidence_reference": 0.5,
    "switch_cost": 0.12,
    "motion_radius": 0.07,
    "motion_penalty": 0.35,
    "hotspot_radius": 0.003,
    "hotspot_episode_gap_seconds": 0.25,
    "hotspot_min_episodes": 3,
    "hotspot_min_span_seconds": 1.0,
    "hotspot_stationary_seconds": 1.0,
    "hotspot_penalty": 0.65,
    "support_radius": 0.12,
    "hotspot_support_discount": 0.8
  }
}
```

Checkpoint SHA-256: `cdfac9bf0f7540c874d44ec2b1e258eb303a43b221bf93a5a617a32512a39ce1`. Before/after hashes of the checkpoint, selected videos, labels, metadata and existing inference/tracker/evaluator source files match exactly.

## Consolidated per-rally results

| Rally | Mode | Precision | Recall | F1 | FP | Invisible FP rate | TP mean / median error (px) | Jumps >200 px | Accepted |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rally18 | Original V2 | 60.07% | 71.06% | 65.11% | 462 | 85.52% | 5.84 / 5.25 | 173 | 96.50% |
| rally18 | Temporal | 85.08% | 70.55% | 77.14% | 121 | 21.27% | 5.89 / 5.30 | 5 | 67.64% |
| rally18 | Offline | 85.04% | 73.82% | 79.04% | 127 | 22.62% | 5.84 / 5.24 | 16 | 70.81% |
| rally24 | Original V2 | 82.16% | 61.86% | 70.58% | 150 | 9.96% | 5.61 / 4.85 | 35 | 51.31% |
| rally24 | Temporal | 87.92% | 59.27% | 70.80% | 91 | 6.70% | 5.60 / 4.85 | 0 | 45.94% |
| rally24 | Offline | 88.41% | 60.79% | 72.04% | 89 | 5.94% | 5.61 / 4.84 | 7 | 46.86% |
| rally35 | Original V2 | 72.61% | 73.53% | 73.07% | 175 | 66.67% | 5.87 / 5.48 | 71 | 93.70% |
| rally35 | Temporal | 90.35% | 72.74% | 80.60% | 49 | 15.69% | 5.82 / 5.47 | 5 | 74.49% |
| rally35 | Offline | 91.42% | 74.33% | 81.99% | 44 | 27.45% | 5.82 / 5.46 | 12 | 75.22% |
| rally44 | Original V2 | 75.78% | 85.57% | 80.38% | 163 | 56.80% | 5.58 / 5.06 | 62 | 87.97% |
| rally44 | Temporal | 81.09% | 82.05% | 81.57% | 114 | 37.87% | 5.57 / 5.04 | 5 | 78.82% |
| rally44 | Offline | 79.16% | 85.40% | 82.16% | 134 | 44.38% | 5.58 / 5.05 | 11 | 84.05% |

## Four-rally micro aggregate (rally29 excluded)

| Mode | Pooled TP | Pooled FP | Pooled FN | Pooled TN | Micro precision | Micro recall | Micro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original V2 | 2360 | 950 | 962 | 592 | 71.30% | 71.04% | 71.17% |
| Temporal | 2300 | 375 | 1022 | 809 | 85.98% | 69.24% | 76.71% |
| Offline | 2379 | 394 | 943 | 793 | 85.79% | 71.61% | 78.06% |

Micro precision = ΣTP / (ΣTP + ΣFP); micro recall = ΣTP / (ΣTP + ΣFN); micro F1 = 2ΣTP / (2ΣTP + ΣFP + ΣFN). No percentages or per-rally medians were averaged. Longer rallies contribute more observations; rally24 contributes the most scored frames.

## Recall changes and new errors

Practical recall concern was defined before results as a drop of at least five percentage points. This is a descriptive flag, not a statistical significance test. Frames are temporally correlated and match independence is unverified, so frame-wise p-values would be misleading.

| Rally | Temporal − original recall (pp) | Offline − original recall (pp) | Offline − temporal recall (pp) | Recall concern |
|---|---:|---:|---:|---|
| rally18 | -0.51 | +2.76 | +3.27 | No ≥5 pp drop |
| rally24 | -2.60 | -1.07 | +1.52 | No ≥5 pp drop |
| rally35 | -0.79 | +0.79 | +1.58 | No ≥5 pp drop |
| rally44 | -3.52 | -0.17 | +3.36 | No ≥5 pp drop |

No mode crosses the predeclared five-point recall-drop flag. Offline's largest recall reduction versus original is 1.07 percentage points on rally24; its reduction on rally44 is 0.17 points. Temporal's largest reduction versus original is 3.52 points on rally44. These are observed differences, not claims of statistical equivalence.

Offline transitions relative to each reference mode:

| Rally | Reference | Previously correct TPs lost | TPs gained | Lost TPs near learned hotspots | Invisible FPs removed | Invisible FPs newly added |
|---|---|---:|---:|---:|---:|---:|
| rally18 | Original V2 | 1 | 28 | 1 | 139 | 0 |
| rally18 | Temporal | 3 | 35 | 1 | 6 | 9 |
| rally24 | Original V2 | 19 | 7 | 10 | 21 | 0 |
| rally24 | Temporal | 15 | 32 | 7 | 11 | 7 |
| rally35 | Original V2 | 5 | 10 | 2 | 20 | 0 |
| rally35 | Temporal | 5 | 15 | 2 | 1 | 7 |
| rally44 | Original V2 | 4 | 3 | 2 | 21 | 0 |
| rally44 | Temporal | 4 | 24 | 2 | 5 | 16 |

A lost TP means the reference prediction was within 20 pixels but offline was missing or mislocalized. The hotspot column checks whether that reference TP falls within the learned hotspot radius. It identifies potentially harmful recurrence suppression; it does not prove counterfactual causation. No penalty ablation was run because parameters were frozen. New invisible FPs show errors introduced relative to a reference even when overall FP counts improve. Complete frame lists are saved in `failure_diagnostics.json`.

## Important failure and recovery examples

| Rally | Frame / time | Event | Ground truth | Original / temporal / offline |
|---|---|---|---|---|
| rally18 | 382 / 12.73s | genuine candidate near learned hotspot lost | (624.00, 8.00) | (626.64, 9.79), err 3.19px / (626.64, 9.79), err 3.19px / missing |
| rally18 | 131 / 4.37s | invisible-frame false positive rejected | invisible | (1077.00, 575.56) / missing / missing |
| rally18 | 259 / 8.63s | real shuttle recovered versus temporal | (605.33, 304.00) | (603.25, 304.41), err 2.12px / missing / (603.25, 304.41), err 2.12px |
| rally18 | 170 / 5.67s | new invisible-frame FP versus temporal | invisible | (1076.67, 575.32) / missing / (650.65, 310.00) |
| rally18 | 151 / 5.03s | remaining invisible-frame false positive | invisible | (680.20, 503.94) / (680.20, 503.94) / (680.20, 503.94) |
| rally24 | 201 / 6.70s | genuine candidate near learned hotspot lost | (753.33, 197.33) | (754.11, 201.25), err 4.00px / (754.11, 201.25), err 4.00px / missing |
| rally24 | 56 / 1.87s | genuine shuttle rejected or mislocalized | (594.67, 108.00) | (588.10, 106.86), err 6.67px / missing / missing |
| rally24 | 131 / 4.37s | invisible-frame false positive rejected | invisible | (635.55, 503.85) / missing / missing |
| rally24 | 1 / 0.03s | real shuttle recovered versus temporal | (684.00, 224.00) | (685.84, 228.00), err 4.40px / missing / (685.84, 228.00), err 4.40px |
| rally24 | 499 / 16.63s | new invisible-frame FP versus temporal | invisible | (587.41, 160.70) / missing / (587.41, 160.70) |
| rally24 | 11 / 0.37s | remaining invisible-frame false positive | invisible | (759.55, 382.54) / (759.55, 382.54) / (759.55, 382.54) |
| rally35 | 534 / 17.80s | genuine candidate near learned hotspot lost | (1028.00, 570.00) | (1022.30, 576.96), err 9.00px / (1022.30, 576.96), err 9.00px / missing |
| rally35 | 448 / 14.93s | genuine shuttle rejected or mislocalized | (864.00, 466.00) | (877.33, 460.57), err 14.39px / (877.33, 460.57), err 14.39px / missing |
| rally35 | 75 / 2.50s | invisible-frame false positive rejected | invisible | (1536.54, 837.24) / missing / missing |
| rally35 | 80 / 2.67s | real shuttle recovered versus temporal | (936.00, 438.00) | (1536.20, 837.03), err 720.74px / missing / (929.27, 437.43), err 6.75px |
| rally35 | 137 / 4.57s | new invisible-frame FP versus temporal | invisible | (1015.11, 492.11) / missing / (1015.11, 492.11) |
| rally35 | 78 / 2.60s | remaining invisible-frame false positive | invisible | (1536.47, 836.96) / (970.80, 460.79) / (970.80, 460.79) |
| rally44 | 403 / 13.43s | genuine candidate near learned hotspot lost | (1244.70, 411.22) | (1238.67, 407.66), err 7.00px / (1238.67, 407.66), err 7.00px / (1148.80, 276.48), err 165.38px |
| rally44 | 33 / 1.10s | genuine shuttle rejected or mislocalized | (1008.95, 397.25) | (1004.17, 396.00), err 4.94px / (1004.17, 396.00), err 4.94px / (1785.21, 975.92), err 968.21px |
| rally44 | 4 / 0.13s | invisible-frame false positive rejected | invisible | (810.57, 901.78) / missing / missing |
| rally44 | 41 / 1.37s | real shuttle recovered versus temporal | (1078.87, 441.17) | (1076.67, 440.40), err 2.33px / missing / (1076.67, 440.40), err 2.33px |
| rally44 | 12 / 0.40s | new invisible-frame FP versus temporal | invisible | (749.56, 574.46) / missing / (1785.38, 975.69) |
| rally44 | 0 / 0.00s | remaining invisible-frame false positive | invisible | (1785.23, 977.74) / (1785.23, 977.74) / (1785.23, 977.74) |

Annotated stills are saved at `outputs/shuttle_multi_rally_eval_20261110/rallyXX/failure_examples.jpg`; orange is original, yellow temporal, green offline, and cyan ground truth. Examples are selected from explicit error categories in frame order; they are illustrations, not additional evaluation data or parameter-tuning targets.

Visual inspection confirms a distinct source of clutter in rally44: playback controls are embedded in the input video. At frame 33, original and temporal are 4.94 pixels from the labelled shuttle, while offline selects the lower-right control icon, 968.21 pixels away. The learned cluster centered at (1785.23, 977.74) has 29 observations over 1.5 seconds, but only two episodes and a longest continuous run of ten frames. Under the frozen recurrence rules it has zero hotspot strength, so this clutter receives no background penalty. Frames 0 and 12 also show detections on this icon. No source cropping, cleanup or algorithm change was applied. The diagnostic evidence is saved in `rally44/playback_control_cluster.json` under the run directory.

Across the four rallies, offline loses 29 previously correct original-mode detections and gains 48; 15 of those 29 losses fall near learned hotspots. For example, rally18:382 and rally24:201 are correct in both original and temporal but missing offline. These demonstrate new errors introduced by refinement despite its net F1 gain. The spatial overlap suggests recurrence suppression can be harmful, but no counterfactual parameter ablation was performed.

## Earlier rally29 development example (separate)

| Mode | Precision | Recall | F1 | FP | FN | Invisible FP rate | TP mean / median error (px) | Jumps >200 px | Accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original V2 | 58.79% | 58.21% | 58.50% | 164 | 168 | 33.33% | 8.02 / 7.16 | 45 | 67.00% |
| Temporal | 65.98% | 55.47% | 60.27% | 115 | 179 | 21.88% | 8.11 / 7.30 | 5 | 56.90% |
| Offline | 67.05% | 58.71% | 62.60% | 116 | 166 | 17.71% | 8.02 / 7.16 | 13 | 59.26% |

Rally29 was the earlier development example. It is absent from all aggregate counts, recall-change tables and four-rally winner decisions above.

## Recommendation for preliminary team integration

Use offline refinement as the provisional default for the uploaded-video pilot when the priority is coordinate detection F1 and recall: it wins F1 on all four observed rallies and has the highest micro F1 (78.06%). Keep temporal filtering selectable for conservative absence handling and smoother trails, since it has fewer pooled FPs and roughly one-third as many large jumps. Preserve raw candidates and original predictions for inspection. In particular, check clips containing playback controls or other short-lived overlays before relying on their trajectory output. Missing output must remain missing rather than being interpolated or frozen. This recommendation is preliminary and should be revisited after broader evaluation with verified match provenance.

## Limits

These four rallies are additional exploratory clips, not a leakage-free held-out benchmark. The original matches behind all 48 rallies have not been verified. No match IDs were inferred from rally numbering, stored group declarations or video overlays, and no random rally-level train/validation/test split was introduced. Do not claim independent generalisation from these results.

Twenty-pixel localisation and 200-pixel jump thresholds are identical across runs, as requested, but represent different fractions of 1280-wide and roughly 1920-wide source videos. TP-only error summaries can improve merely by rejecting difficult detections; interpret them with recall. Lower jump counts can reflect more missing frames rather than better trajectories. No physical shot-speed annotations, end-to-end latency benchmark or formal significance test was available. Current offline tracking can penalize genuine repeated positions, accept background candidates as trajectory bridges, and reset spatial memory after a missing state.

## Exact commands and artifacts

The batch invocation used (run from the repository root):

```bash
.venv/bin/python -m src.shuttle.batch_evaluate --rallies rally18 rally24 rally35 rally44 --weights outputs/shuttle_v2/best.pt --output outputs/shuttle_multi_rally_eval_20261110 --device auto
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m outputs.shuttle_multi_rally_eval_20261110.analyse_results
.venv/bin/python -m outputs.shuttle_multi_rally_eval_20261110.build_report
git diff --check
```

The output directory now exists; choose a new output directory when rerunning inference. Existing output directories are refused. The analysis/report scripts are retained with this run. CLI rechecks of each exported JSONL used:

```bash
.venv/bin/python -m src.shuttle.evaluate --labels data/shuttle/clips/rally18/labels.csv --video data/shuttle/videos/rally18.mp4 --predictions outputs/shuttle_multi_rally_eval_20261110/rally18/original/shuttle_tracking.jsonl outputs/shuttle_multi_rally_eval_20261110/rally18/temporal/shuttle_tracking.jsonl outputs/shuttle_multi_rally_eval_20261110/rally18/offline/shuttle_tracking.jsonl --tolerance 20 --jump-threshold 200 --output outputs/shuttle_multi_rally_eval_20261110/rally18/cli_recheck.json
.venv/bin/python -m src.shuttle.evaluate --labels data/shuttle/clips/rally24/labels.csv --video data/shuttle/videos/rally24.mp4 --predictions outputs/shuttle_multi_rally_eval_20261110/rally24/original/shuttle_tracking.jsonl outputs/shuttle_multi_rally_eval_20261110/rally24/temporal/shuttle_tracking.jsonl outputs/shuttle_multi_rally_eval_20261110/rally24/offline/shuttle_tracking.jsonl --tolerance 20 --jump-threshold 200 --output outputs/shuttle_multi_rally_eval_20261110/rally24/cli_recheck.json
.venv/bin/python -m src.shuttle.evaluate --labels data/shuttle/clips/rally35/labels.csv --video data/shuttle/videos/rally35.mp4 --predictions outputs/shuttle_multi_rally_eval_20261110/rally35/original/shuttle_tracking.jsonl outputs/shuttle_multi_rally_eval_20261110/rally35/temporal/shuttle_tracking.jsonl outputs/shuttle_multi_rally_eval_20261110/rally35/offline/shuttle_tracking.jsonl --tolerance 20 --jump-threshold 200 --output outputs/shuttle_multi_rally_eval_20261110/rally35/cli_recheck.json
.venv/bin/python -m src.shuttle.evaluate --labels data/shuttle/clips/rally44/labels.csv --video data/shuttle/videos/rally44.mp4 --predictions outputs/shuttle_multi_rally_eval_20261110/rally44/original/shuttle_tracking.jsonl outputs/shuttle_multi_rally_eval_20261110/rally44/temporal/shuttle_tracking.jsonl outputs/shuttle_multi_rally_eval_20261110/rally44/offline/shuttle_tracking.jsonl --tolerance 20 --jump-threshold 200 --output outputs/shuttle_multi_rally_eval_20261110/rally44/cli_recheck.json
```

Each rally has shared `candidates.jsonl`, three mode directories containing `shuttle_tracking.jsonl`, `shuttle_metadata.json` and `evaluation.json`, plus offline hotspot diagnostics. Run-wide files include `summary.json`, frozen configurations, before/after hashes, failure diagnostics and analysis scripts. All are under `outputs/shuttle_multi_rally_eval_20261110/` and ignored by git. The requested Markdown report is `docs/shuttle_multi_rally_evaluation.md`.

Added `src/shuttle/batch_evaluate.py` for shared-inference orchestration and `tests/test_shuttle_batch_evaluate.py` for count-based pooling tests. Existing model, detector, preprocessing, temporal/offline algorithms and evaluator code were not changed. All 22 tests pass, including unknown-label exclusion and frame/coordinate alignment checks. CLI rechecks match the batch metrics. No retraining occurred; checkpoint, source videos, labels and metadata were preserved.
