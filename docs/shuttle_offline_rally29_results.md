# Offline rally29 exploratory evaluation

Default OfflineConfig, frozen before inspecting offline metrics. No parameter sweep, label-based tuning, training or match split changes. Candidate evidence is identical to the previously verified V2 inference: 567 components over 594 frames (maximum six/frame). The initial run and the review render use identical tracking results; the review render improves comparison-label readability only.

Video: 2196×1112 at 30 FPS, frames 0–593; 402 visible labels and 192 invisible labels, no unknown labels. The existing evaluator checked metadata, coordinates, frame coverage and timestamps. The original baseline lacks a sidecar; its original-coordinate provenance was confirmed in the preceding evaluation and by comparison to the cached raw coordinates/scores (all frames within 0.0001).

This is one exploratory rally. Incomplete match mapping prevents claims of independent generalisation. No parameters were tuned on rally29, so no additional-rally result is implied or presented. If using it to tune future versions, explicitly treat it as development data and evaluate frozen parameters on additional rallies, then on verified held-out matches.

## Identical evaluation settings

Tolerance: 20 original pixels. A visible prediction outside tolerance is both FP and FN. Jumps: displacement >200 pixels on consecutive accepted labelled frames, without bridging gaps. Error statistics are TP-only; acceptance is the percentage of labelled frames with an accepted detection.

| Metric | Original V2 | Temporal V2 | Offline V2 |
|---|---:|---:|---:|
| True positives | 234 | 223 | 236 |
| False positives | 164 | 115 | 116 |
| False negatives | 168 | 179 | 166 |
| True negatives | 128 | 150 | 158 |
| Precision | 58.79% | 65.98% | 67.05% |
| Recall | 58.21% | 55.47% | 58.71% |
| F1 | 58.50% | 60.27% | 62.60% |
| Invisible-frame FP rate | 33.33% | 21.88% | 17.71% |
| Invisible-frame FPs | 64 | 42 | 34 |
| Mean TP localisation error (px) | 8.02 | 8.11 | 8.02 |
| Median TP localisation error (px) | 7.16 | 7.30 | 7.16 |
| Jumps >200 px | 45 | 5 | 13 |
| Accepted detection percentage | 67.00% | 56.90% | 59.26% |

Offline improves precision, recall and F1 versus temporal filtering on this rally, while reducing invisible-frame false positives. Total FPs increase from 115 to 116 because mislocalized visible detections increase from 73 to 82. Large jumps increase from five to thirteen. Compared with original V2, offline has 48 fewer total FPs, 32 fewer jumps and two more TPs. **The advertisement-suppression and recall objectives are supported here; improved stability over the temporal tracker is not supported by the jump metric.**

## Automatically learned advertisement locations

The grid clustering identifies both `.com` dots among six suspicious locations. Visual inspection confirms the following learned centers correspond to the advertisement dots; these coordinates were never supplied to the algorithm.

| Location | Learned center | Candidate observations | Distinct episodes | Span |
|---|---|---:|---:|---:|
| Left dot | (889.13, 158.73) | 19 | 16 | 19.7 s |
| Right dot | (1703.48, 161.43) | 21 | 17 | 19.7 s |

Diagnostic FP counts within ten original pixels of each learned center (using the same visibility/localisation definition as the evaluator):

| Location | Original V2 | Temporal V2 | Offline V2 |
|---|---:|---:|---:|
| Left advertisement | 19 | 11 | 1 |
| Right advertisement | 3 | 3 | 0 |

The right dot is present as a lower-scoring component much more often than it wins raw inference. Whole-video evidence includes those losing candidates. No semantic advertisement classifier or manual coordinate mask was used.

## Specific examples (zero-based indices)

| Frame / time | Outcome | Evidence |
|---|---|---|
| 2 / 0.067 s | Left advertisement rejected | Raw (889.13, 158.73), score 0.979; GT invisible; offline missing. |
| 26 / 0.867 s | Right advertisement rejected | Raw (1703.63, 161.45), score 0.985; GT visible at (892.12, 155.59); offline missing. Correctly removes an FP but still misses the real shuttle. |
| 24 / 0.800 s | Genuine shuttle recovered versus temporal | Offline accepts raw (781.33, 17.87), 17.19 px from GT; temporal had rejected it. |
| 27 / 0.900 s | Reacquisition after missing frames 25–26 | Offline accepts (879.87, 141.80), 15.22 px from GT. |
| 407 / 13.567 s | Genuine shuttle incorrectly rejected | Raw (1186.45, 189.15), score 0.508, error 4.88 px; offline missing. |
| 95 / 3.167 s | Recurrence suppresses a genuine candidate | Raw (1003.95, 198.16) is 15.91 px from GT but clustered as recurring; offline chooses (1063.69, 144.13), 65.04 px away. |
| 368 / 12.267 s | Left advertisement still accepted | Offline (889.14, 158.96), score 0.979; full hotspot penalty 0.65, but the global path accepts it as a bridge between rewarding segments. This is outside 20 px of the visible GT. |

See `outputs/shuttle_v2_offline/rally29_review/examples.jpg`, `examples.json`, and `comparison_frame002.jpg`. The montage includes raw/GT/offline markers; the comparison video contains no ground-truth annotations.

## Reproduce and view

The reviewed artifacts are under `outputs/shuttle_v2_offline/rally29_review/` (ignored by git). `shuttle_tracking.jsonl` and `shuttle_metadata.json` are evaluator-compatible. `shuttle_offline_diagnostics.json` includes hotspot evidence and the DP objective. `comparison.json` stores all three metrics; `performance.json` stores refinement-only timings.

```bash
open outputs/shuttle_v2_offline/rally29_review/shuttle_demo.mp4
open outputs/shuttle_v2_offline/rally29_review/shuttle_comparison.mp4
```

Both videos were verified as 594 frames at 30 FPS. Offline video is 2196×1112; comparison is 1920×388 with readable headers outside the three 640×324 source-image panels. Comparison temporal decisions exactly match the saved temporal decisions on all 594 frames. Refined JSONL matches the earlier offline run exactly.

See [algorithm, configuration, inference/cache/evaluation commands and tests](shuttle_offline_refinement.md).

## Validation, limits and next work

All 20 tests pass, including existing baseline/temporal tests and nine new offline tests. Small-sequence exhaustive enumeration independently verifies the Viterbi objective. Rendering tests verify raw points rejected by refinement are absent from the offline video. `git diff --check` passes. Refinement-only time was 2.3–3.5 ms for all frames on this host; this excludes network inference and rendering and is not a claim of full M2 throughput.

This version optimizes a first-order heuristic, not a physical trajectory model. Negative local rewards can be accepted to avoid disconnecting a segment (frame 368). Repeated genuine positions can become hotspots (frame 95). Missing states reset motion immediately; this enables recovery but can permit distant re-entry. Geometry-only moving support can be fooled by unrelated candidates. No camera compensation or long-video segmentation is implemented. No extra-rally or match-independent performance claim is justified.

Recommended next steps: evaluate the frozen defaults across multiple exploratory rallies; collect per-shot fast-motion recall and camera-pan cases; investigate second-order motion or duration-aware missing states and confidence-calibrated recurrence on separate development data. Evaluate any tuning on additional rallies. Complete canonical match provenance before formal splits or retraining. Do not tune away frame 368 using its known advertisement coordinates.

## Files and preservation

Modified `src/shuttle/demo.py` for opt-in dispatch/CLI. Created `src/shuttle/offline_tracker.py`, `src/shuttle/offline_demo.py`, `tests/test_shuttle_offline.py`, `docs/shuttle_offline_refinement.md` and this report. Reused the existing candidate extraction, temporal tracker and evaluator without modifying their algorithms. Generated fresh directories `outputs/shuttle_v2_offline/rally29/` and `rally29_review/`; the first render was retained rather than overwritten.

No retraining occurred. Existing pretrained weights, labels, videos, baseline/temporal outputs, match metadata and unrelated player/pose/court/phase components were preserved. The pre-existing labeler edit and untracked metadata were left untouched. No match IDs were invented and no splits were created.
