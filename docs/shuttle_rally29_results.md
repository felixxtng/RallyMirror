# Rally29 exploratory comparison

Recovered V2 checkpoint, default heatmap threshold 0.5; temporal defaults fixed before examining these metrics. This is one exploratory rally, not a leakage-free benchmark. No parameter sweep, training or checkpoint selection was performed.

594 frames, 2196×1112, 30 FPS; 402 visible and 192 invisible labels, no unknown labels. Localisation tolerance 20 original pixels; large jump >200 pixels on consecutive accepted labelled frames. No gaps are bridged.

| Metric | Saved raw baseline | Temporal filter |
|---|---:|---:|
| True positives | 234 | 223 |
| False positives | 164 | 115 |
| False negatives | 168 | 179 |
| True negatives | 128 | 150 |
| Precision | 0.5879 | 0.6598 |
| Recall | 0.5821 | 0.5547 |
| F1 | 0.5850 | 0.6027 |
| TP mean error (px) | 8.02 | 8.11 |
| TP median error (px) | 7.16 | 7.30 |
| Invisible false positives | 64 | 42 |
| Invisible FP rate | 0.3333 | 0.2188 |
| Large jumps | 45 | 5 |
| Accepted labelled frames (%) | 67.00 | 56.90 |

False positives fell by 49 (29.9%) and large jumps by 40 (88.9%). Precision and F1 improved, but recall and localisation error worsened. This supports a precision/stability tradeoff on this rally, not an overall accuracy or fast-shot preservation claim. Twenty baseline TPs were lost (fifteen became missing and five selected a different, incorrect candidate); nine previously incorrect detections were corrected by candidate selection, giving a net loss of eleven TPs.

## Alignment evidence

Video dimensions, frame count, FPS and label bounds agree with prepared metadata. Both prediction files cover exactly frames 0–593 with matching timestamps. Recomputed raw coordinates/scores match the saved baseline on every frame within 0.0001. Prepared JPEGs at frames 0, 24, 27, 105 and 593 match decoded/letterboxed video frames with mean absolute pixel differences of 1.25–1.32 on the 0–255 scale, consistent with JPEG compression. Label coordinates are original pixels according to preparation/labeling code; selected overlays were inspected. Legacy baseline provenance was explicitly confirmed for evaluation.

## Examples (zero-based frames)

| Event | Frame / time | Observation |
|---|---|---|
| Correct false-positive rejection | 105 / 3.50 s | Label invisible; raw (823.54, 473.65), score 0.679; filtered missing. The marked location is on the court/background. |
| Genuine detection incorrectly rejected | 24 / 0.80 s | Visible GT (793.76, 29.74); raw (781.33, 17.87), error 17.19 px, score 0.730; filtered missing. |
| Successful reacquisition | 27 / 0.90 s | After missing frames 24–26, accepts (879.87, 141.80), error 15.22 px from GT (889.84, 153.30); reason reacquired. |

Annotated example montage: `outputs/shuttle_v2_filtered/rally29/examples.jpg`. Video: `outputs/shuttle_v2_filtered/rally29/shuttle_demo.mp4`. Raw candidates and decisions: `shuttle_tracking.jsonl`. Metrics and example frame lists: `comparison.json`, `examples.json`.

## Diagnostic ablations, without parameter tuning

Replaying identical cached candidates with only the quality threshold (0.6), with both temporal costs disabled, yields precision 0.5979, recall 0.5697, F1 0.5834, 154 FPs and 41 jumps. Disabling only the hotspot penalty yields exactly the full filter metrics. Thus this rally demonstrates gains from motion ranking plus thresholding; it does **not** demonstrate additional benefit from hotspot suppression. The synthetic persistent-hotspot test does exercise that mechanism.

The advertisement dot at roughly (889,159) is still accepted on frames 2 and 8. Its intermittent appearances do not accumulate sufficient evidence under the default grace/window. This important failure remains. Simply increasing persistence sensitivity on rally29 would risk rejecting real shuttle detections near the same position, including frame 27; no such tuning was performed.

## Performance, limits and next steps

A single cached-candidate replay took approximately 0.011 ms per frame for tracking only on this host. This excludes network inference, candidate extraction and video writing and is not an M2 end-to-end benchmark. Auto device selected CPU here because MPS was unavailable in this environment.

Evaluate more diverse rallies with frozen defaults and complete match provenance before making general claims. Collect dedicated persistent and intermittent background examples plus stationary-apex, fast-shot, occlusion and camera-pan cases. Consider decaying longer-term hotspot occupancy, multi-frame confirmation of remote reacquisitions, velocity uncertainty and lightweight camera compensation. Compare precision–recall tradeoffs on development matches, freeze parameters, then evaluate on held-out verified matches. In particular, compare with a confidence-only baseline and quantify fast-shot recall rather than optimizing overall F1 alone.

See [algorithm, configuration, exact inference/evaluation commands](shuttle_temporal_filtering.md) and [match mapping audit](shuttle_match_mapping.md).

## Files and validation

Modified `src/shuttle/demo.py`; created `src/shuttle/tracker.py`, `src/shuttle/evaluate.py`, `tests/test_shuttle_tracking.py` and the three documentation files. Generated a new output directory `outputs/shuttle_v2_filtered/rally29/` (ignored by git). All 11 tests pass via `.venv/bin/python -m unittest discover -s tests -v`; `git diff --check` passes.

No retraining occurred. No labels, videos, checkpoints, existing baseline outputs or match metadata were modified. Existing unrelated edits to `src/shuttle/labeler.py` and untracked rally metadata were left untouched.
