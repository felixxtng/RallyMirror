# Real two-racket ownership trial (v004)

This trial uses the completed ownership pilot. It does not alter original racket
labels, the MediaPipe `.task`, teammate YOLO/body code, court code or temporal
code. The new `.pt` predicts a separate string-bed-centre point for a selected
player. MediaPipe remains frozen; its features were cached locally using the
manually verified player boxes.

## Dataset and evaluation

- Train: 354 approved real frames; **351 usable frames / 702 targets** after
  three exclusions described below. Each image is queried for both players.
  Clips: rally01,02,03,05,09,13,25,31,46 (seven source matches).
- Validation: 77 approved real frames, **154 targets**: rally34 (match11) and
  rally43 (match13). These source matches do not enter training.
- The 116 pilot frames whose other racket was marked unlocatable are excluded
  from paired training/evaluation. This is not an absence detector or full-video
  both-racket recall benchmark. Unreviewed frames are also excluded.
- Training review excludes rally09 frame64 (far box selects a seated official),
  rally25 frame296 (far box selects the umpire and excludes its racket target),
  and rally31 frame4 (far racket target above the body-derived crop). No source
  annotation is changed. `training_review_exclusions.json` preserves the reasons.
  No validation frame is removed.
- Legacy: retain all 578 previous validation queries, with byte-identical old
  JPEGs, body boxes and pose features, as a separate regression benchmark.
- Final test rallies14-16 remain untouched. Repeatedly reviewed validation clips
  support model selection; only an untouched final test can assess the final
  result independently of those choices.

The manifest records the explicit owner of the locked point. Original points
can belong to either side; the code never assigns them all to near by default.
Each approved image is presented twice with identical RGB context but different
selected-player boxes/pose features and different racket targets. The context
is the common square union of both expanded body boxes, independent of racket
coordinates. No numeric player ID or near/far class is learned. A tracker can
provide its own player IDs later.

The model starts from the returned v002 checkpoint and fine-tunes on real pairs
only. It uses spatial localization loss plus a 0.25-weight ranking loss that
favours the selected racket over the annotated opponent. No output mask forces
the racket inside the body box, since the head often extends outside it.

Metrics include near/far and per-match PCK20/PCK50 in **original video pixels**,
misses, and overlap subsets. Wrong-racket rate is counted where annotated heads
are more than 40px apart: prediction is more than 20px from the selected head
and within 20px of the other head. Other localization errors and missing outputs
are separate. We also report frames where both selections collapse onto the
same point and evaluate with player conditioning zeroed.

The paired/common-context scores and legacy578 scores describe different
inputs/cohorts; their percentages must not be compared directly. The package
evaluates the original v002 model on both inputs before training, providing
matched before/after comparisons.

## Run on a fresh GPU PyTorch pod

Upload `racket_ownership_v004.zip` to `/workspace`. In its terminal:

```bash
cd /workspace
python -m zipfile -e racket_ownership_v004.zip racket_ownership_v004
cd racket_ownership_v004
python setup_runpod.py
python run_racket_ownership.py --verify-only
python -u run_racket_ownership.py --device cuda --epochs 50 --batch-size 4
```

`setup_runpod.py` preserves the pod's CUDA PyTorch. Training needs no video files,
MediaPipe installation or online pretrained-weight download: images, frozen
features, v002 weights and the Python runtime are bundled. Initial setup requires
internet for missing Python dependencies. The runner prints the GPU name.
Batch size 4 means four images / eight selected-player queries.

The runner first scores v002 and then overfits four **training-only** real images
at 256px, each queried for both players, for at most 150 epochs. This checks that
the ownership/geometry pipeline can learn; it is not held-out accuracy. If that
check fails, it saves diagnostics and explicitly does **not** start full training.
If it passes, its overfit weights are discarded and full training starts again
from the untouched v002 checkpoint. Look for `FULL TRAINING STARTING` and epoch
logs. Full training runs up to 50 epochs, stopping after 15 without validation
improvement; the best checkpoint is restored. Neither extra epochs nor this
ownership pass guarantees 95% accuracy.

Use a fresh `--output` directory if repeating a run; prior outputs are preserved.
Always download the returned ZIP before removing a cloud pod.

## Returned results

Download:

```text
/workspace/racket_ownership_v004/racket_ownership_run_v004_results.zip
```

Place it in RallyMirror's `Runpod/` folder for review. It contains:

- `status.json`: complete, failed, or learning-check-failed/full-not-started.
- `racket_center_real_owner_v004.pt`: selected full-training weights, if completed.
- `comparison.json`: v002 versus new model, paired and legacy578 separately.
- `paired_metrics.json`, predictions, per-side/per-match and ownership metrics.
- `legacy578_metrics.json` and original-input per-frame predictions.
- `conditioning_zeroed_metrics.json`, learning-check results and training history.
- `previews/v002_vs_real_ownership.mp4`: before/after montage of the same 77
  validation frames. Top row selects near, bottom selects far; left v002, right
  new model. Green is selected truth, cyan other truth, magenta prediction,
  yellow selected body box. This sampled montage is not continuous tracking.
- Eight worst validation-frame images for error inspection.

If a stage fails, download its result ZIP for diagnosis before repeating a run.
No final-test score or continuous-video improvement is claimed by packaging.

## Integration contract after reviewing results

Use `src.racket.real_owner_infer.RealOwnerRacketPredictor` with this checkpoint;
old single-body-crop v001/v002 adapters have different context geometry.

```python
predictor = RealOwnerRacketPredictor(weights, device="cuda")
outputs = predictor.predict_players(frame, [
    {"track_id": near_id, "bbox": near_body_box, "keypoints": near_body_points},
    {"track_id": far_id, "bbox": far_body_box, "keypoints": far_body_points},
])
```

Each output preserves its input `track_id`, original body `bbox` and body
`keypoints`, adding a separate `racket` object in original-image pixels. It does
not append to or reorder the 17 body points. Raw body landmarks are preserved
even if clearly wrong-person MediaPipe features are suppressed for the racket
branch. The caller is responsible for correct player boxes and tracking IDs.
`CombinedRealOwnedPlayerPose` optionally computes missing body points with the
frozen body model before calling the predictor. Single-player images are allowed
but were not part of this trial's paired validation cohort. Heatmap concentration
is diagnostic, not a calibrated probability of correctness.

Court mapping can continue reading original body boxes/feet. Temporal code
continues reading its existing body coordinate schema until the teammate
explicitly adds a new racket feature and retrains that classifier. No temporal
code is changed here.
