# Racket-centre model integration

The selected model is **v006, full-training epoch 4**. The runtime checkpoint
is `models/racket_center_v006.pt`; `models/racket_v006.json` is its model card.
The selected weights are committed to Git. Cloning or pulling this branch
includes the checkpoint; no separate model handoff or Runpod download is needed.

## Install and use

Install the repository requirements in your project environment. Keep
`models/racket_center_v006.pt` and the model card together. The test suite verifies
the checkpoint size and SHA-256. MediaPipe's existing body asset remains at
`models/pose_landmarker_lite.task`.

Use the teammate's tracked player boxes and existing MediaPipe keypoints:

```python
from src.racket.real_owner_infer import RealOwnerRacketPredictor

predictor = RealOwnerRacketPredictor("models/racket_center_v006.pt", device="cpu")

# frame_bgr is the original OpenCV frame. All coordinates below are ORIGINAL
# frame pixels, after reversing any upstream resizing, padding or cropping.
players = [
    {
        "track_id": "athlete_A",               # teammate's stable tracker ID
        "bbox": [410, 500, 700, 1000],        # x1, y1, x2, y2
        "keypoints": [
            {"name": "left_wrist", "x": 440, "y": 710,
             "visibility": 0.9, "presence": 0.9},
            # ...existing named MediaPipe landmarks...
        ],
    },
    # ...other tracked player...
]
results = predictor.predict_players(frame_bgr, players)
racket = results[0]["racket"]
if racket["located"]:
    centre_xy = (racket["x"], racket["y"])
```

Initialize the predictor once, not once per frame. Supply both players when
available. An empty keypoint list is supported when MediaPipe fails. If body
poses are not already computed, `CombinedRealOwnedPlayerPose` in the same module
provides an optional combined adapter.

Near/far annotation names describe court sides. They are not learned athlete
classes. Keep the teammate's stable IDs through tracking; do not assign identity
from the predicted racket location. The comparison demo's bottom-edge camera
selection is a heuristic, not the final tracker contract.

## Output contract and teammates

Each returned player keeps all supplied fields, with a separate `racket`:

```python
{
    "name": "head_center",
    "x": 420.5,
    "y": 690.2,
    "located": True,
    "coordinate_system": "original_pixels",
    "concentration": 0.02,
    "score_type": "uncalibrated_heatmap_peak_mass",
    # pose_feature_status, reason, model_format are also returned
}
```

- **Detector/tracker:** supplies person boxes and IDs; the racket model does not
  detect a missing person or replace the tracker.
- **MediaPipe:** stays frozen and supplies body features. Existing landmark names,
  count, order and coordinates remain unchanged.
- **Court mapping:** continue using the original box bottom/feet landmarks.
  The racket point is not a ground-contact coordinate.
- **Temporal classification:** its existing body-only pipeline can continue
  unchanged. Its owner must explicitly add racket features and retrain to use
  them, retaining the same feature layout, coordinate conventions and missing
  value masks in training and inference.

`located` means a finite prediction inside the image. It is not a calibrated
racket-visibility or correctness decision. Peak concentration is not an accuracy
percentage. Keep missing/occluded cases explicit instead of replacing them with
an invented wrist or zero coordinate. A wrist-to-racket display line is a drawing
aid, not another predicted shaft/handle keypoint.

## Measured results and data

PCK20 counts a located manual centre as correct when the prediction is within
20 original-video pixels. Missing predictions count as failures.

| Matched 768px validation inputs | v005 rerun | v006 |
|---|---:|---:|
| Near-side PCK20 | 81.53% | 88.06% |
| Far-side PCK20 | 75.45% | 84.01% |
| All-query PCK20 | 78.49% | 86.04% |
| Mean error | 24.23px | 17.66px |
| Strict wrong-racket hits | 13 / 884 eligible | 12 / 884 eligible |

The 444 validation frames contain 888 located racket queries from match11 and
match13. Three frames have explicitly corrected annotations; original v005
inputs and targets remain a separate native diagnostic. The original v005
640px scores were 84.91% near and 70.95% far, so use the matched rerun above for
the direct comparison.

Training used 3,259 distinct frames across 25 rallies, with 2,970 near-side and
3,159 far-side usable points. Each valid point trains independently; unknown or
unverified queries contribute no localization/ownership loss. Whole source
matches stay in one split:

- Train: match01, match03, match04, match05, match09, match10, match14.
- Validation: match11 (rallies33-40), match13 (rallies41-44).
- Reserved test: match06 (rallies14-16).
- Rallies17-24 have unconfirmed match identities and were excluded.

The selected frozen model was visually inspected on unlabelled test clip rally16.
That footage is now inspected test material. It has no manual reference labels,
so no quantitative test accuracy is claimed. The demo retains occlusions and
person-selection failures. The measured 80% acceptance is a validation result,
not full-video recall.

## Files and reproduction

- `src/racket/real_owner_infer.py`: supported runtime adapter.
- `src/racket/sampled_owner_{data,train,package}.py`: v006 masked training and bundle.
- `scripts/build_racket_ownership_v006.py`: local package build using the
  preserved annotation datasets, v005 feature bundle and returned checkpoint.
- `scripts/demo_racket_ownership_v006.py`: frozen-model rally16 comparison; the historical checkpoint can be extracted from the preserved Runpod result ZIP.
- `data/racket_ownership_sampled_v006/`: latest local annotation sidecars.
- `Runpod/racket_ownership_v006.zip`: preserved self-contained training upload.
- `Runpod/results/`: preserved result ZIPs and the latest extracted JSON reports.
- `outputs/racket_ownership_v006_demo/`: local videos and verified result summary.

The selected v006 checkpoint and model card are tracked. Raw videos, local
annotation versions, other experimental checkpoints, portable datasets, results
and chat exports stay out of Git. The historical Runpod ZIPs remain local; extracted
old run copies and retired experiment code are removed during cleanup. See the [training guide](racket_ownership_v006.md)
and [annotation controls](racket_sampled_full_review.md).
