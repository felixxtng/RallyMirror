# src/court: court detection and court-space mapping

Turns a tracked player's foot pixel into (x, y) metres on a BWF court.

## Manual labels and your own trained court model

The new pipeline imports distinct court views, lets you manually click twelve court keypoints,
and trains a custom CNN from random weights without a pretrained backbone.
Start from the project root with:

```powershell
python court_module/court_learning.py label
```

See [the step-by-step training guide](../../TRAINING_GUIDE.md). The separate
OpenCV baseline below is not used for this pipeline's labels or predictions.
The twelve points include the two TOP net endpoints and two centre/service-line
intersections. Raised net points are excluded from the floor homography.
Labelling and training default to one image per fixed view, and training-only
runs can defer evaluation until you collect different test courts.

## Automatic detection baseline (Windows)

From the **RallyMirror project root**, with `.venv` active:

```powershell
python court_module/detect_court.py rally33-40/videos/rally33.mp4
```

No new packages are needed: this uses NumPy and OpenCV. The root-level launcher
handles the separate `court_module/src` directory automatically.

Results are in `court_module/outputs/rally33/frame000000/`:

- `source.jpg`: original video frame.
- `region.png`, `white_mask.png`, `candidates.png`: intermediate detection steps.
- `overlay.jpg`: fitted court lines in green; the orange line is the **ground
  projection of the net**, not its raised top cable.
- `topdown.jpg`: court warped to a top-down view.
- `calibration.json`: image-to-court homography compatible with `load_calibration`.

To test another frame or clip:

```powershell
python court_module/detect_court.py rally33-40/videos/rally33.mp4 --frame 120
python court_module/detect_court.py rally33-40/videos/rally34.mp4
```

Existing results are protected. Add `--overwrite` to replace the selected
clip/frame's results. Failed detection saves diagnostics but no calibration.

The detector locates a green floor region, extracts locally bright low-saturation
markings, merges Hough line candidates, and scores boundary hypotheses against
**all painted lines** in the court model. It uses no clicked points or fixed
corner coordinates. B0 is the near end of the camera view.

This is an OpenCV baseline, not a trained neural network. It assumes an end-on
view with the whole court visible, a green floor, and reasonably straight lines.
Side views, camera cuts, severe occlusion and other floor colours need further
work. The line-support score is a fit score, not an accuracy percentage. Inspect
the overlay and measure error against independent labelled intersections before
claiming accuracy. Eight clips from the same match do not establish performance
on other matches or camera setups. This command fits one frame, not a full video
or automatic camera-change detection.

Run the existing geometry and new detector checks:

```powershell
python -m unittest discover -s court_module/tests -p "test_court*.py"
```

## Workflow
1. Run the detection demo as usual: `python -m src.detection.demo clip.mp4` (makes `outputs/pose.jsonl`).
2. Calibrate once per clip (fixed camera): `python -m src.court.calibrate clip.mp4`
   Click the prompted court points, check the green overlay, press Enter to save `calibration/<clip>.json`.
3. Map and render: `python -m src.court.demo clip.mp4 --calibration calibration/<clip>.json`
   Writes `outputs/court.jsonl` (pose records plus court fields) and `outputs/court_demo.mp4`.

## Added fields in court.jsonl
`foot_px`, `foot_source` (ankles | ankle | bbox), `court_xy_raw`, `court_xy` (smoothed, metres),
`in_bounds`. Court axes: x across (0 to 6.10), y along (0 to 13.40), net at y = 6.70.

## Conventions
Stand at the B0 end (y = 0) looking at the net. DL/DR = left/right doubles side line.
Run `python -m src.court.calibrate --list-points` for all point names.
The homography is valid for ground-plane points only (feet), never the shuttle or hands.

## Files
court_model.py (court geometry) | homography.py (DLT, error measures) | calibration.py (build/save/load)
calibrate.py (click tool) | mapper.py (foot point, smoothing, JSONL mapping) | draw.py (overlay, minimap) | demo.py
