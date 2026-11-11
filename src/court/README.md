# src/court: court detection and court-space mapping

Turns a tracked player's foot pixel into (x, y) metres on a BWF court.

## A. Manual calibration (works today, no training)
1. `python -m src.detection.demo clip.mp4`   (makes outputs/pose.jsonl)
2. `python -m src.court.calibrate clip.mp4 --points DL-B0,DR-B0,DL-B1,DR-B1,DL-SS0,DR-SS0,DL-SS1,DR-SS1,C-B0,C-SS0,C-SS1,C-B1`
   (images work too). Click the points, check the green overlay, Enter to save calibration/<name>.json.
3. `python -m src.court.demo clip.mp4 --calibration calibration/<name>.json`

## B. Learned court detector (trained from scratch)
Name labelled files `<venue>_<number>` (venueA_01 ...). B0 must ALWAYS be the end nearest the camera.
1. `python -m src.court.export_dataset`        calibration/*.json -> dataset/court (split BY VENUE)
2. `python -m src.court.train --epochs 60`     -> checkpoints/court_net.pt  (use a GPU if you can)
3. `python -m src.court.evaluate`              test-venue pixel/metre error + overlays in outputs/court_eval
4. `python -m src.court.predict image.jpg`     model -> calibration_pred/<name>.json (+ overlay)
   Videos: `--frames 15` takes the median over frames (FIXED camera only).
Use a predicted calibration in the demo exactly like a manual one.

## Fields added to court.jsonl
`foot_px`, `foot_source` (ankles | ankle | bbox), `court_xy_raw`, `court_xy` (smoothed, metres),
`in_bounds`. Court axes: x across (0 to 6.10), y along (0 to 13.40), net at y = 6.70.

## Conventions
Stand at the B0 end (y = 0) looking at the net. DL/DR = left/right doubles side line.
`python -m src.court.calibrate --list-points` shows all names.
The homography is valid for ground-plane points only (feet), never the shuttle or hands.

## Files
court_model, homography (DLT + RANSAC), calibration, calibrate (click tool), mapper, draw, demo
export_dataset, augment, court_data, net, train, predict, evaluate
