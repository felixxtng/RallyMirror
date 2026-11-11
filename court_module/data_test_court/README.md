# Manual assessment of test_court.mp4 frame 29

The unchanged 40-epoch model was applied to frames 29, 116 and 290. On visual
inspection, floor lines are broadly aligned but the predicted top net endpoints
sit above the actual net tape in all three frames. This is not a numerical
accuracy result. No retraining was performed on this video.

This video has Sydney branding/layout resembling training view 15. The physical
court may overlap training, so these images do not establish unseen-court
generalisation. The annotation record marks this uncertainty explicitly.

One blank image from frame 29 is prepared for independent manual labels. No
network predictions were inserted into the ground truth. From RallyMirror:

```powershell
python court_module/court_learning.py label --root court_module/data_test_court
```

Place all twelve points, including the actual left/right TOP NET TAPE endpoints.
H marks a genuinely hidden point; do not click a predicted point just because
the model placed it there. Each edit saves immediately. Q quits/resumes normally.

After annotation, point error can be measured with:

```powershell
python court_module/court_learning.py evaluate court_module/outputs/courts17_run01/last.pt --root court_module/data_test_court --split test
```

This assesses point accuracy on one new video frame, not verified independent
physical-court accuracy. The evaluator states that scope explicitly. Keep this
annotation dataset separate from `data_courts17` and do not train on it while
using it for assessment. Future genuinely different courts are needed to assess
generalisation, and more than one frame/court is needed for a useful benchmark.
