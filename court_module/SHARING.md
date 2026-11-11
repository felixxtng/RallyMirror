# Share the court detector

The repository holds the court code, instructions and annotation JSON files.
Generated `outputs/`, model weights (`*.pt`), dataset images and the old local
`archive/` are excluded by Git. A normal code push does not include the trained
weights or demo MP4s. The `court_module/` code must be added and committed before
it appears in a push.

The local `training videos/` collection and root-level `test_court*.mp4` inputs
are also excluded. Existing tracked `data/shuttle/clips/` annotations belong to
the shared shuttle workflow; leaving them unchanged adds nothing to this commit.
Deleting tracked files would instead create deletions for the team.

To stage just the court work and its setup/docs changes:

```powershell
git add -- .gitignore README.md requirements.txt court_module
git diff --cached --stat
```

Review that list before committing and pushing. Original videos, extracted rally
frames, model weights, demo videos, archived pilots and dataset photos are not
included by this command. Send the model/demo ZIP separately.

## For teammates who want to watch the results

Share these annotated videos directly, or send the prepared demo ZIP:

- `court_module/outputs/courts100_run01/test_court_video/prediction.mp4`
- `court_module/outputs/courts100_run01/test_court2_video/prediction.mp4`

The original root-level `test_court.mp4` and `test_court2.mp4` have no overlays.

## For teammates who want to run the trained model

Share the code through Git and the model/demo ZIP separately. Teammates should
extract the ZIP into their RallyMirror project root. It recreates the expected
`court_module/outputs/courts100_run01/` paths and includes weights, training
metadata, the single-labelled-frame evaluation summary and two annotated videos.
It does not include the training pictures, raw input videos or virtual environment.

From RallyMirror, with a Python virtual environment active:

```powershell
python -m pip install numpy opencv-python torch imageio-ffmpeg
python court_module/court_learning.py predict-video court_module/outputs/courts100_run01/last.pt path/to/their_video.mp4 --output court_module/outputs/their_video_run01
```

Open `prediction.mp4` in the new output folder. Choose a fresh output directory
for each run. No training or manual clicks are needed to generate predictions.
These weights were trained from scratch on 90 usable images out of 100 labelled
pictures. They are not third-party pretrained weights. Prediction quality still
needs evaluation on independently identified courts; the saved evaluation covers
one labelled frame, whose physical court independence is unverified.

To reproduce training, teammates additionally need the matching
`court_module/dataset/images/` pictures and `annotations.json`; share those together
separately. Annotation JSON alone is not enough to retrain.

## Local cleanup

- The two `courts100_run01/test_court*_frame.../` folders are regenerable still
  previews; the annotated videos are in separate `test_court*_video/` folders.
- `outputs/courts17_run01/` is an optional older baseline. Keep a backup if you
  want to reproduce the old/new comparison.
- `archive/pilot_data/` contains unique older labels. Back it up before removing
  it. It is no longer used by the current training commands.
- Python `__pycache__/` directories can be regenerated.

Keep the active `dataset/`, clicked `data_test_court/`, current `courts100_run01/`
checkpoint and results, source code, tests, configuration, `.git` and `.venv`.
Original videos and shuttle annotations under `training videos/` may be needed
for other parts of the project; keep them or move a verified backup elsewhere.
