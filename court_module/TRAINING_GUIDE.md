# Train your own 12-point court detector

Use **one active dataset: `court_module/dataset/`**. Its `images/` folder already
contains your 17 labelled examples. Add new training pictures to that same
folder; labels live together in `annotations.json`.

Follow [the dataset instructions](dataset/README.md) for the complete workflow.
Run commands from RallyMirror with `.venv` active:

```powershell
python court_module/court_learning.py sync-images
python court_module/court_learning.py label --all-frames
python court_module/court_learning.py status
```

First copy new JPG/PNG photos into `court_module/dataset/images/`. Sync registers
new photos without changing existing labels. The labeler resumes unfinished
images. To reach 100 pictures total from the existing 17, add 83 more. Prioritize
different court views; file counts and automatic IDs do not verify court diversity.

Once all intended training images are complete/usable, run:

```powershell
python court_module/court_learning.py train --train-only --all-frames --epochs 40 --output court_module/outputs/courts100_run01
```

This starts from random weights, uses all complete/usable training images and
saves `last.pt`. No pretrained weights are loaded. Choose a fresh output name for
each run. Incomplete or excluded images are not used. Forty epochs is a starting
experiment; training loss alone does not measure performance on new courts.
This mode produces no validation or test score.

## The twelve points

B0 is the near end. Left/right are as seen looking from B0 towards the net.
Use the OUTER doubles boundary even during singles. Click the centre of the
painted intersection. Mark hidden/offscreen points with H instead of guessing.

| Order | Name | Location |
|---|---|---|
| 1 | DL-B0 | Near left outer corner |
| 2 | DR-B0 | Near right outer corner |
| 3 | DL-B1 | Far left outer corner |
| 4 | DR-B1 | Far right outer corner |
| 5 | DL-SS0 | Near short-service line / left outer sideline |
| 6 | DR-SS0 | Near short-service line / right outer sideline |
| 7 | DL-SS1 | Far short-service line / left outer sideline |
| 8 | DR-SS1 | Far short-service line / right outer sideline |
| 9 | NL-TOP | Left end of the TOP net edge at the post |
| 10 | NR-TOP | Right end of the TOP net edge at the post |
| 11 | C-SS0 | Near centre line / short-service line |
| 12 | C-SS1 | Far centre line / short-service line |

Points 9 and 10 are elevated net points. They are learned and displayed but
excluded from court-floor fitting. The other ten points lie on the floor.
The centre points identify service-area divisions; court geometry alone does
not identify the server, score or serve legality.

Click the requested point to save and advance. Click a sidebar row to correct
a point. H marks hidden; U undoes; A/D moves between images; Enter advances
after review; X excludes/restores an image; Q/Esc exits. All edits autosave.
At least four visible floor points and all twelve reviewed statuses are needed
for a usable image. Back up `images/` and `annotations.json` together.

## Training and testing stay separate

### Watch an annotated video

The `predict` command saves one annotated JPG; it does not change the input MP4.
To generate a separate H.264 MP4 with court predictions on every frame, install
the video encoder helper once:

```powershell
python -m pip install imageio-ffmpeg
```

Then run:

```powershell
python court_module/court_learning.py predict-video court_module/outputs/courts100_run01/last.pt test_court2.mp4 --output court_module/outputs/courts100_run01/test_court2_video
```

Open `prediction.mp4` inside that output directory. Source audio is included when
present. The original MP4 and trained weights are unchanged. Each frame is
predicted independently, so lines may jitter; no temporal smoothing is applied.
Frames with an implausible floor fit show point predictions and a message instead
of floor lines. `summary.json` and `frames.jsonl` record those decisions; accepted
fits are not an accuracy score. Choose a new output directory for another run.

Photos added by `sync-images` are training examples. Automatic court/view IDs
are explicitly unverified. For a future validation/test split, review the actual
recording and physical court identities, then keep every image from the same
court in a single split. The `import-sources` command still accepts CSVs with
explicit `source,group,court_id,view_id,split,frame` fields when identities are known.

Your clicked test reference remains in `court_module/data_test_court/`; neither
it nor `test_court.mp4` / `test_court2.mp4` was added to training. Test labels
provide reference answers for evaluation; they do not update weights.

Older training folders are preserved in `court_module/archive/`. Use the new
dataset for all further work. Commands default to it; `--root` can override this
when evaluating a separate test dataset.
