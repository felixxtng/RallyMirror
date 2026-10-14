# Shuttlecock training and labelling

This module implements our three-frame heatmap approach. It is independent of the
existing YOLO player pipeline. The uploaded ZIP contained no previous shuttle
module, video or labels, so this is a new implementation, not an edit of your
local experimental model. No tests, training or video inference were run while
preparing this package. No trained weights or video files are included.

## Which videos to use

**Best source: the original TrackNet / TrackNetV2 rally-video dataset.** Its
authors distribute complete rally clips from professional matches and amateur
games, with original trajectory labels. The documentation describes 1280x720,
30-FPS court-view footage and identifies individual rally files. You can label
the raw videos yourself with our tool; importing their annotations is optional.

- Dataset description and the authors' download link:
  <https://hackmd.io/@TUIK/rJkRW54cU>
- Authors' original implementation:
  <https://gitlab.nol.cs.nycu.edu.tw/open-source/TrackNetv2>

The download's availability was not confirmed: the source page is accessible,
but its SharePoint download could not be fetched in this environment. Use its
Download link in your browser. The extracted directory name may differ; locate
the `rally_video` folders. Do not download entire BWF matches just to label a
handful of rallies if the authors' clips are available.

| Choice | Where to find it in the authors' dataset | Why choose it |
|---|---|---|
| Amateur games, matches 1-3 | `amateur_dataset/match*/rally_video/` | Start here if your original camera is closer to court level; preview the actual angle before choosing. Match 1 includes `1_00_01.mp4`. |
| Tai Tzu Ying vs Chen Yufei, Indonesia Open 2018 | `profession_dataset/match2/rally_video/` | Another source of complete singles rallies; green court with blue surrounds. Example clips: `1_00_02.mp4`, `1_02_03.mp4`. |
| Momota vs Chou Tien Chen, Korea Open 2019 | `profession_dataset/match5/rally_video/` | Different players and match footage; same broad court colour category. Example clips: `1_01_01.mp4`, `1_01_02.mp4`. |
| Momota vs Wang Tzu We, World Tour Finals 2019 | `profession_dataset/match17/rally_video/` | Useful held-out source with a red/grey court background. Example clips: `1_02_02.mp4`, `1_15_13.mp4`. |

These choices are supported by the authors' dataset documentation. I have not
watched the individual clips or confirmed which contains a one-second off-screen
lob. Preview each candidate and choose rallies containing clears/lobs, returns
from outside the frame, fast hits, background clutter, and periods of absence.

**Official BWF alternatives:**

- Indonesia Open 2018, Tai Tzu Ying vs Chen Yufei:
  <https://www.youtube.com/watch?v=Vs3ZUem_nT0>
- Korea Open 2019 men's singles final, Momota vs Chou Tien Chen:
  <https://www.youtube.com/watch?v=eugfCRwSBJo>
- All England 2018, Tai Tzu Ying vs Chen Yufei:
  <https://www.youtube.com/watch?v=xDBA9R01RZw>

The search listings identified these matches; playback, local availability,
resolution, FPS and exact rally timestamps were not verified. Use local video
copies you can access. Choose continuous wide-view rallies at the highest
available resolution. Exclude replays, slow motion, close-ups and camera cuts.
Each prepared clip should contain one uninterrupted rally. A camera cut inside
the clip makes neighbouring-frame motion misleading.

## Data collection plan

1. Start with your existing view plus another similar camera view. Broader
   broadcast footage adds useful backgrounds, but a higher broadcast camera
   does not automatically teach a model to handle your original lower view.
2. Aim initially for roughly 1,000-2,000 labelled training frames from several
   complete rallies across at least two source recordings. This is a planning
   target, not a demonstrated minimum or an accuracy guarantee.
3. Prepare 300-500 or more labelled validation frames from a separate recording.
   Label the entire selected rallies, including invisible frames, so exits and
   re-entries are represented. Keep a third recording aside for testing later.
4. All clips from the same original match/recording must share one `source-group`
   and one split. Do not give near-duplicate frames from one match to both train
   and validation. Training refuses overlapping train/validation source groups.

If using the author clips: select two or more rallies from match2 and match5
for training, and a few from match17 for validation. If using their amateur
clips: hold out a whole recording, not a random set of frames. You do not have
to label whole hour-long matches. Complete 10-30-second rallies are enough to
start collecting examples. Count labelled frames after preparation; durations
and frame rates differ between your 60-FPS video and these 30-FPS videos.

## 1. Install from the project root on your Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use your existing environment if it is already working. The new module only
uses dependencies already listed in the uploaded requirements. The labeller
needs a desktop OpenCV window; run it locally, not inside Colab. Training picks
Apple MPS automatically where supported. CPU and CUDA are also available.

## 2. Prepare local rally clips

Place local clips in `data/shuttle/videos/`. Give each rally a unique name.
The following filenames are examples: put your downloaded or selected local
video there first. Use an actual match/recording ID for `source-group`.

```bash
python -m src.shuttle.prepare data/shuttle/videos/indonesia_rally01.mp4 --name indonesia_01 --source-group indonesia2018 --split train
python -m src.shuttle.prepare data/shuttle/videos/korea_rally01.mp4 --name korea_01 --source-group korea2019 --split train
python -m src.shuttle.prepare data/shuttle/videos/worldtour_rally01.mp4 --name worldtour_01 --source-group worldtour2019 --split val
```

Repeat for more rallies, keeping the same group and split for each match.
Preparation writes numbered, resized JPEGs, clip metadata and a blank CSV under
`data/shuttle/clips/<name>/`. It preserves the image's proportions with padding;
default model input is 960x544. Use the same dimensions for every clip.

To select a rally inside a longer local video, specify original-video frame
indices with `--start-frame N --end-frame M` (M is exclusive). For a video whose
actual FPS is 30, the interval 10-25 seconds is frames 300-750. Use the actual
video FPS; the tool does not assume 30 or 60. The prepared rally's local frame
number starts at zero.

To preserve existing centre labels from your old pipeline:

```bash
python -m src.shuttle.prepare data/sample/singles_short_01.mp4 --name original_01 --source-group original_recording --split train --import-labels data/shuttle/shuttle_train.csv
```

Imported CSV columns can be `frame,x,y,visible` (or `visibility`). Coordinates
must be pixels in the original image, as in your prior labels. With cropping,
imported frame numbers still refer to the original video and are adjusted to
the cropped clip. Verify imported dots in the labeller before training. Do not
import previously half-scaled coordinates as original coordinates.

Existing directories are not overwritten. If a preparation command is
interrupted, its partial directory cannot yet be labelled; choose a new name
or remove only that incomplete prepared directory before preparing again.

## 3. Label all prepared frames

```bash
python -m src.shuttle.labeler data/shuttle/clips/indonesia_01
```

| Action | Control |
|---|---|
| Label visible shuttle centre and advance | Left click in the large image |
| Magnify an area | Hover in the large image; see the panel on the right |
| Mark shuttle not visible and advance | N |
| Previous / next frame | A / D (Space also advances) |
| Clear an uncertain or incorrect label | U |
| Save / save and quit | S / Q (Escape also quits) |

Each edit saves immediately. Reopening resumes at the first unlabelled frame.
At the final frame, it stays there so you can review it. Previous/next thumbnails
help find a tiny moving shuttle. Click the identifiable shuttle centre, not the
middle of a long blur streak; use a consistent convention across all clips.

- `visible=1`: identifiable shuttle, with original-image x/y coordinates.
- `visible=0`: shuttle outside the image or fully hidden; x/y are empty.
- Blank visibility: unknown/unlabelled; excluded as a training target.

Do not invent an x/y point outside the image. Do not mark a difficult but
identifiable visible shuttle absent. If you cannot decide reliably, leave it
unlabelled with U and revisit it. Off-screen and hidden frames teach absence;
they do not teach an exact unseen trajectory.

This project uses **point labels**, so you do not need YOLO bounding boxes.
Label the remaining train and val rally folders the same way. Existing labels
appear as green dots; check several at different image positions. The same
letterbox transform maps clicks, targets and inference back to original pixels.
No hard-coded `x*0.5` assumption is used.

## 4. Train after labelling

```bash
python -m src.shuttle.train --epochs 40 --batch-size 2
```

This command has been provided for you to run later; it was not run here.
For memory pressure on your M2, use `--batch-size 1`. If a backend operation is
unsupported, try `--device cpu`. To keep a separate run, use a fresh output:

```bash
python -m src.shuttle.train --epochs 40 --batch-size 1 --output outputs/shuttle_run02
```

The model sees previous/current/next RGB frames together (9 input channels)
and predicts one heatmap for the middle frame. At clip endpoints, it repeats
the closest frame. A small encoder/decoder with skip connections preserves
spatial detail. Target heatmaps are binary disks of radius 3 in prepared-image
pixels, centred on the transformed label. Invisible frames have empty targets.

The loss balances the disk against the 1,024 hardest background pixels. This
makes strong false peaks on text, lines or other clutter expensive rather than
letting them disappear inside an average over hundreds of thousands of easy
background pixels. Brightness changes and horizontal flips add modest variety.
Neither feature guarantees correct detection; this implementation is untrained.

Training reads all labelled clips marked `train` and evaluates loss on `val`.
It saves `best.pt` (lowest validation loss), `last.pt`, `history.csv` and
`run.json` under `outputs/shuttle/`. Validation loss selects a checkpoint; it
does not measure tracking accuracy. Later we should measure centre error,
missed visible frames, false detections during absence and time to reacquire.
Prepared `test` clips are ignored by training. This package includes no test run.

## 5. Run inference later

```bash
python -m src.shuttle.demo data/sample/singles_short_01.mp4 --weights outputs/shuttle/best.pt --output outputs/shuttle_demo01
```

The command saves a video overlay and a JSONL row for **every** input frame:
frame/time, `detected` or `missing`, x/y, and heatmap score. Coordinates are in
original-image pixels. The score is a neural network output, not a calibrated
probability that the position is correct. This is offline processing with a
one-frame look-ahead; real-time 60-FPS performance is not claimed.

During absence, the output is `missing` with no invented coordinate. The next
frames are searched across the whole image, so re-entry does not depend on a
small window around the last position. Trails break at missing detections.
The detector may still produce a false positive; unseen-clip evaluation is
needed to judge whether it reliably reacquires the shuttle.

**Exact re-entry prediction through a one-second absence is not implemented.**
It needs a separate longer-term trajectory method and appropriate event data.
The current module implements detection, explicit gaps and full-frame search
after a gap. More labels cannot give three frames a memory of a 60-frame gap.

The detector is reusable from code as
`src.shuttle.demo.ShuttleDetector.predict(previous, current, following)`. Its
standalone CLI does not edit or run your existing player tracker. All video,
prepared data, checkpoints and outputs remain ignored by the existing git rules.
