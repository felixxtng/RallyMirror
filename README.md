# RallyMirror: pose proof of concept

The existing YOLO + ByteTrack pipeline selects a target player. This POC adds:

1. **Crop:** take the target box, add 15% space around it, and keep it inside the frame.
2. **Pose:** pass the clean crop to MediaPipe's pretrained Pose Landmarker Lite.
3. **Coordinates:** turn crop-relative joint locations into original-frame pixels:
   `frame_x = crop_left + landmark_x * crop_width` (likewise for y).
4. **Overlay:** draw visible joints and body connections on the original frame.
5. **Save:** write an annotated MP4 and JSONL (one JSON record per target frame).

No training, movement classification, action recognition or analytics is included.

## Setup (PowerShell, from the RallyMirror folder)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task' -OutFile models\pose_landmarker_lite.task
```

If the existing environment has no pip, run ` .\.venv\Scripts\python.exe -m ensurepip --upgrade ` before installation. YOLO downloads `yolo11n.pt` on its first run, so setup/first run need internet. Downloaded weights and outputs are ignored by Git.

## Run a short demo

Place a real badminton clip at `data/sample/singles_short_01.mp4` (none was supplied in this checkout), then:

```powershell
.\.venv\Scripts\python.exe -m src.detection.demo --max-frames 150
```

For another clip or the full video:

```powershell
.\.venv\Scripts\python.exe -m src.detection.demo "C:\path\to\clip.mp4"
```

Outputs: `outputs/player_pose_demo.mp4` and `outputs/pose.jsonl`.
The terminal prints the processed, target and pose frame counts.
Green box = tracked target; yellow lines and purple dots = pose.

Each existing tracking record gains `keypoints`: 33 named joints when detected,
with `x`, `y` in full-frame pixels, `visibility` and `presence` scores. An empty
list means no pose was found. Frames without a target have no JSON record,
matching the original tracking format. Low-confidence joints are saved but not
rendered; model estimates may fall outside the frame. No 3D coordinates are claimed.

## Explain it during the demo

“Felix's stage finds and tracks the player. My stage crops that player and uses
an already-trained pose model to estimate body joints. I convert the joints back
to the original image coordinates and draw a skeleton. This proves that tracking
can feed a pose component; it does not yet recognise badminton strokes.”

## Limits and verification

The existing target selector chooses the person whose box extends lowest in the
image; it can switch people. Each crop is processed independently (no smoothing
or temporal action model). Occlusion, fast motion, small players and overlapping
people can cause missing or incorrect joints. A 15% crop margin helps retain
limbs but can also include another person. Output video has no audio.

Implementation reference: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python

At implementation time there was no sample clip. Dependency installation was
blocked by local Windows temporary-directory permissions. Model download and
runtime validation status are recorded in the chat; do not treat syntax/unit
checks as evidence of successful real-video inference.

Verified locally: Python syntax checks, Git whitespace checks, and two geometry
unit tests pass (crop padding/clipping, invalid boxes, coordinate mapping and
missing-pose handling). The pretrained Lite model is downloaded in `models/`.
Actual MediaPipe/YOLO inference and video encoding have NOT been verified.
