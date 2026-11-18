# Temporal / Phase Recognition

Given what a player is doing over several frames, what phase of play are they
in? A single frame cannot tell you whether a player is moving forward or
recovering backward — you need a **sequence**. This module turns a stream of
per-frame pose keypoints into a per-frame **phase** label.

Per the assignment brief, the deliverable right now is the **interface**, not a
trained model. A rule-based classifier stands in for a future learned model
behind a swappable interface.

## Scope and ownership

This module (`src/temporal/`) sits at the end of the
pipeline and depends on two upstream workstreams that are not finished yet:

```
Tracking                 Pose                        Temporal (this module)
raw clips + player  -->  frame + player box     -->  keypoint sequence --> phase
boxes (+ shuttle)        -> 17 COCO keypoints
```

This module does **not** implement data collection/annotation or pose
estimation. The phase demo here is intentionally separate from
`src/detection/demo.py` so the workstreams do not collide.

## Upstream contracts (`contracts.py`)

**Contract 1 — tracked-clip record:**

```json
{"frame": 0, "player_id": 1, "bbox": [x1, y1, x2, y2], "confidence": 0.9}
```

**Contract 2 — per-frame pose record:**

```json
{"frame": 0, "player_id": 1, "keypoints": [[x, y], ... 17 entries ...]}
```

17 COCO keypoints, each `[x, y]` or `[x, y, score]`, **normalized to the player
bbox** (coords in ~[0, 1]). This is the single input the temporal interface
reasons over.

## Components

| File | Role |
|------|------|
| `schema.py` | Our annotation label vocabulary + the 4 POC phases + the label→phase **projection** |
| `config.py` | Loads `configs/default.yaml` |
| `contracts.py` | Validates upstream records (tolerant); provides deterministic stub streams |
| `windower.py` | Buffers per-frame features into fixed windows, per `player_id`, resets on gaps |
| `classifier.py` | `PhaseClassifier` interface; `RuleBasedClassifier` (now), `TrainableClassifier` (future seam) |
| `pipeline.py` | `PhasePipeline` — the public API: `process(pose_record) -> (frame, player_id, phase)` |
| `demo_phase.py` | End-to-end demo: writes a phase JSONL and (if OpenCV present) an overlay video |

## Phases

Five mutually-exclusive, duration-based phases make up the first-class
vocabulary annotators label and the model trains on (`schema.PHASES`):

1. `ready` — neutral/waiting stance, poised for the opponent's hit.
2. `movement` — travelling toward the shuttle to play it.
3. `stroke` — the hitting action as a window: loading + swing + follow-through
   around contact.
4. `recovery` — regaining balance / landing / repositioning toward base after a
   stroke, before settling back into `ready`.
5. `idle` — dead time, no active play (between points, after a score).

Natural rally cycle: `ready → movement → stroke → recovery → ready`, with `idle`
as the out-of-rally state. Phases are *durations*, deliberately kept distinct
from *instantaneous events* like a hit's exact contact frame — this stops any
class from being starved of frames.

The schema keeps a projection hook (`schema.IDENTITY_PROJECTION`, configurable
via `configs/default.yaml -> projection`). With phases annotated directly it is
the identity; it exists so a **future** fine-grained vocabulary can project down
to these phases without code changes.

### Future layers (not trained yet)

Added only once the phase model performs well, each as a **separate** classifier
or parallel track — not folded into the five phases:
- **Stroke type** (`schema.FUTURE_STROKE_TYPES`: smash / clear / drop / ...,
  BFMD-aligned), conditioned on a detected `stroke` phase.
- **Rally state** (`schema.FUTURE_RALLY_STATES`: between_points /
  serve_initiation / rally_active / point_end).

## Running the demo

Use the project virtual environment (it has OpenCV, torch, etc.):

```powershell
# From the repo root
.\.venv\Scripts\python.exe -m src.temporal.demo_phase
```

Outputs:
- `outputs/phase_demo.jsonl` — one `{frame, player_id, phase}` record per frame.
- `outputs/phase_demo.mp4` — overlay video (only if OpenCV is installed; the
  JSONL is always written regardless).

### Note on the rule-based output

The stub + rule-based classifier emit only `stroke`, `movement`, and `ready` —
the three phases separable from pose motion alone (a swing → stroke, body
translation → movement, settled → ready). It cannot distinguish `ready` from
`recovery` (both are non-swing states, differing by context — direction
relative to the last hit), nor active-play from dead-time `idle`. That is
expected: resolving those is exactly the job of the future **trained**
classifier. The rule-based stand-in returns a valid but coarse subset of the
five phases; it is not a bug.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Covers the schema/projection, the contracts (including tolerant extra-field
handling), the windower (sliding, gap reset, per-track isolation), the
rule-based classifier, the pipeline, and the demo (JSONL + optional render).

## Future work

- **Richer labels via projection.** Swap the active projection in
  `configs/default.yaml` to target the full fine-grained shot vocabulary (e.g. a
  stroke-type classifier) over the same annotations — no re-annotation needed.
- **External pretraining (optional).** The BFMD dataset
  (<https://github.com/Ning-D/BFMD>) ships 17-keypoint COCO poses and shot-type
  labels close to our schema; it could pretrain the temporal model before
  fine-tuning on our own annotated clips. Treated as a cited external aid, not a
  substitute for our own data.
- **Doubles.** The POC is singles-only. Doubles adds a second player per side and
  a harder tracking/association problem; deferred.

## Curation scope (honest boundaries)

This module does structural work — a unified schema, contract validation,
coordinate normalization expectations, deterministic stubs. It does **not**
perform exhaustive label quality-control; that is a separate data-curation
effort. The POC demonstrates the pipeline and interface, not classifier
accuracy.
