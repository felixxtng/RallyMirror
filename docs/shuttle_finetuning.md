# Shuttle fine-tuning preparation

The trainer now supports V2 model initialization, fresh Adam optimization,
validation-loss scheduling, early stopping and epoch-boundary resume. ShuttleNet,
its nine-channel RGB input, heatmap loss and inference checkpoint fields are
unchanged. No tracker algorithms were modified.

## Future fine-tuning command — not executed

Run from the repository root, using a fresh output directory:

```bash
.venv/bin/python -m src.shuttle.train \
  --root data/shuttle/clips \
  --split-manifest data/shuttle/split_manifest.csv \
  --init-weights outputs/shuttle_v2/best.pt \
  --output outputs/shuttle_v3_finetune \
  --epochs 40 --batch-size 1 --lr 0.0001 \
  --patience 8 --min-delta 0 \
  --lr-patience 2 --lr-factor 0.5 --min-lr 0.000001 \
  --device auto
```

This is a proposed development run, not an independently verified benchmark.
Match09/match10 are not verified single-match identities. Canonical source
verification remains unresolved as documented in the Task 2 audit.

The split manifest overrides the original `meta.split` values. It selects
22,454 training labels and 4,432 validation labels; candidate-test groups
match12/match14 and unlisted auxiliary validation01 are excluded. Unknown
visibility is excluded from targets. Declared source groups and leakage blocks
cannot cross splits; metadata groups and manifest counts must agree. Missing
selected clips, labels or frame directories fail immediately; missing/corrupt
individual frame files fail when loaded, without silently skipping a sample.
Only metadata/labels are inspected up front; there is no video or exhaustive
frame audit.

The Task 2 staging root can alternatively replace `--root`:
`outputs/shuttle_dataset_audit_20261110/staged_clips`. Keep the explicit split
manifest argument. Test clips need not exist in a train/validation staging root.

## CLI and saved state

| Argument | Behavior |
|---|---|
| `--init-weights PATH` | Strictly load compatible model weights; start fresh optimizer, scheduler, history and stopping state |
| `--resume PATH` | Restore this output directory's `last.pt`; mutually exclusive with initialization |
| `--split-manifest PATH` | Select manifest assignments rather than original metadata splits |
| `--patience N` | Stop after N consecutive epochs without meaningful validation-loss improvement; default 8; 0 disables stopping |
| `--min-delta FLOAT` | Minimum absolute validation-loss improvement for stopping/scheduling; default 0 |
| `--lr-patience N` | ReduceLROnPlateau patience, default 2 |
| `--lr-factor FLOAT` | LR reduction multiplier, default 0.5 |
| `--min-lr FLOAT` | LR floor, default 1e-6 |
| `--epochs N` | Total target epoch count, including epochs already completed on resume |
| `--smoke-test` | Force CPU and batch size 1; process one training batch and one validation batch, then stop |

Without initialization/resume, the model still starts from scratch. Existing
batch-size/LR defaults remain 2/0.001. For the canonical default clip root, the
new split manifest is automatically used when present. Custom/staged legacy
roots can still use metadata splits; declared overlap and known candidate-test
inclusion are rejected. Explicit manifest selection is recommended.

Checkpoint loading verifies input dimensions, base channels, architecture/input
metadata where available, and every state-dict tensor name and shape. Legacy V2
model-only checkpoints work for initialization, not optimizer resume.

`last.pt` stores current model, Adam state, scheduler state, completed epoch,
lowest validation loss, best model snapshot, early-stopping state, history,
configuration, run identity, metadata/label/manifest signatures, and Torch/NumPy/
Python RNG state (with device RNG state when applicable). It is the authoritative
resume point. Atomic replacement avoids partially written checkpoint files.

`best.pt` contains the lowest-validation-loss model, including improvements
smaller than `min_delta`; it retains the existing inference-compatible layout.
Resume uses `last.pt`, not `best.pt`. An embedded best-model snapshot lets resume
repair missing/stale best/history files after interruption between writes.
`history.csv` retains epoch/train_loss/val_loss and adds the LR used that epoch.
The scheduler steps after validation; its next LR is saved in optimizer state.

## Future resume command — not executed

```bash
.venv/bin/python -m src.shuttle.train \
  --root data/shuttle/clips \
  --split-manifest data/shuttle/split_manifest.csv \
  --resume outputs/shuttle_v3_finetune/last.pt \
  --output outputs/shuttle_v3_finetune \
  --epochs 40 --device auto
```

Use the same data root, manifest and output directory. Resume rejects changed
metadata/labels/manifest and incompatible explicitly supplied optimization
settings. Batch size, optimizer LR, scheduler and stopping policy restore from
the checkpoint when omitted. `run.json` retains the original plan; checkpoint
and history record completed epochs if the target is later extended.

Resume is at completed-epoch boundaries, not mid-batch: an interrupted partial
epoch is repeated from the prior committed checkpoint. Before the first
checkpoint exists, start in a fresh output directory. Exact continuation was
tested on CPU with unchanged data; bitwise equality across different hardware
backends is not promised. Frame bytes are not hashed during training preparation.
An already early-stopped checkpoint takes no further steps; changing that policy
requires a new initialized run. Existing run artifacts cannot be overwritten by
starting a fresh run in their directory; the V2 output directory is protected.

## CPU smoke test performed

Only these bounded commands were run against the real dataset/checkpoint:

```bash
.venv/bin/python -m src.shuttle.train --root data/shuttle/clips --split-manifest data/shuttle/split_manifest.csv --init-weights outputs/shuttle_v2/best.pt --output outputs/shuttle_finetune_smoke_20261110 --epochs 1 --batch-size 1 --lr 0.0001 --device cpu --smoke-test

.venv/bin/python -m src.shuttle.train --root data/shuttle/clips --split-manifest data/shuttle/split_manifest.csv --resume outputs/shuttle_finetune_smoke_20261110/last.pt --output outputs/shuttle_finetune_smoke_20261110 --epochs 2 --device cpu --smoke-test
```

Each command processes exactly one CPU training batch and one validation batch.
Together: two optimizer steps and two validation batches, no full dataset epoch.
Smoke checkpoints are marked as partial cycles and cannot resume as real
training runs. They must not be used as production checkpoints; initialize the
future full run from the unchanged V2 checkpoint.

The optimizer step counter advanced from one to two after resume, scheduler
state advanced from one to two validation observations, history retained both
cycles, and best.pt retained cycle one when cycle two had a higher validation
loss. This verifies mechanics, not fine-tuned accuracy. Validation evidence and
artifacts are under `outputs/shuttle_finetune_smoke_20261110/`.

```bash
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

Tests cover strict initialization, fresh optimizer initialization even from a
full training checkpoint, exact tiny CPU uninterrupted-versus-resumed training,
scheduler reduction, early stopping, best/history recovery, changed-data/output
protection, inference loading of best/last checkpoints, manifest overrides,
unknown/test exclusion and missing-data/group-overlap rejection. Synthetic tests
use 32×32 fixture frames only. No GPU/MPS training or full-dataset training was
run. Original V2 weights, labels, metadata and manifests were not changed.
