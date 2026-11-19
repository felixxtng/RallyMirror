# v006: completed sampled labels, independent queries, 80% per side

Upload only **racket_ownership_v006.zip** to **/workspace** on a Runpod PyTorch
GPU pod. The ZIP contains images, frozen MediaPipe features, the v005 checkpoint,
training code and verification hashes. No source videos or extra label files
are needed. A GPU with 24 GB VRAM is suitable; start with batch size 4.

In the Runpod terminal:

```bash
cd /workspace
python -m zipfile -t racket_ownership_v006.zip
python -m zipfile -e racket_ownership_v006.zip racket_ownership_v006
cd racket_ownership_v006
python setup_runpod.py
python run_racket_ownership.py --verify-only
python run_racket_ownership.py --device cuda
```

The last command runs the complete workflow:

1. Verify package hashes, match-separated splits and the unchanged 444-frame
   validation membership. Evaluate v005 as a baseline on matched inputs.
2. Run a four-training-image learning check. Discard its overfit weights and
   reload v005 before full training. A failed check produces a diagnostics ZIP.
3. Train at 768 pixels for up to 100 epochs, early stopping after 25 without
   improvement. Every independently usable near/far point contributes; unknown
   rackets contribute no localization or opponent-ranking loss. All coordinates,
   body boxes and poses receive the same training augmentation.
4. Refine the best checkpoint at 960 pixels for up to 30 epochs with smaller
   learning rates, stopping after 12 without improvement.
5. Keep the candidate with the best weaker-side PCK20 across the original
   640px baseline, 768px baseline/full stage, and 960px refinement.
   Ties use combined PCK20, fewer wrong-racket selections, PCK50 and mean error.
   A worse refinement does not replace a better full-stage checkpoint.
6. Save both side scores, per-match results, wrong-racket counts, missing outputs,
   a matched v005/v006 comparison video, the chosen checkpoint and a results ZIP.

If the GPU runs out of memory, preserve the diagnostics and rerun with a fresh
output folder and a smaller batch:

```bash
python run_racket_ownership.py --device cuda --batch-size 2 --output racket_ownership_run_v006_batch2
```

Default result to download **before removing the pod**:

```text
/workspace/racket_ownership_v006/racket_ownership_run_v006_results.zip
```

Put that ZIP in RallyMirror's **Runpod/** folder. Key outputs:

- **acceptance.json**: passes only if near AND far PCK20 are each **>= 80%**.
- **paired_metrics.json**: each side, each match/rally, misses and ownership errors.
- **comparison.json** and **baseline_matched_resolution_metrics.json**: comparisons.
- **racket_center_real_owner_v006.pt**: selected deployable racket extension.
- **previews/v005_vs_v006.mp4**: matched validation montage, not continuous tracking.
- **status.json**, **run_config.json**, **full/history.json**, **refine/history.json**.

PCK20 means within 20 original-video pixels of the labelled racket centre.
Missing predictions count as failures. Validation covers located rackets with
verified ownership; it does not establish visibility detection or continuous
video stability. The 444 validation frame IDs stay fixed, with 3 explicit
annotation corrections. Original v005 images and targets remain a separate
native diagnostic. No final test video is used for model selection.

This trains the separate image/pose/selected-body racket extension. MediaPipe
weights, the teammates' detector, body keypoints, temporal code and court code
are unchanged. Use the existing RealOwnerRacketPredictor with the resulting
checkpoint; its saved input size and padding are loaded automatically.
Higher resolution and more labels improve the experiment's prospects, but the
80% result must be measured after training.

## Rebuild from the preserved local history

The selected v006 model is already tracked at `models/racket_center_v006.pt`.
Inference does not require the old checkpoints or any training bundle.
The existing `Runpod/racket_ownership_v006.zip` remains the complete reproducible
training package. Cleanup removes extracted experiment duplicates.

Only when rebuilding features from the local annotation versions, restore the
frozen v005 inputs and returned warm-start from their preserved ZIPs first:

```powershell
python -m zipfile -e Runpod/racket_ownership_v005.zip outputs/racket_ownership_v005/bundle
python -m zipfile -e Runpod/results/racket_ownership_run_v005_results.zip Runpod/racket_ownership_v005/racket_ownership_run_v005
python scripts/build_racket_ownership_v006.py --output outputs/racket_ownership_v006_rebuild --package Runpod/racket_ownership_v006_rebuild.zip
```

The build script preserves existing builds and packages. Use fresh `--output`
and `--package` paths instead of overwriting the preserved v006 upload.
Raw clips and annotation sidecars are local prerequisites.
