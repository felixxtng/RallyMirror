# One court training dataset

Put every selected **training picture** in `court_module/dataset/images/`.
The first 17 pictures are already here as `court_001.jpg` through `court_017.jpg`.
Their manually clicked points are preserved in `annotations.json`; no relabelling
is needed. They represent 17 reviewed visual setups, not verified distinct
physical courts. `consolidation.json` records the original image mapping.

To reach 100 pictures total, add 83 more JPG/PNG images. Choose genuinely varied
court views. Use unique filenames, for example `court_018.jpg` onward; other names
also work. Do not overwrite or crop an image that already has labels. Keep test
footage and test reference images outside this training folder.

Run these commands from RallyMirror, with the virtual environment active:

1. Copy new photos into `court_module/dataset/images/` using File Explorer.
2. Register them (safe to repeat; existing labels are preserved):

   ```powershell
   python court_module/court_learning.py sync-images
   ```

   Exact duplicate decoded images are skipped, without deleting either file.
   Automatic IDs identify photos; they do not verify distinct physical courts.
   This shortcut assigns new photos to training only. For repeated views or
   future held-out splits, review recording, court and camera identities first.
   The explicit `import-sources` CSV workflow remains available for known IDs.

3. Label the 12 points on new pictures:

   ```powershell
   python court_module/court_learning.py label --all-frames
   ```

   The window resumes at the first unfinished image. Click the point requested
   in the sidebar. H marks hidden/offscreen points; Enter advances after review;
   A/D changes images; U undoes a click; Q/Esc closes. Every edit saves immediately.

4. Check progress:

   ```powershell
   python court_module/court_learning.py status
   ```

   Wait until all intended pictures are complete/usable (or deliberately excluded).
   The image count is not proof of court diversity.

5. Once labelling is finished, train fresh random weights:

   ```powershell
   python court_module/court_learning.py train --train-only --all-frames --epochs 40 --output court_module/outputs/courts100_run01
   ```

   This uses all complete, usable training images and saves `last.pt`. Incomplete
   or excluded pictures are not used. Choose another output name for the next
   run. No pretrained weights are loaded. Forty epochs is a starting run length,
   not a guarantee of accurate predictions. This training-only run produces no
   held-out accuracy score; use separate labelled courts for evaluation later.

The default dataset is now this folder. You can still specify another dataset
explicitly with `--root`. Back up **both** `images/` and `annotations.json` together.
Your old training folders are in `court_module/archive/`; the separate test labels
remain in `court_module/data_test_court/`.

See [the training guide](../TRAINING_GUIDE.md) for the 12 point definitions.
