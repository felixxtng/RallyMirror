# RallyMirror

Badminton player tracking, body pose, racket-centre prediction and temporal
phase recognition.

- [Local website and cloud deployment setup](docs/deployment.md)
- [Shuttle annotations and training](docs/shuttle_training.md)
- [Temporal phase-recognition interface](src/temporal/README.md)
- [Racket model integration and results](docs/racket_integration.md)
- [Racket v006 training package](docs/racket_ownership_v006.md)
- [Racket annotation and review controls](docs/racket_sampled_full_review.md)

The selected racket model is **v006, full-training epoch 4, 768px**. On 444
labelled validation frames it reached **88.1% near-side and 84.0% far-side PCK20**
(within 20 original-video pixels, with missing predictions counted as failures).
Both sides passed the agreed 80% target. The rally16 test-match video is an
unlabelled visual check, not another measured accuracy result.

The racket predictor adds a separate `racket` result to each supplied player.
It preserves the existing player ID, bounding box and MediaPipe body keypoints.
It does not automatically change court mapping or temporal features.

The selected checkpoint is included in Git at `models/racket_center_v006.pt`.
A normal clone or pull includes it; no separate handoff download is required.
Its SHA-256, input format and validation results are recorded in
[the model card](models/racket_v006.json). See the integration guide for the
player input contract and working Python example.

Raw videos, annotations and preserved Runpod returns remain local. Retired
single-player models, synthetic-player training, old annotation interfaces and
experiment launchers have been removed. The current annotation entry point is
`python -m src.racket.ownership_fast label`; the training entry point is
`python -m src.racket.sampled_owner_train`.

Run the source checks from the repository root:

```bash
python -m pytest
```
