# Shuttle match mapping and split preparation

Audit date: 2026-11-10 (Asia/Singapore). This supersedes the earlier metadata-only audit. No model was trained, no original metadata was corrected, and no match identities were invented.

## Findings

- All 48 numbered rallies have nonempty, syntactically valid `source_group` values matching the supplied mapping. The 13 declared groups are internally consistent as metadata assignments. This does not establish that each ID denotes one real match.
- `match09` and `match10` contain visibly different matches/venues. They are conservative ingestion buckets rather than verified single-match identities. They remain intact in the proposed split.
- `match02` is not unused: it belongs to the existing `validation01` clip (741 labelled frames), which was V2 validation/model-selection data.
- `validation01` and rally26 show the same Hangzhou broadcast/player/score cues. Treat `match02` and `match09` as one conservative leakage block until exact source/clip overlap is resolved. This is a safety linkage, not a new canonical match ID.
- V2 records explicitly list rally01–rally16 as training clips (11,827 labels), and `validation01` as validation (741 labels). Old group IDs were later consolidated and reused; historical IDs must not be interpreted through the current mapping without Git history.
- There is no certified clean independent test set yet. `match12` and `match14` are reserved candidate groups with no recorded V2 training or prediction evaluation; their source independence is still unverified.

## Complete group inventory and proposed split

Total frames include unknown visibility. Labelled = visible + invisible, excluding unknown. The primary target population is the 48 numbered rallies; `validation01` is auxiliary and excluded from split percentages.

| Source group | Rallies | Total frames | Visible | Invisible | Unknown | Labelled | Proposed split | Status |
|---|---|---:|---:|---:|---:|---:|---|---|
| match01 | rally01, rally02, rally03, rally04 | 4061 | 3403 | 658 | 0 | 4061 | train | Canonical source unverified |
| match02 | validation01 | 741 | 715 | 26 | 0 | 741 | Historical val; excluded | V2 model selection; linked to match09 |
| match03 | rally05, rally06, rally07, rally08 | 1439 | 1064 | 375 | 0 | 1439 | train | Canonical source unverified |
| match04 | rally09, rally10 | 865 | 655 | 210 | 0 | 865 | train | Canonical source unverified |
| match05 | rally11, rally12, rally13 | 3181 | 2121 | 1060 | 0 | 3181 | train | Canonical source unverified |
| match06 | rally14, rally15, rally16 | 2281 | 2166 | 115 | 0 | 2281 | train | Canonical source unverified |
| match07 | rally17, rally18, rally19, rally20, rally21 | 3223 | 2680 | 543 | 0 | 3223 | train | Canonical source unverified |
| match08 | rally22, rally23, rally24 | 2875 | 1986 | 886 | 3 | 2872 | val | Canonical source unverified |
| match09 | rally25, rally26, rally27, rally28 | 1560 | 1198 | 362 | 0 | 1560 | val | Mixed-source bucket |
| match10 | rally29, rally30, rally31, rally32 | 3302 | 2063 | 1237 | 2 | 3300 | train | Mixed-source bucket |
| match11 | rally33, rally34, rally35, rally36 | 1986 | 1821 | 165 | 0 | 1986 | train | Canonical source unverified |
| match12 | rally37, rally38, rally39, rally40 | 3329 | 2754 | 575 | 0 | 3329 | test | Candidate reserve; verify sources |
| match13 | rally41, rally42, rally43, rally44 | 2122 | 1750 | 368 | 4 | 2118 | train | Canonical source unverified |
| match14 | rally45, rally46, rally47, rally48 | 2217 | 1846 | 371 | 0 | 2217 | test | Candidate reserve; verify sources |

The 48 rallies contain **32,441 total frames: 25,507 visible, 6,925 invisible, nine unknown; 32,432 labelled frames**. Auxiliary `validation01` adds 741 frames (715 visible, 26 invisible, no unknowns). The dataset manifest therefore has 49 rows, distinguished by `scope`; the split manifest has exactly 48 rows.

## Evidence and unresolved match identities

Contact sheets were generated from source frames and manually reviewed for every clip. These are observed text/appearance cues, not externally verified tournament or original-recording identities:

| Declared group | Local visual evidence | Remaining uncertainty |
|---|---|---|
| match01 | Scoreboard TAI T.Y. / CHEN Y.F.; consistent broadcast | Original recording URL/date not verified |
| match03 | Consistent low-angle red Hangzhou court and player pair | Complete canonical match identity not verified |
| match04 | Consistent low-angle red court; different camera/player cues | Canonical identity not verified |
| match05 | Consistent green court and dingding badminton watermark | Canonical identity not verified |
| match06 | Wide low-angle view and Badminton Rush 4K watermark | Canonical identity not verified; group consolidation documented in Git |
| match07 | Tokyo 2020 graphics; scoreboard AXELSEN / CHEN | Original recording identity/time ranges not verified |
| match08 | Scoreboard K. VITIDSARN / KOK JIN HONG | Original recording identity/time ranges not verified |
| match09 | rally25 China Open graphics/green court, rally26 Hangzhou/red court, rally27 Singapore graphics, rally28 close-up | Multiple sources; not one match; some individual identities unresolved |
| match10 | rally29 lower-third Naraoka vs Loh; rally30 Lin Chun Yi vs Christo Popov; rally32 Lakshya Sen vs Victor Lai; rally31 Guangzhou red court | Explicitly different named matches; original recordings/time ranges unresolved |
| match11 | Paris 2024 graphics; scoreboard CHOU / SEN | Original recording/time ranges not verified |
| match12 | TANAKA / SEN scoreboard and SYDNEY branding | Candidate test only; event/date/recording not independently verified |
| match13 | AXELSEN / LEE C.W. scoreboard and Malaysia branding | Original recording not verified; source contains playback controls |
| match14 | Jonatan CHRISTIE / Kean Yew LOH scoreboard and Aichi-Nagoya 2026 banner | Candidate test only; source recording/time ranges not independently verified |
| match02 auxiliary | Hangzhou red court and score 7–1, visually matching rally26 source cues | Exact overlapping segment/time offset unresolved |

No new `source_group` values were assigned. In particular, different existing IDs, different crops or different clip filenames are not proof of source independence. `leakage_block=match02+match09` in the manifest names a constraint linking two existing IDs, not a fabricated real-match identity. All other blocks currently mirror declared IDs and remain subject to source verification.

Rally40 and rally47 include close-up/camera-change footage in inspected samples. No frames were removed or silently reassigned. Review continuity and original time ranges before using them for a formal trajectory benchmark.

## Duplicate and overlapping-content checks

- SHA-256 of all **49 canonical video files**: no identical video files found.
- SHA-256 of all **33,182 canonical prepared JPEGs**: no byte-identical JPEGs across different clips found. Within-clip repeats are not cross-split leakage and were not reported as duplicate rallies.
- **920 perceptual hashes**, sampled at approximately one frame per second after removing preparation padding, were compared across different declared groups.
- The closest perceptual candidate (Hamming distance six) is rally21:60 versus rally43:330. Visual review shows different broadcasts/opponents despite similar court layout. This illustrates why image-hash proximity alone cannot merge matches.
- rally26:60 versus validation01:0 has Hamming distance ten; independent visual inspection of multiple source timestamps shows matching Hangzhou camera/player/7–1 score cues. This is sufficient to reserve a common leakage block, but not to assert byte duplication or an exact time alignment.

Negative byte/hash checks do **not** rule out rescaled, recropped, re-encoded, temporally offset or partially overlapping source footage. The audit did not perform exhaustive decoded-video temporal registration. Original URLs/recording IDs and clip start/end times are still needed, especially for mixed groups 09/10 and candidate test groups 12/14.

Evidence is retained under `outputs/shuttle_dataset_audit_20261110/`: `overview_01.jpg` through `overview_05.jpg`, `validation_rally26_comparison.jpg`, `source_evidence.json`, `content_audit.json`, and `perceptual_fingerprints.json`.

## Historical V2 exposure and overlap audit

Only one training `run.json` was found under `outputs/`: `outputs/shuttle_v2/run.json`. `configs/default.yaml` is empty and provides no shuttle split configuration. The current training CLI derives splits from clip metadata, not from a central manifest.

The associated recovered checkpoint has input 960×544, base channels 16, best epoch 11 and validation loss 0.3880765411. `run.json` plans 40 epochs, batch size 2 and learning rate 0.001. No `history.csv` or `last.pt` is present in the recovered V2 directory, so completion of all planned epochs and other historical runs cannot be reconstructed. Neither checkpoint nor run summary records exact label/video hashes or a canonical match manifest.

Recorded training clips are rally01–16, totalling 11,827 current labelled frames, exactly matching the run count. Recorded validation is `validation01`, 741 labels, also matching. Treat these as the known recorded exposures; exact historical label versions cannot be proved solely from these counts.

### Why the run lists match07 and match08

The run records training groups `match01, match03, match04, match05, match06, match07, match08` and validation group `match02`. Git history resolves the apparent mismatch with today’s groups:

- At commit `d93268f`, rally14 was `match06`, rally15 `match07`, and rally16 `match08`.
- Commit `afa9b140e19204ee679b6090d566d2e9b67aa3fa` (“Group rallies 14 to 16 under the same source match”) changes rally15 and rally16 to `match06`.
- Today’s `match07` / `match08` denote rally17–21 / rally22–24. Their names reuse historical identifiers; the run’s old group strings do not show that these later clips received V2 gradient training.

The recorded train/validation group strings are disjoint, and the current groups derived from the recorded clip lists are also disjoint. No proven initial V2 train/validation content overlap was found in the sampled footage. This is not a proof of independence: original-source provenance is incomplete. The newly identified validation01/rally26 source link concerns later data, and would become a leakage risk if today’s default all-rally training root were used.

### Development versus future testing

V2 predictions and comparative metrics have already been inspected on rally18, rally24, rally29, rally35 and rally44. Their entire current groups (07, 08, 10, 11, 13) must be treated as development-exposed for future checkpoint/post-processing decisions, not promoted to untouched test groups by choosing an unevaluated sibling rally. Earlier evaluation reports remain historical exploratory results. Merely reserving different siblings now does not retroactively make them independent.

No local prediction outputs or training clip records were found for groups 12 or 14. The present audit examines their content only for provenance and does not evaluate model predictions. They are candidate reserves, **not certified unbiased test matches**. If subsequent source verification links them to trained/development footage, merge the relevant blocks and recalculate. If no untouched verified blocks remain, collect new match recordings; do not claim a clean test set from the existing 48.

## Paths and current loader behavior

All manifest paths are current repository-relative paths such as `data/shuttle/videos/rally26.mp4` and `data/shuttle/clips/rally26/labels.csv`. Original metadata is preserved byte-for-byte, including old absolute paths.

Of 49 stored historical video paths, 25 resolve on this machine (including files in the old macOS checkout); the remaining 24 are historical Windows paths. Existing old-checkout copies do not make absolute paths portable and could diverge from the current repository.

`ShuttleDataset` reads labels and prepared frames relative to the supplied clip root and does not open `meta["video"]`. Demo inference opens its explicit video argument; the evaluator also receives an explicit video. Thus stale metadata video paths do not directly break these current code paths while prepared files exist. Consumers that directly open that field, or attempts to recreate data from historical paths, can fail or silently read an obsolete copy. Staged metadata copies use the current relative video paths.

All 49 video-container dimensions, FPS values and frame counts agree with metadata; labels cover exactly the zero-based ranges with unique indices, valid visibility and finite in-bounds visible coordinates. Preparation transforms match the shared transform function. Every canonical numbered JPEG exists, and sampled boundary/middle JPEG dimensions agree with model input. Source videos were not exhaustively decoded as part of this inventory.

One warning was found: `data/shuttle/clips/rally33/frames/000279_broken.jpg` is an unreferenced noncanonical backup. The required `000279.jpg` exists. The backup was preserved and excluded from canonical hashing/staging counts; no labels or frames were repaired automatically.

## Reproducible proposed split

The policy `v2_continuation_group_frame_balance_v1` uses exhaustive enumeration of **whole declared groups**. It minimizes the sum of squared deviations of labelled-frame fractions from 0.70 / 0.15 / 0.15, with a deterministic lexicographic group-assignment tie-break. It does not optimize using model accuracy, and excludes unknown frames from balancing.

Constraints:

- Every recorded V2 training clip remains in training; therefore groups 01, 03, 04, 05 and 06 are forced to training for any V2 continuation.
- Test is allowed only on candidate groups 12/14; groups already used for training, validation/model selection, exploratory model evaluation or linked to those sources are ineligible.
- Group09 is forced to development validation because of its source link to historical group02; the auxiliary validation01 remains historical validation and is not used in the proposed 48-rally loader.
- Whole groups and linked source blocks cannot cross split boundaries. Original per-clip `meta.split` fields are not changed.

| Proposed split | Groups | Rallies | Total frames | Labelled frames | Unknown | Labelled-frame percentage |
|---|---|---:|---:|---:|---:|---:|
| train | match01, match03, match04, match05, match06, match07, match10, match11, match13 | 33 | 22460 | 22454 | 6 | 69.23% |
| val | match08, match09 | 7 | 4435 | 4432 | 3 | 13.67% |
| test | match12, match14 | 8 | 5546 | 5546 | 0 | 17.10% |

The 17.10% candidate-test share is the practical group-level approximation to 15%; moving individual rallies to hit an exact percentage would violate the grouping requirement. The proposed split is a data reservation/development plan, not a statement that cross-source independence has already been proven.

## Safe preparation for the next fine-tuning run

**Do not run training on the default `data/shuttle/clips` root yet.** Its current metadata still assigns all 48 numbered rallies to train and `validation01` to val. Merely creating CSV manifests does not change `train.py`, which does not read them. Its existing group-string overlap check would miss group02/group09 shared-source risk.

A separate staging tree has already been created at:

`outputs/shuttle_dataset_audit_20261110/staged_clips/`

It contains 33 training rallies and seven validation rallies, with metadata copies carrying proposed splits and current repository-relative video paths. Labels and prepared frame directories are relative symlinks to their original read-only-by-convention data; no labels or videos were copied or changed. The eight reserved test rallies and auxiliary validation01 are excluded entirely, preventing accidental loading by the current trainer. Do not use the staging links to edit annotations.

Data-loader-only validation loaded **22,454 training labels and 4,432 validation labels**, with disjoint group sets and all unknowns excluded. No model, optimizer or training epoch was run.

This staging layout is mechanically group-disjoint, but cannot guarantee unknown cross-source relationships are absent. Before the next fine-tune, verify the constituent recordings in mixed groups 09/10 against both training and reserved test footage. Resolve discovered links by joining whole declared groups and recomputing the split; do not move individual rallies to hide overlaps. Until then, this is a development-only split proposal.

For a future V2 fine-tune, initialize from the unchanged V2 checkpoint, use this manifest/staging layout, record input hashes and canonical source evidence, and keep test groups inaccessible until source verification and final model/post-processing selection are frozen. The current `train.py` always initializes a new ShuttleNet and has no checkpoint-resume/fine-tune argument: using the staging root alone would start training from scratch, not fine-tune V2. Checkpoint initialization support must be added explicitly in a separate training task. No training command was executed or scheduled here.

## Manifests, reproduction and validation

- `data/shuttle/dataset_manifest.csv`: 49 records (48 numbered rallies plus auxiliary validation01), with current paths, metadata group/block, frame/visibility counts, content hashes, exposure flags, source evidence and unresolved eligibility status.
- `data/shuttle/split_manifest.csv`: 48 numbered rallies, one proposed split each, labelled/unknown counts, paths, leakage block, policy and `test_independence_verified=0`. The `test` value means a reserved candidate holdout; it is not a certification.
- `.gitignore` now permits these two CSVs so they can later be versioned. No git staging, commit or push was performed.

Audit command used:

```bash
.venv/bin/python -m src.shuttle.dataset_audit --output outputs/shuttle_dataset_audit_20261110 --evidence outputs/shuttle_dataset_audit_20261110/source_evidence.json --stage
.venv/bin/python -m src.shuttle.dataset_audit --validate-manifests
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

The read-only `--validate-manifests` command rechecks input hashes, counts, relative paths, exposure constraints and the deterministic split without replacing files. The CSVs/output audit now exist; generation refuses replacement. For a new independent audit, select fresh output locations and deliberately review any requested manifest replacement instead of deleting existing results automatically. Visual evidence was stored separately in `source_evidence.json`; the code does not infer identities from group numbering or image hashes.

Checks passed: expected 48-rally membership/group mapping; all label/frame ranges and visibility totals; all video metadata/transform checks; all canonical frame-file presence; video/JPEG content hashing; complete one-split-per-rally coverage; no source-group or leakage-block crossing; no recorded V2 training clip outside train; no development-exposed group in candidate test; staged loader counts; and before/after hashes for all 49 metadata files, label files, canonical videos and the V2 checkpoint.

All 27 tests pass, including five new group-split tests. Original metadata, labels, videos and checkpoint hashes are unchanged. The extra broken-image backup remains untouched. No training, inference, parameter tuning, commit or push occurred in this task.
