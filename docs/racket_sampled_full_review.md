# Full racket points and player boxes review

Close any older annotation window with Q. From RallyMirror:

```powershell
.\.venv\Scripts\python.exe -m src.racket.ownership_fast label
```

The title must say **FULL FRAME REVIEW**. This is one queue containing
**all 1,000 sampled frames in the original order**: the six earlier cases and
994 new sampled frames. No body-only filtering or priority reordering hides
the original first frame. Previously saved annotations are loaded for editing.

## Controls

- **1:** select the NEAR player's racket point.
- **2:** select the FAR player's racket point.
- **Click:** place or replace the selected racket centre.
- **H in racket mode:** mark that racket unlocatable.
- After a NEAR racket click or H, FAR racket is selected automatically on the
  same frame. A FAR racket click or H validates, saves and automatically moves
  to the next unconfirmed frame. **1/2** still selects either player manually;
  body edits do not switch players.
- **B:** draw the selected player's BODY using two opposite corner clicks.
- **P**, or **1/2:** return to racket-point editing.
- **Tab:** cycle the selected player's body proposals.
- **Space / Enter:** validate/save the whole frame and advance.
- **A / D:** previous / next frame, including earlier completed frames.
- **C:** clear the currently selected point or body, according to the NOW prompt.
- **U / Backspace:** undo the current frame's last edit.
- **H in body mode:** mark that body unlocatable.
- **X:** exclude an ambiguous sample without inventing labels.
- **Wheel:** zoom; **R:** reset; **G:** show/hide grey proposals.
- **Q / Escape:** save/quit. The same command resumes.

Check and correct body boxes **before** labelling the rackets. Every edit
autosaves a draft. The FAR racket click/H checks both racket states and body
associations before advancing. Space runs the same checks and accepts a frame
whose existing labels already look correct. Incomplete or invalid frames stay
open; fix them and press Space or edit the FAR racket to retry.
The status line shows **DRAFT - NOT CONFIRMED** until confirmation succeeds, then
automatically opens the next unconfirmed frame and shows
**LAST SAVE: CONFIRMED [rally and frame]**. Returning with A/D shows
**CONFIRMED** on saved frames. Editing them changes the status back to DRAFT
until you confirm again. Invalid/incomplete records stay on screen with
an explanation; they are never labelled confirmed.
You can return with A, select 1/2, and replace either point immediately; no
separate unlocking command is needed.

Example with correct body boxes: click NEAR centre, click FAR centre: auto-next.
Use H instead of either click when that racket cannot be located.
To correct boxes, **1**, **B**, draw NEAR body; **2**, **B**, draw FAR body;
then **1**, click NEAR racket, click FAR racket: auto-next.

To start at **queue item 35/1000** (rally04, source frame 1317):

```powershell
.\.venv\Scripts\python.exe -m src.racket.ownership_fast label --id rally04_01317
```

## Returning to the first frames

The original manifest first frame is **rally34 frame186**:

```powershell
.\.venv\Scripts\python.exe -m src.racket.ownership_fast label --id rally34_00186
```

The first new sampled frame is **rally01 frame0**:

```powershell
.\.venv\Scripts\python.exe -m src.racket.ownership_fast label --id rally01_00000
```

Without --id, the command resumes the first unfinished frame of this full
review pass. A/D still reaches every frame; completed frames are not filtered out.

## Preservation

Corrections go to `data/racket_ownership_sampled_v006/labels/`.
Original single-point labels, the earlier ownership datasets, video files and
Runpod results are unchanged. Even the earlier referenced points can now be
corrected in this new annotation version: their original coordinates and
reference signatures are retained in per-point correction provenance.

The full review has its own `full_review_progress.json`; earlier box-only or
six-frame progress does not hide frames from it. Training/validation splits stay
unchanged, and corrected validation targets remain a versioned benchmark.
