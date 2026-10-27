# lab_scripts — Kennedy lab tools for the sDANNCE / rat social-behavior data

Scripts and notes from the Kennedy lab's use of the labeler on sDANNCE rat social-behavior data. The
labeler changes listed under "What we changed in the labeler itself" are in `src/`.

## Scripts

| script | what it does |
|---|---|
| `dannce_to_slp.py` | Converts one camera view of an sDANNCE `save_data_AVG0.mat` (3D poses, mm) into a per-camera 2D `.slp` the labeler can load next to that camera's video, using the cage's `calibration.json`. It reads `sDANNCE/predict02` when that folder exists, else `predict01`. Some sessions have both, and there `predict01` stops at frame 60000. |
| `import_social_window.py` | Sets a clip's *social window* (frames where the second rat is in the arena) from the QC spreadsheet, through a running labeler's HTTP API. |
| `review_stats.py` | Precision of a head's proposals from your accept and reject decisions in review. |
| `characterize_approach_head.py` | ROC, precision-recall and review precision over time for every video in a project, scored against that project's own labels. |
| `compare_to_handlabels.py` | Scores a held-out video's predictions against hand labels kept in a different project. Suggests a slider threshold and keeps a scoreboard across fine-tunes in `<project>/analysis/`. Add `--full-coverage` when every real bout was painted. |
| `hidra_patches/` | Changes to HiDRA that the labeler needs. Right now only `reviewed-frames-only.patch` (see below). |

## The pixel scale (px/cm) for HiDRA is measured automatically

HiDRA works in centimetres, so it needs each camera's px/cm. `dannce_to_slp.py` measures it while
converting (true 3D size in mm vs. projected size in pixels, over long head/trunk segments of the whole
session) and writes `<poses>.slp.scale.json` next to the `.slp`. **Keep the two files together** when you
copy poses between machines. The labeler then fills the px/cm box by itself:

- a value you type always wins; clear the box to go back to the automatic one;
- a video that is a uniformly resized copy gets the scale multiplied by the same factor;
- a video of a different shape (e.g. cropped) is refused with the reason shown, never guessed;
- no sidecar → nothing is guessed, and HiDRA still asks for the number.

To measure it for poses you already converted: `python dannce_to_slp.py <cage_dir> --camera CameraN --out
<poses>.slp --scale-only`. It is one number per camera, so it is approximate: the sidecar records the
spread (`iqr`), which is wide because a tilted camera sees the arena at different scales near and far.

## What we changed in the labeler itself

- **Social window** (per clip `social_start`/`social_end`): features, training, review candidates and label
  writes are restricted to the frames where a second animal is actually present; the GUI dims the rest.
- Adding a video with no poses (labeling-only mode) no longer fails.
- `HIDRA_GPU=<n>` picks the GPU for HiDRA jobs. Set it when you start the labeler.
- **Train on** (menu next to the HiDRA review rate): every frame, inside the social window, or
  reviewed frames only. See below.
- **Hold this clip out of training** (checkbox next to ⚙ Train). Use it for test videos. The clip keeps
  its labels and can still be predicted and reviewed. Every training run skips it, HiDRA fine-tune and
  the project's own Train alike. Its tab in the clip bar says "held out".
- **Every fine-tune is kept.** A finished run goes to `<project>/hidra/runs/<date>_<time>__<lab>__<action>/`
  with its checkpoints, the exact spans sent to HiDRA (`bouts.csv`, `reviewed.csv`), HiDRA's full log
  (`hidra_log.txt`) and `manifest.json`. The manifest lists every clip the run trained on and every clip
  it skipped, with the reason. Predict uses the newest run. To go back to an older one, point the
  behavior's `weights` in `project.json` at that run's `models/` (stop the labeler first).

## Choosing which frames a fine-tune learns from

The "train on" menu has three choices. The run's `manifest.json` records which one was used.

- **every frame** (the old behavior): see the next paragraph.
- **inside social window**: the same, but only frames inside each clip's social window. Time when
  only one rat is in the arena is left out.
- **reviewed frames only**: only the frames you answered yes or no.

By default HiDRA counts every frame you did not mark as an approach as a "no". That includes real
approaches the model never showed you. It also includes the time before and after the social window. With
**reviewed frames only** chosen, the next fine-tune learns from the frames you answered and nothing
else. Accepted candidates are the yeses. Rejected candidates are the nos. Frames marked Unknown and
frames you never looked at are left out. A clip whose candidates you all rejected still counts, since
its nos are real information.

The last two need a small change to HiDRA that the labeler checks for. It is in `hidra_patches/`. To put it
on a fresh HiDRA checkout:

```bash
cd ~/HiDRA
git checkout -b reviewed-frames-only
git apply ~/code/AndysAnnotator/lab_scripts/hidra_patches/reviewed-frames-only.patch
.venv/bin/python -m pytest -q tests/test_reviewed_frames.py     # 9 tests should pass
```

Without the patch, those two choices stop with a message that says so. With "every frame" HiDRA
behaves exactly as before. The run's `hidra_log.txt` lists the number of reviewed
frames for each clip. Its `[REVIEWED]` lines give the total each model trained on.

Things to keep in mind:

- The nos you gave were mostly rejected candidates, so they are hard cases. The model sees no easy
  "rats far apart" frames this way. Compare it on the held-out video before trusting it.
- HiDRA keeps about 15% of the staged clips out of each of its 5 models for validation (2 of 9 clips
  per model in our runs, a different pair for each). Each `[LABTAIL]` line in `hidra_log.txt` says which.
  This happens in both modes.

## Checking that a clip was never trained on

1. In the GUI the clip's tab says "held out". When the clip is open, "hold this clip out of training" is ticked.
2. Look in `<project>/labels/`. A clip with no `<clip>.parquet` there has no labels in this project, so
   there is nothing of it to train on.
3. Open the run folder in `<project>/hidra/runs/`. In `manifest.json` the clip must be under
   `held_out_clips` (or `skipped_clips`) and not under `trained_clips`. `bouts.csv` and `reviewed.csv`
   name every file sent to HiDRA. Clip `0` would appear as `0.parquet` and clip `0-11` as `0-11_0.parquet`.

   ```bash
   cd ~/my-projects/approach_defs/hidra/runs/<run>
   grep -c "^0\.parquet\|^0-11_0\.parquet" bouts.csv reviewed.csv      # both should print 0
   ```
4. The server event log `<project>/events/server.jsonl` has a `job_done` line for every train job. Since
   2026-10-23 it lists the `clips` used and the `held_out` ones.

Every fine-tune starts from HiDRA's published weights, never from an earlier fine-tune. So a run can
only learn from the clips in its own manifest.

## Browsing the model's detections (read only)

**👁 Browse detections** (step ④, next to Review candidates) plays the model's bouts on the open clip
one at a time. It is for getting a feel for the model on a held-out clip. You cannot accept, reject or
trim anything.

1. Open the clip, pick the behavior and the actor's track (track 2 for approach). Run Predict first if
   the clip has no predictions.
2. Pick what to step through.
   - all detections
   - detections on your labels (they overlap a frame you labeled Happening)
   - detections with no label (they overlap no Happening frame)
   - your bouts it missed (Happening bouts no detection touches)
3. Pick the order: time, highest score, lowest score or random. Random gives a fair sample if you only
   watch a few.
4. Pick the labels to compare with. Clip 0's hand labels live in the separate `cage1rb_day1` project, so
   choose "labels: cage1RB_day1 · approach" there. Another project is offered only if its clip has the
   same name, length, frame size, fps and social window. Its labels are read and never changed.
5. Keys: ⏎ next, Shift+⏎ back, r replay, ← → scrub, space play, Esc done.

Bouts are found the same way as the timeline lane. That means the detect slider, min bout, 5-frame
smoothing and a 3-frame gap bridge, all inside the social window. Moving the slider rebuilds the list.

While browsing, the GUI refuses every write. Painting, undo, delete, trim, Train, Predict, Review and
Import show "read only" instead. The page also blocks any other write request to the server except the
event log and the detect slider. Browsing logs only `browse_*` events and never `candidate_accept` or
`candidate_reject`, so the review-precision numbers in the metrics notebook are unaffected.

## Bugs we found and fixed

**Social windows started up to 19 s too early.** Fixed 2026-10-23.
The QC sheet `asd_session_qc_social` writes entry and exit times as minutes:seconds in a number cell,
so 9.41 means 9 min 41 s. They were read as decimal minutes (9.41 min is 9 min 24.6 s), and
`Sheet1` has no frame numbers for these sessions. So most windows began before the partner entered.
In that time the tracker puts the missing rat's pose on top of the tethered rat, which looks like
contact, and those frames were offered as candidates. `import_social_window.py` now reads only the
main sheet as minutes:seconds. Its frames match the `partner_present` record in ugne_clusters exactly.

**Day 2 to 5 poses stopped at frame 60000.** Fixed 2026-10-23.
`dannce_to_slp.py` always read `sDANNCE/predict01`. On the day 2 to 5 sessions that file only covers
the first 60000 frames, while `predict02` holds the same poses for the whole session. The partner is in
the arena until about frame 79000, so a third of the social time had no poses. HiDRA still scored those
empty frames, and all-frames fine-tunes counted them as "no". The script now prefers `predict02`.

**A failed fine-tune deleted the model in use.** Fixed 2026-10-23.
Each fine-tune first emptied `hidra/_finetune/<lab>__<action>/`. The current checkpoint lived in that
same folder. So a run that crashed partway left Predict pointing at files that were gone. Finished
runs now live in their own folders under `hidra/runs/` and the next run cannot touch them.

**Changing the review rate undid the fine-tune.** Fixed 2026-10-20.
Editing the "review rate" box or the "scores" menu saved a new copy of the head's settings. That copy
left out the link to the fine-tuned model. Predict then went back to the stock model without saying
so. The next re-tune also ran in the slow tail mode. Now the link is kept as long as you stay on the
same head. Picking a different head still starts fresh, because the old fine-tune belongs to the old head.

**Fine-tuning on several videos only used one of them.** Fixed 2026-10-09.
Every camera video in this dataset is named `0.mp4`. Fine-tuning named each video's exported poses
after that file name, so each video overwrote the one before it. A fine-tune that seemed to use five
videos really used only the last one. Exports are now named by the clip's ID instead. Redo any
analysis that came from a fine-tune before this date. To check a fine-tune, open
`<project>/hidra/_finetune/<lab>__<action>/tracking/`. It should hold one file per video.

**HiDRA ignored the GPU we picked.** Fixed 2026-10-06.
HiDRA always ran on GPU 0, even when the labeler was started with `CUDA_VISIBLE_DEVICES`. On a shared
machine that GPU is often full and the job crashed out of memory. Use `HIDRA_GPU` instead (see above).

**Joint names came out as `kp00` to `kp22`.** Fixed 2026-10-06.
`dannce_to_slp.py` now writes the real names (nose, ear_left and so on). Without them the labeler
dropped about 20 features unless you renamed the nodes by hand.

**The trim handle sat one frame past the playhead.** Fixed 2026-10-05.
When you dragged the end of a candidate, its handle was drawn one frame to the right of the playhead.
Only the drawing was off. The saved labels were always right.

## Data

Videos, poses and projects are **not** in this repo. They live on the lab server
(`smb://labdata.scripps.edu/kennedy/...`) and in each user's `~/my-projects`.
