# laras-labeler

A local, pose-based, human-in-the-loop behavior classifier — a lightweight, modern
[JAABA](https://jaaba.sourceforge.net/). Annotate positive/negative **frames** across videos, derive
features from multi-animal poses (`sleap-io` + `movement`), train a `scikit-learn` classifier, and
get per-frame predictions back in the annotation UI to guide the next labels.

**See [`PLAN.md`](PLAN.md) for the full design** (data contracts, feature/ML pipeline, API, on-disk
format, milestones).

## Run it

Needs [uv](https://docs.astral.sh/uv/); uv fetches its own Python (3.12+), so you need no system
Python, virtualenv or pip. The most reliable path is one command to install, then a normal command
to run:

```bash
uv tool install git+https://github.com/talmolab/laras-labeler
laras-labeler ~/my-projects
```

`~/my-projects` is a directory of on-disk projects, created on first run. It starts empty: create a
project in the UI, then add clips by uploading a video + `.slp`, or by giving server-side paths
(`POST /api/projects/{pid}/videos` with `video_path` / `slp_path`).

Without installing anything at all:

```bash
uvx --from git+https://github.com/talmolab/laras-labeler laras-labeler ~/my-projects
```

That resolves into a throwaway environment each time. **On macOS this form can fail** with
`realpath: command not found`: uv runs the console script straight out of its cache via a shim that
needs `realpath`, which older macOS does not ship. It is not specific to this project (any `uvx`
package fails the same way). Either use `uv tool install` above, or bypass the shim:

```bash
uvx --from git+https://github.com/talmolab/laras-labeler python -m laras_labeler.cli ~/my-projects
```

Or as an editable checkout, to work on the code:

```bash
git clone https://github.com/talmolab/laras-labeler && cd laras-labeler
uv sync && uv run laras-labeler ~/my-projects
```

On Intel macOS the dependency pins in `pyproject.toml` matter — see the note there; without them
several of `movement`'s transitive dependencies build from source and fail. Intel macOS also builds
`bottleneck` from source, which needs the Xcode command-line tools; Apple Silicon and Linux get
wheels for everything.

Verify the core pipeline headlessly (needs a two-mouse SLEAP file with a `nose` node — the
`mice.tracked.slp` sample lives in [`slp-viewer/`](https://github.com/talmolab/vibes/tree/main/slp-viewer)
in the vibes repo):

```bash
uv run scripts/verify_pipeline.py path/to/mice.tracked.slp
```

## HiDRA classifiers (`hidra-in-the-loop`)

[HiDRA](https://github.com/talmolab/HiDRA) ships 82 pretrained behaviour classifiers. A behavior in
this labeler can be bound to one of them, and the two buttons the GUI already has then mean
something different for that behavior: **▶ Predict** runs the head over the clip, **⚙ Train**
fine-tunes it (LABTAIL) on the bouts you accepted in review. Everything between is unchanged — a
HiDRA lane and a lane from the project's own model are the same artifact by the time the timeline
and the review queue see them.

```bash
uv tool install git+https://github.com/talmolab/laras-labeler
hidra-in-the-loop ~/my-projects
```

`hidra-in-the-loop` is the same app as `laras-labeler` — one codebase, one install, not a fork. It
differs only in printing the HiDRA runtime at startup, so a missing checkout is named on the
terminal as well as in the GUI. Use either command.

It is **feature-detected**: with no HiDRA checkout the labeler is unchanged and trains its own model
on pose features, which needs no setup. When a checkout is missing the GUI shows a setup box saying
what is missing, with fields for the two paths:

- **checkout** — the directory holding HiDRA's `predict.py`
- **interpreter with JAX** — HiDRA is invoked as a subprocess in its own interpreter, never
  imported, so this labeler never needs JAX itself

Those are saved to `<your projects folder>/settings.json`, so they survive a restart and travel with
the projects folder. `HIDRA_HOME` / `HIDRA_PYTHON` still work and apply when nothing is set in the
GUI. `GET /api/hidra/status` reports which source is in effect and, when inference cannot run, why.

## Arena outlines from the HCM database (Salk)

When a clip opens, its cage, spout and feeder outlines are filled from the HCM database. The labeler
uses the clip's own recording and falls back to the camera's typical outline (never for the feeder).
This needs two things:

- the Salk network (on site, the VPN, or Tailscale)
- `HCM_DATABASE_URL`, set in the terminal before you start the labeler. Use the URL from the
  `DATABASE_URL` line of the lab's db-query skill (talmolab/pheno-claude-plugins,
  `plugins/pheno/skills/db-query/SKILL.md`). Either driver name in it works. Never commit it here:

  ```bash
  export HCM_DATABASE_URL='postgresql://…'
  uv run hidra-in-the-loop
  ```

Clips from other experiments (not cut from an HCM recording) have no database to ask: the video
toolbar shows **◎ Set spout**, **◇ Set spout ROI** and **▢ Set cage ROI** to draw them by hand, with
a ✕ to clear. The default objects (feeder top / side, cage wall, spout) are HCM's arena. Remove the
ones another experiment doesn't have with ✕ in the Objects card. Adding one back by name restores it.

When an HCM clip's outlines can't be loaded, the video toolbar shows **⚠ outlines not loaded from HCM**. Hover over it
to see why. `uv run python scripts/check_hcm_roi.py --clip <part of the clip name>` walks the lookup
one step at a time for that clip.

## Pose data for tailcyclenet

[tailcyclenet](https://github.com/AI-HHMI/tailcyclenet) is a video pose estimator, with a box
detector and a label-free track-quality scorer. `scripts/slp_to_tailcycle.py` converts SLEAP `.slp`
files into the dataset format it trains on. It is separate from the labeler: run it in the
environment tailcyclenet is installed in.

```bash
# human-labelled frames (a SLEAP training project) -> a pose / detector training set
python scripts/slp_to_tailcycle.py labels.v001.slp --out ~/tailcycle/hcm-mice \
    --replace /old/video/prefix/ /where/the/videos/are/

# SLEAP tracking over whole videos -> a "tracked" set, e.g. for `tailcyclenet score`
python scripts/slp_to_tailcycle.py *.predictions.slp --out ~/tailcycle/hcm-tracked --tracked
```

It writes one 2D session per video, takes the keypoint names, edges and left/right symmetries from
the SLEAP skeleton, and runs tailcyclenet's own validator at the end. The script's docstring says
how occlusion, identity and unlabelled animals are carried over. Training then runs on a GPU
(`tailcyclenet train --data ~/tailcycle/hcm-mice --out runs/hcm-pose`; about a day on one L4).
`tests/test_slp_to_tailcycle.py` checks the converter and skips when tailcyclenet isn't installed.

## Status

- **v0a foundation — DONE & verified.** `uv` package scaffold; core `SLP → Labels.numpy() → movement
  clean → kinematics/pairwise` pipeline verified on the mice sample; FastAPI backend serving
  **authoritative frames + pose blob**; minimal browser viewer (frame + skeleton overlay, scrub,
  play, arrow-key step).
- **Next:** labeling (paint pos/neg ranges → per-frame parquet store, ethogram timeline, undo/redo)
  → on-disk project layer (PLAN §9) → feature cache job (PLAN §4) → **v0b** train/predict loop +
  prediction heatstrip.

## Taking stock of the classifier library

Across many projects it gets hard to say which behaviors run on a HiDRA head and which on the
project's own model, how many bouts each has and where they came from, and how they score.
`library_inventory.py` reads a projects folder (read-only) and writes one row per project ×
behavior. It is a standalone `uv` script, so it needs no checkout or labeler install:

```bash
uv run https://raw.githubusercontent.com/talmolab/laras-labeler/main/scripts/library_inventory.py ~/laras-projects --md library.md --csv library.csv
```

- **Predicted by** — HiDRA lab/action/collapse, zero-shot or fine-tuned and how many fine-tune runs;
  or the native random forest and its feature set.
- **+ bouts** — split into painted by hand / accepted in review / imported; plus − bouts, and
  coverage over clips and (clip, animal) lanes.
- **Native CV AP** — the random forest's grouped-CV score on its own labels.
- **Test clip** — every stored model's AP/AUROC on the project's held-out clip: zero-shot,
  each fine-tune, each native version. A model that trained on the clip is marked in-sample.
- **Suggested start** — a guess from the name: *head first* (partner-directed), *bake-off*
  (a solo action HiDRA also has), *native + landmarks* (defined by a place in the cage), or
  *native*.

### Within-mouse vs within-cage variation

`bout_kinematics_cv.py` averages the feature cache's per-frame kinematics over each bout (speed,
turning, body length and bend, posture extents, plus duration). For each behavior and kinematic, it
then compares the CV across one mouse's bouts with the CV across all bouts in its cage. It also
reports the share of within-cage variance that is explained by which mouse did the bout, with a
permutation p.

```bash
uv run scripts/bout_kinematics_cv.py ~/laras-projects/hidralabel --out-bouts bouts.csv --csv cv.csv
```

A "mouse" is a (clip, track): tracks are not linked across clips. Add `--baseline` to divide each
bout's posture by the same mouse's median posture outside the behavior. That cancels differences in
build (or in how one animal's keypoints are placed), leaving how the behavior itself changes each mouse.

`plot_mouse_vs_cage.py` draws it: for each behavior, the bout-to-bout, mouse-to-mouse and
cage-to-cage CV of every kinematic against what bout noise alone would give, plus the bouts
themselves grouped cage → mouse.

```bash
uv run scripts/plot_mouse_vs_cage.py bouts_rel.csv --baseline --out variation.png
```

`mouse_signature.py` asks whether each mouse has a behavioral signature, within each clip. It
tests **what** a mouse does (bouts per mouse vs equal shares; fair only with equal labeling coverage)
and **how** it does it: can a classifier tell which mouse made a bout from its kinematics, can it
still do so when trained on earlier bouts and tested on later ones, and does a mouse's style in one
behavior identify it in another?

```bash
uv run scripts/mouse_signature.py bouts_rel.csv --baseline
```

`behavior_places.py` maps where in the cage each behavior happens: every bout's nose position
and path on the clip's mouse-free background, one panel per clip × behavior, with nearby bouts
circled as places (A, B, …) and their distance to the water and wall — a first count of how many
location-defined sub-behaviors each behavior has.

```bash
uv run scripts/behavior_places.py ~/laras-projects/hidralabel --out places.png
```

## Measuring annotation time (human-in-the-loop vs. by hand)

Every labeling session writes an append-only event log to `<project>/events/*.jsonl`: each painted
bout, each candidate accepted or rejected, each Train and Predict, timestamped, plus a heartbeat that
makes it possible to tell working time from a coffee break. The server appends job durations and
label writes itself, so compute time and every label that reached disk survive a closed browser tab.
See [`PLAN.md` §9.1](PLAN.md) for the event list and the exact definition of "active time".

The rollup groups the log into **rounds** — the stretch of work between two Trains of a behavior —
and separates the human's time into *labeling by hand* and *reviewing model proposals*:

```bash
python scripts/annotation_timing.py ~/laras-projects --pid my-project --csv rounds.csv
```

```
  # behavior         started              work   label  review   wait  other    cpu  hand  shown   ok   no  s/bout  s/dec     AP
  1 drinking         01-15 08:00          5:44    5:44    0:00   0:17   0:28   0:17    11      0    0    0    31.3      -  0.441
  2 drinking         01-15 08:07          3:26    0:22    3:04   0:15   0:22   0:15     1     14   11    3    22.0    13.1  0.812

annotation   9:10   (labeling 6:06 | reviewing 3:04)
other time   0:50 setting up / navigating / looking  +  0:32 waiting on jobs   =  10:32 at the machine in total

BY HAND    12 bouts / 900 frames (30s of video) in 6:06  ->  30.5s per bout
IN REVIEW  14 decisions (11 accepted) on 41s of proposed video in 3:04  ->  13.1s per decision, 16.7s per accepted bout
==> a bout cost 1.83x LESS human time through review (30.5s by hand vs 16.7s accepted)
```

It also prints accuracy against **cumulative minutes of annotation** — the axis that actually answers
"how long to a usable model", where the Stats panel's learning curve plots accuracy per *bout*. In the app,
**⤓ rounds** (top right) downloads the same per-round table; `GET /api/projects/{pid}/timing` returns
the full rollup as JSON.

Time that is neither labeling nor reviewing — opening the project, picking a clip, reading Stats,
looking at what a Train just produced — is reported as `other_s` and is a denominator for nothing.
It used to fall into `label_s` by default, which inflated the by-hand arm and biased the headline
toward the workflow. The accuracy curve is plotted against `work_s` (labeling + reviewing), not
against time at the machine, so waiting on a slow Train does not read as annotation effort.

Both arms are priced the same way: **all the seconds the arm consumed, over the positive bouts it
produced.** Painting *Not-happening* and rejecting a candidate are real work and are charged, but
neither produces a bout, so neither goes in a denominator (negatives are reported beside the count,
never folded into it). Review also reports how much *fixing* the proposals needed — the share of
decisions that ended with the model's bounds edited, plus replays — because a model whose bounds
always need trimming costs an edit, not just a decision, and that is invisible in dwell time alone.

Caveat worth repeating in any writeup: review only ever visits bouts the model already proposed, so
part of why it is fast is that the *search* was done for you. That is the point of the workflow, but
it makes "seconds per bout" a cost ratio, not an accuracy claim — read it next to the accuracy curve.

### Verifying it on a machine with no data

The loop can be driven end to end without any real recordings — useful because the numbers above are
only as good as the events behind them:

```bash
uv run python scripts/make_synthetic_clip.py /tmp/synth     # video + .slp + ground truth
uv pip install playwright                                   # a browser to drive
uv run python scripts/verify_event_log.py /tmp/synth
```

`verify_event_log.py` starts a server, drives the real UI in Chromium (paint bouts by hand, Train,
review the model's candidates, Train again — with a break in the middle), keeps its own independent
ledger of every action and when, and then checks the rollup against it: counts, per-phase
attribution, round boundaries, dwell times, the break not being billed as work, and each surface
(`/timing`, `/timing.csv`, `/events.jsonl`, the **⤓ rounds** button, `annotation_timing.py`).

## History

This started as a subdirectory of [talmolab/vibes](https://github.com/talmolab/vibes)
([#69](https://github.com/talmolab/vibes/pull/69),
[#72](https://github.com/talmolab/vibes/pull/72),
[#73](https://github.com/talmolab/vibes/pull/73)) and was extracted into its own repository on
2026-09-04 with history preserved. It is a Python-backed local app rather than a client-side-only
vibe, so it does not belong on vibes.tlab.sh. Mentions of `../event-annotator/`, `../slp-viewer/`
etc. in `PLAN.md` refer to sibling directories in that repo.
