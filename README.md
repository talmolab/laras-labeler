# laras-labeler

A local, pose-based, human-in-the-loop behavior classifier — a lightweight, modern
[JAABA](https://jaaba.sourceforge.net/). Annotate positive/negative **frames** across videos, derive
features from multi-animal poses (`sleap-io` + `movement`), train a `scikit-learn` classifier, and
get per-frame predictions back in the annotation UI to guide the next labels.

**See [`PLAN.md`](PLAN.md) for the full design** (data contracts, feature/ML pipeline, API, on-disk
format, milestones).

## Run it

No clone, no virtualenv, no install step:

```bash
uvx --from git+https://github.com/talmolab/laras-labeler laras-labeler ~/my-projects
```

That fetches the code, resolves dependencies into a throwaway environment, starts the server and
opens a browser. `~/my-projects` is a directory of on-disk projects; it is created on first run.

```bash
uvx --from git+https://github.com/talmolab/laras-labeler laras-labeler --help
```

To keep it around instead of re-resolving each time:

```bash
uv tool install git+https://github.com/talmolab/laras-labeler
laras-labeler ~/my-projects
```

Or as a normal editable checkout:

```bash
git clone https://github.com/talmolab/laras-labeler && cd laras-labeler
uv sync && uv run laras-labeler ~/my-projects
```

Needs Python 3.12+ (`movement` 0.17 does not support 3.11). On Intel macOS the dependency pins
in `pyproject.toml` matter — see the note there; without them several of `movement`'s
transitive dependencies build from source and fail.


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

## Status

- **v0a foundation — DONE & verified.** `uv` package scaffold; core `SLP → Labels.numpy() → movement
  clean → kinematics/pairwise` pipeline verified on the mice sample; FastAPI backend serving
  **authoritative frames + pose blob**; minimal browser viewer (frame + skeleton overlay, scrub,
  play, arrow-key step).
- **Next:** labeling (paint pos/neg ranges → per-frame parquet store, ethogram timeline, undo/redo)
  → on-disk project layer (PLAN §9) → feature cache job (PLAN §4) → **v0b** train/predict loop +
  prediction heatstrip.

## Run (dev)

The heavy part is the `movement` install; on Intel macOS it needs the viz-free recipe (PLAN §1):

```bash
git clone https://github.com/talmolab/laras-labeler.git
cd laras-labeler
uv venv --python 3.12 .venv
uv pip install --python .venv sleap-io scikit-learn fastapi "uvicorn[standard]" python-multipart \
    pandas pyarrow joblib pydantic numpy scipy xarray
uv pip install --python .venv movement==0.17.0 --no-deps
uv pip install --python .venv attrs pooch tqdm shapely PyYAML loguru orjson bottleneck
uv pip install --python .venv -e . --no-deps

# launch — pass a folder for your projects (created on first run); it prints the URL and opens the browser
.venv/bin/laras-labeler ~/laras-projects
```

Verify the core pipeline headlessly:

```bash
.venv/bin/python scripts/verify_pipeline.py
```

The app starts with no projects. Create one in the UI, then add clips either by uploading a
video + `.slp` or by giving server-side paths (`POST /api/projects/{pid}/videos` with
`video_path` / `slp_path`). A small two-mouse SLEAP sample (`mice.tracked.slp` + `mice.mp4`)
lives in [`slp-viewer/`](https://github.com/talmolab/vibes/tree/main/slp-viewer) in the vibes
repo; `verify_pipeline.py` takes its path as the first argument (or via `LARAS_SAMPLE_SLP`).

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
