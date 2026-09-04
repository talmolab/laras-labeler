# laras-labeler

A local, pose-based, human-in-the-loop behavior classifier — a lightweight, modern
[JAABA](https://jaaba.sourceforge.net/). Annotate positive/negative **frames** across videos, derive
features from multi-animal poses (`sleap-io` + `movement`), train a `scikit-learn` classifier, and
get per-frame predictions back in the annotation UI to guide the next labels.

**See [`PLAN.md`](PLAN.md) for the full design** (data contracts, feature/ML pipeline, API, on-disk
format, milestones).

## Run it

Needs Python 3.12+ and [uv](https://docs.astral.sh/uv/). The most reliable path — one command to
install, then a normal command to run:

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

That resolves into a throwaway environment each time (~7 s warm). **On macOS this form can fail**
with `realpath: command not found` — uv runs the console script straight out of its archive cache
via a shim that needs `realpath`, which older macOS does not ship. It is not specific to this
project (any `uvx` package fails the same way). Either use `uv tool install` above, or bypass the
shim:

```bash
uvx --from git+https://github.com/talmolab/laras-labeler python -m laras_labeler.cli ~/my-projects
```

Or as an editable checkout, to hack on it:

```bash
git clone https://github.com/talmolab/laras-labeler && cd laras-labeler
uv sync && uv run laras-labeler ~/my-projects
```

uv fetches its own Python, so none of these need a system Python, a virtualenv, or pip. On Intel
macOS the dependency pins in `pyproject.toml` matter — see the note there; without them several of
`movement`'s transitive dependencies build from source and fail. Intel macOS also builds
`bottleneck` from source, which needs the Xcode command-line tools; Apple Silicon and Linux get
wheels for everything.

Verify the core pipeline headlessly (needs a two-mouse SLEAP file with a `nose` node — the
`mice.tracked.slp` sample lives in [`slp-viewer/`](https://github.com/talmolab/vibes/tree/main/slp-viewer)
in the vibes repo):

```bash
uv run scripts/verify_pipeline.py path/to/mice.tracked.slp
```

## Status

- **v0a foundation — DONE & verified.** `uv` package scaffold; core `SLP → Labels.numpy() → movement
  clean → kinematics/pairwise` pipeline verified on the mice sample; FastAPI backend serving
  **authoritative frames + pose blob**; minimal browser viewer (frame + skeleton overlay, scrub,
  play, arrow-key step).
- **Next:** labeling (paint pos/neg ranges → per-frame parquet store, ethogram timeline, undo/redo)
  → on-disk project layer (PLAN §9) → feature cache job (PLAN §4) → **v0b** train/predict loop +
  prediction heatstrip.

## History

This started as a subdirectory of [talmolab/vibes](https://github.com/talmolab/vibes)
([#69](https://github.com/talmolab/vibes/pull/69),
[#72](https://github.com/talmolab/vibes/pull/72),
[#73](https://github.com/talmolab/vibes/pull/73)) and was extracted into its own repository on
2026-09-04 with history preserved. It is a Python-backed local app rather than a client-side-only
vibe, so it does not belong on vibes.tlab.sh. Mentions of `../event-annotator/`, `../slp-viewer/`
etc. in `PLAN.md` refer to sibling directories in that repo.
