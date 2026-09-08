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

## History

This started as a subdirectory of [talmolab/vibes](https://github.com/talmolab/vibes)
([#69](https://github.com/talmolab/vibes/pull/69),
[#72](https://github.com/talmolab/vibes/pull/72),
[#73](https://github.com/talmolab/vibes/pull/73)) and was extracted into its own repository on
2026-09-04 with history preserved. It is a Python-backed local app rather than a client-side-only
vibe, so it does not belong on vibes.tlab.sh. Mentions of `../event-annotator/`, `../slp-viewer/`
etc. in `PLAN.md` refer to sibling directories in that repo.
