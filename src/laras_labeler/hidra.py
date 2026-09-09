"""HiDRA heads as a source of review candidates, built into the labeler.

HiDRA (the MABe/DooM behaviour classifier) ships 82 pretrained per-lab heads. Point the labeler at
the output of a HiDRA run and any of those heads can propose bouts for a behaviour, which you then
review and correct exactly as you would the labeler's own model's proposals. No training, no GPU:
this reads predictions someone else already computed.

Input is what `HiDRA/predict.py --out` writes, per video:

    <stem>.frames.parquet   frame, subject, target, lab, action, prob, call
    <stem>.bouts.csv        subject, target, lab, action, start_frame, stop_frame, mean_prob, threshold

Videos are matched to the project by FILE STEM, so nothing has to carry a numeric id.

THREE THINGS THIS GETS RIGHT, because each fails silently otherwise:

COLLAPSE. HiDRA scores every ordered (subject -> target) pair; a lane shows one number per animal.
    SELF      self-directed (selfgroom, jumpdown): animal k gets its own `self` row
    SCENE     "is this happening at all": max over every pair, same on every lane
    DIRECTED  "is this animal the acting one": animal k's best score toward anyone
Every option yields a plausible lane. SCENE and DIRECTED differ by more than 0.2 AUROC on the same
head and the same labels.

CALIBRATION. The labeler turns probabilities into candidates at a fixed threshold. HiDRA's
probabilities are not on that scale out of domain -- they compress toward zero while their ORDER
survives, so a head with excellent ranking can put its 99th percentile below the cut and propose
nothing at all. Proposals are therefore rescaled monotonically (ranking untouched, AUROC identical)
so a chosen fraction of frames lands above it.

PREVALENCE. That fraction has to be at or above how common the behaviour is. Below that, no rate
surfaces most bouts however good the head: on one real behaviour at 12.5% prevalence, offering 1%
of frames surfaced 16% of bouts and 15% surfaced 90%.
"""

from __future__ import annotations

import csv
import enum
import json
import pickle
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

# ==============================================================================================
# heads
# ==============================================================================================


import numpy as np
import pandas as pd

FRAMES_SUFFIX = ".frames.parquet"
BOUTS_SUFFIX = ".bouts.csv"


class Collapse(Enum):
    SELF = "self"
    SCENE = "scene"
    DIRECTED = "directed"


def track_index(subject: str) -> int:
    """`mouse1` -> 0. HiDRA numbers animals from 1, in the column order of the export."""
    s = str(subject)
    if not s.startswith("mouse") or not s[5:].isdigit():
        raise ValueError(f"unexpected subject {subject!r}; expected mouseN")
    return int(s[5:]) - 1


def outputs(directory: str | Path) -> dict[str, Path]:
    """{video stem: frames parquet} for a HiDRA results directory."""
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"no HiDRA output directory at {d}")
    found = {p.name[: -len(FRAMES_SUFFIX)]: p for p in sorted(d.glob("*" + FRAMES_SUFFIX))}
    if not found:
        raise FileNotFoundError(
            f"no *{FRAMES_SUFFIX} in {d}. That is what `predict.py --out` writes; point this at "
            f"the directory you gave --out.")
    return found


def read(path: str | Path, lab: str | None = None, action: str | None = None) -> pd.DataFrame:
    """One video's per-frame table, optionally narrowed to one head."""
    filters = []
    if lab is not None:
        filters.append(("lab", "==", lab))
    if action is not None:
        filters.append(("action", "==", action))
    return pd.read_parquet(path, filters=filters or None)


def available(path: str | Path) -> pd.DataFrame:
    """Which (lab, action) heads this video carries, and how confidently each fires."""
    d = pd.read_parquet(path, columns=["lab", "action", "prob", "call"])
    g = d.groupby(["lab", "action"], observed=True)
    return (g.agg(p99=("prob", lambda s: float(np.percentile(s, 99))),
                  called=("call", "mean"))
             .reset_index().sort_values(["lab", "action"]))


def shipped_thresholds(path: str | Path) -> dict[tuple[str, str], float]:
    """HiDRA's own threshold per (lab, action), from the bouts csv beside the frames parquet."""
    b = Path(str(path).replace(FRAMES_SUFFIX, BOUTS_SUFFIX))
    if not b.exists():
        return {}
    d = pd.read_csv(b)
    if "threshold" not in d:
        return {}
    return {(r.lab, r.action): float(r.threshold)
            for r in d[["lab", "action", "threshold"]].drop_duplicates().itertuples()}


def probabilities(df: pd.DataFrame, lab: str, action: str, how: Collapse,
                  n_tracks: int | None = None, n_frames: int | None = None) -> np.ndarray | None:
    """(n_frames, n_tracks) float32, or None if this head carries nothing for that action."""
    d = df[(df["lab"] == lab) & (df["action"] == action)]
    if d.empty:
        return None
    n_frames = int(n_frames or d["frame"].max() + 1)
    n_tracks = int(n_tracks or max(track_index(s) for s in d["subject"].unique()) + 1)

    def grid(sub: pd.DataFrame) -> np.ndarray:
        """(frames, tracks) of the max prob per (frame, subject) in `sub`."""
        out = np.zeros((n_frames, n_tracks), np.float32)
        if sub.empty:
            return out
        g = sub.groupby(["frame", "subject"], observed=True)["prob"].max().reset_index()
        g = g[g["frame"] < n_frames]
        cols = np.array([track_index(s) for s in g["subject"]])
        keep = cols < n_tracks
        out[g["frame"].to_numpy()[keep], cols[keep]] = g["prob"].to_numpy()[keep]
        return out

    if how is Collapse.SELF:
        return grid(d[d["target"] == "self"])

    cross = d[d["target"] != "self"]
    if cross.empty:                      # a self-directed action asked for at scene level
        cross = d
    if how is Collapse.DIRECTED:
        return grid(cross)

    scene = cross.groupby("frame", observed=True)["prob"].max()
    out = np.zeros(n_frames, np.float32)
    idx = scene.index.to_numpy()
    out[idx[idx < n_frames]] = scene.to_numpy()[idx < n_frames]
    return np.repeat(out[:, None], n_tracks, axis=1)

# ==============================================================================================
# calibrate
# ==============================================================================================

import numpy as np


def rescale(p: np.ndarray, thr: float, hi: float = 0.6) -> np.ndarray:
    """Piecewise-linear, monotone: 0->0, thr->hi, max->1. Order is preserved exactly."""
    p = np.asarray(p, np.float32)
    top = float(np.nanmax(p))
    thr = float(np.clip(thr, 1e-6, max(top - 1e-6, 1e-6)))
    out = np.where(p <= thr,
                   hi * p / thr,
                   hi + (1.0 - hi) * (p - thr) / max(top - thr, 1e-6))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def by_rate(p: np.ndarray, rate: float = 0.01, hi: float = 0.6) -> tuple[np.ndarray, float]:
    """Rescale so ~`rate` of frames exceed `hi`. Returns (rescaled, threshold_used)."""
    if not 0 < rate < 1:
        raise ValueError("rate must be in (0, 1)")
    thr = float(np.nanquantile(p, 1.0 - rate))
    return rescale(p, thr, hi), thr


def by_threshold(p: np.ndarray, thr: float, hi: float = 0.6) -> tuple[np.ndarray, float]:
    return rescale(p, thr, hi), float(thr)


def firing_rate(p: np.ndarray, hi: float = 0.6) -> float:
    """Fraction of entries the labeler would treat as candidate-worthy."""
    return float((np.asarray(p) >= hi).mean())

# ==============================================================================================
# budget
# ==============================================================================================


import numpy as np



@dataclass
class Point:
    rate: float
    bouts_surfaced: int
    bouts_total: int
    frames_offered: float

    @property
    def recall(self) -> float:
        return self.bouts_surfaced / self.bouts_total if self.bouts_total else float("nan")

    def review_minutes(self, n_frames: int, fps: float) -> float:
        return self.frames_offered * n_frames / fps / 60.0


def rasterise(frames, n_frames: int, what: str = "labels") -> np.ndarray:
    """Frame indices -> a boolean mask, refusing to index out of bounds.

    A label beyond the registered length means project.json and the label file disagree about how
    long the video is -- usually a project assembled by hand. Crashing with an IndexError says
    nothing useful; this says exactly what disagrees.
    """
    f = np.asarray(frames, dtype=np.int64)
    y = np.zeros(int(n_frames), bool)
    over = f >= n_frames
    if over.all():
        raise ValueError(
            f"every one of the {len(f)} {what} lies beyond the registered length "
            f"({n_frames} frames; highest label is {f.max()}). project.json and the label file "
            f"disagree about this video.")
    if over.any():
        import warnings
        warnings.warn(f"{over.sum()} of {len(f)} {what} lie beyond the registered length "
                      f"({n_frames} frames; highest is {f.max()}) and were ignored", stacklevel=2)
    y[f[~over]] = True
    return y


def _runs(mask) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, np.asarray(mask, bool).view(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def curve(per_video: dict[str, tuple[np.ndarray, np.ndarray]],
          rates=(0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30),
          hi: float = 0.6) -> list[Point]:
    """per_video: {video: (labels, probabilities)}. Labels may be (frames,) or (frames, tracks)."""
    out = []
    for rate in rates:
        tot = hit = 0
        offered = []
        for y, p in per_video.values():
            q, _ = by_rate(p, rate, hi)
            cand = (q >= hi)
            cand = cand.any(axis=1) if cand.ndim == 2 else cand
            offered.append(float(cand.mean()))
            flat = np.asarray(y)
            flat = flat.any(axis=1) if flat.ndim == 2 else flat.astype(bool)
            for s, e in _runs(flat):
                tot += 1
                hit += int(cand[s:e + 1].any())
        out.append(Point(rate, hit, tot, float(np.mean(offered))))
    return out


def prevalence(per_video: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    ys = [np.asarray(y) for y, _ in per_video.values()]
    return float(np.mean([(y.any(axis=1) if y.ndim == 2 else y.astype(bool)).mean() for y in ys]))


def report(per_video, n_frames: int, fps: float, rates=None) -> str:
    pts = curve(per_video, rates) if rates else curve(per_video)
    base = prevalence(per_video)
    lines = [f"prevalence {base:.1%} — a review rate below this cannot surface most bouts", "",
             f"  {'rate':>5}  {'bouts surfaced':>16}  {'frames offered':>14}  {'review min/clip':>15}"]
    for p in pts:
        lines.append(f"  {p.rate:>4.0%}  {p.bouts_surfaced:>4d}/{p.bouts_total} = {p.recall:>6.0%}"
                     f"  {p.frames_offered:>13.1%}  {p.review_minutes(n_frames, fps):>14.1f}")
    good = next((p for p in pts if p.recall >= 0.9), None)
    if good:
        lines += ["", f"  {good.rate:.0%} surfaces {good.recall:.0%} of bouts for "
                      f"{good.review_minutes(n_frames, fps):.1f} min of review per clip"]
    return "\n".join(lines)

# ==============================================================================================
# propose
# ==============================================================================================


import numpy as np



@dataclass
class Proposal:
    """One behaviour in a labeler project, answered by one HiDRA head."""

    behaviour_id: int
    lab: str
    action: str
    collapse: Collapse = Collapse.SCENE
    rate: float | None = 0.05        # fraction of frames offered for review
    threshold: float | None = None   # or a threshold you already trust; overrides `rate`

    def label(self) -> str:
        return f"{self.action} (HiDRA {self.lab})"


@dataclass
class Written:
    video_id: str
    behaviour_id: int
    path: Path
    threshold: float
    firing_rate: float
    snapshotted: bool = False


def load_project(project_dir: str | Path) -> dict:
    p = Path(project_dir) / "project.json"
    if not p.exists():
        raise FileNotFoundError(f"not a labeler project: {p}")
    return json.loads(p.read_text())


def match(project_dir: str | Path, results_dir: str | Path) -> tuple[dict, list[str]]:
    """({video_id: frames parquet}, unmatched video_ids). Matching is by stem, exact then prefix."""
    manifest = load_project(project_dir)
    avail = outputs(results_dir)
    hit, miss = {}, []
    for v in manifest.get("videos", []):
        vid = v["video_id"]
        if vid in avail:
            hit[vid] = avail[vid]
            continue
        near = [k for k in avail if k.startswith(vid) or vid.startswith(k)]
        if len(near) == 1:
            hit[vid] = avail[near[0]]
        else:
            miss.append(vid)
    return hit, miss


def propose(project_dir: str | Path, results_dir: str | Path, plan: list[Proposal],
            n_tracks: int | None = None, force: bool = False,
            log=lambda s: None) -> list[Written]:
    """Write every (video x behaviour) in `plan` into the project. Returns what it wrote."""
    project_dir = Path(project_dir)
    manifest = load_project(project_dir)
    frames_by_id = {v["video_id"]: int(v["n_frames"]) for v in manifest.get("videos", [])}
    hit, miss = match(project_dir, results_dir)
    if miss:
        log(f"  {len(miss)} project video(s) have no HiDRA output and were skipped: "
            f"{', '.join(m[:24] for m in miss[:3])}{' …' if len(miss) > 3 else ''}")
    if not hit:
        raise ValueError(
            "no project video matched a HiDRA output file. The labeler's video_id must equal the "
            "stem of a <stem>.frames.parquet in the results directory.")

    out: list[Written] = []
    for vid, pq in hit.items():
        df = read(pq)
        for prop in plan:
            p = probabilities(df, prop.lab, prop.action, prop.collapse,
                                    n_tracks=n_tracks, n_frames=frames_by_id.get(vid))
            if p is None:
                log(f"  skip {vid[:26]:26s} {prop.lab}/{prop.action}: not in this output")
                continue

            if prop.threshold is not None:
                q, thr = by_threshold(p, prop.threshold)
            else:
                q, thr = by_rate(p, prop.rate)

            dest = project_dir / "predictions" / vid / f"{prop.behaviour_id}.npy"
            dest.parent.mkdir(parents=True, exist_ok=True)
            snap = False
            if dest.exists():
                if not force:
                    raise FileExistsError(
                        f"{dest} already exists. Re-proposing would replace a proposal that may "
                        f"already have been reviewed. Pass --force if that is what you mean.")
                keep = project_dir / "predictions_hidra_round0" / vid / f"{prop.behaviour_id}.npy"
                if not keep.exists():
                    keep.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dest, keep)
                    snap = True

            np.save(dest, q.astype(np.float32))
            w = Written(vid, prop.behaviour_id, dest, thr, firing_rate(q), snap)
            out.append(w)
            log(f"  {vid[:26]:26s} {prop.label():32s} thr {thr:.4f} -> "
                f"{w.firing_rate:.2%} offered" + ("  [snapshotted]" if snap else ""))
    return out


# ==============================================================================================
# running HiDRA: video + tracking -> per-frame probabilities the review flow already understands
# ==============================================================================================
# The labeler does NOT import HiDRA. HiDRA is JAX and picks its backend (CUDA or CPU) at import,
# so pulling it into this process would put a heavyweight, hardware-specific dependency in the way
# of an install whose only job might be labeling. It is invoked as a subprocess in its own
# interpreter instead -- which is what HiDRA's own predict.py does internally anyway. Configure:
#
#   HIDRA_HOME    the HiDRA checkout (the directory holding predict.py)
#   HIDRA_PYTHON  an interpreter with JAX installed (default: HIDRA_HOME/.venv/bin/python)
#
# CPU works and is the honest default -- JAX installs CPU-only where there is no CUDA, and the
# heads import and run under it. It is a speed difference, not a capability one.

# HiDRA's own bodypart vocabulary (solution.py BODYPARTS). Any node whose name is in here is
# passed through unchanged -- which covers most SLEAP skeletons, since these are the conventional
# names. The first seven are HiDRA's canonical set.
HIDRA_BODYPARTS = (
    "tail_base", "ear_right", "ear_left", "nose", "neck", "body_center", "tail_tip",
    "tail_midpoint", "forepaw_left", "forepaw_right", "hindpaw_left", "hindpaw_right",
    "hip_right", "hip_left", "lateral_right", "lateral_left", "head", "spine_1", "spine_2",
    "tail_middle_1", "tail_middle_2",
    "headpiece_topfrontright", "headpiece_topbackright", "headpiece_topfrontleft",
    "headpiece_topbackleft", "headpiece_bottomfrontright", "headpiece_bottombackright",
    "headpiece_bottombackleft", "headpiece_bottomfrontleft",
)

# Only for names that differ from HiDRA's. A pass-through is always preferred: mapping a node onto
# a different anatomical name moves a keypoint, which changes every distance the model computes.
NODE_ALIASES = {
    "snout": "nose",
    "earl": "ear_left", "earr": "ear_right",
    "left_ear": "ear_left", "right_ear": "ear_right",
    "thorax": "neck",
    "centroid": "body_center", "center": "body_center", "trunk": "body_center",
    "abdomen": "body_center", "body": "body_center", "thorax_center": "body_center",
    "tailbase": "tail_base", "tail_start": "tail_base",
    # TTI is the tail-torso interface, i.e. the tail base. Worth naming explicitly: tail_base is one
    # of HiDRA's canonical seven, and without this entry a skeleton that calls it TTI silently
    # supplies only six of them — our own 15-node skeleton did exactly that.
    "tti": "tail_base", "tailtorso": "tail_base", "tail_torso": "tail_base",
    "haunch_left": "hip_left", "haunch_right": "hip_right",
    "haunchl": "hip_left", "haunchr": "hip_right",
    "tailtip": "tail_tip", "tail_end": "tail_tip",
    "tail_mid": "tail_midpoint", "tailmid": "tail_midpoint",
    "forepaw_l": "forepaw_left", "forepaw_r": "forepaw_right",
    "hindpaw_l": "hindpaw_left", "hindpaw_r": "hindpaw_right",
    "side_left": "lateral_left", "side_right": "lateral_right",
}


def map_nodes(node_names: list[str]) -> tuple[dict[str, str], list[str]]:
    """This skeleton's node names -> HiDRA's, plus the ones with no equivalent.

    Nodes with no HiDRA name are dropped rather than guessed at: HiDRA indexes its input by bodypart
    name, so a node sent under the wrong name is not a missing feature but a wrong one, and the
    classifier has no way to tell.

    EXACT NAMES WIN OVER ALIASES, in two passes, because otherwise a perfectly ordinary skeleton
    could not run at all. `thorax` aliases to `neck` and `neck` is also a HiDRA name of its own, so
    a rig carrying both -- common in SLEAP mouse skeletons -- mapped two nodes onto one bodypart and
    the whole Predict failed with a collision. The exact `neck` is unambiguously the right occupant
    of that slot; the aliased `thorax` is redundant and is reported as dropped. An alias losing to a
    real name costs one feature. Refusing to run costs the user the feature entirely."""
    # Matching ignores case and separators, so earL, ear_l and EAR-L all reach the same entry --
    # skeletons name the same keypoint every one of those ways.
    norm = lambda x: "".join(ch for ch in x.lower() if ch.isalnum())   # noqa: E731
    known = {norm(b): b for b in HIDRA_BODYPARTS}
    alias = {norm(k): v for k, v in NODE_ALIASES.items()}
    keep, dropped = {}, []
    #: How each kept node was resolved. An exact HiDRA name is stronger evidence about what a
    #: keypoint IS than an alias guess, and the difference decides collisions below.
    exact: dict[str, bool] = {}
    for n in node_names:
        k = norm(n)
        if k in known:
            keep[n.lower()] = known[k]
            exact[n.lower()] = True
        elif k in alias:
            keep[n.lower()] = alias[k]
            exact[n.lower()] = False
        else:
            dropped.append(n)

    # Two source nodes can land on one HiDRA name -- a skeleton carrying both `neck` and `thorax`
    # hits it, and that includes this repo's own synthetic clip. Refusing the whole clip is too
    # strong when the tie is not actually a tie: `neck` IS HiDRA's neck, while `thorax` only
    # reaches it through an alias, so the exact match wins and the alias is dropped like any other
    # node with no equivalent. That is the conservative direction -- dropping a keypoint costs a
    # feature, whereas keeping the alias could move one, which is a WRONG feature the classifier
    # cannot detect. A collision between two names of equal standing is still left to the caller,
    # because there is no principled way to pick.
    by_target: dict[str, list[str]] = {}
    for src, tgt in keep.items():
        by_target.setdefault(tgt, []).append(src)
    for tgt, srcs in by_target.items():
        if len(srcs) < 2:
            continue
        winners = [s for s in srcs if exact[s]]
        if len(winners) == 1:
            for s in srcs:
                if s != winners[0]:
                    dropped.append(s)
                    keep.pop(s)
    return keep, dropped


# Where the checkout is. Normally HIDRA_HOME / HIDRA_PYTHON, but the GUI can set these and persist
# them (config.py -> <projects_root>/settings.json), because requiring a terminal to find two paths
# is the difference between "a new user can use HiDRA" and "a new user never learns it is there".
# An explicit setting wins over the environment: the env var came from whatever shell happened to
# launch the server, and silently overriding what someone just typed into the GUI is worse than
# ignoring a stale variable. `source` says which one won, so it is never a mystery.
_OVERRIDE: dict[str, str] = {}

DEFAULT_HOME = Path.home() / "code/hidra-review/HiDRA"


def configure(home: str | None = None, python: str | None = None) -> None:
    """Set (or clear, with an empty value) the paths the integration uses."""
    import os
    for key, val in (("home", home), ("python", python)):
        val = (val or "").strip()
        if val:
            _OVERRIDE[key] = str(Path(val).expanduser())
        else:
            _OVERRIDE.pop(key, None)
    os.environ.pop("_HIDRA_CACHE", None)   # nothing cached today; kept as the one place to clear


def configured() -> dict:
    """The two paths in effect, and where each came from: 'gui', 'env' or 'default'."""
    import os
    if "home" in _OVERRIDE:
        home, home_src = Path(_OVERRIDE["home"]), "gui"
    elif os.environ.get("HIDRA_HOME"):
        home, home_src = Path(os.environ["HIDRA_HOME"]).expanduser(), "env"
    else:
        home, home_src = DEFAULT_HOME, "default"
    if "python" in _OVERRIDE:
        py, py_src = Path(_OVERRIDE["python"]), "gui"
    elif os.environ.get("HIDRA_PYTHON"):
        py, py_src = Path(os.environ["HIDRA_PYTHON"]).expanduser(), "env"
    else:
        # A venv puts its interpreter at Scripts\python.exe on Windows and bin/python elsewhere, so
        # the posix layout alone made the default unreachable on Windows -- and it is the default
        # that decides whether HiDRA is found with no configuration at all. Both candidates are
        # tried so a checkout laid out either way is picked up; the posix one is the fallback so the
        # reported path stays recognisable when neither exists.
        cands = [home.parent / "Scripts" / "python.exe", home.parent / ".venv" / "Scripts" / "python.exe",
                 home.parent / ".venv" / "bin" / "python", home / ".venv" / "Scripts" / "python.exe",
                 home / ".venv" / "bin" / "python"]
        py = next((c for c in cands if c.exists()), home.parent / ".venv" / "bin" / "python")
        py_src = "default"
    return {"home": home, "python": py, "home_source": home_src, "python_source": py_src}


def runtime() -> dict:
    """Can inference actually run here, and if not, exactly what is missing.

    Reported to the GUI so a bound head whose runtime is absent says so in the picker rather than
    failing only once the user presses Predict -- and so the GUI can offer to fix it, which is why
    `why` is a sentence a user can act on rather than a boolean."""
    import subprocess

    cfg = configured()
    home, py = cfg["home"], cfg["python"]
    info = {"home": str(home), "python": str(py), "can_infer": False, "backend": None, "why": None,
            "home_source": cfg["home_source"], "python_source": cfg["python_source"]}

    if not (home / "predict.py").exists():
        info["why"] = f"no predict.py under {home} — set HIDRA_HOME to your HiDRA checkout"
        return info
    if not py.exists():
        info["why"] = f"no interpreter at {py} — set HIDRA_PYTHON to one with JAX installed"
        return info
    try:
        r = subprocess.run([str(py), "-c", "import jax; print(jax.default_backend())"],
                           capture_output=True, text=True, timeout=120)
    except (subprocess.SubprocessError, OSError) as e:
        info["why"] = f"could not probe {py}: {e}"
        return info
    if r.returncode != 0:
        info["why"] = f"JAX is not importable in {py} — install it there ({r.stderr.strip().splitlines()[-1:] or ['']}[0])"
        return info
    info["backend"] = r.stdout.strip()
    info["can_infer"] = True
    return info


def runnable_labs(home: Path | None = None) -> set[str]:
    """The labs HiDRA will actually run, read from predict.py's ALL_LABS.

    Necessary because the shipped threshold table contains rows the model cannot serve -- notably
    `pooled`, an aggregate across labs. predict.py silently drops any lab outside ALL_LABS, then
    exits 0 having run nothing, so a head offered from the table alone produces an empty result
    that looks like a behaviour that never occurred. Parsed statically: this must work in the
    labeler's own interpreter, which has no JAX."""
    import ast
    home = home or Path(runtime()["home"])
    src = home / "predict.py"
    if not src.exists():
        return set()
    try:
        tree = ast.parse(src.read_text())
    except SyntaxError:
        return set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "ALL_LABS" for t in node.targets):
            try:
                return set(ast.literal_eval(node.value))
            except (ValueError, TypeError):
                return set()
    return set()


def catalog(thresholds_csv: Path | None = None) -> list[dict]:
    """Every (lab, action) head that ships, for the behavior->classifier picker.

    Read from HiDRA's own shipped threshold table rather than a list maintained here, so the picker
    cannot drift from the checkout. `threshold` is the shipped operating point -- kept because it is
    what a user would otherwise assume applies to their videos, and out of domain it usually does
    not (see the module docstring on calibration)."""
    import os
    p = thresholds_csv or Path(os.environ.get(
        "HIDRA_THRESHOLDS", str(configured()["home"] / "derived_thresholds_train.csv")))
    if not p.exists():
        return []
    runnable = runnable_labs()
    # The shipped table keys each head as `Lab__action` in an unnamed first column.
    out, seen = [], set()
    with p.open() as f:
        for row in csv.reader(f):
            if len(row) < 2 or "__" not in row[0]:
                continue                                          # header, or a malformed line
            lab, _, action = row[0].partition("__")
            lab, action = lab.strip(), action.strip()
            if not lab or not action or (lab, action) in seen:
                continue
            if runnable and lab not in runnable:
                continue                                          # e.g. `pooled` — cannot be served
            seen.add((lab, action))
            try:
                thr = float(row[1])
            except ValueError:
                thr = float("nan")
            out.append({"lab": lab, "action": action, "threshold": thr,
                        "collapse": default_collapse(action), "category": category(action)})
    return sorted(out, key=lambda h: (h["action"], h["lab"]))


def default_collapse(action: str) -> str:
    """A first guess at what one number per animal should mean for this action.

    A guess only, and surfaced as an editable control: the same head scored SCENE vs DIRECTED can
    differ by more than 0.2 AUROC, so this is the one setting worth checking by eye before trusting
    a lane. Actions naming a self-directed act take their own `self` row; anything naming a partner
    is directed; the rest default to scene, which is the safe read when the label's own semantics
    are unknown."""
    a = action.lower()
    # An act aimed at the cage or an object is an individual behavior however transitive its verb
    # sounds, so it is settled before the partner keywords below can claim it.
    if a.endswith("object") or "cage" in a or "wall" in a:
        return "self"
    # Directed is tested FIRST: "allogroom" and "huddle" contain self-directed-looking stems while
    # naming acts done to another animal, and getting that backwards silently scores the wrong
    # question -- the exact failure this control exists to make visible.
    if any(k in a for k in ("allo", "sniff", "attack", "mount", "chase", "approach", "follow",
                            "dominan", "escape", "avoid", "flee", "defend", "intromi", "ejacul",
                            "anogenital", "partner", "social", "huddle", "nose", "bite")):
        return "directed"
    if any(k in a for k in ("self", "groom", "rear", "jump", "climb", "dig", "freeze", "immobil",
                            "run", "walk", "rest", "sleep", "explor", "eat", "drink", "scratch")):
        return "self"
    return "scene"


def export_tracking(poses: np.ndarray, node_names: list[str], fps: float,
                    pix_per_cm: float, stem: str, work: Path) -> dict:
    """(F, T, N, 3) poses -> the parquet folder HiDRA's predict.py takes, plus its metadata.csv.

    Dense on purpose -- every (frame, mouse, bodypart) row is written even where the point is
    missing. HiDRA derives its mouse and bodypart axes from the parquet's own index levels, so a
    mouse with no surviving rows would quietly vanish from the model's input instead of being
    treated as unobserved, and dropping trailing frames would shorten the clip."""
    keep, dropped = map_nodes(node_names)
    if not keep:
        raise ValueError(
            f"none of this skeleton's nodes are HiDRA bodyparts: {node_names}. "
            f"Rename them to HiDRA's names: {', '.join(HIDRA_BODYPARTS[:7])} (and others).")
    cols = [i for i, n in enumerate(node_names) if n.lower() in keep]
    names = [keep[node_names[i].lower()] for i in cols]
    if len(set(names)) != len(names):
        clash: dict[str, list[str]] = {}
        for i, nm in zip(cols, names):
            clash.setdefault(nm, []).append(node_names[i])
        raise ValueError("node mapping collision: " + "; ".join(
            f"{v} both map to {k!r}" for k, v in clash.items() if len(v) > 1))

    sub = poses[:, :, cols, :]                                    # (F, T, k, 3)
    n_f, n_t, n_k = sub.shape[0], sub.shape[1], sub.shape[2]
    frame = np.repeat(np.arange(n_f, dtype=np.int64), n_t * n_k)
    mouse = np.tile(np.repeat([f"mouse{t + 1}" for t in range(n_t)], n_k), n_f)
    part = np.tile(np.asarray(names, dtype=object), n_f * n_t)
    df = pd.DataFrame({
        "video_frame": frame, "mouse_id": mouse, "bodypart": part,
        "x": sub[..., 0].reshape(-1).astype("float32"),
        "y": sub[..., 1].reshape(-1).astype("float32"),
    })

    work.mkdir(parents=True, exist_ok=True)
    (work / f"{stem}.parquet").unlink(missing_ok=True)
    df.to_parquet(work / f"{stem}.parquet", index=False)
    # Per-file metadata beats the CLI's global flags: one folder can hold clips from different rigs.
    with (work / "metadata.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "fps", "pix_per_cm"])
        w.writerow([f"{stem}.parquet", fps, pix_per_cm])
    return {"rows": len(df), "frames": n_f, "animals": n_t,
            "bodyparts": names, "dropped_nodes": dropped}


def infer(work: Path, out: Path, lab: str, action: str, fps: float, pix_per_cm: float,
          progress=lambda p, m: None) -> Path:
    """Run one (lab, action) head over the exported folder. Returns the frames parquet.

    Scoped to a single head by a one-row job sheet: the default is all 82, which on CPU is the
    difference between minutes and hours for a result the user did not ask for."""
    import os
    import subprocess

    rt = runtime()
    if not rt["can_infer"]:
        raise RuntimeError(rt["why"])

    jobs_csv = work / "jobs.csv"
    with jobs_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "lab", "action", "subject", "target"])
        # "*" is predict.py's wildcard, and the ONLY thing its filter accepts as "any":
        #     keep() -> any((fs in ("*", subj)) and (ft in ("*", tgt)) for fs, ft in filt)
        # An empty cell matches no subject and no target, so every (subject, target) pair was
        # rejected, `frames` stayed empty, and predict.py wrote no frames parquet at all -- the
        # per-frame probabilities this integration exists to read. It looked like a head that
        # found nothing rather than a job sheet that asked for nothing.
        w.writerow(["1", lab, action, "*", "*"])                  # every pair; we collapse after

    out.mkdir(parents=True, exist_ok=True)
    cmd = [rt["python"], str(Path(rt["home"]) / "predict.py"), str(work),
           "--jobs", str(jobs_csv), "--out", str(out),
           "--fps", str(fps), "--pix-per-cm", str(pix_per_cm), "--output", "both"]
    progress(5, f"{action} ({lab}) on {rt['backend']}")

    proc = subprocess.Popen(cmd, cwd=rt["home"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    tail: list[str] = []
    for line in proc.stdout or []:
        line = line.rstrip()
        if not line:
            continue
        tail = (tail + [line])[-25:]
        progress(min(90, 5 + len(tail) * 3), line[:120])
    if proc.wait() != 0:
        raise RuntimeError("HiDRA inference failed:\n" + "\n".join(tail[-12:]))

    # predict.py exits 0 after running nothing if the job sheet selected no servable lab, so a
    # zero-work run has to be caught here or it reads downstream as a behaviour that never occurred.
    joined = "\n".join(tail)
    if "/dev/shm" in joined:
        raise RuntimeError(
            "HiDRA's predict.py hardcodes its scratch directory under /dev/shm "
            "(predict.py: PERLAB_WORKDIR=f\"/dev/shm/doom_predict_{os.getpid()}\"), which exists "
            "only on Linux. It overrides the environment, so it cannot be redirected from here, "
            "and /dev/shm cannot be created on macOS. Change that one line to a temp directory — "
            "e.g. tempfile.mkdtemp(prefix=\"doom_predict_\") — and inference runs. "
            "Nothing else in the pipeline is platform-specific.")
    if "running 0 lab classifier set" in joined:
        raise RuntimeError(
            f"HiDRA ran no classifier for ({lab}, {action}) — {lab!r} is not one of the labs it can "
            f"serve. Runnable: {', '.join(sorted(runnable_labs())) or 'none found'}")

    hits = sorted(out.glob("*.frames.parquet"))
    if not hits:
        raise RuntimeError("HiDRA wrote no frames parquet:\n" + "\n".join(tail[-12:]))
    return hits[0]


def to_lanes(frames_parquet: Path, lab: str, action: str, collapse: str,
             n_frames: int, n_animals: int, rate: float | None = 0.15) -> np.ndarray:
    """HiDRA's per-pair frame table -> the (F, T) array the labeler's review flow reads.

    Two steps, each of which silently produces a plausible-looking wrong answer if skipped:

    COLLAPSE, because HiDRA scores ordered (subject -> target) pairs and a lane holds one number per
    animal. `self` takes the animal's own row, `scene` the max over every pair (identical on every
    lane), `directed` the animal's best score toward anyone.

    CALIBRATE, because probabilities trained elsewhere are not on this GUI's threshold scale. They
    compress toward zero out of domain while their ORDER survives, so an excellent head can rank
    perfectly and still propose nothing. Rescaling is monotone -- AUROC is untouched -- and puts
    `rate` of frames above the candidate cut. Pass rate=None to keep the raw probabilities."""
    df = pd.read_parquet(frames_parquet)
    sel = df[(df["lab"] == lab) & (df["action"] == action)]
    if sel.empty:
        raise RuntimeError(f"no rows for ({lab}, {action}) in {frames_parquet.name}; "
                           f"it holds {sorted(set(zip(df['lab'], df['action'])))[:8]}")

    lanes = np.zeros((n_frames, n_animals), dtype="float32")
    mode = Collapse(collapse)
    if mode is Collapse.SCENE:
        g = sel.groupby("frame")["prob"].max()
        col = np.zeros(n_frames, dtype="float32")
        col[np.clip(g.index.to_numpy(), 0, n_frames - 1)] = g.to_numpy(dtype="float32")
        lanes[:] = col[:, None]
    else:
        for t in range(n_animals):
            who = f"mouse{t + 1}"
            rows = sel[sel["subject"] == who]
            rows = rows[rows["target"] == "self"] if mode is Collapse.SELF else rows
            if rows.empty:
                continue
            g = rows.groupby("frame")["prob"].max()
            lanes[np.clip(g.index.to_numpy(), 0, n_frames - 1), t] = g.to_numpy(dtype="float32")

    return lanes if rate is None else by_rate(lanes, float(rate))[0]


def export_labels(store, labels, pid: str, bid: int, head: dict, work: Path) -> dict:
    """Reviewed labels for one behavior -> the per-frame table LABTAIL fine-tuning trains on.

    Only frames a human actually adjudicated are written. A frame nobody looked at is not a
    negative -- treating it as one teaches the head that its own correct detections are wrong,
    which is the fastest way to make fine-tuning worse than the head it started from. The
    `labeled` column carries that distinction so the trainer can mask.

    Reads the LabelStore, which is where labels actually live. This used to look for
    `labels/<vid>/<bid>.json` spans, or a `Project.labels()` method -- neither of which exists: the
    app writes `labels/<vid>.parquet`, a per-frame table, and `Project` has no such method. So the
    export found nothing for every project, always, and the fine-tune refused with "no reviewed
    labels for this behavior yet" no matter how much had just been reviewed. It was never possible
    for this path to succeed.

    Stored values are 1 = Happening, 0 = Not-happening, 2 = Unknown (app.py's span schema, and how
    predict.py reads them). Unknown is saved but never trained on."""
    proj = store.get(pid)
    rows: list[dict] = []
    n_bouts = 0
    for v in proj.videos:
        vid = v["video_id"]
        try:
            df = labels.rows_for_behavior(pid, vid, bid)
        except (KeyError, FileNotFoundError):
            continue
        if df is None or df.empty:
            continue
        stem = Path(str(v.get("video_path") or vid)).stem or vid
        for t in sorted({int(x) for x in df["track"].unique()}):
            for run in labels.get_runs(pid, vid, t, bid).get(bid, []):
                a, b, val = int(run[0]), int(run[1]), int(run[2])
                if val not in (0, 1) or b <= a:
                    continue
                n_bouts += val == 1
                rows.append({"file": stem,
                             "subject": f"mouse{t + 1}", "target": "self",
                             "lab": head["lab"], "action": head["action"],
                             "start_frame": a, "stop_frame": b - 1,   # inclusive, as predict.py writes
                             "label": val, "labeled": 1})

    work.mkdir(parents=True, exist_ok=True)
    out = work / "labels.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "subject", "target", "lab", "action",
                                          "start_frame", "stop_frame", "label", "labeled"])
        w.writeheader()
        w.writerows(rows)
    return {"bouts": n_bouts, "spans": len(rows), "path": str(out),
            "videos": len({r["file"] for r in rows})}


def finetune(script: Path, rt: dict, head: dict, work: Path, counts: dict,
             progress=lambda p, m: None) -> dict:
    """HiDRA's LABTAIL adaptation on this project's reviewed labels.

    LABTAIL trains the lab embedding, three tail blocks and the per-lab output projection while the
    self-supervised trunk stays frozen. That is what makes it viable here: a few hundred reviewed
    bouts is nowhere near enough to move a trunk, but it is enough to re-aim a tail."""
    import os
    import subprocess

    ckpt = work / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    cmd = [rt["python"], str(script), "--mode", "labtail",
           "--labels", str(work / "labels.csv"), "--lab", head["lab"],
           "--action", head["action"], "--out", str(ckpt)]
    progress(0, "starting LABTAIL")

    proc = subprocess.Popen(cmd, cwd=rt["home"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1,
                            env={**os.environ, "PYTHONUNBUFFERED": "1",
                                 "PERLAB_WORKDIR": str(work / "_perlab")})
    tail: list[str] = []
    for line in proc.stdout or []:
        line = line.rstrip()
        if not line:
            continue
        tail = (tail + [line])[-40:]
        progress(min(95, len(tail) * 2), line[:120])
    if proc.wait() != 0:
        raise RuntimeError("LABTAIL fine-tuning failed:\n" + "\n".join(tail[-15:]))

    progress(100, "done")
    return {"mode": "labtail", "lab": head["lab"], "action": head["action"],
            "checkpoint": str(ckpt), "backend": rt["backend"], **counts,
            "log": tail[-15:]}


# Behaviour categories, for grouping the classifier picker the way the ethogram is organised rather
# than alphabetically -- 99 heads over 34 actions is too many to scan as a flat list. Navigation
# only: nothing downstream reads the category, and an action absent here still appears, under
# "other", so an unmapped head is never hidden by this table.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "investigative": ("approach", "follow", "sniff", "sniffall", "sniffbody", "sniffface",
                      "sniffgenital", "reciprocalsniff"),
    "aggressive": ("attack", "chase", "chaseattack", "dominance", "dominancegroom", "tussle",
                   "shepherd"),
    "reproductive": ("mount", "attemptmount", "intromit"),
    "defensive": ("avoid", "escape", "defend", "flinch", "submit", "freeze"),
    "affiliative": ("allogroom", "huddle"),
    "nonsocial": ("climb", "dig", "rear", "rest", "run", "selfgroom", "exploreobject",
                  "biteobject"),
}

_CATEGORY_OF = {a: c for c, actions in CATEGORIES.items() for a in actions}


def category(action: str) -> str:
    return _CATEGORY_OF.get(action.lower(), "other")
