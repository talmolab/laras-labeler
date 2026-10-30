#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "pyarrow"]
# ///
"""How much a behavior's kinematics vary within one mouse vs across the mice of a cage.

For every bout of a behavior it averages the labeler's own per-frame kinematics over the bout --
speed, turning, body length and bend, posture extents, ... (the `__raw` columns of the feature
cache: cleaned, body-length normalized, rotation-invariant, exactly what the classifier sees) --
plus the bout's duration. Then, per behavior and kinematic:

  CV within mouse   SD / mean across one mouse's bouts; median over mice
  CV within cage    SD / mean across all bouts of all mice in one cage; median over cages
  cage / mouse      the two, per cage (within-cage CV / median within-mouse CV), median over cages.
                    ~1: cage-mates are as alike as a mouse is to itself. >1: the mice differ.
  mouse share       of the variation inside a cage, the fraction explained by WHICH mouse did the
                    bout (one-way ICC, 0 = mice interchangeable); median over cages
  p                 permutation test of that: shuffle which mouse each bout belongs to, within its
                    cage, and count how often the shuffled between-mouse spread is as large

Only cages with at least two mice that each have --min-bouts bouts are used, and every number for a
kinematic is computed on the same bouts, so the two CVs are comparable.

    uv run scripts/bout_kinematics_cv.py ~/laras-projects/hidralabel
    uv run scripts/bout_kinematics_cv.py ~/laras-projects/hidralabel --behaviors eating,"supported rearing" \\
        --out-bouts bouts.csv --csv cv_summary.csv
    # predicted rather than labeled bouts: export them first, then pass the CSV
    python scripts/export_bouts.py --pid hidralabel --out predicted.csv
    uv run scripts/bout_kinematics_cv.py ~/laras-projects/hidralabel --bouts predicted.csv

WHAT A "MOUSE" IS HERE. Tracks are numbered per clip, so a mouse is one (clip, track): the same
animal in two clips counts as two mice. A cage is the camera in the clip id (cam_07), or the clip
itself with --cage clip. An identity swap inside a clip blends two mice into both tracks, which can
only make them look MORE alike -- so a difference that shows up is not an artifact of swaps.

--baseline divides each bout's posture (widths, lengths, bend, spread) by the same mouse's median
posture over the rest of the clip: its tracked frames outside its bouts of that behavior. A mouse
built differently from its cage-mates -- or one whose keypoints the pose model places in its own
peculiar way -- looks different in every behavior; this cancels that, leaving how the behavior
changes each mouse's posture. Speeds and duration are left as they are. Per-bout `<posture>_rel`
columns carry the ratios in --out-bouts.

Needs the feature cache (`features/<clip>.npy`), which the labeler builds when it opens a project
or trains. CVs are unaffected by the 30-vs-50 fps registration of HCM clips: a constant frame-rate
factor scales a kinematic's SD and mean alike. Standalone: `uv run` resolves its dependencies.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

# Base kinematics that are magnitudes (>= 0), where SD/mean means something. Signed ones
# (forward_speed, lateral_speed, angular_velocity) average toward zero; pass them with --signals
# and they are reported without a CV.
DEFAULT_SIGNALS = ("centroid_speed", "nose_speed", "tail_speed", "abs_turn", "accel_mag",
                   "body_length", "body_bend", "ego_len", "ego_width", "elongation", "spread_area",
                   "spine_len", "haunch_width", "shoulder_width", "ear_width")
CAM_RE = re.compile(r"cam[_-]?0*(\d+)", re.I)
# Posture: the shape of a body, which a mouse carries into every behavior. With --baseline these are
# divided by the mouse's own median outside the behavior, so its build -- and any keypoint placement
# peculiar to that one animal -- cancels. Speeds are left alone: a mostly resting mouse's median speed
# is ~0, and dividing by it would mostly measure how much it sleeps.
POSTURE = {"body_length", "body_bend", "ego_len", "ego_width", "elongation", "spread_area",
           "spine_len", "haunch_width", "shoulder_width", "ear_width"}


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _runs(frames: np.ndarray) -> list[tuple[int, int]]:
    """Sorted unique frame indices -> contiguous [start, end) runs."""
    if len(frames) == 0:
        return []
    breaks = np.flatnonzero(np.diff(frames) > 1)
    starts = np.r_[frames[0], frames[breaks + 1]]
    ends = np.r_[frames[breaks], frames[-1]] + 1
    return list(zip(starts.tolist(), ends.tolist()))


# ----------------------------------------------------------------------------- bouts
def labeled_bouts(proj: Path, manifest: dict, wanted: set[str] | None) -> pd.DataFrame:
    """Contiguous Happening runs per (clip, behavior, track), from the project's label store."""
    names = {int(b["id"]): b["name"].strip() for b in manifest.get("behaviors", [])}
    rows = []
    for v in manifest.get("videos", []):
        p = proj / "labels" / f"{v['video_id']}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df = df[df["value"] == 1]
        if "track" not in df.columns:
            df = df.assign(track=0)
        for (bid, track), g in df.groupby(["behavior_id", "track"]):
            name = names.get(int(bid))
            if name is None or (wanted and _norm(name) not in wanted):
                continue
            for s, e in _runs(np.unique(g["frame"].to_numpy(np.int64))):
                rows.append({"video_id": v["video_id"], "behavior": name, "track": int(track),
                             "start_frame": s, "end_frame": e})
    return pd.DataFrame(rows, columns=["video_id", "behavior", "track", "start_frame", "end_frame"])


def csv_bouts(path: Path, wanted: set[str] | None) -> pd.DataFrame:
    """Bouts from scripts/export_bouts.py (or any CSV with those columns)."""
    df = pd.read_csv(path)
    need = {"video_id", "behavior", "track", "start_frame", "end_frame"}
    if need - set(df.columns):
        sys.exit(f"{path} lacks columns {sorted(need - set(df.columns))}; export_bouts.py writes them")
    df["behavior"] = df["behavior"].astype(str).str.strip()
    if wanted:
        df = df[df["behavior"].map(_norm).isin(wanted)]
    return df[sorted(need)].copy()


# ----------------------------------------------------------------------------- per-bout kinematics
def bout_kinematics(proj: Path, manifest: dict, bouts: pd.DataFrame, signals: list[str],
                    cage_by: str, min_tracked: float, baseline: bool = False) -> pd.DataFrame:
    """One row per bout: its clip, mouse, cage, duration and the mean of each kinematic over it.
    With `baseline`, also `<posture>_rel`: that mean over the same mouse's median posture in the
    rest of the clip (tracked frames outside its bouts of this behavior)."""
    fps = {v["video_id"]: float(v.get("fps") or 30.0) for v in manifest.get("videos", [])}
    out = []
    for vid, g in bouts.groupby("video_id"):
        npy, meta = proj / "features" / f"{vid}.npy", proj / "features" / f"{vid}.meta.json"
        if not npy.exists() or not meta.exists():
            print(f"  skipped {vid}: no feature cache — open the project in the labeler (it builds "
                  f"one per clip) and re-run", file=sys.stderr)
            continue
        X = np.load(npy, mmap_mode="r")                               # (F, T, D)
        col = {n: i for i, n in enumerate(json.loads(meta.read_text())["feature_names"])}
        have = [s for s in signals if f"{s}__raw" in col]
        idx = [col[f"{s}__raw"] for s in have]
        present = col.get("centroid_speed__raw")
        m = CAM_RE.search(str(vid))
        cage = f"cam_{int(m.group(1)):02d}" if (cage_by == "camera" and m) else str(vid)
        rel = [s for s in have if s in POSTURE] if baseline else []
        rel_idx = [col[f"{s}__raw"] for s in rel]
        own: dict[tuple, np.ndarray] = {}

        def own_baseline(beh: str, t: int) -> np.ndarray:
            """This mouse's median posture over the clip's tracked frames outside this behavior."""
            if (beh, t) not in own:
                keep = np.ones(X.shape[0], bool)
                for o in g[(g["behavior"] == beh) & (g["track"] == t)].itertuples(index=False):
                    keep[int(o.start_frame):int(o.end_frame)] = False
                if present is not None:
                    keep &= np.isfinite(np.asarray(X[:, t, present]))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    own[(beh, t)] = np.nanmedian(np.asarray(X[:, t, rel_idx])[keep], axis=0)
            return own[(beh, t)]

        for b in g.itertuples(index=False):
            s, e, t = int(b.start_frame), int(b.end_frame), int(b.track)
            if t >= X.shape[1] or s >= X.shape[0] or e <= s:
                continue
            seg = np.asarray(X[s:min(e, X.shape[0]), t, :])
            tracked = float(np.isfinite(seg[:, present]).mean()) if present is not None else 1.0
            row = {"behavior": b.behavior, "video_id": vid, "track": t, "mouse": f"{vid} track {t + 1}",
                   "cage": cage, "start_frame": s, "end_frame": e,
                   "duration_s": (e - s) / fps.get(vid, 30.0), "tracked_frac": round(tracked, 3)}
            with warnings.catch_warnings():                        # all-NaN column -> NaN, quietly
                warnings.simplefilter("ignore", RuntimeWarning)
                means = np.nanmean(seg[:, idx], axis=0) if idx else []
            row.update({s_: float(v_) for s_, v_ in zip(have, means)})
            if rel:
                for s_, base in zip(rel, own_baseline(b.behavior, t)):
                    row[f"{s_}_rel"] = row[s_] / base if base > 0 else float("nan")
            out.append(row)
    df = pd.DataFrame(out)
    if not df.empty:
        dropped = int((df["tracked_frac"] < min_tracked).sum())
        if dropped:
            print(f"  dropped {dropped} bout(s) whose animal was tracked in < {min_tracked:.0%} of "
                  f"their frames", file=sys.stderr)
        df = df[df["tracked_frac"] >= min_tracked]
    return df


# ----------------------------------------------------------------------------- statistics
def _cv(x: np.ndarray) -> float:
    if len(x) < 2 or (x < 0).any() or x.mean() <= 0:
        return float("nan")
    return float(x.std(ddof=1) / x.mean())


def _icc(x: np.ndarray, g: np.ndarray) -> float:
    """One-way random-effects ICC(1) for unbalanced groups, clipped to [0, 1]."""
    k, n = len(np.unique(g)), len(x)
    if k < 2 or n <= k:
        return float("nan")
    _, gi = np.unique(g, return_inverse=True)
    ni = np.bincount(gi)
    mi = np.bincount(gi, x) / ni
    msb = float((ni * (mi - x.mean()) ** 2).sum() / (k - 1))
    msw = float(((x - mi[gi]) ** 2).sum() / (n - k))
    n0 = (n - (ni ** 2).sum() / n) / (k - 1)
    den = msb + (n0 - 1) * msw
    return float(np.clip((msb - msw) / den, 0, 1)) if den > 0 else float("nan")


def _between_share(cages: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Pooled between-mouse sum of squares over total, within cages (cage means removed)."""
    ssb = sst = 0.0
    for x, gi in cages:
        ni = np.bincount(gi)
        mi = np.bincount(gi, x) / ni
        ssb += float((ni * (mi - x.mean()) ** 2).sum())
        sst += float(((x - x.mean()) ** 2).sum())
    return ssb / sst if sst > 0 else float("nan")


def summarize(df: pd.DataFrame, signal: str, min_bouts: int, n_perm: int, rng,
              label: str | None = None) -> dict:
    d = df[["mouse", "cage", signal]].dropna()
    counts = d.groupby("mouse").size()
    d = d[d["mouse"].isin(counts[counts >= min_bouts].index)]
    mice_per_cage = d.groupby("cage")["mouse"].nunique()
    d = d[d["cage"].isin(mice_per_cage[mice_per_cage >= 2].index)]
    out = {"signal": label or signal, "bouts": len(d), "mice": d["mouse"].nunique(), "cages": d["cage"].nunique(),
           "cv_within_mouse": np.nan, "cv_within_cage": np.nan, "cage_over_mouse": np.nan,
           "mouse_share": np.nan, "p": np.nan, "note": ""}
    if d.empty:
        out["note"] = f"no cage with 2+ mice of {min_bouts}+ bouts"
        return out
    # A kinematic that goes negative anywhere has no CV at all -- not even for a mouse whose own bouts
    # happen to be positive -- so the two CVs never compare different subsets.
    signed = bool((d[signal] < 0).any())
    if signed:
        out["note"] = "signed: CV undefined"
    cv = (lambda x: float("nan")) if signed else _cv
    mouse_cv = d.groupby("mouse")[signal].apply(lambda s: cv(s.to_numpy(float)))
    cage_rows, cage_cvs, ratios, iccs = [], [], [], []
    for cage, c in d.groupby("cage"):
        x = c[signal].to_numpy(float)
        _, gi = np.unique(c["mouse"].to_numpy(), return_inverse=True)
        cage_rows.append((x, gi))
        cage_cvs.append(cv(x))
        ratios.append(cage_cvs[-1] / mouse_cv[c["mouse"].unique()].median())
        iccs.append(_icc(x, gi))
    with warnings.catch_warnings():                                # all-NaN medians -> NaN, quietly
        warnings.simplefilter("ignore", RuntimeWarning)
        out["cv_within_mouse"] = float(np.nanmedian(mouse_cv))
        out["cv_within_cage"] = float(np.nanmedian(cage_cvs))
        out["cage_over_mouse"] = float(np.nanmedian(ratios))
        out["mouse_share"] = float(np.nanmedian(iccs))
    if n_perm:
        obs = _between_share(cage_rows)
        hits = sum(_between_share([(x, rng.permutation(gi)) for x, gi in cage_rows]) >= obs - 1e-12
                   for _ in range(n_perm))
        out["p"] = (hits + 1) / (n_perm + 1)
    return out


# ----------------------------------------------------------------------------- report
def _f(x, nd=2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def report(beh: str, rows: list[dict], df: pd.DataFrame, min_bouts: int) -> str:
    b = df[df["behavior"] == beh]
    per_clip = [f"- {vid} ({cage}): " + " · ".join(f"track {t + 1}: {n}" for t, n in g.groupby("track").size().items())
                for (vid, cage), g in b.groupby(["video_id", "cage"])]
    head = (f"## {beh}\n\n{len(b)} bouts · {b['mouse'].nunique()} mice (clip × track) · "
            f"{b['cage'].nunique()} cage(s). Bouts per mouse (GUI track numbers):\n\n" + "\n".join(per_clip) + "\n")
    lines = [head, "| kinematic | bouts | mice | cages | CV within mouse | CV within cage | cage / mouse "
                   "| mouse share | p | note |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['signal']} | {r['bouts']} | {r['mice']} | {r['cages']} | {_f(r['cv_within_mouse'])} "
                     f"| {_f(r['cv_within_cage'])} | {_f(r['cage_over_mouse'])} | {_f(r['mouse_share'])} "
                     f"| {_f(r['p'], 3)} | {r['note']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Within-mouse vs within-cage variation of bout kinematics.")
    ap.add_argument("project", help="one labeler project folder, e.g. ~/laras-projects/hidralabel")
    ap.add_argument("--behaviors", help="comma-separated behavior names (default: every labeled one)")
    ap.add_argument("--bouts", help="CSV of bouts (export_bouts.py) instead of the labeled bouts")
    ap.add_argument("--signals", help=f"comma-separated base kinematics (default: {', '.join(DEFAULT_SIGNALS)})")
    ap.add_argument("--min-bouts", type=int, default=3, help="bouts a mouse needs to count (default 3)")
    ap.add_argument("--min-tracked", type=float, default=0.5,
                    help="drop bouts whose animal was tracked in fewer of their frames (default 0.5)")
    ap.add_argument("--cage", choices=("camera", "clip"), default="camera",
                    help="what a cage is: the camera in the clip id (default) or each clip")
    ap.add_argument("--permutations", type=int, default=2000, help="for p; 0 to skip (default 2000)")
    ap.add_argument("--baseline", action="store_true",
                    help="divide each bout's posture by the same mouse's median posture outside the "
                         "behavior, so differences in build cancel (speeds and duration unchanged)")
    ap.add_argument("--out-bouts", help="write the per-bout kinematics table here")
    ap.add_argument("--csv", help="write the summary table here")
    args = ap.parse_args(argv)

    proj = Path(args.project).expanduser()
    if not (proj / "project.json").exists():
        sys.exit(f"not a labeler project (no project.json): {proj}")
    manifest = json.loads((proj / "project.json").read_text(encoding="utf-8"))
    wanted = {_norm(s) for s in args.behaviors.split(",")} if args.behaviors else None
    signals = ([s.strip() for s in args.signals.split(",") if s.strip()] if args.signals
               else list(DEFAULT_SIGNALS))

    bouts = csv_bouts(Path(args.bouts), wanted) if args.bouts else labeled_bouts(proj, manifest, wanted)
    if bouts.empty:
        sys.exit("no bouts found for those behaviors")
    df = bout_kinematics(proj, manifest, bouts, signals, args.cage, args.min_tracked, args.baseline)
    if df.empty:
        sys.exit("no bouts with kinematics — is the feature cache built for these clips?")

    present = ["duration_s"] + [s for s in signals if s in df.columns and df[s].notna().any()]
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"# Bout kinematics: within mouse vs within cage\n\n`{proj}` · cage = {args.cage} · a mouse "
          f"needs {args.min_bouts}+ bouts · only cages with 2+ such mice are compared"
          + (" · posture ÷ each mouse's own median outside the behavior (`÷ own`)" if args.baseline else "")
          + "\n")

    def one(beh: str, s: str) -> dict:
        # A seed per (behavior, kinematic), so a p does not move with whichever others share the run.
        rng = np.random.default_rng([0, zlib.crc32(f"{_norm(beh)}|{s}".encode())])
        if args.baseline and s in POSTURE and f"{s}_rel" in df.columns:
            r = summarize(df[df["behavior"] == beh], f"{s}_rel", args.min_bouts, args.permutations, rng, s)
            r["note"] = "; ".join(x for x in ("÷ own", r["note"]) if x)
            return r
        return summarize(df[df["behavior"] == beh], s, args.min_bouts, args.permutations, rng)

    summary = []
    for beh in sorted(df["behavior"].unique(), key=_norm):
        rows = [one(beh, s) for s in present]
        print(report(beh, rows, df, args.min_bouts))
        summary += [{"behavior": beh, **r} for r in rows]
    print("cage / mouse ~1 and mouse share ~0: cage-mates are as alike as one mouse is to itself. "
          "cage / mouse > 1 with a high mouse share and small p: the mice of a cage differ.")
    if args.out_bouts:
        df.to_csv(args.out_bouts, index=False)
        print(f"wrote {args.out_bouts}", file=sys.stderr)
    if args.csv:
        pd.DataFrame(summary).to_csv(args.csv, index=False)
        print(f"wrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
