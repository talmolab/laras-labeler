#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "pyarrow", "matplotlib", "scikit-learn", "sleap-io",
#                 "opencv-python-headless"]
# ///
"""Where in the cage each behavior happens -- to see which behaviors split by place (at the feeder,
at the water, against a wall, out in the open) and so may want to be separate sub-behaviors.

One row per clip (each camera has its own view), one column per behavior. Each panel is that
clip's cage with the mice removed (the median of frames spread across the clip), and on it every
bout: a dot where the mouse's nose mostly was during the bout, a faint line for the path it took,
colored by mouse. Bouts within --radius body lengths of each other are circled as one PLACE
(A, B, ...), so the number of places per behavior is a first count of how many location-defined
variants it has; bouts far from every other are left uncircled (scattered).

    uv run scripts/behavior_places.py ~/laras-projects/hidralabel --out places.png
    uv run scripts/behavior_places.py ~/laras-projects/hidralabel --behaviors eating,"climb up" --radius 1 --csv places.csv

When a clip carries its water (spout) or cage-outline ROI -- the labeler fills these from the HCM
database -- the outlines are drawn and the table gives each place's distance to the water and to
the wall in body lengths. The feeder has no ROI in the labeler, so it is read off the image.

Places are a heuristic grouping (DBSCAN on nose positions, radius in body lengths), a starting
point for naming sub-behaviors by eye, not a definition. Needs the clips' .slp files, and their
videos for the background (without them the panels are blank but the bouts still plot).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

NOSE = ("nose", "snout", "nose_tip", "head")
TAIL_BASE = ("tail_base", "tailbase", "tti", "tail_start", "tailtorso", "tail_torso")
CAM_RE = re.compile(r"cam[_-]?0*(\d+)", re.I)
# "where" tags in the table; distances in body lengths from the place's center
AT_WATER_BL, AT_WALL_BL = 1.0, 0.75

# Reference palette (dataviz skill) slots 1-3, validated all-pairs: one color per mouse (HCM cages
# hold 1-3). Each dot gets a white ring so it reads on the cage image.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
MOUSE = ["#2a78d6", "#eb6834", "#1baf7a"]


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _runs(frames: np.ndarray) -> list[tuple[int, int]]:
    if len(frames) == 0:
        return []
    breaks = np.flatnonzero(np.diff(frames) > 1)
    return list(zip(np.r_[frames[0], frames[breaks + 1]].tolist(), (np.r_[frames[breaks], frames[-1]] + 1).tolist()))


def labeled_bouts(proj: Path, manifest: dict, wanted: set[str] | None) -> pd.DataFrame:
    """Contiguous Happening runs per (clip, behavior, track) from the project's labels."""
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


# ----------------------------------------------------------------------------- clip data
def _find(names: list[str], cands) -> int | None:
    low = {n.lower(): i for i, n in enumerate(names)}
    return next((low[c] for c in cands if c in low), None)


def load_poses(slp: str):
    """-> (nose-or-centroid (F, T, 2), body length in px, which point was used)."""
    import sleap_io as sio
    lab = sio.load_slp(slp)
    P = lab.numpy()                                               # (F, T, N, 2), NaN where missing
    names = list(lab.skeletons[0].node_names)
    ni, ti = _find(names, NOSE), _find(names, TAIL_BASE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        point = P[:, :, ni, :] if ni is not None else np.nanmean(P, axis=2)
        if ni is not None and ti is not None:
            bl = float(np.nanmedian(np.linalg.norm(P[:, :, ni] - P[:, :, ti], axis=-1)))
        else:                                                     # extent of the skeleton instead
            bl = float(np.nanmedian(np.nanmax(P[..., 0], 2) - np.nanmin(P[..., 0], 2)))
    return point, (bl if np.isfinite(bl) and bl > 0 else 40.0), names[ni] if ni is not None else "centroid"


def background(path: str | None, n: int = 15) -> np.ndarray | None:
    """Median of n frames spread over the clip: the cage with the (moving) mice removed."""
    if not path or not Path(path).exists():
        return None
    try:
        import sleap_io as sio
        vid = sio.load_video(path)
        idx = np.linspace(0, vid.shape[0] - 1, n).astype(int)
        frames = [np.asarray(vid[int(i)], dtype=np.float32) for i in idx]
        frames = [f.mean(axis=-1) if f.ndim == 3 else f for f in frames]
        return np.median(np.stack(frames), axis=0)
    except Exception as e:                                        # a background is optional
        print(f"  no background for {Path(path).name}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _edge_dist(p: np.ndarray, poly) -> float:
    """Distance from point p to the nearest edge of a polygon."""
    A = np.asarray(poly, float)
    B = np.roll(A, -1, axis=0)
    AB = B - A
    t = np.clip(((p - A) * AB).sum(1) / np.maximum((AB ** 2).sum(1), 1e-9), 0, 1)
    return float(np.min(np.linalg.norm(A + t[:, None] * AB - p, axis=1)))


def _inside(p: np.ndarray, poly) -> bool:
    from matplotlib.path import Path as MPath
    return bool(MPath(np.asarray(poly, float)).contains_point(p))


# ----------------------------------------------------------------------------- places
def places(xy: np.ndarray, radius_px: float) -> np.ndarray:
    """Group labels (0, 1, ... by size; -1 = scattered) for bout locations."""
    from sklearn.cluster import DBSCAN
    if len(xy) < 2:
        return np.full(len(xy), -1)
    lab = DBSCAN(eps=radius_px, min_samples=2).fit_predict(xy)
    order = [g for g, _ in sorted(((g, (lab == g).sum()) for g in set(lab) if g >= 0), key=lambda x: -x[1])]
    remap = {g: i for i, g in enumerate(order)}
    return np.array([remap.get(g, -1) for g in lab])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Where in the cage each behavior happens, per clip.")
    ap.add_argument("project", help="one labeler project folder, e.g. ~/laras-projects/hidralabel")
    ap.add_argument("--behaviors", help="comma-separated behavior names (default: every labeled one)")
    ap.add_argument("--bouts", help="CSV of bouts (export_bouts.py) instead of the labeled bouts")
    ap.add_argument("--radius", type=float, default=1.5,
                    help="bouts within this many body lengths of each other are one place (default 1.5)")
    ap.add_argument("--out", default="behavior_places.png", help="figure file (.png, .pdf or .svg)")
    ap.add_argument("--csv", help="also write one row per bout (with its place) here")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proj = Path(args.project).expanduser()
    if not (proj / "project.json").exists():
        sys.exit(f"not a labeler project (no project.json): {proj}")
    manifest = json.loads((proj / "project.json").read_text(encoding="utf-8"))
    wanted = {_norm(s) for s in args.behaviors.split(",")} if args.behaviors else None
    if args.bouts:
        bouts = pd.read_csv(args.bouts)
        bouts["behavior"] = bouts["behavior"].astype(str).str.strip()
        if wanted:
            bouts = bouts[bouts["behavior"].map(_norm).isin(wanted)]
    else:
        bouts = labeled_bouts(proj, manifest, wanted)
    if bouts.empty:
        sys.exit("no bouts found for those behaviors")
    videos = {v["video_id"]: v for v in manifest.get("videos", [])}
    fc = manifest.get("feature_config") or {}

    # --- per bout: where the nose mostly was, and the path ------------------------------------
    clips, rows = {}, []
    for vid, g in bouts.groupby("video_id"):
        v = videos.get(vid, {})
        slp = v.get("slp_path")
        if not slp or not Path(slp).exists():
            print(f"  skipped {vid}: its .slp is not reachable here ({slp})", file=sys.stderr)
            continue
        point, bl, used = load_poses(slp)
        bg = background(v.get("video_path") or v.get("playback_path"))
        water, cage = v.get("spout_roi") or fc.get("spout_roi"), v.get("cage_roi") or fc.get("cage_roi")
        wh = (bg.shape[1], bg.shape[0]) if bg is not None else (v.get("width") or 1280, v.get("height") or 1024)
        clips[vid] = {"bg": bg, "bl": bl, "water": water, "cage": cage, "used": used, "wh": wh}
        for b in g.itertuples(index=False):
            s, e, t = int(b.start_frame), int(b.end_frame), int(b.track)
            if t >= point.shape[1] or s >= point.shape[0]:
                continue
            path = point[s:min(e, point.shape[0]), t]
            path = path[np.isfinite(path).all(axis=1)]
            if len(path) == 0:
                continue
            rows.append({"video_id": vid, "behavior": b.behavior, "track": t, "start_frame": s,
                         "end_frame": e, "x": float(np.median(path[:, 0])), "y": float(np.median(path[:, 1])),
                         "path_xy": path})
    if not rows:
        sys.exit("no bout had a tracked nose — are the .slp files reachable from this machine?")
    d = pd.DataFrame(rows)

    # --- places -----------------------------------------------------------------------------
    d["place"] = -1
    for (vid, beh), g in d.groupby(["video_id", "behavior"]):
        d.loc[g.index, "place"] = places(g[["x", "y"]].to_numpy(), args.radius * clips[vid]["bl"])
    table = []
    for (vid, beh, pl), g in d[d["place"] >= 0].groupby(["video_id", "behavior", "place"]):
        c, bl = clips[vid], clips[vid]["bl"]
        center = g[["x", "y"]].mean().to_numpy()
        to_water = (0.0 if _inside(center, c["water"]) else _edge_dist(center, c["water"]) / bl) if c["water"] else np.nan
        to_wall = _edge_dist(center, c["cage"]) / bl if c["cage"] else np.nan
        where = ("at water" if to_water <= AT_WATER_BL else "at wall" if to_wall <= AT_WALL_BL
                 else "open floor" if np.isfinite(to_wall) else "—")
        table.append({"clip": vid, "behavior": beh, "place": chr(65 + int(pl)), "bouts": len(g),
                      "mice": " ".join(str(t + 1) for t in sorted(g["track"].unique())),
                      "x": round(center[0]), "y": round(center[1]),
                      "to_water_bl": round(to_water, 2) if np.isfinite(to_water) else np.nan,
                      "to_wall_bl": round(to_wall, 2) if np.isfinite(to_wall) else np.nan, "where": where})
    table = pd.DataFrame(table)

    # --- figure ---------------------------------------------------------------------------------
    behs = sorted(d["behavior"].unique(), key=str.lower)
    vids = [v for v in clips if v in set(d["video_id"])]
    W, H = (float(x) for x in clips[vids[0]]["wh"])
    pw = 3.4
    fig, axes = plt.subplots(len(vids), len(behs), squeeze=False, facecolor=SURFACE,
                             figsize=(pw * len(behs) + 0.6, pw * H / W * len(vids) + 1.4))
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8})
    for r, vid in enumerate(vids):
        c = clips[vid]
        for col, beh in enumerate(behs):
            ax = axes[r][col]
            ax.set_facecolor("#f0efec")
            w, h = c["wh"]
            if c["bg"] is not None:
                ax.imshow(c["bg"], cmap="gray", extent=(0, w, h, 0), alpha=0.6, zorder=0)
            for poly, name in ((c["cage"], "cage"), (c["water"], "water")):
                if poly:
                    p = np.asarray(poly, float)
                    ax.plot(*np.vstack([p, p[:1]]).T, color=INK2, lw=0.9, zorder=1)
                    if name == "water":
                        # below the outline: place labels sit at a circle's upper right
                        ax.text(p[:, 0].mean(), p[:, 1].max() + 0.25 * c["bl"], "water", ha="center",
                                va="top", fontsize=7, color=INK2, zorder=5)
            g = d[(d["video_id"] == vid) & (d["behavior"] == beh)]
            for row in g.itertuples(index=False):
                color = MOUSE[row.track] if row.track < len(MOUSE) else MUTED
                ax.plot(row.path_xy[:, 0], row.path_xy[:, 1], color=color, lw=0.8, alpha=0.4, zorder=2)
            for t, gt in g.groupby("track"):
                ax.scatter(gt["x"], gt["y"], s=34, color=MOUSE[t] if t < len(MOUSE) else MUTED,
                           edgecolor="white", linewidth=1.2, zorder=4)
            n_places = int(g["place"].max() + 1) if len(g) and g["place"].max() >= 0 else 0
            for pl, gp in g[g["place"] >= 0].groupby("place"):
                ctr = gp[["x", "y"]].mean().to_numpy()
                rad = float(np.max(np.hypot(gp["x"] - ctr[0], gp["y"] - ctr[1]))) + 0.5 * c["bl"]
                ax.add_patch(plt.Circle(ctr, rad, fill=False, color=INK, lw=1.0, zorder=3))
                ax.text(ctr[0] + rad * 0.72, ctr[1] - rad * 0.72, f"{chr(65 + int(pl))} ({len(gp)})",
                        fontsize=8, color=INK, fontweight="bold", zorder=5,
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8})
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            scattered = int((g["place"] < 0).sum())
            sub = (f"{len(g)} bouts · {n_places} place{'s' * (n_places != 1)}"
                   + (f" · {scattered} scattered" if scattered else "")) if len(g) else "no bouts"
            ax.set_title(f"{beh}\n", fontsize=10, color=INK, fontweight="bold", loc="left", pad=2)
            ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
            if col == 0:
                m = CAM_RE.search(vid)
                ax.set_ylabel((f"cam_{int(m.group(1)):02d}\n" if m else "") + vid.split(".predictions")[0][:28],
                              fontsize=8, color=INK2)
    tracks = sorted(d["track"].unique())
    handles = [plt.Line2D([], [], ls="", marker="o", ms=7, mec="white", color=MOUSE[t] if t < len(MOUSE) else MUTED,
                          label=f"mouse (track {t + 1})") for t in tracks]
    handles.append(plt.Line2D([], [], ls="", marker="o", ms=11, mfc="none", mec=INK,
                              label=f"place: bouts within {args.radius:g} body lengths of each other"))
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False, fontsize=8,
               bbox_to_anchor=(0.005, 1.0), labelcolor=INK2)
    fig.text(0.005, 0.004, f"Dot = where the {clips[vids[0]]['used']} mostly was during a bout (median); line = "
             "its path. Background = median of frames across the clip (mice removed). A mouse is a track "
             "within one clip.", fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}", file=sys.stderr)

    # --- table ------------------------------------------------------------------------------------
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("| clip | behavior | place | bouts | mice | to water (BL) | to wall (BL) | where |\n|---|---|---|---|---|---|---|---|")
    for r in table.itertuples(index=False):
        f = lambda x: "—" if pd.isna(x) else f"{x:.1f}"           # noqa: E731
        print(f"| {r.clip.split('.predictions')[0][:24]} | {r.behavior} | {r.place} | {r.bouts} | {r.mice} | "
              f"{f(r.to_water_bl)} | {f(r.to_wall_bl)} | {r.where} |")
    for (vid, beh), g in d.groupby(["video_id", "behavior"]):
        n_sc = int((g["place"] < 0).sum())
        if n_sc:
            print(f"  {vid.split('.predictions')[0][:24]} · {beh}: {n_sc} scattered bout(s), no place")
    if args.csv:
        d.drop(columns="path_xy").assign(place=d["place"].map(lambda p: chr(65 + int(p)) if p >= 0 else "")) \
            .to_csv(args.csv, index=False)
        print(f"wrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
