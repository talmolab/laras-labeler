#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Figure: how much a behavior's kinematics differ bout to bout, mouse to mouse, and cage to cage.

Reads the per-bout table `bout_kinematics_cv.py --out-bouts` writes. One row of panels per behavior:

  left    for every kinematic, three spreads on one CV axis --
            within mouse   bout-to-bout CV inside one mouse (median over mice)
            across mice    CV of the mouse means inside one cage (median over cages)
            across cages   CV of the cage means (needs 2+ cages)
          and, as a tick in the same color, what the across-mice / across-cages spread would be from
          bout-to-bout noise alone if mice (or cages) did not differ at all. A dot near its tick is
          noise; a dot well right of it is a real difference.
  right   the bouts themselves for a few kinematics, grouped cage -> mouse: blue dots are bouts,
          orange bars mouse means, aqua lines cage means -- the same three spreads, drawn.

    uv run scripts/plot_mouse_vs_cage.py bouts_rel.csv --baseline --out variation.png
    uv run scripts/plot_mouse_vs_cage.py bouts.csv --show duration_s,centroid_speed,ego_width --out raw.pdf

--baseline plots posture as `<posture>_rel` (each bout over the same mouse's own median outside the
behavior), which --out-bouts writes when bout_kinematics_cv.py ran with --baseline. --csv writes
the numbers behind the left panels. A mouse is a (clip, track) and a cage is a camera, as in
bout_kinematics_cv.py; with one clip per cage, cage-to-cage also carries day and clip differences.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

MOVEMENT = ["duration_s", "centroid_speed", "nose_speed", "tail_speed", "abs_turn", "accel_mag"]
POSTURE = ["body_length", "body_bend", "ego_len", "ego_width", "elongation", "spread_area", "spine_len",
           "haunch_width", "shoulder_width", "ear_width"]
DEFAULT_SHOW = ["duration_s", "centroid_speed", "ego_width", "body_bend"]

# Reference palette (dataviz skill), light mode; slots 1-3 validated all-pairs. Aqua is under 3:1 on
# the surface, so each spread also has its own marker shape and a labeled legend.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SPREADS = [  # key, label, color, marker
    ("within_mouse", "within mouse (bout to bout)", "#2a78d6", "o"),
    ("across_mice", "across mice in a cage", "#eb6834", "s"),
    ("across_cages", "across cages", "#1baf7a", "D"),
]


def _cv(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.std(ddof=1) / s.mean()) if len(s) >= 2 and s.mean() > 0 else np.nan


def spreads(b: pd.DataFrame, col: str, min_bouts: int) -> dict:
    """The three spreads (and their noise floors) for one behavior x kinematic."""
    d = b[["mouse", "cage", col]].dropna()
    n = d.groupby("mouse").size()
    d = d[d["mouse"].isin(n[n >= min_bouts].index)]
    out = {"bouts": len(d), "mice": d["mouse"].nunique(), "cages": d["cage"].nunique()}
    if d.empty or (d[col] < 0).any():                 # empty, or signed: CV has no meaning
        return {**out, **{k: np.nan for k in ("within_mouse", "across_mice", "across_cages",
                                              "floor_mice", "floor_cages")}}
    g = d.groupby("mouse")[col]
    mice = pd.DataFrame({"cage": d.groupby("mouse")["cage"].first(), "mean": g.mean(),
                         "cv": g.apply(_cv), "n": g.size()})
    by_cage = mice.groupby("cage")["mean"]
    across_mice = [_cv(s) for _, s in by_cage if len(s) >= 2]
    cage_means = by_cage.mean()
    out["within_mouse"] = float(mice["cv"].median())
    out["across_mice"] = float(np.nanmedian(across_mice)) if across_mice else np.nan
    out["across_cages"] = _cv(cage_means) if len(cage_means) >= 2 else np.nan
    # If mice did not differ, a mouse mean would still wobble by its own CV / sqrt(its bouts); if
    # cages did not differ, a cage mean would wobble by the across-mice spread / sqrt(its mice).
    out["floor_mice"] = float((mice["cv"] / np.sqrt(mice["n"])).median())
    out["floor_cages"] = (out["across_mice"] / np.sqrt(by_cage.size().median())
                          if len(cage_means) >= 2 else np.nan)
    return out


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8, length=0)


def dot_panel(ax, table: pd.DataFrame, kins: list[str], label, lims: tuple[float, float]) -> None:
    y = np.arange(len(kins))[::-1]
    for yi in y:
        ax.axhline(yi, color=GRID, lw=0.8, zorder=0)
    rows = table.set_index("kinematic").reindex(kins)
    for key, name, color, marker in SPREADS:
        ax.scatter(rows[key], y, s=46, marker=marker, color=color, edgecolor=SURFACE, linewidth=1.5,
                   zorder=3, label=name)
    for key, floor in (("across_mice", "floor_mice"), ("across_cages", "floor_cages")):
        color = next(c for k, _, c, _ in SPREADS if k == key)
        ax.scatter(rows[floor], y, s=90, marker="|", color=color, alpha=0.55, linewidth=1.6, zorder=2)
    split = sum(k in MOVEMENT for k in kins)                  # movement rows above, posture below
    if 0 < split < len(kins):
        ax.axhline(len(kins) - split - 0.5, color="#c3c2b7", lw=1, zorder=1)
    ax.set_yticks(y, [label(k) for k in kins])
    # Log: what matters is each dot's RATIO to its noise tick, and a log axis draws equal ratios as
    # equal gaps on every row -- posture CVs (~0.01-0.1) are otherwise crushed against zero by the
    # movement CVs (~0.5).
    ax.set_xscale("log")
    lo, hi = lims                                             # shared by every behavior's row
    ax.set_xlim(lo, hi)
    ticks = [t for t in (0.003, 0.01, 0.03, 0.1, 0.3, 1, 3) if lo <= t <= hi]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("CV (SD / mean), log scale", color=INK2, fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)


def strip_panel(ax, b: pd.DataFrame, col: str, title: str, min_bouts: int, rng) -> None:
    d = b[["mouse", "cage", "track", col]].dropna()
    n = d.groupby("mouse").size()
    d = d[d["mouse"].isin(n[n >= min_bouts].index)]
    x, ticks, labels = 0.0, [], []
    for cage, c in d.groupby("cage", sort=True):
        start = x
        for mouse, m in c.groupby("mouse", sort=True):
            v = m[col].to_numpy(float)
            ax.scatter(x + rng.uniform(-0.17, 0.17, len(v)), v, s=22, color=SPREADS[0][2], alpha=0.75,
                       edgecolor=SURFACE, linewidth=0.8, zorder=2)
            ax.hlines(v.mean(), x - 0.3, x + 0.3, color=SPREADS[1][2], lw=2.6, zorder=3)
            ticks.append(x)
            labels.append(f"{int(m['track'].iloc[0]) + 1}")
            x += 1
        mean_of_mice = c.groupby("mouse")[col].mean().mean()
        ax.hlines(mean_of_mice, start - 0.45, x - 0.55, color=SPREADS[2][2], lw=2, zorder=4)
        ax.text((start + x - 1) / 2, -0.075, cage, transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=8, color=INK2)
        x += 0.8                                                    # gap between cages
    ax.set_xticks(ticks, labels)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Bout-to-bout vs mouse-to-mouse vs cage-to-cage spread.")
    ap.add_argument("bouts", help="per-bout CSV from bout_kinematics_cv.py --out-bouts")
    ap.add_argument("--out", default="mouse_vs_cage.png", help="figure file (.png, .pdf or .svg)")
    ap.add_argument("--baseline", action="store_true",
                    help="plot posture relative to each mouse's own baseline (the *_rel columns)")
    ap.add_argument("--show", help=f"kinematics for the bout panels (default: {','.join(DEFAULT_SHOW)})")
    ap.add_argument("--behaviors", help="comma-separated behaviors (default: all in the CSV)")
    ap.add_argument("--min-bouts", type=int, default=3, help="bouts a mouse needs to count (default 3)")
    ap.add_argument("--csv", help="also write the numbers behind the left panels here")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = pd.read_csv(args.bouts)
    b["behavior"] = b["behavior"].astype(str).str.strip()
    if args.behaviors:
        want = {s.strip().lower() for s in args.behaviors.split(",")}
        b = b[b["behavior"].str.lower().isin(want)]
    if args.baseline and not any(c.endswith("_rel") for c in b.columns):
        sys.exit(f"{args.bouts} has no *_rel columns: run bout_kinematics_cv.py with --baseline --out-bouts")

    def column(k: str) -> str:
        return f"{k}_rel" if args.baseline and k in POSTURE and f"{k}_rel" in b.columns else k

    def label(k: str) -> str:
        return k.replace("_s", " (s)") if k == "duration_s" else k + (" ÷ own" if column(k) != k else "")

    kins = [k for k in MOVEMENT + POSTURE if column(k) in b.columns and b[column(k)].notna().any()]
    show = [k.strip() for k in (args.show.split(",") if args.show else DEFAULT_SHOW) if k.strip() in kins]
    rows, keep = [], []
    for beh, g in sorted(b.groupby("behavior"), key=lambda x: x[0].lower()):
        t = pd.DataFrame([{"behavior": beh, "kinematic": k, **spreads(g, column(k), args.min_bouts)}
                          for k in kins])
        if t["across_mice"].notna().any():          # needs a cage with 2+ mice of enough bouts
            rows.append(t)
            keep.append(beh)
        else:
            print(f"  skipped {beh}: no cage with 2+ mice of {args.min_bouts}+ bouts", file=sys.stderr)
    if not keep:
        sys.exit("nothing to plot")
    table = pd.concat(rows, ignore_index=True)
    vals = table[["within_mouse", "across_mice", "across_cages", "floor_mice", "floor_cages"]].to_numpy(float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    lims = (vals.min() / 1.5, vals.max() * 1.5) if vals.size else (0.01, 1.0)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8})
    fig, axes = plt.subplots(len(keep), 1 + len(show), squeeze=False, facecolor=SURFACE,
                             figsize=(4.6 + 2.5 * len(show), (0.27 * len(kins) + 1.3) * len(keep) + 1.2),
                             gridspec_kw={"width_ratios": [1.7] + [1] * len(show)})
    rng = np.random.default_rng(0)
    for r, beh in enumerate(keep):
        t, g = table[table["behavior"] == beh], b[b["behavior"] == beh]
        dot_panel(axes[r][0], t, kins, label, lims)
        cages = int(t["cages"].max())
        axes[r][0].set_title(beh, fontsize=10, color=INK, loc="left", fontweight="bold", pad=16)
        axes[r][0].text(0, 1.01, f"{int(t['bouts'].max())} bouts · {int(t['mice'].max())} mice · {cages} cage"
                        + ("s" if cages != 1 else " (no cage-to-cage spread)"),
                        transform=axes[r][0].transAxes, fontsize=8, color=INK2, va="bottom")
        for c, k in enumerate(show, start=1):
            strip_panel(axes[r][c], g, column(k), label(k), args.min_bouts, rng)

    handles = [plt.Line2D([], [], ls="", marker=m, ms=7, color=c, label=name) for _, name, c, m in SPREADS]
    handles += [plt.Line2D([], [], ls="", marker="|", ms=9, mew=1.6, color=MUTED,
                           label="tick: that spread if mice / cages did not differ (bout noise alone)")]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.01, 1.0), labelcolor=INK2)
    one_clip = b.groupby("cage")["video_id"].nunique().max() == 1
    note = ("Right panels: dots = bouts, orange bars = mouse means, aqua lines = cage means; x labels are "
            "GUI track numbers. A mouse is one track in one clip; a cage is a camera"
            + (" (here one clip each, so cage-to-cage also includes day and clip)." if one_clip else "."))
    if args.baseline:
        note += "  Posture (÷ own) is each bout over that mouse's own median outside the behavior."
    fig.text(0.01, 0.005, note, fontsize=7.5, color=MUTED, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}", file=sys.stderr)

    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}", file=sys.stderr)
    with pd.option_context("display.width", 140, "display.max_rows", 200):
        cols = ["behavior", "kinematic", "within_mouse", "across_mice", "floor_mice", "across_cages", "floor_cages"]
        print(table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
