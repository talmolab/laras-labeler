#!/usr/bin/env python3
"""Per-video performance of a HiDRA-bound behavior, scored against the project's own labels.

Only frames you labeled or reviewed count as ground truth. Those frames are not a random sample.
They cluster where you or the review queue looked. Precision is reliable on every video. Recall
and ROC are only reliable on videos with high label coverage, so check coverage_pct first.

Outputs (in --out, default <project>/analysis/<timestamp>/):
  roc_curves.png            ROC per video
  pr_curves.png             precision-recall per video, with the live threshold marked
  operating_point.png       precision and recall at the live threshold
  precision_over_time.png   rolling review precision with fine-tune times marked. Use --since to
                            leave out old review rounds.
  summary.csv               per-video and pooled metrics

    python lab_scripts/characterize_approach_head.py ~/my-projects/approach_defs --behavior 0 --track 1

Needs pandas/numpy/sklearn/matplotlib (the labeler's own venv has all four).
"""
import argparse
import glob
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

# Fixed color order. Colors follow the video, not its rank.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD = "#0ca30c"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "system-ui, -apple-system, Helvetica, Arial, sans-serif",
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "legend.frameon": False,
})


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def bout_runs(mask: np.ndarray):
    """Maximal contiguous True runs -> [(s, e), ...], e exclusive."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def predicted_bouts(proba: np.ndarray, smooth=5, hi=0.6, lo=0.6, min_bout=25, min_gap=3):
    """Same smoothing/hysteresis the labeler itself uses to turn proba into bouts (predict.py)."""
    p = pd.Series(proba).rolling(max(1, smooth), center=True, min_periods=1).mean().to_numpy()
    p = np.nan_to_num(p, nan=0.0)
    F = len(p)
    above_lo, above_hi = p >= lo, p >= hi
    d = np.diff(above_lo.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if above_lo[0]:
        starts = [0] + starts
    if above_lo[-1]:
        ends = ends + [F]
    bouts = [(s, e) for s, e in zip(starts, ends) if above_hi[s:e].any() and e - s >= min_bout]
    merged = []
    for s, e in bouts:
        if merged and s - merged[-1][1] < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(int(s), int(e)) for s, e in merged]


def video_metrics(proj: Path, vid: str, bid: int, track: int, thr: float, postproc: dict):
    lab = pd.read_parquet(proj / "labels" / f"{vid}.parquet")
    lab = lab[(lab.track == track) & (lab.behavior_id == bid) & lab.value.isin([0, 1])]
    if lab.empty:
        return None
    proba = np.load(proj / "predictions" / vid / f"{bid}.npy")[:, track]
    frames = lab.frame.to_numpy()
    y_true = lab.value.to_numpy().astype(int)
    y_score = proba[frames]

    span = int(frames.max() - frames.min()) if len(frames) else 0
    coverage = len(frames) / span * 100 if span else 0.0

    y_pred = (y_score >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    accuracy = (tp + tn) / len(y_true)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_score) if y_true.min() != y_true.max() else (None, None, None)
    auroc = roc_auc_score(y_true, y_score) if y_true.min() != y_true.max() else float("nan")
    pr_p, pr_r, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # Bout recall: share of labeled positive runs that any detected bout touches.
    pos_mask = np.zeros(int(frames.max()) + 1, dtype=bool)
    pos_mask[frames[y_true == 1]] = True
    true_bouts = bout_runs(pos_mask)
    pred = predicted_bouts(proba, **postproc)
    hit = 0
    for s, e in true_bouts:
        if any(ps < e and pe > s for ps, pe in pred):
            hit += 1
    bout_recall = hit / len(true_bouts) if true_bouts else float("nan")

    return dict(video_id=vid, n_pos=int((y_true == 1).sum()), n_neg=int((y_true == 0).sum()),
                coverage_pct=coverage, accuracy=accuracy, precision=precision, recall=recall,
                specificity=specificity, f1=f1, auroc=auroc, ap=ap, n_true_bouts=len(true_bouts),
                bout_recall=bout_recall, tp=tp, fp=fp, tn=tn, fn=fn,
                fpr=fpr, tpr=tpr, pr_p=pr_p, pr_r=pr_r, y_true=y_true, y_score=y_score)


def load_events(proj: Path):
    ev = []
    for f in sorted(glob.glob(str(proj / "events" / "*.jsonl"))):
        for line in open(f):
            try:
                ev.append(json.loads(line))
            except ValueError:
                pass
    return ev


def small_multiples(n, figsize_per=(3.1, 2.7)):
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for a in axes[n:]:
        a.axis("off")
    return fig, axes[:n]


def plot_roc(results, out):
    fig, axes = small_multiples(len(results))
    for ax, r in zip(axes, results):
        if r["fpr"] is None:
            ax.text(0.5, 0.5, "only one class\nlabeled here", ha="center", va="center",
                    fontsize=9, color=MUTED, transform=ax.transAxes)
        else:
            ax.plot(r["fpr"], r["tpr"], color=CAT[0], lw=2, solid_capstyle="round")
            ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls=(0, (4, 3)), zorder=0)
            ax.fill_between(r["fpr"], r["tpr"], step=None, color=CAT[0], alpha=0.08, lw=0)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.grid(True, lw=0.6, alpha=0.6)
        auroc = r["auroc"]
        ax.set_title(f"{r['video_id']}\nAUROC {auroc:.2f}" if auroc == auroc else f"{r['video_id']}\nAUROC n/a",
                     fontsize=10, pad=6)
        ax.set_xlabel("false positive rate", fontsize=8.5)
        ax.set_ylabel("true positive rate", fontsize=8.5)
    fig.suptitle("ROC per video, current fine-tuned head (approach · LyricalHare)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.6, w_pad=1.8)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pr(results, thr, out):
    fig, axes = small_multiples(len(results))
    for ax, r in zip(axes, results):
        base = r["n_pos"] / (r["n_pos"] + r["n_neg"]) if (r["n_pos"] + r["n_neg"]) else float("nan")
        ax.plot(r["pr_r"], r["pr_p"], color=CAT[0], lw=2, solid_capstyle="round")
        ax.axhline(base, color=GRID, lw=1.5, ls=(0, (4, 3)), zorder=0)
        ax.plot(r["recall"], r["precision"], marker="o", ms=6, color=INK, mec=SURFACE, mew=1,
                zorder=5, label=f"operating pt. (thr={thr:.2f})")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.grid(True, lw=0.6, alpha=0.6)
        ax.set_title(f"{r['video_id']}\nAP {r['ap']:.2f}", fontsize=10, pad=6)
        ax.set_xlabel("recall", fontsize=8.5)
        ax.set_ylabel("precision", fontsize=8.5)
    fig.suptitle(f"Precision-recall per video\nBlack dot: the live threshold ({thr:.2f}). "
                 "Dashed line: the positive rate, what chance would get.",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.91], h_pad=2.6, w_pad=1.8)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_operating_point(results, thr, out):
    """One row per video. A P-R scatter put several videos on top of each other."""
    order = sorted(results, key=lambda r: r["precision"])
    pooled_p = sum(r["tp"] for r in results) / max(1, sum(r["tp"] + r["fp"] for r in results))
    pooled_r = sum(r["tp"] for r in results) / max(1, sum(r["tp"] + r["fn"] for r in results))
    rows = order + [dict(video_id="POOLED", precision=pooled_p, recall=pooled_r)]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.5, 0.56 * len(rows) + 2.2))
    for yi, r in zip(y, rows):
        ax.plot([r["precision"], r["recall"]], [yi, yi], color=GRID, lw=2, zorder=1,
               solid_capstyle="round")
    ax.scatter([r["precision"] for r in rows], y, s=90, color=CAT[0], zorder=5,
              edgecolor=SURFACE, linewidth=1, label="precision")
    ax.scatter([r["recall"] for r in rows], y, s=90, color=CAT[1], zorder=5,
              edgecolor=SURFACE, linewidth=1, label="recall")
    ax.axhline(len(order) - 0.5, color=BASE, lw=1, ls=(0, (2, 2)))
    ax.set_yticks(y, [r["video_id"] for r in rows])
    for lbl, r in zip(ax.get_yticklabels(), rows):
        if r["video_id"] == "POOLED":
            lbl.set_fontweight("bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("value at the live detection threshold")
    ax.grid(True, axis="x", lw=0.6, alpha=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_title(f"Precision vs. recall by video (threshold ≥ {thr:.2f})",
                fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_precision_over_time(proj: Path, bid: int, track: int, video_order: list, out, since=None):
    local = datetime.now().astimezone().tzinfo
    since = pd.Timestamp(since).tz_localize(local) if since else None

    def to_local(ts):   # event times are stored in UTC
        t = pd.to_datetime(ts, utc=True)
        return t.dt.tz_convert(local) if isinstance(t, pd.Series) else t.tz_convert(local)

    ev = load_events(proj)
    dec = [e for e in ev if e.get("type") in ("candidate_accept", "candidate_reject")
          and e.get("behavior_id") == bid and e.get("track") == track and e.get("t_srv")]
    trains = [to_local(e["t_srv"]) for e in ev if e.get("type") == "job_done" and e.get("kind") == "train"
              and e.get("status") == "done"]
    if since is not None:
        dec = [e for e in dec if to_local(e["t_srv"]) >= since]
        trains = [t for t in trains if t >= since]
    if not dec:
        return False
    trains = sorted(t.tz_localize(None) for t in trains)
    df = pd.DataFrame(dec)
    df["t"] = to_local(df["t_srv"]).dt.tz_localize(None)
    df = df.sort_values("t").reset_index(drop=True)
    df["accept"] = (df["type"] == "candidate_accept").astype(int)
    win = max(10, min(30, len(df) // 6 or 10))
    df["rolling"] = df["accept"].rolling(win, min_periods=max(5, win // 3)).mean()

    fig, ax = plt.subplots(figsize=(9.5, 5))
    vid_color = {v: CAT[i % len(CAT)] for i, v in enumerate(video_order)}
    for vid, g in df.groupby("video_id"):
        ax.scatter(g["t"], g["accept"] + np.random.uniform(-0.03, 0.03, len(g)),
                  s=14, color=vid_color.get(vid, MUTED), alpha=0.55, linewidth=0,
                  label=vid, zorder=3)
    ax.plot(df["t"], df["rolling"], color=INK, lw=2.2, zorder=5,
           label=f"rolling precision (window={win})")
    for i, t in enumerate(trains):
        ax.axvline(t, color=GOOD, lw=1.2, ls=(0, (3, 2)), alpha=0.8, zorder=1,
                   label="fine-tune completed" if i == 0 else None)
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("accepted (1) or rejected (0), jittered")
    ax.set_xlabel("time of review decision (local time)")
    ax.grid(True, axis="y", lw=0.6, alpha=0.6)
    when = f" since {since:%Y-%m-%d %H:%M}" if since is not None else ", all time"
    ax.set_title(f"Review precision over time{when}", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="video", title_fontsize=9)
    fig.autofmt_xdate(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--behavior", type=int, required=True)
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--since", default=None,
                    help="only plot review decisions after this local time, e.g. '2026-10-09 17:40'")
    a = ap.parse_args()

    proj = Path(a.project).expanduser()
    manifest = json.load(open(proj / "project.json"))
    beh = next(b for b in manifest["behaviors"] if b["id"] == a.behavior)
    pp = {"smooth": 5, "hi": 0.6, "lo": 0.6, "min_bout": 25, "min_gap": 3, **(beh.get("postproc") or {})}
    thr = pp["hi"]
    out = Path(a.out) if a.out else proj / "analysis" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    vids = [v["video_id"] for v in manifest["videos"]]
    results = []
    for vid in vids:
        try:
            r = video_metrics(proj, vid, a.behavior, a.track, thr, pp)
        except FileNotFoundError:
            r = None
        if r:
            results.append(r)
    if not results:
        raise SystemExit("no labeled frames found for this behavior/track in any video")

    plot_roc(results, out / "roc_curves.png")
    plot_pr(results, thr, out / "pr_curves.png")
    plot_operating_point(results, thr, out / "operating_point.png")
    had_time = plot_precision_over_time(proj, a.behavior, a.track, [r["video_id"] for r in results],
                                       out / "precision_over_time.png", a.since)

    rows = []
    for r in results:
        rows.append({k: v for k, v in r.items()
                    if k not in ("fpr", "tpr", "pr_p", "pr_r", "y_true", "y_score")})
    pooled = dict(video_id="POOLED", n_pos=sum(r["n_pos"] for r in results),
                 n_neg=sum(r["n_neg"] for r in results),
                 tp=sum(r["tp"] for r in results), fp=sum(r["fp"] for r in results),
                 tn=sum(r["tn"] for r in results), fn=sum(r["fn"] for r in results))
    pooled["precision"] = pooled["tp"] / max(1, pooled["tp"] + pooled["fp"])
    pooled["recall"] = pooled["tp"] / max(1, pooled["tp"] + pooled["fn"])
    pooled["accuracy"] = (pooled["tp"] + pooled["tn"]) / max(1, sum(pooled[k] for k in ("tp", "fp", "tn", "fn")))
    pooled["f1"] = (2 * pooled["precision"] * pooled["recall"] / (pooled["precision"] + pooled["recall"])
                   if (pooled["precision"] + pooled["recall"]) else float("nan"))
    rows.append(pooled)
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)

    cols = ["video_id", "n_pos", "n_neg", "coverage_pct", "accuracy", "precision", "recall",
           "specificity", "f1", "auroc", "ap", "n_true_bouts", "bout_recall"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print(f"\nWrote roc_curves, pr_curves, operating_point" + (" and precision_over_time" if had_time else "")
          + f" (.png) and summary.csv to {out}")
    if not had_time:
        print("No review decisions in the chosen time range, so there is no precision_over_time plot.")
    base_rate = pooled["n_pos"] / max(1, pooled["n_pos"] + pooled["n_neg"])
    print(f"\nPooled positive rate on labeled frames is {base_rate:.1%}. At this imbalance accuracy says "
         f"little. Use precision, recall and AUROC.")


if __name__ == "__main__":
    main()
