#!/usr/bin/env python3
"""Score one project's HiDRA predictions against hand labels kept in another project.

Run it on a held-out video after every fine-tune and predict. Each run:
  - prints AUROC, AP, and frame and bout precision and recall at the live threshold
  - suggests the slider value with the best bout F1
  - adds a row to <project>/analysis/heldout_log.csv and redraws heldout_scoreboard.png there

Files written to --out:
  vs_handlabels_snapshot.png   ROC and precision-recall of the current predictions
  vs_handlabels_snapshot.csv   the numbers behind it
  threshold_sweep.csv          bout and frame scores at each slider value

By default only frames with an explicit label are scored. Pass --full-coverage if every real bout
in the social window was painted. Then every other frame in the window counts as a negative.

    python lab_scripts/compare_to_handlabels.py \\
        --project ~/my-projects/approach_defs --video 0 \\
        --gt-project ~/my-projects/cage1rb_day1 --gt-video 0 \\
        --behavior 0 --track 1 --full-coverage

Run it with the labeler's own venv. It uses the labeler's bout detection.
"""
import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from laras_labeler.predict import DEFAULT_POSTPROC, bouts_from_proba

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.frameon": False,
})

SLIDER = np.round(np.arange(0.30, 0.951, 0.05), 2)   # the GUI slider moves in 0.05 steps


def runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def score(y_lookup, proba, thr, pp):
    """Frame and bout scores at one threshold, on frames where y_lookup is 0 or 1."""
    scored = y_lookup >= 0
    y = y_lookup[scored]
    pred = proba[scored] >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    true_b = runs(y_lookup == 1)
    pred_b = [(s, e) for s, e in bouts_from_proba(proba, pp["smooth"], thr, thr, pp["min_bout"], pp["min_gap"])
              if scored[s:e].any()]
    hit_pred = sum(any(s < te and e > ts for ts, te in true_b) for s, e in pred_b)
    hit_true = sum(any(ps < e and pe > s for ps, pe in pred_b) for s, e in true_b)

    def ratio(a, b):
        return a / b if b else float("nan")

    def f1(p, r):
        return 2 * p * r / (p + r) if p + r else 0.0

    fp_, fr_ = ratio(tp, tp + fp), ratio(tp, tp + fn)
    bp, br = ratio(hit_pred, len(pred_b)), ratio(hit_true, len(true_b))
    return dict(threshold=thr, frame_precision=fp_, frame_recall=fr_, frame_f1=f1(fp_, fr_),
                bout_precision=bp, bout_recall=br, bout_f1=f1(bp, br),
                n_pred_bouts=len(pred_b), n_true_bouts=len(true_b))


def checkpoint_time(beh):
    w = (beh.get("hidra") or {}).get("weights")
    files = glob.glob(w.replace("{config}", "*")) if w else []
    if not files:
        return "zero-shot"
    return datetime.fromtimestamp(max(os.path.getmtime(f) for f in files)).strftime("%Y-%m-%d %H:%M")


def plot_scoreboard(log, video, gt_video, path):
    d = log[(log.video.astype(str) == str(video)) & (log.gt_video.astype(str) == str(gt_video))]
    d = d.drop_duplicates("checkpoint_time", keep="last").sort_values("checkpoint_time")
    if len(d) < 2:
        return False
    x = np.arange(1, len(d) + 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(d) + 3), 4.4))
    for col, name, c in [("auroc", "AUROC", CAT[0]), ("ap", "average precision", CAT[1]),
                         ("best_bout_f1", "best bout F1", CAT[2])]:
        ax.plot(x, d[col], color=c, lw=2, marker="o", ms=7, mec=SURFACE, mew=1, label=name)
    ax.set_xticks(x, [f"{t[:10]}\n{m}" for t, m in zip(d.checkpoint_time, d.finetune_mode)])
    ax.set_xlabel("fine-tune (date it finished, mode)")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", lw=0.6, alpha=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(f"Held-out video {video}: scores after each fine-tune", pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="project whose HiDRA predictions you're scoring")
    ap.add_argument("--video", required=True, help="video_id in --project")
    ap.add_argument("--gt-project", required=True, help="project holding the hand labels (ground truth)")
    ap.add_argument("--gt-video", default=None, help="video_id in --gt-project (default: same as --video)")
    ap.add_argument("--behavior", type=int, required=True)
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--full-coverage", action="store_true",
                    help="count every unpainted frame in the social window as a negative. Only use "
                         "this if every real bout was painted.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    proj = Path(a.project).expanduser()
    gt_proj = Path(a.gt_project).expanduser()
    gt_vid = a.gt_video or a.video
    out = Path(a.out) if a.out else proj / "analysis" / "vs_handlabels"
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(proj / "project.json"))
    beh = next(b for b in manifest["behaviors"] if b["id"] == a.behavior)
    pp = {**DEFAULT_POSTPROC, **(beh.get("postproc") or {})}
    thr = pp["hi"]
    n_frames = next(v["n_frames"] for v in manifest["videos"] if v["video_id"] == a.video)

    # Ground truth from the hand-labeled project.
    gt_all = pd.read_parquet(gt_proj / "labels" / f"{gt_vid}.parquet")
    gt = gt_all[(gt_all.track == a.track) & (gt_all.behavior_id == a.behavior) & gt_all.value.isin([0, 1])]
    if gt.empty:
        raise SystemExit(f"no hand labels found in {gt_proj}/labels/{gt_vid}.parquet for "
                         f"behavior {a.behavior}, track {a.track}")

    y_lookup = np.full(n_frames, -1, dtype=int)
    if a.full_coverage:
        gt_manifest = json.load(open(gt_proj / "project.json"))
        gt_entry = next(v for v in gt_manifest["videos"] if v["video_id"] == gt_vid)
        ws = gt_entry.get("social_start"); we = gt_entry.get("social_end")
        if ws is None or we is None:
            raise SystemExit(f"--full-coverage needs a social window on {gt_vid} in {gt_proj}. "
                             "It marks the part of the video you reviewed.")
        y_lookup[ws:we] = 0
        pos = gt[gt.value == 1].frame.to_numpy()
        y_lookup[pos[(pos >= ws) & (pos < we)]] = 1
        print(f"--full-coverage: scoring all {we - ws} frames in the social window [{ws},{we}). "
             f"{int((y_lookup[ws:we]==1).sum())} are positive. The rest count as negative.")
    else:
        y_lookup[gt.frame.to_numpy()] = gt.value.to_numpy()
        span = int(gt.frame.max() - gt.frame.min())
        coverage = len(gt) / span * 100 if span else 0.0
        print(f"Scoring only the {len(gt)} labeled frames ({coverage:.1f}% of the labeled span). "
             "Pass --full-coverage if you reviewed the whole window.")

    proba = np.load(proj / "predictions" / a.video / f"{a.behavior}.npy")[:, a.track]
    frames = np.flatnonzero(y_lookup >= 0)
    y_true = y_lookup[frames]
    y_score = proba[frames]
    base_rate = (y_true == 1).mean()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    pr_p, pr_r, _ = precision_recall_curve(y_true, y_score)
    ap_score = average_precision_score(y_true, y_score)
    live = score(y_lookup, proba, thr, pp)
    sweep = pd.DataFrame([score(y_lookup, proba, t, pp) for t in SLIDER])
    sweep.to_csv(out / "threshold_sweep.csv", index=False)
    best = sweep.loc[sweep.bout_f1.idxmax()]
    ckpt = checkpoint_time(beh)
    mode = (beh.get("hidra") or {}).get("finetune_mode", "") if ckpt != "zero-shot" else ""

    print(f"{proj.name}/{a.video} predictions vs. {gt_proj.name}/{gt_vid} hand labels. "
         f"Checkpoint {ckpt} {mode}")
    print(f"{len(frames)} frames scored, positive rate {base_rate:.1%}, {live['n_true_bouts']} true bouts")
    print(f"AUROC {auroc:.3f}   average precision {ap_score:.3f} (chance {base_rate:.3f})")
    print(f"At the live threshold {thr:.2f}:")
    print(f"  bouts:  precision {live['bout_precision']:.2f}  recall {live['bout_recall']:.2f}  "
         f"F1 {live['bout_f1']:.2f}  ({live['n_pred_bouts']} detected)")
    print(f"  frames: precision {live['frame_precision']:.2f}  recall {live['frame_recall']:.2f}  "
         f"F1 {live['frame_f1']:.2f}")
    print(f"Best slider value for bout F1: {best.threshold:.2f} (precision {best.bout_precision:.2f}, "
         f"recall {best.bout_recall:.2f}, F1 {best.bout_f1:.2f}). It was picked on this video, so expect "
         "a bit less on new ones.")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ax = axes[0]
    ax.plot(fpr, tpr, color=CAT[0], lw=2.4, solid_capstyle="round")
    ax.fill_between(fpr, tpr, color=CAT[0], alpha=0.08, lw=0)
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls=(0, (4, 3)), zorder=0)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.grid(True, lw=0.6, alpha=0.6)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC vs. hand labels\nAUROC {auroc:.2f}")

    ax = axes[1]
    ax.plot(pr_r, pr_p, color=CAT[0], lw=2.4, solid_capstyle="round")
    ax.axhline(base_rate, color=GRID, lw=1.5, ls=(0, (4, 3)), zorder=0)
    ax.plot(live["frame_recall"], live["frame_precision"], marker="o", ms=7, color=INK, mec=SURFACE,
           mew=1, zorder=5, label=f"live threshold ({thr:.2f})")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.grid(True, lw=0.6, alpha=0.6)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title(f"Precision-recall vs. hand labels\nAP {ap_score:.2f} (chance = {base_rate:.2f})")
    ax.legend(loc="upper right")

    fig.suptitle(f"{a.video} in {proj.name}: HiDRA predictions vs. hand labels in {gt_proj.name}",
                fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "vs_handlabels_snapshot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    row = dict(run_at=datetime.now().strftime("%Y-%m-%d %H:%M"), video=a.video, gt_project=gt_proj.name,
               gt_video=gt_vid, full_coverage=a.full_coverage, checkpoint_time=ckpt, finetune_mode=mode,
               positive_rate=base_rate, auroc=auroc, ap=ap_score, **{k: v for k, v in live.items()},
               best_threshold=best.threshold, best_bout_f1=best.bout_f1)
    pd.DataFrame([row]).to_csv(out / "vs_handlabels_snapshot.csv", index=False)

    log_path = proj / "analysis" / "heldout_log.csv"
    log = pd.concat([pd.read_csv(log_path, dtype=str), pd.DataFrame([row]).astype(str)]) \
        if log_path.exists() else pd.DataFrame([row]).astype(str)
    log.to_csv(log_path, index=False)
    numeric = log.copy()
    for col in ("auroc", "ap", "best_bout_f1"):
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    board = proj / "analysis" / "heldout_scoreboard.png"
    drew = plot_scoreboard(numeric, a.video, gt_vid, board)
    print(f"\nWrote {out}/ and added a row to {log_path}")
    print(f"Scoreboard: {board}" if drew else
          "The scoreboard plot appears once the log has two different checkpoints for this video.")


if __name__ == "__main__":
    main()
