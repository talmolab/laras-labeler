#!/usr/bin/env python3
"""Score a MODEL'S RAW PREDICTIONS against ground truth — the transfer number, and the accuracy
point for each fine-tuning round.

`score_vs_ground_truth.py` grades the *labels a human produced* (0/1 spans). This grades the
*model's output before a human touched it* — the per-frame probability lane the labeler writes to
<project>/predictions/<video_id>/<behavior_id>.npy on every Predict (HiDRA-bound or native alike).
Two uses, same command:

  ZERO-SHOT TRANSFER   run Predict once with a shipped head, then score its lane vs your manual
                       labels: does HiDRA's sniff/groom classifier rank YOUR frames correctly?
  ROUNDS CURVE         after each predict->review->fine-tune round, score the NEXT clip's raw lane
                       before reviewing it. The trace across rounds is "how many rounds to plateau".

    python scripts/score_predictions.py --truth doom-manual --pred doom-hitl --behaviors sniff
    python scripts/score_predictions.py --truth doom-manual --pred doom-hitl \
        --behaviors jump-down --videos cam4,cam5 --round 3 --csv rounds.csv   # append one round's point

WHY THRESHOLD-FREE METRICS LEAD
  HiDRA's probabilities are miscalibrated out of domain — they compress toward zero while their
  ORDER survives (see hidra.py, CALIBRATION). So a head that transfers well can still put every
  frame below its shipped cutoff. AUROC and average precision (AP) judge the RANKING, independent of
  any threshold, and are the honest "does it transfer" numbers. Best-F1 (swept threshold) is
  reported too — what you'd get after a one-line calibration — with its threshold, so the gap
  between raw-threshold and best-F1 shows how much a recalibration would buy.

Matches behaviours by NAME (truth and pred projects may number them differently), videos by
video_id, tracks by lane column. value == 1 in the truth labels is positive. Reads parquet + npy +
JSON directly — no labeler install needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _behavior_ids(manifest: dict) -> dict[str, int]:
    return {_norm(b.get("name", "")): int(b["id"]) for b in manifest.get("behaviors", [])}


def _manifest(proj: Path) -> dict:
    p = proj / "project.json"
    if not p.exists():
        sys.exit(f"ERROR: not a labeler project (no project.json): {proj}")
    return json.loads(p.read_text())


def _gt_mask(truth: Path, vid: str, bid: int, track: int, n: int) -> np.ndarray:
    """Positive (value == 1) frame mask for one (video, behaviour, track)."""
    m = np.zeros(int(n), bool)
    p = truth / "labels" / f"{vid}.parquet"
    if not p.exists():
        return m
    df = pd.read_parquet(p)
    if "track" not in df.columns:
        df = df.assign(track=0)
    df = df[(df["behavior_id"] == bid) & (df["track"] == int(track)) & (df["value"] == 1)]
    f = df["frame"].to_numpy().astype(np.int64)
    f = f[(f >= 0) & (f < n)]
    m[f] = True
    return m


def _lane(pred: Path, vid: str, bid: int) -> np.ndarray | None:
    """The prediction probability lane (n_frames,) or (n_frames, n_tracks), or None if absent."""
    p = pred / "predictions" / vid / f"{bid}.npy"
    if not p.exists():
        return None
    a = np.load(p)
    return a.astype(np.float64)


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUROC with tie-averaged ranks (no sklearn)."""
    y = y.astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    ranks = np.empty(len(p), float)
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _ap(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision = area under the precision-recall curve (step interpolation)."""
    y = y.astype(bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")[::-1]
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / y.sum()
    rec = np.concatenate(([0.0], rec))
    prec = np.concatenate(([1.0], prec))
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _iou(a, b) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if not inter:
        return 0.0
    return inter / ((a[1] - a[0]) + (b[1] - b[0]) - inter)


def _bout_f1(y: np.ndarray, pred_mask: np.ndarray, iou: float) -> float:
    truth, pred = _runs(y), _runs(pred_mask)
    if not truth and not pred:
        return float("nan")
    pairs = sorted(((_iou(t, p), ti, pi) for ti, t in enumerate(truth) for pi, p in enumerate(pred)),
                   reverse=True)
    ut, up, tp = set(), set(), 0
    for v, ti, pi in pairs:
        if v < iou:
            break
        if ti in ut or pi in up:
            continue
        ut.add(ti); up.add(pi); tp += 1
    fp, fn = len(pred) - len(up), len(truth) - len(ut)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def _best_f1(y: np.ndarray, p: np.ndarray, iou: float, grid: int = 200):
    """Sweep thresholds; return (best frame-F1, threshold, precision, recall, bout-F1 at that thr)."""
    lo, hi = float(np.min(p)), float(np.max(p))
    if hi <= lo:
        return 0.0, hi, 0.0, 0.0, 0.0
    best = (0.0, hi, 0.0, 0.0, 0.0)
    npos = int(y.sum())
    for thr in np.linspace(lo, hi, grid):
        c = p >= thr
        tp = int(np.count_nonzero(c & y)); fp = int(np.count_nonzero(c & ~y)); fn = npos - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best[0]:
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            best = (f1, float(thr), prec, rec, _bout_f1(y, c, iou))
    return best


def score(truth: Path, pred: Path, behaviours, videos, iou):
    tman, pman = _manifest(truth), _manifest(pred)
    t_ids, p_ids = _behavior_ids(tman), _behavior_ids(pman)
    nfr = {v["video_id"]: int(v.get("n_frames") or 0) for v in tman.get("videos", [])}
    vids = [v["video_id"] for v in tman.get("videos", [])
            if not videos or v["video_id"] in set(videos)]

    want = {_norm(b) for b in behaviours} if behaviours else set(t_ids)
    out = []
    for nm in sorted(want):
        if nm not in t_ids:
            print(f"  note: behaviour {nm!r} not in truth project — skipped"); continue
        disp = next((b.get("name") for b in tman["behaviors"] if int(b["id"]) == t_ids[nm]), nm)
        if nm not in p_ids:
            print(f"  note: behaviour {disp!r} has no predictions in the arm — skipped"); continue
        P, Y = [], []
        for vid in vids:
            lane = _lane(pred, vid, p_ids[nm])
            if lane is None:
                continue
            n = nfr.get(vid) or lane.shape[0]
            n = min(n, lane.shape[0])
            ntr = lane.shape[1] if lane.ndim == 2 else 1
            for tr in range(ntr):
                col = lane[:n, tr] if lane.ndim == 2 else lane[:n]
                y = _gt_mask(truth, vid, t_ids[nm], tr, n)
                if y.sum() == 0 and col.max(initial=0) == 0:
                    continue
                P.append(col); Y.append(y)
        if not P:
            print(f"  note: no prediction lanes found for {disp!r} — did you Predict this arm?"); continue
        p = np.concatenate(P); y = np.concatenate(Y)
        f1, thr, prec, rec, bf1 = _best_f1(y, p, iou)
        out.append({"behavior": disp, "n_frames": int(y.size), "prevalence": round(float(y.mean()), 4),
                    "auroc": round(_auroc(y, p), 4), "ap": round(_ap(y, p), 4),
                    "best_f1": round(f1, 4), "best_thr": round(thr, 4),
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "bout_f1": round(bf1, 4)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, type=Path, help="ground-truth project (doom-manual)")
    ap.add_argument("--pred", required=True, type=Path, help="project whose predictions/ lanes to score")
    ap.add_argument("--behaviors", help="comma-separated behaviour names (default: all in truth)")
    ap.add_argument("--videos", help="comma-separated video_ids, e.g. a held-out set (default: all)")
    ap.add_argument("--iou", type=float, default=0.5, help="bout-match IoU for the bout-F1 column (default 0.5)")
    ap.add_argument("--round", type=int, help="tag these scores with a round number (for the rounds curve)")
    ap.add_argument("--csv", type=Path, help="append the scores here (round curve accumulates across runs)")
    args = ap.parse_args()

    behaviours = [b.strip() for b in args.behaviors.split(",") if b.strip()] if args.behaviors else None
    videos = [v.strip() for v in args.videos.split(",") if v.strip()] if args.videos else None
    rows = score(args.truth, args.pred, behaviours, videos, args.iou)
    if not rows:
        sys.exit("ERROR: nothing scored — check --behaviors, and that the arm has predictions/ lanes.")

    hdr = f"{'behaviour':<18}{'AUROC':>8}{'AP':>8}{'best F1':>9}{'@thr':>7}{'prec':>7}{'rec':>7}{'bout F1':>9}{'prev':>7}"
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['behavior']:<18}{r['auroc']:>8.3f}{r['ap']:>8.3f}{r['best_f1']:>9.3f}"
              f"{r['best_thr']:>7.3f}{r['precision']:>7.3f}{r['recall']:>7.3f}{r['bout_f1']:>9.3f}"
              f"{r['prevalence']:>7.3f}")
    print()
    if args.csv:
        for r in rows:
            if args.round is not None:
                r["round"] = args.round
        df = pd.DataFrame(rows)
        cols = (["round"] if args.round is not None else []) + \
               ["behavior", "auroc", "ap", "best_f1", "best_thr", "precision", "recall", "bout_f1",
                "prevalence", "n_frames"]
        df = df[[c for c in cols if c in df.columns]]
        if args.csv.exists():
            df.to_csv(args.csv, mode="a", header=False, index=False)
        else:
            df.to_csv(args.csv, index=False)
        print(f"{'appended' if args.csv.exists() else 'wrote'} {len(rows)} row(s) -> {args.csv}")


if __name__ == "__main__":
    main()
