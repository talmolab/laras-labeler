#!/usr/bin/env python3
"""Score one arm's annotations against the ground-truth project — the accuracy axis of the
HiDRA-in-the-loop efficiency trial.

The annotation event log measures TIME precisely (per-round label/review minutes; see events.py),
but it fills in accuracy (ap/f1) only for the labeler's own model — a HiDRA-bound behavior's rounds
come back blank by design. So the quality of every arm is computed HERE instead, the same way for
all of them: take the labels an arm produced and compare them, frame for frame and bout for bout,
against a frozen ground-truth project (for the DooM home-cage work that is `doom-manual`).

    python scripts/score_vs_ground_truth.py --truth /path/doom-manual --pred /path/doom-hitl
    # restrict to a held-out test set, and to one behaviour, at a stricter bout overlap:
    python scripts/score_vs_ground_truth.py --truth GT --pred ARM \
        --videos cam0,cam3 --behaviors sniff --iou 0.5 --csv sniff_hitl.csv

WHAT IT COMPARES
  Both projects store labels as <project>/labels/<video_id>.parquet, per-frame tri-state rows
  (frame, track, behavior_id, value; value 1 = Happening), with the manifest in project.json.
  Behaviours are matched by NAME (an arm may assign different behavior_ids than the truth project);
  videos by video_id; tracks by index (both derive from the same tracking, so mouse1 is track 0 in
  each). Only value == 1 counts as positive — negatives and unlabeled frames are both "not the
  behaviour", which is what an ethogram means by absence.

TWO SCORES, because they answer different questions and can disagree
  FRAME  positive-class precision/recall/F1 over every frame. Sensitive to boundary jitter — a bout
         found but trimmed a little short still costs here.
  BOUT   each ground-truth bout is matched to a predicted bout when their overlap (intersection over
         union) is at least --iou; matched = TP, unmatched truth = FN (a miss), unmatched prediction
         = FP (a false alarm). This is the "did it catch the events" number, forgiving of small edge
         differences. Matching is one-to-one and greedy by descending IoU.

  Report both. A high frame-F1 with a low bout-F1 means fragmented detections; the reverse means the
  right events with sloppy edges.

Standalone: reads parquet + JSON directly, so it runs wherever pandas does — no labeler install.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- loading
def _manifest(proj: Path) -> dict:
    p = proj / "project.json"
    if not p.exists():
        sys.exit(f"ERROR: not a labeler project (no project.json): {proj}")
    return json.loads(p.read_text())


def _behavior_ids_by_name(manifest: dict) -> dict[str, int]:
    """{normalised name: behavior_id}. Names collide rarely; last one wins with a warning."""
    out: dict[str, int] = {}
    for b in manifest.get("behaviors", []):
        name = _norm(b.get("name", ""))
        if name in out:
            print(f"  warning: two behaviours named {b.get('name')!r}; using id {b['id']}")
        out[name] = int(b["id"])
    return out


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _positive_runs(proj: Path, vid: str, bid: int, track: int) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of value == 1 for one (video, behaviour, track).

    Missing label file / no rows -> no positives (the arm proposed nothing here, which is a real
    answer, not an error)."""
    p = proj / "labels" / f"{vid}.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p, columns=None)
    if "track" not in df.columns:                     # pre-per-track parquet: everything is track 0
        df = df.assign(track=0)
    df = df[(df["behavior_id"] == bid) & (df["track"] == int(track)) & (df["value"] == 1)]
    if df.empty:
        return []
    f = np.sort(df["frame"].to_numpy().astype(np.int64))
    runs, s, prev = [], int(f[0]), int(f[0])
    for x in f[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
        else:
            runs.append((s, prev + 1)); s = prev = x
    runs.append((s, prev + 1))
    return runs


def _tracks_present(proj: Path, vid: str, bid: int) -> set[int]:
    p = proj / "labels" / f"{vid}.parquet"
    if not p.exists():
        return set()
    df = pd.read_parquet(p, columns=None)
    if "track" not in df.columns:
        return {0}
    df = df[df["behavior_id"] == bid]
    return {int(t) for t in df["track"].unique()}


# ----------------------------------------------------------------------------- scoring
def _mask(runs: list[tuple[int, int]], n: int) -> np.ndarray:
    m = np.zeros(int(n), bool)
    for s, e in runs:
        m[max(0, s):min(n, e)] = True
    return m


def _frame_counts(truth: list[tuple[int, int]], pred: list[tuple[int, int]], n: int) -> tuple[int, int, int]:
    t, p = _mask(truth, n), _mask(pred, n)
    tp = int(np.count_nonzero(t & p))
    return tp, int(np.count_nonzero(p & ~t)), int(np.count_nonzero(t & ~p))   # tp, fp, fn


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def _bout_counts(truth: list[tuple[int, int]], pred: list[tuple[int, int]], thr: float) -> tuple[int, int, int]:
    """Greedy one-to-one IoU match -> (tp, fp, fn)."""
    pairs = sorted(
        ((_iou(t, p), ti, pi) for ti, t in enumerate(truth) for pi, p in enumerate(pred)),
        reverse=True)
    used_t, used_p, tp = set(), set(), 0
    for iou, ti, pi in pairs:
        if iou < thr:
            break
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti); used_p.add(pi); tp += 1
    return tp, len(pred) - len(used_p), len(truth) - len(used_t)   # tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> dict:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


# ----------------------------------------------------------------------------- driver
def score(truth_dir: Path, pred_dir: Path, behaviours: list[str] | None,
          videos: list[str] | None, iou: float) -> dict:
    tman, pman = _manifest(truth_dir), _manifest(pred_dir)
    t_bids, p_bids = _behavior_ids_by_name(tman), _behavior_ids_by_name(pman)
    t_nframes = {v["video_id"]: int(v.get("n_frames") or 0) for v in tman.get("videos", [])}

    want = {_norm(b) for b in behaviours} if behaviours else None
    vid_filter = set(videos) if videos else None
    truth_vids = [v["video_id"] for v in tman.get("videos", [])
                  if vid_filter is None or v["video_id"] in vid_filter]
    if vid_filter:
        missing = vid_filter - set(truth_vids)
        if missing:
            print(f"  warning: --videos not in the truth project, skipped: {sorted(missing)}")

    results = []
    for name_norm, t_bid in sorted(t_bids.items()):
        if want is not None and name_norm not in want:
            continue
        disp = next((b.get("name") for b in tman["behaviors"] if int(b["id"]) == t_bid), name_norm)
        if name_norm not in p_bids:
            print(f"  note: behaviour {disp!r} not present in the arm project — scored as all-missed")
        p_bid = p_bids.get(name_norm)

        fr = {"tp": 0, "fp": 0, "fn": 0}
        bt = {"tp": 0, "fp": 0, "fn": 0}
        per_video = []
        for vid in truth_vids:
            tracks = _tracks_present(truth_dir, vid, t_bid)
            if p_bid is not None:
                tracks |= _tracks_present(pred_dir, vid, p_bid)
            n = t_nframes.get(vid, 0)
            vfr = {"tp": 0, "fp": 0, "fn": 0}
            vbt = {"tp": 0, "fp": 0, "fn": 0}
            for tr in sorted(tracks) or [0]:
                truth_runs = _positive_runs(truth_dir, vid, t_bid, tr)
                pred_runs = _positive_runs(pred_dir, vid, p_bid, tr) if p_bid is not None else []
                nn = n or (max([e for _, e in truth_runs + pred_runs], default=0))
                tp, fp, fn = _frame_counts(truth_runs, pred_runs, nn)
                vfr["tp"] += tp; vfr["fp"] += fp; vfr["fn"] += fn
                tp, fp, fn = _bout_counts(truth_runs, pred_runs, iou)
                vbt["tp"] += tp; vbt["fp"] += fp; vbt["fn"] += fn
            for k in fr:
                fr[k] += vfr[k]; bt[k] += vbt[k]
            if vfr["tp"] or vfr["fn"] or vfr["fp"]:
                per_video.append({"video_id": vid, "frame": _prf(**vfr), "bout": _prf(**vbt)})

        results.append({"behavior": disp, "frame": _prf(**fr), "bout": _prf(**bt),
                        "per_video": per_video})

    return {"truth": str(truth_dir), "pred": str(pred_dir), "iou": iou,
            "videos_scored": truth_vids, "behaviors": results}


def _print(rep: dict) -> None:
    print(f"\nground truth : {rep['truth']}")
    print(f"arm          : {rep['pred']}")
    print(f"videos       : {len(rep['videos_scored'])}   bout IoU >= {rep['iou']}\n")
    hdr = f"{'behaviour':<20}{'bout P':>8}{'bout R':>8}{'bout F1':>9}" \
          f"{'frame P':>9}{'frame R':>9}{'frame F1':>10}   bouts(tp/fp/fn)"
    print(hdr); print("-" * len(hdr))
    for b in rep["behaviors"]:
        bt, fr = b["bout"], b["frame"]
        print(f"{b['behavior']:<20}{bt['precision']:>8.3f}{bt['recall']:>8.3f}{bt['f1']:>9.3f}"
              f"{fr['precision']:>9.3f}{fr['recall']:>9.3f}{fr['f1']:>10.3f}"
              f"   {bt['tp']}/{bt['fp']}/{bt['fn']}")
    print()


def _to_rows(rep: dict) -> list[dict]:
    rows = []
    for b in rep["behaviors"]:
        rows.append({"behavior": b["behavior"], "level": "bout", "iou": rep["iou"], **b["bout"]})
        rows.append({"behavior": b["behavior"], "level": "frame", "iou": rep["iou"], **b["frame"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, type=Path, help="ground-truth project dir (e.g. doom-manual)")
    ap.add_argument("--pred", required=True, type=Path, help="an arm's project dir to score")
    ap.add_argument("--behaviors", help="comma-separated behaviour names to score (default: all in truth)")
    ap.add_argument("--videos", help="comma-separated video_ids to score, e.g. the held-out test set (default: all)")
    ap.add_argument("--iou", type=float, default=0.5, help="min bout overlap to count a match (default 0.5)")
    ap.add_argument("--csv", type=Path, help="also write the per-behaviour scores here")
    ap.add_argument("--json", type=Path, help="also write the full report (incl. per-video) here")
    args = ap.parse_args()

    behaviours = [b.strip() for b in args.behaviors.split(",") if b.strip()] if args.behaviors else None
    videos = [v.strip() for v in args.videos.split(",") if v.strip()] if args.videos else None
    if not 0 < args.iou <= 1:
        sys.exit("ERROR: --iou must be in (0, 1]")

    rep = score(args.truth, args.pred, behaviours, videos, args.iou)
    if not rep["behaviors"]:
        sys.exit("ERROR: no behaviours scored — check --behaviors names against the truth project.")
    _print(rep)
    if args.csv:
        pd.DataFrame(_to_rows(rep)).to_csv(args.csv, index=False)
        print(f"wrote per-behaviour scores -> {args.csv}")
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2))
        print(f"wrote full report -> {args.json}")


if __name__ == "__main__":
    main()
