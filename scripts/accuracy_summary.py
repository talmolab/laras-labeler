#!/usr/bin/env python3
"""One combined accuracy table for the HITL trial — raw-model quality AND label quality, per behavior
(and optionally per clip), in one command. The thing you run at each milestone.

It calls the two existing scorers and merges them by behavior:
  - score_predictions.score  -> the MODEL'S raw lane vs ground truth: AUROC, AP, best-F1 @threshold
    (threshold-free ranking quality — the honest "does HiDRA transfer" number).
  - score_vs_ground_truth.score -> the LABELS an arm produced vs ground truth: bout & frame P/R/F1
    (how good the reviewed annotation is).

    python scripts/accuracy_summary.py --truth <GT_PROJECT> --pred <ARM_PROJECT> --behaviors grooming
    # per-clip rows too, and a CSV that grows into the accuracy figure:
    python scripts/accuracy_summary.py --truth GT --pred ARM --behaviors grooming,jump down --per-clip --csv acc.csv

--truth is your ground-truth / hand-labeled project; --pred the arm being scored (HITL or hand).
--behaviors matches by NAME; --videos / per-clip match video_id by substring. Needs pandas+numpy —
run with the labeler env python. Append rows across rounds by pointing --csv at the same file and
adding a --round tag.
"""
from __future__ import annotations

import argparse
import csv as _csv
import sys
from pathlib import Path

_SCR = Path(__file__).resolve().parent
if str(_SCR) not in sys.path:
    sys.path.insert(0, str(_SCR))
import score_predictions as sp        # noqa: E402
import score_vs_ground_truth as sg    # noqa: E402


def _clips(truth: Path, videos):
    vids = [v["video_id"] for v in sp._manifest(truth).get("videos", [])]
    return [v for v in vids if not videos or any(f in v for f in videos)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Combined raw-model + label accuracy per behavior (and clip).")
    ap.add_argument("--truth", required=True, help="ground-truth / hand-labeled project dir")
    ap.add_argument("--pred", required=True, help="arm project dir to score (HITL or hand)")
    ap.add_argument("--behaviors", help="comma-separated behavior names (default: all in truth)")
    ap.add_argument("--videos", help="comma-separated video_id substrings (default: all clips)")
    ap.add_argument("--per-clip", action="store_true", help="also emit one row per (behavior, clip)")
    ap.add_argument("--iou", type=float, default=0.5, help="bout-match IoU (default 0.5)")
    ap.add_argument("--round", help="optional round/label tag written into each row (e.g. r0, r1)")
    ap.add_argument("--csv", help="write (append if it exists) rows to this CSV — grows into the figure")
    args = ap.parse_args(argv)

    truth, pred = Path(args.truth).expanduser(), Path(args.pred).expanduser()
    behaviours = [b.strip() for b in args.behaviors.split(",")] if args.behaviors else None
    vfilter = [v.strip() for v in args.videos.split(",")] if args.videos else None

    scopes = [("all", vfilter)]                     # aggregate over all matched clips
    if args.per_clip:
        scopes += [(c[:22], [c]) for c in _clips(truth, vfilter)]

    rows = []
    for scope, vids in scopes:
        raw = {r["behavior"]: r for r in sp.score(truth, pred, behaviours, vids, args.iou)}
        lab = {b["behavior"]: b for b in sg.score(truth, pred, behaviours, vids, args.iou)["behaviors"]}
        for beh in sorted(set(raw) | set(lab)):
            r, l = raw.get(beh, {}), lab.get(beh, {})
            bout, frame = l.get("bout", {}), l.get("frame", {})
            rows.append({
                "round": args.round or "", "scope": scope, "behavior": beh,
                "prevalence": r.get("prevalence"),
                "auroc": r.get("auroc"), "ap": r.get("ap"),
                "best_f1": r.get("best_f1"), "best_thr": r.get("best_thr"),
                "label_bout_p": bout.get("precision"), "label_bout_r": bout.get("recall"),
                "label_bout_f1": bout.get("f1"), "label_frame_f1": frame.get("f1"),
                "bouts_tp_fp_fn": f"{bout.get('tp','')}/{bout.get('fp','')}/{bout.get('fn','')}" if bout else "",
            })

    if not rows:
        sys.exit("no behaviors scored — check --behaviors / --videos / that the arm has predictions.")

    cols = ["round", "scope", "behavior", "prevalence", "auroc", "ap", "best_f1", "best_thr",
            "label_bout_f1", "label_bout_p", "label_bout_r", "label_frame_f1", "bouts_tp_fp_fn"]
    disp = [c for c in cols if not (c == "round" and not args.round)]
    w = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in disp}
    print("  ".join(c.ljust(w[c]) for c in disp))
    print("  ".join("-" * w[c] for c in disp))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w[c]) for c in disp))
    print("\nraw-model: auroc/ap/best_f1@thr (threshold-free ranking) · label: your arm's bouts vs GT "
          "(bout_f1, tp/fp/fn) · a high AUROC with low bout_f1 = good ranking, wrong cut.")

    if args.csv:
        p = Path(args.csv)
        exists = p.exists()
        with p.open("a", newline="", encoding="utf-8") as f:
            wr = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if not exists:
                wr.writeheader()
            wr.writerows(rows)
        print(f"\n{'appended to' if exists else 'wrote'} {args.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
