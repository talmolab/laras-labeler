#!/usr/bin/env python3
"""Copy one behavior's labels for specific clips from one project into another — e.g. seed the HITL
arm with the hand arm's jump-down labels on 2 clips, without dragging the other behaviors along.

The label store is per-clip parquet (labels/<video_id>.parquet) holding EVERY behavior for that clip.
This reads the source project's parquet for the chosen clips, keeps only the chosen behavior's rows
(all tracks, all values, plus the directed `target`), and merges them into the destination project's
parquet for the same clips — replacing any existing rows of that behavior there, leaving other
behaviors untouched.

Usage
    python scripts/transfer_labels.py --from <SRC_PROJECT> --to <DST_PROJECT> --behavior N --clips clip_001,clip_002

    # seed HITL jump-down (behavior 0) from the hand arm on clips 001 + 002:
    python scripts/transfer_labels.py `
      --from C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-test `
      --to   C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-hitl `
      --behavior 0 --clips clip_001,clip_002

STOP the labeler first (it caches label frames in memory and rewrites parquet on save). Needs pandas —
run with the labeler env python. --clips matches by substring against each labels/<video_id>.parquet
filename, so a prefix like `clip_001` is enough.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Copy one behavior's labels for chosen clips between projects.")
    ap.add_argument("--from", dest="src", required=True, help="source project dir (has the labels to copy)")
    ap.add_argument("--to", dest="dst", required=True, help="destination project dir (receives them)")
    ap.add_argument("--behavior", type=int, required=True, help="behavior_id to copy (e.g. 0 = jump down)")
    ap.add_argument("--clips", required=True, help="comma-separated clip id substrings to match (e.g. clip_001,clip_002)")
    ap.add_argument("--dry-run", action="store_true", help="report what would be copied, write nothing")
    args = ap.parse_args(argv)

    src = Path(args.src).expanduser() / "labels"
    dst = Path(args.dst).expanduser() / "labels"
    if not src.is_dir():
        sys.exit(f"no labels/ in source: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    wanted = [c.strip() for c in args.clips.split(",") if c.strip()]

    total = 0
    for p in sorted(src.glob("*.parquet")):
        if not any(w in p.name for w in wanted):
            continue
        sdf = pd.read_parquet(p)
        rows = sdf[sdf["behavior_id"] == args.behavior]
        if rows.empty:
            print(f"  {p.name[:32]}… : no behavior-{args.behavior} rows in source, skipped")
            continue
        n_pos = int(((rows["value"] == 1)).sum())
        dp = dst / p.name
        if dp.exists():
            ddf = pd.read_parquet(dp)
            ddf = ddf[ddf["behavior_id"] != args.behavior]          # replace this behavior only
            out = pd.concat([ddf, rows], ignore_index=True)
        else:
            out = rows.copy()
        print(f"  {p.name[:32]}… : copying {len(rows)} frame-rows ({n_pos} positive) → {dp.name[:32]}…"
              + ("  [dry-run]" if args.dry_run else ""))
        if not args.dry_run:
            out.to_parquet(dp, index=False)
        total += len(rows)

    print(f"\n{'would copy' if args.dry_run else 'copied'} {total} frame-rows of behavior {args.behavior} "
          f"for clips matching {wanted}.")
    if not args.dry_run:
        print("RESTART the labeler; the destination clips now carry the seeded behavior. "
              "(Seed labeling TIME is not transferred — it stays in the source project's event log; add it by hand.)")


if __name__ == "__main__":
    main()
