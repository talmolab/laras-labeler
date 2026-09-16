#!/usr/bin/env python3
"""Clear one behavior's labels on chosen clips, for a clean re-label — in the SAME project.

Labels live in per-clip parquets (labels/<video_id>.parquet: frame, track, behavior_id, value,
source). This removes every row of the chosen behavior on the chosen clips, so you re-label them from
a blank slate while the other clips keep their labels. Each parquet is backed up to .parquet.bak
first, and a `label_clear` marker is appended to the event log so timing_per_clip.py automatically
counts ONLY the re-label (not the discarded first pass) — no manual timestamps.

    # STOP the labeler first (it caches labels in memory and would re-save the old ones), then:
    python scripts/clear_labels.py C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-test \\
        --behavior 0 --clips clip_001,clip_010
    # ...restart the labeler and re-label those clips.

--behavior is the behavior_id (jump down = 0). --clips is comma-separated substrings matched against
each clip's video_id. Reversible: the removed rows are in the .bak files.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Clear a behavior's labels on chosen clips for a clean re-label.")
    ap.add_argument("project", help="the project directory (contains labels/ and events/)")
    ap.add_argument("--behavior", type=int, required=True, help="behavior_id to clear (jump down = 0)")
    ap.add_argument("--clips", required=True,
                    help="comma-separated video_id substrings, e.g. clip_001,clip_010")
    ap.add_argument("--no-marker", action="store_true",
                    help="don't append a label_clear event (timing won't auto-split at the redo)")
    args = ap.parse_args(argv)

    proj = Path(args.project).expanduser()
    labels_dir = proj / "labels"
    if not labels_dir.is_dir():
        sys.exit(f"no labels/ folder in {proj} (point me at the project directory)")
    subs = [s.strip() for s in args.clips.split(",") if s.strip()]
    if not subs:
        sys.exit("--clips is empty")
    bid = args.behavior

    now = datetime.now(timezone.utc)
    iso, ms = now.isoformat(), now.timestamp() * 1000.0
    markers = []

    matched = 0
    for pq in sorted(labels_dir.glob("*.parquet")):
        vid = pq.stem
        if not any(s in vid for s in subs):
            continue
        matched += 1
        df = pd.read_parquet(pq)
        n = int((df["behavior_id"] == bid).sum()) if "behavior_id" in df.columns else 0
        if n == 0:
            print(f"{vid[:40]}: no behavior {bid} rows — nothing to clear")
            continue
        bak = pq.with_suffix(".parquet.bak")
        if not bak.exists():
            shutil.copy2(pq, bak)
        df[df["behavior_id"] != bid].reset_index(drop=True).to_parquet(pq, index=False)
        print(f"{vid[:40]}: cleared {n} frame-rows of behavior {bid}  (backup -> {bak.name})")
        markers.append({"session": "server", "src": "server", "t_srv": iso, "type": "label_clear",
                        "video_id": vid, "behavior_id": bid, "t": iso, "t_ms": ms,
                        "note": "cleared by clear_labels.py for re-label"})

    if matched == 0:
        sys.exit(f"no clip parquet matched {subs} in {labels_dir}")

    if markers and not args.no_marker:
        events_dir = proj / "events"
        events_dir.mkdir(parents=True, exist_ok=True)
        sp = events_dir / "server.jsonl"
        with sp.open("a", encoding="utf-8") as f:
            for mk in markers:
                f.write(json.dumps(mk, separators=(",", ":")) + "\n")
        print(f"\nappended {len(markers)} label_clear marker(s) to events/{sp.name} "
              f"— timing_per_clip.py will count only the re-label for these clips.")

    if markers:
        print("\nDone. Restart the labeler and re-label the cleared clips. Old labels are in the .bak files.")
    else:
        print("\nNothing cleared.")


if __name__ == "__main__":
    main()
