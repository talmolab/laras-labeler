#!/usr/bin/env python3
"""Count the user ACTIONS logged per behavior — manual labeling gestures vs candidate-review decisions.

The annotation event log records every gesture the annotator makes. This tallies them per behavior so
you can compare "how many actions did labeling take" between a hand-labeled arm and a HITL-review arm.
Run it on each project (the manual/GT project reports manual gestures; the HITL arm reports review
decisions) and read the matching behavior rows off each.

    python scripts/action_counts.py C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-test
    python scripts/action_counts.py C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-hitl

Buckets:
  manual gestures  = paint_start, paint_commit, paint_cancel, label_trim, label_delete
  review decisions = candidate_{accept,reject,merge,split,reclassify,skip,undo}
  review other     = candidate_show, candidate_trim, candidate_replay, bout_review_open (navigation within review)
`paint_commit` and `candidate_accept`+`reject`… are the headline "one action per bout / per decision"
counts; the fuller gesture totals include boundary trims and cancels. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

MANUAL = {"paint_start", "paint_commit", "paint_cancel", "label_trim", "label_delete"}
DECISIONS = {f"candidate_{k}" for k in ("accept", "reject", "merge", "split", "reclassify", "skip", "undo")}
REVIEW_OTHER = {"candidate_show", "candidate_trim", "candidate_replay", "bout_review_open", "candidate_replay"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Tally manual vs review actions per behavior from the event log.")
    ap.add_argument("project", help="project dir (contains events/)")
    ap.add_argument("--names", action="store_true", help="also print behavior names from project.json")
    args = ap.parse_args(argv)

    proj = Path(args.project).expanduser()
    evdir = proj / "events"
    if not evdir.is_dir():
        sys.exit(f"no events/ under {proj}")
    names = {}
    pj = proj / "project.json"
    if pj.exists():
        names = {int(b["id"]): b.get("name", "") for b in json.loads(pj.read_text(encoding="utf-8")).get("behaviors", [])}

    # per behavior_id: counts per bucket + a raw type histogram
    manual = defaultdict(int); commits = defaultdict(int)
    decisions = defaultdict(int); accepts = defaultdict(int); rejects = defaultdict(int)
    review_other = defaultdict(int)
    seen_bids = set()
    for p in sorted(evdir.glob("*.jsonl")):
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            typ = str(ev.get("type", ""))
            bid = ev.get("behavior_id")
            if bid is None:
                continue
            bid = int(bid); seen_bids.add(bid)
            if typ in MANUAL:
                manual[bid] += 1
                if typ == "paint_commit":
                    commits[bid] += 1
            elif typ in DECISIONS:
                decisions[bid] += 1
                if typ == "candidate_accept":
                    accepts[bid] += 1
                elif typ == "candidate_reject":
                    rejects[bid] += 1
            elif typ in REVIEW_OTHER:
                review_other[bid] += 1

    print(f"{proj.name}")
    hdr = f"{'behavior':<22}{'paint_commit':>13}{'manual_gest':>13}{'decisions':>11}{'accept':>8}{'reject':>8}{'rev_other':>11}"
    print(hdr); print("-" * len(hdr))
    for bid in sorted(seen_bids):
        nm = (names.get(bid, "") or f"id {bid}")[:20]
        print(f"{nm:<22}{commits[bid]:>13}{manual[bid]:>13}{decisions[bid]:>11}"
              f"{accepts[bid]:>8}{rejects[bid]:>8}{review_other[bid]:>11}")
    print("\nmanual gestures = paint_*/label_* · decisions = candidate_accept/reject/merge/split/… · "
          "paint_commit & (accept+reject) are the 'one action per bout / decision' counts.")


if __name__ == "__main__":
    main()
