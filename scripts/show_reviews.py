#!/usr/bin/env python3
"""Dump every candidate REVIEW decision (accept/reject/…) for one behavior, one row per decision.

The per-clip timing table (timing_per_clip.py) gives aggregates; this shows the individual calls so
you can sanity-check them: which bout, how long you looked (dwell), the model's confidence, whether
you trimmed it, and — for a directed behavior — the actor→target pair you committed. Reads the same
per-project event log, pure stdlib (runs with base python).

Usage
    python scripts/show_reviews.py <PROJECT_DIR> --behavior 1
    python scripts/show_reviews.py C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-hitl --behavior 1 --fps 50

--behavior is the behavior_id (see the timing table). Omit it to list every behavior's decisions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DECISIONS = ("accept", "reject", "merge", "split", "reclassify", "skip")


def _read(events_dir: Path) -> list[dict]:
    recs: list[dict] = []
    for p in sorted(events_dir.glob("*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    recs.sort(key=lambda r: (float(r.get("t_ms") or 0), int(r.get("seq") or 0)))
    return recs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Per-decision candidate review dump for one behavior.")
    ap.add_argument("project", help="project directory (its events/ folder) or an events/ folder")
    ap.add_argument("--behavior", type=int, default=None, help="behavior_id to show (default: all)")
    ap.add_argument("--fps", type=float, default=50.0, help="fps for the video-seconds column")
    ap.add_argument("--csv", help="also write the rows to this CSV path")
    args = ap.parse_args(argv)

    root = Path(args.project).expanduser()
    events_dir = root / "events" if (root / "events").is_dir() else root
    if not events_dir.is_dir():
        sys.exit(f"no events found: {events_dir} is not a directory")

    rows = []
    for ev in _read(events_dir):
        typ = str(ev.get("type"))
        if not typ.startswith("candidate_"):
            continue
        kind = typ.split("_", 1)[1]
        if kind not in _DECISIONS:
            continue
        bid = ev.get("behavior_id")
        if args.behavior is not None and bid != args.behavior:
            continue
        s = ev.get("trim_start") if ev.get("trim_start") is not None else ev.get("start")
        e = ev.get("trim_end") if ev.get("trim_end") is not None else ev.get("end")
        nf = ev.get("n_frames")
        dwell = ev.get("dwell_ms")
        tgt = ev.get("target")
        rows.append({
            "clip": str(ev.get("video_id") or "?")[:20],
            "track": ev.get("track"),
            "decision": kind,
            "start": s, "end": e,
            "frames": nf,
            "video_s": round((nf or 0) / (args.fps or 50.0), 2),
            "dwell_s": round(dwell / 1000.0, 1) if isinstance(dwell, (int, float)) else None,
            "proba": round(ev.get("proba"), 2) if isinstance(ev.get("proba"), (int, float)) else None,
            "trimmed": bool(ev.get("trimmed")),
            "replays": ev.get("replays"),
            "target": tgt if (isinstance(tgt, int) and tgt >= 0) else "",
            "behavior_id": bid,
        })

    if not rows:
        sys.exit("no review decisions found (nothing reviewed yet, or wrong --behavior).")

    cols = ["clip", "track", "decision", "start", "end", "frames", "video_s", "dwell_s",
            "proba", "trimmed", "replays", "target"]
    w = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))

    n = len(rows)
    acc = sum(1 for r in rows if r["decision"] in ("accept", "merge", "split"))
    rej = sum(1 for r in rows if r["decision"] == "reject")
    dwells = [r["dwell_s"] for r in rows if r["dwell_s"] is not None]
    med = sorted(dwells)[len(dwells) // 2] if dwells else None
    print()
    print(f"{n} decisions · accepted {acc} · rejected {rej}"
          + (f" · accept rate {acc / n:.0%}" if n else "")
          + (f" · median dwell {med}s" if med is not None else ""))

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            wr = _csv.DictWriter(f, fieldnames=cols + ["behavior_id"], extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
