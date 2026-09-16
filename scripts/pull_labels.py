#!/usr/bin/env python3
"""Pull annotated BOUTS out of a project's label store into a tidy CSV (one row per bout).

The labels live as per-frame rows in labels/<vid>.parquet (frame, track, behavior_id, value, source,
target). This reconstructs bouts by run-length-encoding consecutive frames that share the same
(value, source, target) within a (clip, behavior, track) — the same reconstruction the labeler uses —
and writes them out with behavior names and seconds, so you can inspect or share what's been labeled.

Usage
    python scripts/pull_labels.py <PROJECT_DIR> [--csv out.csv] [--fps 50]
                                  [--source manual] [--value pos|all] [--behavior N]

    # everything hand-labeled, positives only (the default), printed + written:
    python scripts/pull_labels.py C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-test --csv manual_bouts.csv

value: pos (default) = Happening bouts only; all = also Not-happening (0) and Unknown (2).
source: any (default) = every bout; manual = hand-labeled only (excludes HITL candidate-accepted).
Needs pandas — run with the labeler's env python:
    & "C:\\Users\\TalmoLab\\AppData\\Roaming\\uv\\tools\\laras-labeler\\Scripts\\python.exe" scripts\\pull_labels.py ...
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_VALUE_NAME = {1: "happening", 0: "not-happening", 2: "unknown"}


def _bouts(df: pd.DataFrame) -> list[dict]:
    """RLE one clip's per-frame rows into bouts: a maximal run of consecutive frames with the same
    (track, behavior_id, value, source, target)."""
    out: list[dict] = []
    if df.empty:
        return out
    for (track, bid), g in df.groupby(["track", "behavior_id"], sort=True):
        g = g.sort_values("frame")
        fr = g["frame"].to_numpy()
        val = g["value"].to_numpy()
        src = g["source"].astype("object").to_numpy()
        tgt = g["target"].to_numpy()
        # a run breaks where the frame is not contiguous OR any of value/source/target changes
        brk = np.ones(len(fr), dtype=bool)
        brk[1:] = (fr[1:] != fr[:-1] + 1) | (val[1:] != val[:-1]) | (src[1:] != src[:-1]) | (tgt[1:] != tgt[:-1])
        starts = np.flatnonzero(brk)
        ends = np.append(starts[1:], len(fr))
        for s, e in zip(starts, ends):
            out.append({"track": int(track), "behavior_id": int(bid),
                        "value": int(val[s]), "source": str(src[s]), "target": int(tgt[s]),
                        "start": int(fr[s]), "end": int(fr[e - 1]) + 1})   # end EXCLUSIVE
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Pull annotated bouts from a project's label store to CSV.")
    ap.add_argument("project", help="project directory (must contain project.json and labels/)")
    ap.add_argument("--csv", help="write the bouts to this CSV path")
    ap.add_argument("--fps", type=float, default=None, help="override fps (default: per-clip from project.json, else 50)")
    ap.add_argument("--source", default="any", help="'any' (default) or 'manual' (hand-labeled only)")
    ap.add_argument("--value", default="pos", choices=["pos", "all"], help="'pos' = Happening only (default); 'all' = include 0/2")
    ap.add_argument("--behavior", type=int, default=None, help="only this behavior_id")
    args = ap.parse_args(argv)

    root = Path(args.project).expanduser()
    pj = root / "project.json"
    if not pj.exists():
        sys.exit(f"no project.json in {root}")
    man = json.loads(pj.read_text(encoding="utf-8"))
    names = {int(b["id"]): b.get("name") for b in man.get("behaviors", [])}
    default_fps = float(man.get("fps") or 50.0)

    rows: list[dict] = []
    for v in man.get("videos", []):
        vid = v.get("video_id")
        p = root / "labels" / f"{vid}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "target" not in df.columns:          # tolerate pre-directed label files
            df["target"] = np.int16(-1)
        if "source" not in df.columns:
            df["source"] = "manual"
        if args.behavior is not None:
            df = df[df["behavior_id"] == args.behavior]
        fps = float(args.fps or v.get("fps") or default_fps)
        for b in _bouts(df):
            if args.value == "pos" and b["value"] != 1:
                continue
            if args.source != "any" and b["source"] != args.source:
                continue
            n = b["end"] - b["start"]
            rows.append({
                "clip": str(vid)[:24],
                "behavior_id": b["behavior_id"],
                "behavior": names.get(b["behavior_id"], "?"),
                "track": b["track"],
                "value": _VALUE_NAME.get(b["value"], b["value"]),
                "start": b["start"], "end": b["end"], "n_frames": n,
                "seconds": round(n / (fps or 50.0), 2),
                "source": b["source"],
                "target": b["target"] if b["target"] >= 0 else "",
                "clip_full": vid,
            })

    if not rows:
        sys.exit("no bouts found (nothing labeled yet, or filters excluded everything).")

    rows.sort(key=lambda r: (r["behavior_id"], r["clip"], r["track"], r["start"]))
    cols = ["clip", "behavior", "track", "value", "start", "end", "n_frames", "seconds", "source", "target"]
    w = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))

    # summary per behavior (positives)
    print()
    pos = [r for r in rows if r["value"] == "happening"]
    by: dict[str, list] = {}
    for r in pos:
        by.setdefault(r["behavior"], []).append(r)
    for name, rs in sorted(by.items()):
        secs = sum(r["seconds"] for r in rs)
        clips = len({r["clip_full"] for r in rs})
        print(f"{name}: {len(rs)} bouts · {clips} clips · {secs / 60:.1f} min of behavior")
    print(f"TOTAL: {len(rows)} rows ({len(pos)} positive bouts)")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            wr = _csv.DictWriter(f, fieldnames=cols + ["clip_full", "behavior_id"], extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
