#!/usr/bin/env python3
"""Per-CLIP annotation timing, for the hand-vs-HITL efficiency experiment.

The built-in rollup (`/api/projects/{pid}/timing`, events.summarize) groups by ROUND — the work
between two Trains of one behavior. In the hand arm there is no Train, so every clip's labeling of a
behavior accumulates into one open round and you cannot read off "how long did clip 10 take". This
script re-runs the SAME active-time accounting (it imports events._phase / _PHASE_BUCKET /
_paint_state and the same gap-cap / idle-break constants, so it can never drift from the labeler)
but buckets by (video_id, behavior) — and optionally by track — so each clip gets its own label_s,
review_s, and bout counts. That per-clip number is what the paired experiment compares.

Usage
    python scripts/timing_per_clip.py <PROJECT_DIR or its events/ dir> [--fps 50] [--behavior N]
                                      [--by-track] [--csv out.csv]

    # e.g. on the labeler machine (default projects root is ~/laras-projects):
    python scripts/timing_per_clip.py ~/laras-projects/doom-annotation-time-test --fps 50

`--fps` only scales the *_video_s columns (seconds of video annotated); label_s/review_s do not
depend on it. Point it at the project directory (it finds the events/ folder) or straight at an
events/ folder full of <session>.jsonl files.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Reuse the labeler's own phase logic so this stays bit-for-bit consistent with the in-app rollup.
# events.py is pure-stdlib, so this import works without the heavy runtime deps installed.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from laras_labeler.events import (  # noqa: E402
    ACTIVE_GAP_CAP_S, IDLE_BREAK_S, _PHASE_BUCKET, _POS, _NEG, _paint_state, _phase,
)


def _read_events(events_dir: Path) -> list[dict]:
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
                    continue  # a torn last line costs one event, not the file
    # Same global ordering the rollup uses: by client timestamp, then sequence.
    recs.sort(key=lambda r: (float(r.get("t_ms") or 0), int(r.get("seq") or 0)))
    return recs


def _blank(vid, bid) -> dict:
    return {"video_id": vid, "behavior_id": bid, "active_s": 0.0, "label_s": 0.0, "review_s": 0.0,
            "wait_s": 0.0, "other_s": 0.0, "manual_bouts": 0, "manual_frames": 0,
            "manual_video_s": 0.0, "manual_neg_bouts": 0, "manual_unknown_bouts": 0,
            "deletes": 0, "trims": 0, "decisions": 0, "candidates_shown": 0}


def per_clip(recs: list[dict], fps: float = 50.0, behavior_id: int | None = None,
             gap_cap_s: float = ACTIVE_GAP_CAP_S, idle_break_s: float = IDLE_BREAK_S,
             by_track: bool = False) -> list[dict]:
    """Mirror events.summarize's gap accounting, but charge each gap to the (video, behavior[, track])
    that was open when the gap STARTED — exactly how the rollup charges it to the round that was open."""
    buckets: dict[tuple, dict] = {}

    def B(vid, bid, track=None) -> dict:
        key = (vid, bid, track) if by_track else (vid, bid)
        b = buckets.get(key)
        if b is None:
            b = _blank(vid, bid)
            if by_track:
                b["track"] = track
            buckets[key] = b
        return b

    phase = "label"
    prev_ms = None
    prev_key = None  # (video_id, bid[, track]) of the previous client event — open when a gap starts

    for ev in recs:
        t = float(ev.get("t_ms") or 0)
        typ = str(ev.get("type"))
        raw_bid = ev.get("behavior_id")
        bid = int(raw_bid) if isinstance(raw_bid, (int, float)) else None
        vid = ev.get("video_id")
        track = ev.get("track") if by_track else None
        if behavior_id is not None and bid != behavior_id:
            continue

        # A clear_labels.py marker (a label_clear carrying a `note`) marks a clip+behavior whose labels
        # were wiped for a clean re-label: drop everything accumulated for it BEFORE the marker (the
        # discarded first pass), so only post-clear time counts. NOTE: routine edits also emit
        # label_clear (the labeler DELETEs a range before each PUT) — those have NO `note` and must be
        # ignored here, or every paint would reset the count. Events are time-ordered, so resetting the
        # bucket at the marker does exactly the right thing.
        if typ == "label_clear" and ev.get("note") and vid is not None:
            for k in [k for k in buckets if k[0] == vid and k[1] == bid]:
                del buckets[k]
            if prev_key is not None and prev_key[0] == vid and prev_key[1] == bid:
                prev_ms = None  # don't charge the gap that straddles the clear
            continue

        # --- time accounting (client events only; server records are instantaneous notes) ---
        if ev.get("src") != "server":
            if prev_ms is not None and t >= prev_ms and prev_key is not None:
                charged = min((t - prev_ms) / 1000.0, gap_cap_s)
                b = buckets.get(prev_key)
                if b is not None:
                    b["active_s"] += charged
                    b[_PHASE_BUCKET.get(phase, "other_s")] += charged
            idle_ms = ev.get("idle_ms")
            hidden = ev.get("visible") is False or typ == "tab_hidden"
            broke = hidden or (isinstance(idle_ms, (int, float)) and idle_ms / 1000.0 > idle_break_s)
            prev_ms = None if broke else t

        phase = _phase(typ, phase)
        if ev.get("src") != "server":
            prev_key = (vid, bid, track) if by_track else (vid, bid)
            B(vid, bid, track)  # ensure the open bucket exists so the NEXT gap can charge to it

        # --- what was produced (charged to the clip the event names) ---
        if typ == "paint_commit":
            n = int(ev.get("n_frames") or max(0, int(ev.get("end") or 0) - int(ev.get("start") or 0)))
            b = B(vid, bid, track)
            state = _paint_state(ev)
            if state == _POS:
                b["manual_bouts"] += 1
                b["manual_frames"] += n
                b["manual_video_s"] += n / (fps or 50.0)
            elif state == _NEG:
                b["manual_neg_bouts"] += 1
            else:
                b["manual_unknown_bouts"] += 1
        elif typ == "label_delete":
            B(vid, bid, track)["deletes"] += 1
        elif typ == "label_trim":
            B(vid, bid, track)["trims"] += 1
        elif typ == "candidate_show":
            B(vid, bid, track)["candidates_shown"] += 1
        elif typ.startswith("candidate_") and typ.split("_", 1)[1] in (
                "accept", "reject", "merge", "split", "reclassify", "skip"):
            B(vid, bid, track)["decisions"] += 1

    rows = []
    for b in buckets.values():
        for k in ("active_s", "label_s", "review_s", "wait_s", "other_s", "manual_video_s"):
            b[k] = round(b[k], 1)
        b["s_per_manual_bout"] = round(b["label_s"] / b["manual_bouts"], 1) if b["manual_bouts"] else None
        rows.append(b)
    rows.sort(key=lambda r: (str(r["video_id"]), r["behavior_id"] if r["behavior_id"] is not None else -1,
                             r.get("track") if r.get("track") is not None else -1))
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Per-clip annotation timing (hand-vs-HITL experiment).")
    ap.add_argument("project", help="a project directory (its events/ folder is used) or an events/ folder")
    ap.add_argument("--fps", type=float, default=50.0, help="fps for the *_video_s columns (default 50)")
    ap.add_argument("--behavior", type=int, default=None, help="only this behavior_id")
    ap.add_argument("--by-track", action="store_true", help="also split by track (bout counts per track; "
                    "note: label time cannot be cleanly attributed to a track)")
    ap.add_argument("--csv", help="also write the table to this CSV path")
    args = ap.parse_args(argv)

    root = Path(args.project).expanduser()
    events_dir = root / "events" if (root / "events").is_dir() else root
    if not events_dir.is_dir():
        sys.exit(f"no events found: {events_dir} is not a directory (point me at the project dir or its events/ dir)")

    recs = _read_events(events_dir)
    rows = per_clip(recs, fps=args.fps, behavior_id=args.behavior, by_track=args.by_track)
    if not rows:
        sys.exit(f"no client events in {events_dir} (nothing labeled yet, or wrong --behavior).")

    cols = ["video_id", "behavior_id"] + (["track"] if args.by_track else []) + [
        "label_s", "review_s", "other_s", "active_s", "manual_bouts", "manual_frames",
        "manual_video_s", "s_per_manual_bout"]
    w = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))
    print()
    print(f"{len(recs)} events · total label_s {round(sum(r['label_s'] for r in rows), 1)} "
          f"· total review_s {round(sum(r['review_s'] for r in rows), 1)} "
          f"· total manual_bouts {sum(r['manual_bouts'] for r in rows)}")
    print("(total label_s should match the sum of the rounds CSV's label_s, minus rounding.)")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
