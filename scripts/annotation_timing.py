#!/usr/bin/env python
"""How long does annotation actually take — and is the human-in-the-loop loop cheaper than by hand?

Reads the append-only event log a labeling session writes (`<project>/events/*.jsonl`, see
`src/laras_labeler/events.py`) and rolls it up into ROUNDS: one round is the stretch of work between
two Trains of the same behavior — label / review, Train, look, repeat. Per round it separates

    label_s   human seconds spent painting labels from scratch     (the MANUAL workflow)
    review_s  human seconds spent judging model proposals          (the HITL workflow)
    wait_s    human seconds spent watching a train/predict bar
    compute_s machine seconds the human waited through
    feature_s machine seconds spent in the background (pre-warm) — nobody waited for these

and reports what each bought: bouts painted, candidates shown/accepted/rejected, and the model that
came out. The comparison the whole thing exists for is then just a division:

    seconds per bout painted by hand   vs   seconds per bout confirmed in review

Both numbers come from the same person on the same behavior in the same sitting, so the ratio is a
paired measurement, not a comparison across studies.

    python scripts/annotation_timing.py ~/laras-projects --pid single-cage-test
    python scripts/annotation_timing.py ~/laras-projects --pid single-cage-test --behavior 0 --csv rounds.csv

Caveats worth stating in any writeup this feeds:
  * "active" time is inferred from event gaps capped at --gap-cap seconds (the browser heartbeats
    every 10 s while its tab is visible). A gap longer than the cap is assumed to be a break and is
    NOT counted — so active time is a slight UNDER-estimate, in both arms equally.
  * a reviewed bout and a painted bout are not the same product: review only ever visits bouts the
    model already proposed, so the HITL arm is fast partly because the search is done for it. That is
    the point of the workflow, but it means "seconds per bout" is a cost ratio, not an accuracy claim.
    Read it next to the accuracy-per-minute curve, which this prints too.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from laras_labeler.events import EventLog, rounds_csv, summarize  # noqa: E402


class _Proj:
    """Just enough of a Project for the rollup (behavior names + per-clip fps), read straight from
    project.json so this script never needs sleap-io or the rest of the app installed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        m = json.loads((path / "project.json").read_text())
        self.name = m.get("name", path.name)
        self.behaviors = m.get("behaviors", [])
        self.videos = m.get("videos", [])


class _Store:
    def __init__(self, proj: _Proj) -> None:
        self.proj = proj

    def get(self, pid):
        return self.proj


def _fmt(sec: float | None) -> str:
    if sec is None:
        return "-"
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projects_root", help="the directory you pass to `laras-labeler`")
    ap.add_argument("--pid", required=True, help="project id (the folder name under projects_root)")
    ap.add_argument("--behavior", type=int, default=None, help="restrict to one behavior_id")
    ap.add_argument("--session", default=None, help="restrict to one session id (see --list)")
    ap.add_argument("--list", action="store_true", help="list the recorded sessions and exit")
    ap.add_argument("--gap-cap", type=float, default=15.0,
                    help="seconds; a gap between events longer than this is a break, not work (default 15)")
    ap.add_argument("--idle-break", type=float, default=120.0,
                    help="seconds; a heartbeat idle longer than this ends the active stretch (default 120)")
    ap.add_argument("--csv", type=Path, default=None, help="write the per-round table here")
    ap.add_argument("--json", type=Path, default=None, help="write the full rollup here")
    args = ap.parse_args(argv)

    root = Path(args.projects_root).expanduser() / args.pid
    if not (root / "project.json").exists():
        print(f"no project.json under {root}", file=sys.stderr)
        return 2
    proj = _Proj(root)
    log = EventLog(_Store(proj))

    if args.list:
        for s in log.sessions(args.pid):
            print(f"{s['session']:<28} {s['n_events']:>6} events  {s['start']} .. {s['end']}")
        return 0

    recs = log.read(args.pid, [args.session] if args.session else None)
    if not recs:
        print(f"no events recorded for {args.pid} - nothing has been annotated with event logging on "
              f"(look for {root / 'events'})", file=sys.stderr)
        return 1
    s = summarize(recs, proj, gap_cap_s=args.gap_cap, idle_break_s=args.idle_break,
                  behavior_id=args.behavior)

    print(f"{proj.name}  ({args.pid})   {s['n_events']} events, {s['n_rounds']} rounds")
    print()
    hdr = (f"{'#':>3} {'behavior':<16} {'started':<17} {'active':>7} {'label':>7} {'review':>7} "
           f"{'wait':>6} {'cpu':>6} {'hand':>5} {'shown':>6} {'ok':>4} {'no':>4} {'s/bout':>7} "
           f"{'s/dec':>6} {'AP':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in s["rounds"]:
        started = (r["start"] or "")[5:16].replace("T", " ")
        d = r["decisions"]
        spb = "-" if r["s_per_manual_bout"] is None else r["s_per_manual_bout"]
        spd = "-" if r["s_per_decision"] is None else r["s_per_decision"]
        ap = "-" if r["ap"] is None else format(r["ap"], ".3f")
        print(f"{r['round']:>3} {str(r['behavior'] or '-')[:16]:<16} {started:<17} "
              f"{_fmt(r['active_s']):>7} {_fmt(r['label_s']):>7} {_fmt(r['review_s']):>7} "
              f"{_fmt(r['wait_s']):>6} {_fmt(r['compute_s']):>6} "
              f"{r['manual_bouts']:>5} {r['candidates_shown']:>6} {r['accepted']:>4} {d.get('reject', 0):>4} "
              f"{spb:>7} {spd:>6} {ap:>6}")

    t, m, rv = s["totals"], s["totals"]["manual"], s["totals"]["review"]
    print()
    print(f"human time   {_fmt(t['active_s'])}   (labeling {_fmt(t['label_s'])} | "
          f"reviewing {_fmt(t['review_s'])} | waiting on jobs {_fmt(t['wait_s'])})")
    print(f"machine time {_fmt(t['compute_s'])} waited-for + {_fmt(t['feature_s'])} background")
    print()
    print("BY HAND    "
          f"{m['bouts']} bouts / {m['frames']} frames ({m['video_s']:.0f}s of video) in {_fmt(t['label_s'])}"
          + (f"  ->  {m['s_per_bout']}s per bout, {m['s_per_video_s']}s of work per second of video"
             if m["s_per_bout"] else "  ->  n/a"))
    print("IN REVIEW  "
          f"{rv['decisions']} decisions ({rv['accepted']} accepted) on {rv['video_s']:.0f}s of proposed "
          f"video in {_fmt(t['review_s'])}"
          + (f"  ->  {rv['s_per_decision']}s per decision, {rv['s_per_accepted_bout']}s per accepted bout"
             if rv["s_per_decision"] else "  ->  n/a"))
    if t["speedup_per_bout"]:
        print()
        print(f"==> a bout cost {t['speedup_per_bout']}x less human time through review "
              f"({m['s_per_bout']}s by hand vs {rv['s_per_accepted_bout']}s accepted)")
    elif not (m["bouts"] and rv["accepted"]):
        print()
        print("==> not comparable yet: the log needs BOTH hand-painted bouts and accepted candidates "
              "(one arm is empty)")

    if s["curve"]:
        print()
        print("accuracy per minute of human time:")
        for c in s["curve"]:
            print(f"  round {c['round']:>2}  {c['cum_active_min']:>6.1f} min  "
                  f"AP {c['ap'] if c['ap'] is not None else '-'}  f1 {c['f1'] if c['f1'] is not None else '-'}  "
                  f"({c['n_pos_bouts']} pos bouts: {c['n_seed_bouts']} seed + {c['n_candidate_bouts']} from review)")
    for ms in s["milestones"]:
        print(f"  reached AP {ms['ap']} after {ms['cum_active_min']} min of human time (round {ms['round']})")

    if args.csv:
        args.csv.write_text(rounds_csv(s), encoding="utf-8")
        print(f"\nwrote {args.csv}")
    if args.json:
        args.json.write_text(json.dumps(s, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
