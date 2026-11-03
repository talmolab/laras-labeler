#!/usr/bin/env python3
"""Pick 5-min clips across a cage's lifespan from re-ID identity tables, then cut them.

Hierarchical sampling, coarse to fine, each level spread evenly and drawn with a fixed seed:

    age bin (default: each 7-day week of the recording, counted from the first recorded day)
      -> day      (--days-per-bin days, spread across the bin; days with too few hours skipped)
        -> hour   (one hour per --hour-strata stratum, e.g. "12-17,18-23" = early and late dark phase)
          -> window (a --minutes window inside the hour in which every mouse carries an identity
                     in >= --min-coverage of frames; drawn at random among the qualifying windows)

The window rule is what keeps clips off an empty cage (the first hour of a batch), a huddle under the
hut, or a stretch the re-ID could not decode. It does bias toward times the mice are visible, which is
what a clip to label needs; say so if clips are used to estimate how often a behavior happens.

If a chosen hour has no qualifying window, the next hour in the same stratum is tried, then the next
day in the bin. Every choice is written to `plan.csv`, so a run can be reviewed (`--dry-run`) and
repeated exactly (same `--seed`).

Needs `reid_to_slp.py` in the same folder (it does the re-labeling and cutting).

    # look at the plan first (reads only the identity tables of the hours it considers)
    uv run --python 3.12 --with sleap-io --with imageio-ffmpeg python sample_reid_clips.py \\
        /Volumes/talmo/than/26-09-reid-v2/studies/reid_remaining_20261007/b21/ident_tables \\
        --out ~/reid_b21_lifespan --dry-run

    # then cut them
    uv run --python 3.12 --with sleap-io --with imageio-ffmpeg python sample_reid_clips.py \\
        /Volumes/talmo/than/26-09-reid-v2/studies/reid_remaining_20261007/b21/ident_tables \\
        --out ~/reid_b21_lifespan
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reid_to_slp

STEM = re.compile(r"^(?P<batch>b\d+)_(?P<day>\d{8})_h(?P<hour>\d{2})(?P<rest>.*)$")


def index(folder: Path) -> list[dict]:
    rows = []
    for p in sorted(folder.glob("*.ident.npz")):
        stem = p.name[: -len(".ident.npz")]
        m = STEM.match(stem)
        if not m:
            continue
        rows.append({"path": p, "stem": stem, "hour": int(m["hour"]),
                     "day": dt.datetime.strptime(m["day"], "%Y%m%d").date()})
    return rows


def spread(items: list, k: int, rng: np.random.Generator) -> list:
    """k items spread evenly over the list, each drawn at random inside its own equal slice."""
    if k >= len(items):
        return list(items)
    edges = np.linspace(0, len(items), k + 1)
    return [items[int(rng.integers(int(np.floor(a)), max(int(np.floor(a)) + 1, int(np.floor(b)))))]
            for a, b in pairwise(edges)]


def pick_window(path: Path, minutes: float, min_cov: float, step_s: float,
                rng: np.random.Generator) -> dict | None:
    t = reid_to_slp.load_table(path)
    ident, meta = t["ident"], t["meta"]
    fps = float(Fraction(str(meta.get("fps") or "50")))           # "50/1" or "50"
    n_mice = int(meta.get("n_mice") or ident.shape[1])
    all_in = (ident >= 0).sum(axis=1) >= n_mice                      # every mouse has an identity
    win, step = round(minutes * 60 * fps), max(1, round(step_s * fps))
    if len(all_in) < win:
        return None
    csum = np.concatenate([[0], np.cumsum(all_in)])
    starts = np.arange(0, len(all_in) - win + 1, step)
    cov = (csum[starts + win] - csum[starts]) / win
    ok = starts[cov >= min_cov]
    if len(ok) == 0:
        return {"best_coverage": float(cov.max()), "start": None}
    s = int(rng.choice(ok))
    return {"start": s, "end": s + win, "coverage": float(cov[starts == s][0]),
            "best_coverage": float(cov.max()), "age_days": meta.get("age_days"), "fps": fps}


def parse_strata(text: str) -> list[range]:
    out = []
    for part in text.split(","):
        a, b = (int(x) for x in part.split("-"))
        out.append(range(a, b + 1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", help="a batch's ident_tables/ folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin-days", type=int, default=7, help="age-bin width in days (default 7)")
    ap.add_argument("--days-per-bin", type=int, default=1)
    ap.add_argument("--hour-strata", default="12-17,18-23",
                    help="hour ranges, one clip from each per chosen day (default dark phase: 12-17,18-23)")
    ap.add_argument("--skip-days", type=int, default=3,
                    help="skip the first N recorded days (cage entry; weakest identities). Default 3")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--min-coverage", type=float, default=0.8,
                    help="share of the window's frames in which every mouse has an identity (default 0.8)")
    ap.add_argument("--min-hours-per-day", type=int, default=12, help="skip days with fewer table hours")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="write plan.csv only, cut nothing")
    ap.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    rows = index(Path(args.tables).expanduser())
    if not rows:
        print("no *.ident.npz tables found", file=sys.stderr)
        return 1
    by_day: dict[dt.date, list[dict]] = {}
    for r in rows:
        by_day.setdefault(r["day"], []).append(r)
    days = sorted(by_day)
    first = days[0]
    usable = [d for d in days[args.skip_days:] if len(by_day[d]) >= args.min_hours_per_day]
    strata = parse_strata(args.hour_strata)

    bins: dict[int, list[dt.date]] = {}
    for d in usable:
        bins.setdefault((d - first).days // args.bin_days, []).append(d)

    plan = []
    for b in sorted(bins):
        bin_days = bins[b]
        chosen = spread(bin_days, args.days_per_bin, rng)
        # fallback order: the chosen days first, then the bin's other days nearest to them
        rest = [d for d in bin_days if d not in chosen]
        for d in chosen:
            for si, stratum in enumerate(strata):
                cands = [d] + sorted(rest, key=lambda x: abs((x - d).days))
                done = False
                for day in cands:
                    hours = [r for r in by_day[day] if r["hour"] in stratum]
                    rng.shuffle(hours)
                    for r in hours:
                        w = pick_window(r["path"], args.minutes, args.min_coverage, 10, rng)
                        if w and w.get("start") is not None:
                            plan.append({"bin": b, "stratum": f"{stratum.start}-{stratum.stop - 1}",
                                         "stem": r["stem"], "day": str(day), "hour": r["hour"],
                                         "age_days": w["age_days"], "start": w["start"], "end": w["end"],
                                         "coverage": round(w["coverage"], 3), "path": str(r["path"])})
                            print(f"bin {b:2d} ({stratum.start:02d}-{stratum.stop - 1:02d}h): {r['stem']}  "
                                  f"P{w['age_days']}  frames {w['start']}-{w['end']}  "
                                  f"all mice identified {w['coverage']:.0%}")
                            done = True
                            break
                    if done:
                        break
                if not done:
                    print(f"bin {b:2d} ({stratum.start:02d}-{stratum.stop - 1:02d}h): no qualifying window "
                          f"(days {bin_days[0]}..{bin_days[-1]})")

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "plan.csv").open("w", newline="") as f:
        if plan:
            w = csv.DictWriter(f, fieldnames=list(plan[0]))
            w.writeheader()
            w.writerows(plan)
    (out / "plan.args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"{len(plan)} clips planned ({len(plan) * args.minutes:.0f} min); plan: {out / 'plan.csv'}")
    if args.dry_run or not plan:
        return 0

    if reid_to_slp.find_ffmpeg() is None:
        print("cutting needs ffmpeg: add `--with imageio-ffmpeg` to the uv command", file=sys.stderr)
        return 1
    path_map = [tuple(m.split("=", 1)) for m in args.path_map]
    if not path_map and Path("/Volumes/talmo").exists():
        path_map = [("/mnt/vast", "/Volumes/talmo"), ("/root/vast", "/Volumes/talmo")]
    results = []
    for i, p in enumerate(plan, 1):
        ns = SimpleNamespace(out=str(out), clip=(int(p["start"]), int(p["end"])))
        row = reid_to_slp.process(Path(p["path"]), ns, path_map)
        results.append({**p, **row})
        print(f"[{i}/{len(plan)}] cut {Path(row['clip_video']).name}")
    (out / "clips.summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"done: {len(results)} clips in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
