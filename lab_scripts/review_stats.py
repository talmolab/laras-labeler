#!/usr/bin/env python3
"""How good were a head's (or the model's) proposals? Statistics from the review decisions you made.

Every time you accept or reject a candidate in "Review candidates", the labeler logs it (events/*.jsonl).
This turns those decisions into numbers:

  * precision  = accepted / (accepted + rejected), with a 95% confidence interval. It is an UNBIASED
    estimate of "what fraction of this head's detections are real" ONLY if you reviewed in order = random
    (the default least-confident order deliberately shows the hard ones first).
  * how many accepted bouts you had to trim (were its start/end right?)
  * what predicted your calls: the head's confidence, and the bout's length
  * (optional) --compare: the same frames in ANOTHER project, e.g. your main labels, and where your two
    sets of judgments disagree, so you can re-watch those bouts and settle on a rule.

    python lab_scripts/review_stats.py ~/my-projects/approach_defs --behavior 0
    python lab_scripts/review_stats.py ~/my-projects/approach_defs --behavior 0 --track 1 \\
        --compare ~/my-projects/cage1rb_day1 --compare-track 1

Tracks are the STORED 0-based numbers (the GUI shows them +1). Needs pandas/numpy (the labeler env).
"""
import argparse, glob, json, math
from pathlib import Path

import numpy as np
import pandas as pd


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def auroc(positive: np.ndarray, score: np.ndarray) -> float:
    """P(a random accepted bout scores higher than a random rejected one) by ranks (Mann-Whitney)."""
    s = pd.Series(score).rank().to_numpy()
    n1, n0 = int(positive.sum()), int((~positive).sum())
    return float((s[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--behavior", type=int, required=True, help="behavior_id (0-based, see project.json)")
    ap.add_argument("--track", type=int, default=None, help="only decisions on this stored track (0-based)")
    ap.add_argument("--compare", default=None, help="another project whose labels to cross-check against")
    ap.add_argument("--compare-track", type=int, default=None, help="track to read in --compare (default: --track)")
    ap.add_argument("--csv", default=None, help="also write the per-decision table here")
    a = ap.parse_args()

    proj = Path(a.project).expanduser()
    manifest = json.load(open(proj / "project.json"))
    beh = next(b for b in manifest["behaviors"] if b["id"] == a.behavior)
    fps = float((manifest["videos"][0].get("fps")) or 50.0)
    ev = []
    for f in sorted(glob.glob(str(proj / "events" / "s-*.jsonl"))):
        for line in open(f):
            try:
                ev.append(json.loads(line))
            except ValueError:
                pass
    dec = [e for e in ev if e["type"] in ("candidate_accept", "candidate_reject") and e.get("behavior_id") == a.behavior
           and (a.track is None or e.get("track") == a.track)]
    if not dec:
        raise SystemExit("no accept/reject decisions found for that behavior/track")
    df = pd.DataFrame([{"video_id": e.get("video_id"), "track": e.get("track"), "decision": e["type"].replace("candidate_", ""),
                        "start": e["start"], "end": e["end"], "proba": e.get("proba"), "trimmed": bool(e.get("trimmed")),
                        "dwell_s": (e.get("dwell_ms") or 0) / 1000, "order": e.get("order")} for e in dec])
    df["dur_s"] = (df.end - df.start) / fps
    acc = (df.decision == "accept").to_numpy()
    n, k = len(df), int(acc.sum())
    lo, hi = wilson(k, n)

    head = beh.get("hidra") or {}
    print(f"{proj.name} · behavior {a.behavior} '{beh['name']}'" + (f" · HiDRA head {head.get('lab')}/{head.get('action')}" if head else " · own model"))
    print(f"reviewed {n} candidates (track {sorted(df.track.unique().tolist())}, order {sorted(set(map(str, df.order)))})")
    print(f"accepted {k}, rejected {n - k}  ->  PRECISION {k / n:.0%}   (95% CI {lo:.0%}-{hi:.0%})")
    if "random" not in set(map(str, df.order)):
        print("  NOTE: you did not review in 'random' order, so this over/under-states the true precision.")
    if n < 60:
        print(f"  NOTE: with {n} decisions the interval is wide; ~100 would narrow it to about +/-10%.")
    print(f"accepted bouts you had to trim: {int(df[acc].trimmed.sum())}/{k}   |   median time per decision {df.dwell_s.median():.0f}s")
    print(f"\nwhat predicted your accept/reject (AUROC, 0.5 = useless, 1.0 = perfect):"
          f"  head's confidence {auroc(acc, df.proba.fillna(0).to_numpy()):.2f}   |   bout length {auroc(acc, df.dur_s.to_numpy()):.2f}")
    for cut in (1.0, 1.5, 2.0):
        m = (df.dur_s >= cut).to_numpy()
        if m.any() and (~m).any():
            print(f"  bouts >= {cut:.1f}s: {int(m.sum()):3d} candidates, {acc[m].mean():4.0%} accepted   |   shorter: {int((~m).sum()):3d} candidates, {acc[~m].mean():4.0%} accepted")

    if a.compare:
        cmp = Path(a.compare).expanduser()
        vid = df.video_id.iloc[0]
        ctrack = a.compare_track if a.compare_track is not None else a.track
        lab = pd.read_parquet(cmp / "labels" / f"{vid}.parquet")
        lab = lab[lab.value.isin([0, 1])]
        if ctrack is not None:
            lab = lab[lab.track == ctrack]
        y = np.full(int(manifest["videos"][0]["n_frames"]), -1)
        y[lab.frame.to_numpy()] = lab.value.to_numpy()
        print(f"\ncross-check against the labels in {cmp.name} (track {ctrack}); bouts where the two disagree — worth re-watching:")
        agree = tot = 0
        for r in df.itertuples():
            seg = y[r.start:r.end]
            seg = seg[seg >= 0]
            if len(seg) < 5:
                continue
            tot += 1
            frac = float((seg == 1).mean())
            if (r.decision == "accept") == (frac >= 0.5):
                agree += 1
            else:
                print(f"  frames {r.start}-{r.end} ({r.dur_s:.1f}s): you {r.decision.upper()}ED it here, but {frac:.0%} of the {len(seg)} frames you labeled there are 'approach' in {cmp.name}")
        print(f"  the two sets of judgments agree on {agree}/{tot} bouts that overlap labeled frames")
    if a.csv:
        df.to_csv(a.csv, index=False)
        print(f"\nper-decision table -> {a.csv}")


if __name__ == "__main__":
    main()
