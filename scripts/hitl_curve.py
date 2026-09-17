#!/usr/bin/env python3
"""HITL learning curve — annotation quality (F1 vs ground truth) as a function of review EFFORT.

Replays the arm's review decisions in the order you made them (from the event log), accumulating the
bouts you accepted, and after each decision recomputes bout-level precision/recall/F1 against the
ground-truth project — over the clip/track lanes you've touched so far. The x-axis is cumulative
review effort: minutes spent, frames reviewed, or decisions made. This is the "how much review buys
how much accuracy" curve.

    python scripts/hitl_curve.py --truth <GT> --pred <ARM> --behaviors grooming \
        --videos clip_002,clip_005,clip_007,clip_010,clip_011 --out groom_curve.html --csv groom_curve.csv

Accepted bouts are read from candidate_accept / merge / split events (their trim range); rejects add
to effort but not to the accepted set. --videos filters by substring. Needs pandas — labeler env
python. Writes a self-contained HTML plot (open/screenshot) and optionally a CSV.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _manifest(p): return json.loads((p / "project.json").read_text(encoding="utf-8"))
def _bids(m): return {b.get("name", "").strip().lower(): int(b["id"]) for b in m.get("behaviors", [])}


def _gt_runs(proj: Path, vid: str, bid: int, track: int):
    p = proj / "labels" / f"{vid}.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    df = df[(df["behavior_id"] == bid) & (df["track"] == track) & (df["value"] == 1)]
    if df.empty:
        return []
    fr = np.sort(df["frame"].to_numpy()); runs = []; s = prev = int(fr[0])
    for f in fr[1:]:
        if f == prev + 1: prev = int(f)
        else: runs.append((s, prev + 1)); s = prev = int(f)
    runs.append((s, prev + 1)); return runs


def _read_events(events_dir: Path):
    recs = []
    for p in sorted(events_dir.glob("*.jsonl")):
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if line:
                try: recs.append(json.loads(line))
                except json.JSONDecodeError: pass
    recs.sort(key=lambda r: (float(r.get("t_ms") or 0), int(r.get("seq") or 0)))
    return recs


def _iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    u = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / u if u else 0.0


def _merge(intervals):
    """Union overlapping/adjacent accepted ranges into distinct bouts — so a merge event or a
    re-accept doesn't count as an extra bout (matches how the label store RLE-collapses runs)."""
    if not intervals:
        return []
    iv = sorted(intervals); out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def _prf(acc_by_lane, gt_by_lane, iou):
    tp = fp = fn = 0
    lanes = set(acc_by_lane) | set(gt_by_lane)
    for lane in lanes:
        gt = list(gt_by_lane.get(lane, [])); pr = _merge(acc_by_lane.get(lane, []))
        pairs = sorted(((_iou(g, p), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pr)), reverse=True)
        gm, pm = set(), set()
        for v, gi, pi in pairs:
            if v >= iou and gi not in gm and pi not in pm:
                gm.add(gi); pm.add(pi)
        tp += len(gm); fn += len(gt) - len(gm); fp += len(pr) - len(pm)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return tp, fp, fn, prec, rec, f1


def main(argv=None):
    ap = argparse.ArgumentParser(description="HITL F1-vs-effort learning curve.")
    ap.add_argument("--truth", required=True); ap.add_argument("--pred", required=True)
    ap.add_argument("--behaviors", required=True, help="single behavior name")
    ap.add_argument("--videos", help="comma-separated video_id substrings (default: all reviewed)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default="hitl_curve.html"); ap.add_argument("--csv")
    args = ap.parse_args(argv)

    truth, pred = Path(args.truth).expanduser(), Path(args.pred).expanduser()
    tb, pbids = _bids(_manifest(truth)), _bids(_manifest(pred))
    bnm = args.behaviors.strip().lower()
    if bnm not in tb or bnm not in pbids:
        sys.exit(f"behavior {args.behaviors!r} not found in both projects")
    t_bid, p_bid = tb[bnm], pbids[bnm]
    vfilter = [v.strip() for v in args.videos.split(",")] if args.videos else None

    recs = _read_events(pred / "events")
    DEC = {"accept", "merge", "split", "reject"}
    acc_by_lane, gt_cache, touched = {}, {}, set()
    pts = []
    cum_ms = cum_frames = 0
    n_dec = 0
    for ev in recs:
        typ = str(ev.get("type", ""))
        if not typ.startswith("candidate_") or typ.split("_", 1)[1] not in DEC:
            continue
        if int(ev.get("behavior_id", -999)) != p_bid:
            continue
        vid = ev.get("video_id"); tr = int(ev.get("track") or 0)
        if vfilter and not any(f in str(vid) for f in vfilter):
            continue
        kind = typ.split("_", 1)[1]
        n_dec += 1
        cum_ms += float(ev.get("dwell_ms") or 0)
        cum_frames += int(ev.get("n_frames") or 0)
        lane = (vid, tr)
        if lane not in touched:
            touched.add(lane)
            gt_cache[lane] = _gt_runs(truth, vid, t_bid, tr)
        if kind in ("accept", "merge", "split"):
            s = ev.get("trim_start"); e = ev.get("trim_end")
            s = int(s if s is not None else ev.get("start")); e = int(e if e is not None else ev.get("end"))
            if e > s:
                acc_by_lane.setdefault(lane, []).append((s, e))
        gt_now = {l: gt_cache[l] for l in touched}
        tp, fp, fn, prec, rec, f1 = _prf(acc_by_lane, gt_now, args.iou)
        pts.append({"decision": n_dec, "cum_min": round(cum_ms / 60000, 2),
                    "cum_frames": cum_frames, "tp": tp, "fp": fp, "fn": fn,
                    "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)})

    if not pts:
        sys.exit("no review decisions found for that behavior — check --behaviors/--videos.")

    last = pts[-1]
    print(f"{len(pts)} decisions · {last['cum_min']} min · {last['cum_frames']} frames reviewed "
          f"→ final F1 {last['f1']} (P {last['precision']}, R {last['recall']})")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(pts[0])); w.writeheader(); w.writerows(pts)
        print(f"wrote {args.csv}")

    # self-contained HTML: F1 (and P, R) vs cumulative review minutes
    import json as _json
    data = _json.dumps(pts)
    html = """<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HITL Curve</title><style>
:root{--s1:#fcfcfb;--s2:#f2f1ec;--ink:#0b0b0b;--muted:#8a8983;--grid:#e8e7e0;--f1:#2a78d6;--p:#eb6834;--r:#1baf7a;}
@media(prefers-color-scheme:dark){:root{--s1:#1a1a19;--s2:#242422;--ink:#fff;--muted:#8f8e86;--grid:#2b2b28;--f1:#3987e5;--p:#d95926;--r:#199e70;}}
body{margin:0;background:var(--s1);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:22px 16px;max-width:820px;margin:0 auto;}
h1{font-size:19px;margin:0 0 3px;} .sub{color:var(--muted);font-size:12.5px;margin:0 0 6px;}
.legend{display:flex;gap:16px;font-size:12.5px;margin:8px 0 6px;} .legend span{display:inline-flex;align-items:center;gap:6px;}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block;} svg{width:100%;height:auto;}
.ax{fill:var(--muted);font-size:10.5px;font-variant-numeric:tabular-nums;} .axt{fill:var(--muted);font-size:12px;}
</style></head><body>
<h1>HITL learning curve — F1 vs review effort</h1>
<p class="sub" id="sub"></p>
<div class="legend"><span><span class="dot" style="background:var(--f1)"></span>F1</span>
<span><span class="dot" style="background:var(--p)"></span>precision</span>
<span><span class="dot" style="background:var(--r)"></span>recall</span></div>
<svg id="plot" xmlns="http://www.w3.org/2000/svg"></svg>
<script>
const D=__DATA__;
const NS="http://www.w3.org/2000/svg",W=820,H=360,PL=48,PR=16,PT=12,PB=40;
const xs=D.map(d=>d.cum_min),xmax=Math.max(...xs,0.1);
const x=v=>PL+v/xmax*(W-PL-PR), y=v=>PT+(1-v)*(H-PT-PB);
const el=(n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;};
const svg=document.getElementById("plot");svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
for(let v=0;v<=1.001;v+=0.2){svg.appendChild(el("line",{x1:PL,y1:y(v),x2:W-PR,y2:y(v),stroke:"var(--grid)","stroke-width":1}));
 const t=el("text",{x:PL-6,y:y(v)+3,"text-anchor":"end",class:"ax"});t.textContent=v.toFixed(1);svg.appendChild(t);}
const step=xmax<=10?2:xmax<=30?5:10;
for(let v=0;v<=xmax+0.01;v+=step){const t=el("text",{x:x(v),y:H-PB+18,"text-anchor":"middle",class:"ax"});t.textContent=v;svg.appendChild(t);}
svg.appendChild(Object.assign(el("text",{x:(PL+W-PR)/2,y:H-6,"text-anchor":"middle",class:"axt"}),{textContent:"cumulative review time (min)"}));
function line(key,col){let d="";D.forEach((p,i)=>{d+=(i?"L":"M")+x(p.cum_min).toFixed(1)+" "+y(p[key]).toFixed(1)+" ";});
 svg.appendChild(el("path",{d,fill:"none",stroke:col,"stroke-width":key==="f1"?2.5:1.5,"stroke-opacity":key==="f1"?1:0.7}));}
line("recall","var(--r)");line("precision","var(--p)");line("f1","var(--f1)");
const L=D[D.length-1];
document.getElementById("sub").textContent=`${D.length} decisions · ${L.cum_min} min · ${L.cum_frames} frames reviewed → final F1 ${L.f1} (P ${L.precision}, R ${L.recall})`;
</script></body></html>""".replace("__DATA__", data)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out} — open or screenshot it.")


if __name__ == "__main__":
    main()
