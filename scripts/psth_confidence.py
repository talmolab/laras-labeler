#!/usr/bin/env python3
"""Peri-event confidence PSTH — average model confidence time-locked to REAL (manual) bout onsets.

For each behavior it takes every ground-truth bout, grabs the model's per-frame confidence lane in a
window around the bout ONSET (and/or OFFSET), averages across all bouts and all clips, and plots
mean ± SEM confidence vs time relative to the event. It answers "does HiDRA's confidence actually
rise when the behavior starts?" — the mechanism behind the F1/AUROC numbers, in one picture.

Reads the same artifacts as score_predictions.py — the ground-truth label parquets and the stored
prediction lanes (predictions/<vid>/<bid>.npy, already on the labeler's rescaled 0-1 display scale) —
so no labeler install is needed, just pandas + numpy.

    python scripts/psth_confidence.py ^
        --truth C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-test ^
        --pred  C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-hitl ^
        --behaviors "grooming,anogenital sniffing,jump down" ^
        --window 3.0 --align onset ^
        --thresholds "grooming=0.75,anogenital sniffing=0.65,jump down=0.75" ^
        --csv psth_confidence.csv --out psth_confidence.html

Outputs a tidy CSV (behavior, align, t_sec, mean, sem, n_events) and a self-contained HTML figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _manifest(proj: Path) -> dict:
    p = proj / "project.json"
    if not p.exists():
        sys.exit(f"ERROR: not a labeler project (no project.json): {proj}")
    return json.loads(p.read_text())


def _behavior_ids(man: dict) -> dict[str, int]:
    return {_norm(b.get("name", "")): int(b["id"]) for b in man.get("behaviors", [])}


def _gt_mask(truth: Path, vid: str, bid: int, track: int, n: int) -> np.ndarray:
    m = np.zeros(int(n), bool)
    p = truth / "labels" / f"{vid}.parquet"
    if not p.exists():
        return m
    df = pd.read_parquet(p)
    if "track" not in df.columns:
        df = df.assign(track=0)
    df = df[(df["behavior_id"] == bid) & (df["track"] == int(track)) & (df["value"] == 1)]
    f = df["frame"].to_numpy().astype(np.int64)
    f = f[(f >= 0) & (f < n)]
    m[f] = True
    return m


def _lane(pred: Path, vid: str, bid: int) -> np.ndarray | None:
    p = pred / "predictions" / vid / f"{bid}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float64)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _snippets(conf: np.ndarray, anchors: list[int], half: int) -> np.ndarray:
    """(n_events, 2*half+1) confidence snippets centered on each anchor; out-of-range filled NaN."""
    L = 2 * half + 1
    out = np.full((len(anchors), L), np.nan, float)
    n = conf.shape[0]
    for i, a in enumerate(anchors):
        s, e = a - half, a + half + 1
        cs, ce = max(0, s), min(n, e)
        out[i, (cs - s):(ce - s)] = conf[cs:ce]
    return out


def collect(truth: Path, pred: Path, behaviours, videos, half: int, align: str):
    tman, pman = _manifest(truth), _manifest(pred)
    t_ids, p_ids = _behavior_ids(tman), _behavior_ids(pman)
    nfr = {v["video_id"]: int(v.get("n_frames") or 0) for v in tman.get("videos", [])}
    vids = [v["video_id"] for v in tman.get("videos", [])
            if not videos or any(f in v["video_id"] for f in videos)]
    want = [_norm(b) for b in behaviours] if behaviours else list(t_ids)

    series = {}                                   # disp_name -> dict(align -> snippet matrix), baseline, n
    for nm in want:
        if nm not in t_ids or nm not in p_ids:
            print(f"  note: {nm!r} missing in truth or pred — skipped"); continue
        disp = next((b.get("name") for b in tman["behaviors"] if int(b["id"]) == t_ids[nm]), nm)
        on_mat, off_mat, base = [], [], []
        for vid in vids:
            lane = _lane(pred, vid, p_ids[nm])
            if lane is None:
                continue
            n = min(nfr.get(vid) or lane.shape[0], lane.shape[0])
            ntr = lane.shape[1] if lane.ndim == 2 else 1
            for tr in range(ntr):
                conf = lane[:n, tr] if lane.ndim == 2 else lane[:n]
                y = _gt_mask(truth, vid, t_ids[nm], tr, n)
                if y.sum() == 0:
                    continue
                base.append(conf)
                runs = _runs(y)
                if align in ("onset", "both"):
                    on_mat.append(_snippets(conf, [s for s, _ in runs], half))
                if align in ("offset", "both"):
                    off_mat.append(_snippets(conf, [e - 1 for _, e in runs], half))
        rec = {"n_events": 0, "baseline": float(np.mean(np.concatenate(base))) if base else float("nan")}
        if on_mat:
            M = np.vstack(on_mat); rec["onset"] = M; rec["n_events"] = M.shape[0]
        if off_mat:
            M = np.vstack(off_mat); rec["offset"] = M; rec["n_events"] = max(rec["n_events"], M.shape[0])
        if "onset" in rec or "offset" in rec:
            series[disp] = rec
            print(f"  {disp}: {rec['n_events']} bouts · baseline conf {rec['baseline']:.3f}")
    return series


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--behaviors")
    ap.add_argument("--videos")
    ap.add_argument("--window", type=float, default=3.0, help="seconds each side of the event (default 3)")
    ap.add_argument("--fps", type=float, help="frames/sec (default: project video fps, else 50)")
    ap.add_argument("--align", choices=["onset", "offset", "both"], default="onset")
    ap.add_argument("--thresholds", help="labeling thresholds `name=val,...` to draw as reference lines")
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--out", type=Path, default=Path("psth_confidence.html"))
    args = ap.parse_args(argv)

    tman = _manifest(args.truth)
    fps = args.fps or float(next((v.get("fps") for v in tman.get("videos", []) if v.get("fps")), 0) or 50.0)
    half = int(round(args.window * fps))
    behaviours = [b.strip() for b in args.behaviors.split(",") if b.strip()] if args.behaviors else None
    videos = [v.strip() for v in args.videos.split(",") if v.strip()] if args.videos else None
    thr = {}
    if args.thresholds:
        for tok in args.thresholds.split(","):
            if "=" in tok:
                k, v = tok.split("=", 1); thr[_norm(k)] = float(v)

    series = collect(args.truth, args.pred, behaviours, videos, half, args.align)
    if not series:
        sys.exit("ERROR: nothing collected — check --behaviors and that --pred has prediction lanes.")

    t = (np.arange(-half, half + 1) / fps).round(4)
    rows, plot = [], []
    for disp, rec in series.items():
        entry = {"behavior": disp, "baseline": rec["baseline"], "n": rec["n_events"],
                 "thr": thr.get(_norm(disp))}
        for al in ("onset", "offset"):
            if al in rec:
                M = rec[al]
                mean = np.nanmean(M, axis=0)
                sem = np.nanstd(M, axis=0) / np.sqrt(np.sum(~np.isnan(M), axis=0).clip(1))
                entry[al] = {"mean": mean.round(4).tolist(), "sem": sem.round(4).tolist()}
                for ti, tv in enumerate(t):
                    rows.append({"behavior": disp, "align": al, "t_sec": float(tv),
                                 "mean": round(float(mean[ti]), 4), "sem": round(float(sem[ti]), 4),
                                 "n_events": int(rec["n_events"])})
        plot.append(entry)

    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")

    payload = {"t": t.tolist(), "series": plot, "align": args.align, "window": args.window}
    html = _HTML.replace("__DATA__", json.dumps(payload))
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} — open or screenshot it.")


_HTML = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Confidence PSTH</title>
<style>
:root{--s1:#fcfcfb;--ink:#0b0b0b;--sec:#52514e;--muted:#8a8983;--grid:#ecebe4;}
@media(prefers-color-scheme:dark){:root{--s1:#1a1a19;--ink:#fff;--sec:#c3c2b7;--muted:#8f8e86;--grid:#2b2b28;}}
body{margin:0;background:var(--s1);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;margin:0 auto;padding:24px 16px 30px;}
h1{font-size:20px;margin:0 0 3px;} .sub{color:var(--sec);font-size:13px;margin:0 0 6px;}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--sec);margin:10px 0 2px;}
.legend .k{display:inline-flex;align-items:center;gap:7px;} .sw{width:16px;height:3px;border-radius:2px;display:inline-block;}
svg{width:100%;height:auto;display:block;} .ax{fill:var(--muted);font-size:10.5px;font-variant-numeric:tabular-nums;} .axt{fill:var(--muted);font-size:12px;}
.note{margin:12px 0 0;font-size:12px;color:var(--sec);border-left:3px solid #2a78d6;padding:9px 13px;background:var(--grid);border-radius:0 8px 8px 0;}
</style></head><body>
<h1>Confidence PSTH &mdash; model confidence time-locked to real bout onsets</h1>
<p class="sub" id="sub"></p>
<div class="legend" id="legend"></div>
<svg id="plot" xmlns="http://www.w3.org/2000/svg"></svg>
<p class="note">Each line is mean HiDRA confidence across every manual bout, aligned to bout onset (t=0), &plusmn;SEM band.
Dashed horizontal lines are each behavior's labeling threshold; the dotted line is that behavior's baseline (mean confidence over all frames).
A confidence that jumps from baseline to above threshold right at t=0 is the model correctly locking onto the behavior.</p>
<script>
const P=__DATA__;
const COLS={"grooming":"#2f8f5b","anogenital sniffing":"#b8480f","anogenital sniff":"#b8480f","jump down":"#2a78d6"};
const fallback=["#2a78d6","#b8480f","#2f8f5b","#8250c4"];
const NS="http://www.w3.org/2000/svg",el=(n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;};
const W=820,H=430,PL=52,PR=168,PT=14,PB=44;
const t=P.t, xmin=t[0], xmax=t[t.length-1];
const x=v=>PL+(v-xmin)/(xmax-xmin)*(W-PL-PR), y=v=>PT+(1-v)*(H-PT-PB);
const svg=document.getElementById("plot");svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
for(let v=0;v<=1.001;v+=0.2){svg.appendChild(el("line",{x1:PL,y1:y(v),x2:W-PR,y2:y(v),stroke:"var(--grid)","stroke-width":1}));
  svg.appendChild(Object.assign(el("text",{x:PL-7,y:y(v)+3,"text-anchor":"end",class:"ax"}),{textContent:v.toFixed(1)}));}
for(let s=Math.ceil(xmin);s<=xmax;s++){svg.appendChild(el("line",{x1:x(s),y1:PT,x2:x(s),y2:H-PB,stroke:"var(--grid)","stroke-width":s===0?1.6:1,"stroke-dasharray":s===0?"":""}));
  svg.appendChild(Object.assign(el("text",{x:x(s),y:H-PB+18,"text-anchor":"middle",class:"ax"}),{textContent:(s>0?"+":"")+s+"s"}));}
svg.appendChild(Object.assign(el("text",{x:PL+(W-PL-PR)/2,y:H-6,"text-anchor":"middle",class:"axt"}),{textContent:"time from bout onset (s)"}));
svg.appendChild(Object.assign(el("text",{x:14,y:PT+(H-PT-PB)/2,"text-anchor":"middle",class:"axt",transform:`rotate(-90 14 ${PT+(H-PT-PB)/2})`}),{textContent:"model confidence"}));
const lg=document.getElementById("legend");
const labels=[];
P.series.forEach((d,i)=>{
  const c=COLS[d.behavior.toLowerCase()]||fallback[i%fallback.length];
  const S=d.onset||d.offset; if(!S)return;
  // SEM band
  let up="",dn="";
  S.mean.forEach((m,j)=>up+=" "+x(t[j]).toFixed(1)+","+y(Math.min(1,m+S.sem[j])).toFixed(1));
  for(let j=S.mean.length-1;j>=0;j--)dn+=" "+x(t[j]).toFixed(1)+","+y(Math.max(0,S.mean[j]-S.sem[j])).toFixed(1);
  svg.appendChild(el("polygon",{points:up+dn,fill:c,"fill-opacity":0.14,stroke:"none"}));
  // baseline (dotted) + threshold (dashed)
  if(d.baseline!=null)svg.appendChild(el("line",{x1:PL,y1:y(d.baseline),x2:W-PR,y2:y(d.baseline),stroke:c,"stroke-width":1,"stroke-dasharray":"1 3","stroke-opacity":0.7}));
  if(d.thr!=null)svg.appendChild(el("line",{x1:PL,y1:y(d.thr),x2:W-PR,y2:y(d.thr),stroke:c,"stroke-width":1,"stroke-dasharray":"5 4","stroke-opacity":0.55}));
  // mean line
  let dd="";S.mean.forEach((m,j)=>dd+=(j?"L":"M")+x(t[j]).toFixed(1)+" "+y(m).toFixed(1)+" ");
  svg.appendChild(el("path",{d:dd,fill:"none",stroke:c,"stroke-width":2.4,"stroke-linejoin":"round"}));
  labels.push({c:c,name:d.behavior,y:y(S.mean[S.mean.length-1])});
  const k=document.createElement("span");k.className="k";k.innerHTML='<span class="sw" style="background:'+c+'"></span> '+d.behavior+' (n='+d.n+')';lg.appendChild(k);
});
// de-collide right-edge labels: enforce a minimum vertical gap, clamped to the plot
labels.sort((a,b)=>a.y-b.y);
const GAP=15, lo=PT+6, hi=H-PB-2;
for(let i=1;i<labels.length;i++) if(labels[i].y-labels[i-1].y<GAP) labels[i].y=labels[i-1].y+GAP;
for(let i=labels.length-1;i>=0;i--){ labels[i].y=Math.min(labels[i].y,hi); if(i<labels.length-1&&labels[i+1].y-labels[i].y<GAP) labels[i].y=labels[i+1].y-GAP; labels[i].y=Math.max(labels[i].y,lo);}
labels.forEach(L=>svg.appendChild(Object.assign(el("text",{x:W-PR+8,y:L.y+4}),{textContent:L.name})).setAttribute("style","fill:"+L.c+";font-weight:700;font-size:11.5px;"));
document.getElementById("sub").textContent="DooM · "+P.window+"s window · aligned to "+P.align+" · mean \u00b1 SEM confidence over all manual bouts";
</script></body></html>"""


if __name__ == "__main__":
    main()
