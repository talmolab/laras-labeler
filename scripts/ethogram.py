#!/usr/bin/env python3
"""Ethogram comparison — which bouts an arm CAUGHT, MISSED, or FALSE-ALARMED vs ground truth.

Renders a self-contained HTML raster: for each clip and track, two stacked lanes over the video
timeline —
  GT lane   : the ground-truth bouts, GREEN if matched by a predicted bout (caught), RED if not (missed / FN).
  ARM lane  : the arm's bouts, BLUE if matched (true positive), ORANGE if not (false alarm / FP).
A bout matches when intersection-over-union >= --iou (greedy, one-to-one). This is the visual behind
the P/R/F1 numbers — you see exactly where HiDRA caught grooming and where it slipped.

    python scripts/ethogram.py --truth <GT_PROJECT> --pred <ARM_PROJECT> --behaviors grooming \
        --videos clip_002,clip_005 --out groom_ethogram.html

--behaviors matches by name; --videos by substring (default all clips). Needs pandas — run with the
labeler env python. Open the HTML (or screenshot it) — no server needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _manifest(p: Path) -> dict:
    return json.loads((p / "project.json").read_text(encoding="utf-8"))


def _bids(man: dict) -> dict:
    return {b.get("name", "").strip().lower(): int(b["id"]) for b in man.get("behaviors", [])}


def _runs(proj: Path, vid: str, bid: int, track: int) -> list[tuple[int, int]]:
    p = proj / "labels" / f"{vid}.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    df = df[(df["behavior_id"] == bid) & (df["track"] == track) & (df["value"] == 1)]
    if df.empty:
        return []
    fr = np.sort(df["frame"].to_numpy())
    runs, s, prev = [], int(fr[0]), int(fr[0])
    for f in fr[1:]:
        if f == prev + 1:
            prev = int(f)
        else:
            runs.append((s, prev + 1)); s = prev = int(f)
    runs.append((s, prev + 1))
    return runs


def _tracks(proj: Path, vid: str, bid: int) -> set[int]:
    p = proj / "labels" / f"{vid}.parquet"
    if not p.exists():
        return set()
    df = pd.read_parquet(p)
    return {int(t) for t in df[df["behavior_id"] == bid]["track"].unique()}


def _iou(a, b) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def _match(gt, pred, iou):
    pairs = sorted(((_iou(g, p), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)),
                   reverse=True)
    gm, pm = set(), set()
    for v, gi, pi in pairs:
        if v < iou or gi in gm or pi in pm:
            continue
        gm.add(gi); pm.add(pi)
    return gm, pm  # matched gt indices, matched pred indices


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Ethogram: caught / missed / false-alarm vs ground truth.")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--behaviors", help="comma-separated names (default: all in truth)")
    ap.add_argument("--videos", help="comma-separated video_id substrings (default: all)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--out", default="ethogram.html")
    args = ap.parse_args(argv)

    truth, pred = Path(args.truth).expanduser(), Path(args.pred).expanduser()
    tman, pman = _manifest(truth), _manifest(pred)
    tb, pb = _bids(tman), _bids(pman)
    want = [b.strip().lower() for b in args.behaviors.split(",")] if args.behaviors else list(tb)
    vfilter = [v.strip() for v in args.videos.split(",")] if args.videos else None
    nfr = {v["video_id"]: int(v.get("n_frames") or 0) for v in tman.get("videos", [])}
    vids = [v["video_id"] for v in tman.get("videos", [])
            if not vfilter or any(f in v["video_id"] for f in vfilter)]

    panels = []
    totals = {"tp": 0, "fn": 0, "fp": 0}
    for bnm in want:
        if bnm not in tb:
            continue
        disp = next((b.get("name") for b in tman["behaviors"] if int(b["id"]) == tb[bnm]), bnm)
        for vid in vids:
            N = nfr.get(vid) or 0
            tracks = sorted(_tracks(truth, vid, tb[bnm]) | (_tracks(pred, vid, pb[bnm]) if bnm in pb else set()))
            lanes = []
            for tr in tracks:
                gt = _runs(truth, vid, tb[bnm], tr)
                pr = _runs(pred, vid, pb[bnm], tr) if bnm in pb else []
                if not gt and not pr:
                    continue
                gm, pm = _match(gt, pr, args.iou)
                N = max([N] + [e for _, e in gt + pr])
                totals["tp"] += len(gm); totals["fn"] += len(gt) - len(gm); totals["fp"] += len(pr) - len(pm)
                lanes.append({"track": tr,
                              "gt": [{"s": s, "e": e, "ok": i in gm} for i, (s, e) in enumerate(gt)],
                              "pred": [{"s": s, "e": e, "ok": i in pm} for i, (s, e) in enumerate(pr)]})
            if lanes:
                panels.append({"behavior": disp, "vid": vid, "N": N or 15000, "lanes": lanes})

    if not panels:
        sys.exit("nothing to draw — check --behaviors/--videos and that both projects have labels.")

    # ---- render HTML ----
    def bars(items, N, color_ok, color_bad, y, h, W, padl):
        out = []
        for it in items:
            x0 = padl + it["s"] / N * (W - padl - 8)
            w = max(1.5, (it["e"] - it["s"]) / N * (W - padl - 8))
            out.append(f'<rect x="{x0:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="2" '
                       f'fill="{color_ok if it["ok"] else color_bad}"/>')
        return "".join(out)

    W, padl, laneH, gap = 900, 74, 11, 4
    secs = []
    for p in panels:
        rows = []
        rowH = laneH * 2 + gap + 22
        svgH = len(p["lanes"]) * rowH + 26
        body = []
        y = 18
        # frame axis ticks (every ~fps*60 = 1 min)
        step = int(args.fps * 60) or 3000
        ticks = []
        for f in range(0, p["N"] + 1, step):
            xt = padl + f / p["N"] * (W - padl - 8)
            ticks.append(f'<line x1="{xt:.1f}" y1="12" x2="{xt:.1f}" y2="{svgH-6}" stroke="var(--grid)" stroke-width="1"/>'
                         f'<text x="{xt:.1f}" y="10" text-anchor="middle" class="ax">{f//int(args.fps)}s</text>')
        for ln in p["lanes"]:
            body.append(f'<text x="{padl-8}" y="{y+laneH-1}" text-anchor="end" class="tk">t{ln["track"]} GT</text>')
            body.append(bars(ln["gt"], p["N"], "var(--tp)", "var(--fn)", y, laneH, W, padl))
            body.append(f'<text x="{padl-8}" y="{y+laneH*2+gap-1}" text-anchor="end" class="tk">t{ln["track"]} HITL</text>')
            body.append(bars(ln["pred"], p["N"], "var(--tp2)", "var(--fp)", y+laneH+gap, laneH, W, padl))
            y += rowH
        secs.append(f'<div class="panel"><div class="ptitle">{p["vid"][:30]}… · <b>{p["behavior"]}</b></div>'
                    f'<svg viewBox="0 0 {W} {svgH}" xmlns="http://www.w3.org/2000/svg">{"".join(ticks)}{"".join(body)}</svg></div>')

    tp, fn, fp = totals["tp"], totals["fn"], totals["fp"]
    rec = tp / (tp + fn) if tp + fn else 0
    prec = tp / (tp + fp) if tp + fp else 0
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ethogram</title><style>
:root{{--surface-1:#fcfcfb;--surface-2:#f2f1ec;--ink:#0b0b0b;--muted:#8a8983;--grid:#e8e7e0;
--tp:#008300;--tp2:#2a78d6;--fn:#e34948;--fp:#eda100;}}
@media (prefers-color-scheme:dark){{:root{{--surface-1:#1a1a19;--surface-2:#242422;--ink:#fff;--muted:#8f8e86;--grid:#2b2b28;
--tp:#2faa4d;--tp2:#3987e5;--fn:#e66767;--fp:#c98500;}}}}
body{{margin:0;background:var(--surface-1);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:22px 16px;max-width:960px;margin:0 auto;}}
h1{{font-size:19px;margin:0 0 4px;}} .sub{{color:var(--muted);font-size:12.5px;margin:0 0 8px;}}
.legend{{display:flex;gap:16px;font-size:12.5px;flex-wrap:wrap;margin:8px 0 14px;}}
.legend span{{display:inline-flex;align-items:center;gap:6px;}} .sw{{width:13px;height:11px;border-radius:2px;display:inline-block;}}
.panel{{background:var(--surface-2);border-radius:8px;padding:10px 12px;margin:10px 0;}}
.ptitle{{font-size:12.5px;color:var(--muted);margin:0 0 2px;}} .ptitle b{{color:var(--ink);}}
svg{{width:100%;height:auto;display:block;}} .tk{{fill:var(--muted);font-size:9.5px;}} .ax{{fill:var(--muted);font-size:9px;}}
.tot{{font-size:13px;margin:2px 0 0;}} .tot b{{font-variant-numeric:tabular-nums;}}
</style></head><body>
<h1>Ethogram: caught vs missed vs false-alarm</h1>
<p class="sub">GT lane = hand-labeled bouts · HITL lane = your reviewed bouts · bout match at IoU ≥ {args.iou}</p>
<div class="legend">
<span><span class="sw" style="background:var(--tp)"></span> GT caught (TP)</span>
<span><span class="sw" style="background:var(--fn)"></span> GT missed (FN)</span>
<span><span class="sw" style="background:var(--tp2)"></span> HITL correct</span>
<span><span class="sw" style="background:var(--fp)"></span> HITL false alarm (FP)</span>
</div>
<p class="tot">totals — caught <b>{tp}</b> · missed <b>{fn}</b> · false alarms <b>{fp}</b> · recall <b>{rec:.2f}</b> · precision <b>{prec:.2f}</b></p>
{"".join(secs)}
</body></html>"""
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out} — {len(panels)} clip panels · caught {tp} / missed {fn} / false {fp} "
          f"(recall {rec:.2f}, precision {prec:.2f})")
    print("open it in a browser (or screenshot) — self-contained, no server.")


if __name__ == "__main__":
    main()
