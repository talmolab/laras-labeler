#!/usr/bin/env python3
"""Sample-efficiency curve for a from-scratch classifier — F1/AP vs number of labeled POSITIVE bouts.

Unlike rf_learning_curve.py (which needs explicit not-happening labels), this treats the UNLABELED
background as the negative class — the standard weakly-supervised setup — so it works on projects that
only ever labeled positives (e.g. a HiDRA HITL arm). It reads the already-cached per-clip features
(`features/<vid>.npy` + `.meta.json`) and the positive labels (`labels/<vid>.parquet`), samples
background frames as negatives (grouped into short pseudo-bouts so grouped CV never leaks), then, for a
sweep of training-set sizes, subsamples that many labeled POSITIVE bouts (with a proportional negative
draw), scores the labeler's own RandomForest by StratifiedGroupKFold, and reports F1/AP vs the number
of training frames — several random subsamples per size give a mean ± spread.

    python scripts/sample_efficiency.py --project C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-test \\
        --behavior "jump down" --out jumpdown_sampeff.html --csv jumpdown_sampeff.csv

PREREQUISITE: features cached for the clips (the labeler pre-warms them on project open; check
`<project>/features/*.npy`). Needs the labeler env python (pandas/numpy/sklearn + laras_labeler).

CAVEAT: background-as-negative is a from-scratch BASELINE, not an identical-methodology match to
HiDRA (which needs only positives). It answers "how many labeled bouts does a plain classifier need",
which is the right sample-efficiency question — just report it as a baseline, not a head-to-head.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a boolean mask, as [start, end) index pairs."""
    if not mask.any():
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return list(zip(idx[0::2].tolist(), idx[1::2].tolist()))


def _grouped_f1_ap(X, y, groups, seed):
    """Out-of-fold F1 (@0.5) and AP over bout-grouped CV. Returns (f1, ap) or (None, None)."""
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.metrics import f1_score, average_precision_score
    from laras_labeler.training import make_model
    y = np.asarray(y)
    posg = len(np.unique(groups[y == 1])); negg = len(np.unique(groups[y == 0]))
    if posg < 2 or negg < 2:
        return None, None
    n_splits = int(min(5, posg, negg))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(make_model(), X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    return float(f1_score(y, (proba >= 0.5).astype(int), zero_division=0)), float(average_precision_score(y, proba))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sample-efficiency curve (background-as-negative): F1/AP vs #labeled positive bouts.")
    ap.add_argument("--project", required=True, help="project dir with features/ and labels/")
    ap.add_argument("--behavior", required=True, help="behavior name or id")
    ap.add_argument("--repeats", type=int, default=3, help="random subsamples per size (default 3)")
    ap.add_argument("--neg-per-pos", type=float, default=8.0, help="background negative FRAMES to keep, as a multiple of positive frames (default 8)")
    ap.add_argument("--neg-chunk", type=int, default=30, help="length of a negative pseudo-bout, frames (default 30)")
    ap.add_argument("--sizes", help="comma-separated positive-bout counts to sweep (default: auto)")
    ap.add_argument("--out", default="sample_efficiency.html"); ap.add_argument("--csv")
    args = ap.parse_args(argv)

    import pandas as pd
    from laras_labeler.featurestore import select_feature_cols

    proj = Path(args.project).expanduser()
    man = json.loads((proj / "project.json").read_text(encoding="utf-8"))
    bmap = {str(b.get("name", "")).strip().lower(): int(b["id"]) for b in man.get("behaviors", [])}
    bid = int(args.behavior) if args.behavior.isdigit() else bmap.get(args.behavior.strip().lower())
    if bid is None:
        sys.exit(f"behavior {args.behavior!r} not found; have {list(bmap)}")
    beh = next((b for b in man["behaviors"] if int(b["id"]) == bid), {})
    feature_set = beh.get("feature_set")
    beh_disp = beh.get("name", args.behavior)

    feats_dir, labels_dir = proj / "features", proj / "labels"
    npys = sorted(feats_dir.glob("*.npy"))
    if not npys:
        sys.exit(f"no cached features in {feats_dir} — open the project in the labeler to pre-warm, or run a native Train.")

    # First pass: materialize positives (small), record negative pseudo-bout descriptors (cheap).
    cols = None; D = None
    pos_X, pos_g = [], []
    neg_desc = []                                  # (vid, track, cs, ce)
    gid = 0
    total_pos_frames = 0
    used, skipped = [], []
    for npy in npys:
        vid = npy.stem
        mp = feats_dir / f"{vid}.meta.json"
        if not mp.exists():
            skipped.append(vid); continue
        meta = json.loads(mp.read_text())
        fnames = meta.get("feature_names") or []
        if cols is None:
            cols = select_feature_cols(fnames, feature_set); D = len(fnames)
        elif len(fnames) != D:
            print(f"  skip {vid[:24]}: {len(fnames)} feature cols != {D}"); skipped.append(vid); continue
        lp = labels_dir / f"{vid}.parquet"
        df = pd.read_parquet(lp) if lp.exists() else None
        F = np.load(npy, mmap_mode="r")            # (nf, T, Dfull)
        nf, T = F.shape[0], F.shape[1]
        colsel = np.asarray(cols, dtype=int)
        got_any = False
        for tr in range(T):
            if df is not None:
                pf = df[(df["behavior_id"] == bid) & (df["track"] == tr) & (df["value"] == 1)]["frame"].to_numpy()
                pf = pf[(pf >= 0) & (pf < nf)]
            else:
                pf = np.empty(0, dtype=int)
            posmask = np.zeros(nf, dtype=bool)
            posmask[pf] = True
            for (s, e) in _mask_runs(posmask):     # positive bouts
                rows = np.asarray(F[s:e, tr, :], dtype="float32")[:, colsel]
                pos_X.append(rows); pos_g.append(np.full(e - s, gid, dtype=np.int64)); gid += 1
                total_pos_frames += (e - s); got_any = True
            for (s, e) in _mask_runs(~posmask):    # background -> pseudo-bout descriptors
                for cs in range(s, e, args.neg_chunk):
                    neg_desc.append((vid, tr, cs, min(cs + args.neg_chunk, e)))
        if got_any:
            used.append(vid)
    if not pos_X:
        sys.exit("no positive labels for this behavior in any cached clip.")

    nP = len(pos_X)
    print(f"{nP} positive bouts · {total_pos_frames} positive frames · {len(used)} clips "
          f"· {len(neg_desc)} background chunks available")
    if skipped:
        print("  skipped: " + ", ".join(s[:20] for s in skipped))
    if nP < 4:
        sys.exit(f"only {nP} positive bouts — too few for a curve.")

    # Sample background chunks to the frame budget, then materialize just those.
    rng = np.random.RandomState(0)
    budget = int(args.neg_per_pos * total_pos_frames)
    order = rng.permutation(len(neg_desc))
    keep, got = [], 0
    for i in order:
        d = neg_desc[i]; keep.append(d); got += d[3] - d[2]
        if got >= budget:
            break
    keep_by_vid: dict[str, list] = {}
    for d in keep:
        keep_by_vid.setdefault(d[0], []).append(d)
    neg_X, neg_g = [], []
    ng = 0
    colsel = np.asarray(cols, dtype=int)
    for vid, ds in keep_by_vid.items():
        F = np.load(feats_dir / f"{vid}.npy", mmap_mode="r")
        for (_, tr, cs, ce) in ds:
            rows = np.asarray(F[cs:ce, tr, :], dtype="float32")[:, colsel]
            neg_X.append(rows); neg_g.append(np.full(ce - cs, 10_000_000 + ng, dtype=np.int64)); ng += 1
    nN = ng
    print(f"sampled {sum(len(a) for a in neg_X)} background frames as {nN} negative pseudo-bouts")

    X = np.concatenate(pos_X + neg_X, axis=0).astype("float32")
    y = np.concatenate([np.ones(sum(len(a) for a in pos_X), dtype=np.int8),
                        np.zeros(sum(len(a) for a in neg_X), dtype=np.int8)])
    groups = np.concatenate(pos_g + neg_g)
    pos_groups = np.unique(groups[y == 1]); neg_groups = np.unique(groups[y == 0])

    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(",")]
    else:
        sizes = sorted({int(round(v)) for v in np.linspace(4, nP, 6)})
    sizes = [s for s in sizes if 4 <= s <= nP]

    pts = []
    for k in sizes:
        f1s, aps, frms = [], [], []
        kneg = max(2, int(round(k * nN / max(nP, 1))))
        for rep in range(args.repeats):
            gp = np.random.RandomState(1000 + rep + k)
            keepP = set(gp.choice(pos_groups, size=min(k, nP), replace=False).tolist())
            keepN = set(gp.choice(neg_groups, size=min(kneg, nN), replace=False).tolist())
            keep_set = keepP | keepN
            mask = np.array([g in keep_set for g in groups])
            f1, apv = _grouped_f1_ap(X[mask], y[mask], groups[mask], seed=rep)
            if f1 is not None:
                f1s.append(f1); aps.append(apv); frms.append(int(mask.sum()))
        if f1s:
            pts.append({"pos_bouts": k, "neg_bouts": kneg, "n_runs": len(f1s),
                        "train_frames": int(round(float(np.mean(frms)))),
                        "f1_mean": round(float(np.mean(f1s)), 3), "f1_sd": round(float(np.std(f1s)), 3),
                        "ap_mean": round(float(np.mean(aps)), 3), "ap_sd": round(float(np.std(aps)), 3)})
            print(f"  {k:>3} pos bouts ({pts[-1]['train_frames']} train frames) → "
                  f"F1 {pts[-1]['f1_mean']:.3f} ± {pts[-1]['f1_sd']:.3f} · AP {pts[-1]['ap_mean']:.3f}")

    if not pts:
        sys.exit("no usable subsample sizes (too few groups per class).")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(pts[0])); w.writeheader(); w.writerows(pts)
        print(f"wrote {args.csv}")

    # minimal self-contained HTML (F1/AP vs train frames); the polished overlay is built from the CSV.
    data = json.dumps(pts)
    html = """<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sample Efficiency</title><style>
:root{--s1:#fcfcfb;--ink:#0b0b0b;--muted:#8a8983;--grid:#e8e7e0;--f1:#2a78d6;--ap:#eb6834;--band:#2a78d61a;}
@media(prefers-color-scheme:dark){:root{--s1:#1a1a19;--ink:#fff;--muted:#8f8e86;--grid:#2b2b28;--f1:#3987e5;--ap:#d95926;--band:#3987e522;}}
body{margin:0;background:var(--s1);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:22px 16px;max-width:820px;margin:0 auto;}
h1{font-size:19px;margin:0 0 3px;} .sub{color:var(--muted);font-size:12.5px;margin:0 0 8px;} svg{width:100%;height:auto;}
.ax{fill:var(--muted);font-size:10.5px;font-variant-numeric:tabular-nums;} .axt{fill:var(--muted);font-size:12px;}
</style></head><body>
<h1>Sample efficiency — __BEH__ (from-scratch, background-as-negative)</h1>
<p class="sub" id="sub"></p>
<svg id="plot" xmlns="http://www.w3.org/2000/svg"></svg>
<script>
const D=__DATA__;
const NS="http://www.w3.org/2000/svg",W=820,H=360,PL=52,PR=16,PT=12,PB=42;
const xmax=Math.max(...D.map(d=>d.train_frames),1);
const x=v=>PL+v/xmax*(W-PL-PR),y=v=>PT+(1-v)*(H-PT-PB);
const el=(n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;};
const svg=document.getElementById("plot");svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
for(let v=0;v<=1.001;v+=0.2){svg.appendChild(el("line",{x1:PL,y1:y(v),x2:W-PR,y2:y(v),stroke:"var(--grid)","stroke-width":1}));
 const t=el("text",{x:PL-6,y:y(v)+3,"text-anchor":"end",class:"ax"});t.textContent=v.toFixed(1);svg.appendChild(t);}
D.forEach(p=>{const t=el("text",{x:x(p.train_frames),y:H-PB+18,"text-anchor":"middle",class:"ax"});t.textContent=(p.train_frames/1000).toFixed(0)+"k";svg.appendChild(t);});
svg.appendChild(Object.assign(el("text",{x:(PL+W-PR)/2,y:H-6,"text-anchor":"middle",class:"axt"}),{textContent:"training frames"}));
function band(){let up="",dn="";D.forEach(p=>up+=" "+x(p.train_frames).toFixed(1)+","+y(Math.min(1,p.f1_mean+p.f1_sd)).toFixed(1));
 for(let i=D.length-1;i>=0;i--){const p=D[i];dn+=" "+x(p.train_frames).toFixed(1)+","+y(Math.max(0,p.f1_mean-p.f1_sd)).toFixed(1);}
 svg.appendChild(el("polygon",{points:up+dn,fill:"var(--band)",stroke:"none"}));}
function line(key,col){let d="";D.forEach((p,i)=>d+=(i?"L":"M")+x(p.train_frames).toFixed(1)+" "+y(p[key]).toFixed(1)+" ");
 svg.appendChild(el("path",{d,fill:"none",stroke:col,"stroke-width":2.5}));D.forEach(p=>svg.appendChild(el("circle",{cx:x(p.train_frames),cy:y(p[key]),r:3.5,fill:col})));}
band();line("ap_mean","var(--ap)");line("f1_mean","var(--f1)");
const L=D[D.length-1];document.getElementById("sub").textContent=`up to ${L.pos_bouts} positive bouts (${(L.train_frames/1000).toFixed(0)}k frames) → F1 ${L.f1_mean} · AP ${L.ap_mean} · blue=F1 orange=AP shaded=\u00b11SD`;
</script></body></html>""".replace("__DATA__", data).replace("__BEH__", str(beh_disp))
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out} — open or screenshot it.")


if __name__ == "__main__":
    main()
