#!/usr/bin/env python3
"""Random-forest learning curve — the native classifier's accuracy vs amount of labeled data.

The HITL/HiDRA curves ask "how much does REVIEW buy"; this asks the classic supervised question:
"how many labeled BOUTS does the labeler's own RF need to reach good F1?" It pools this project's
labeled examples (Trainer.gather — the exact features/labels the Train button uses), then, for a
sweep of training-set sizes, subsamples that many labeled BOUTS and scores the RF by grouped
cross-validation (StratifiedGroupKFold on bouts, so no bout leaks across folds — the labeler's honest
estimate). Several random subsamples per size give a mean ± spread. Output: F1 (and AP) vs number of
labeled bouts.

    python scripts/rf_learning_curve.py --project C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-test \\
        --behavior grooming --out groom_rf_curve.html --csv groom_rf_curve.csv

PREREQUISITE: features must be cached for the clips — run one native Train (classifier = "this
project's own model") on the behavior in the labeler first; that computes + caches the features this
reads. Needs the labeler env python (pandas/numpy/sklearn + laras_labeler importable).
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


def _grouped_f1_ap(X, y, groups, seed):
    """Out-of-fold F1 (at 0.5) and AP over bout-grouped CV. Returns (f1, ap) or (None, None)."""
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
    ap = argparse.ArgumentParser(description="RF learning curve: F1/AP vs number of labeled bouts.")
    ap.add_argument("--project", required=True, help="project dir whose labels train the RF (e.g. the hand arm)")
    ap.add_argument("--behavior", required=True, help="behavior name or id")
    ap.add_argument("--repeats", type=int, default=5, help="random subsamples per size (default 5)")
    ap.add_argument("--sizes", help="comma-separated bout counts to sweep (default: auto)")
    ap.add_argument("--out", default="rf_learning_curve.html"); ap.add_argument("--csv")
    args = ap.parse_args(argv)

    from laras_labeler.project import ProjectStore
    from laras_labeler.labels import LabelStore
    from laras_labeler.featurestore import FeatureStore
    from laras_labeler.training import Trainer

    proj_dir = Path(args.project).expanduser()
    man = json.loads((proj_dir / "project.json").read_text(encoding="utf-8"))
    bmap = {str(b.get("name", "")).strip().lower(): int(b["id"]) for b in man.get("behaviors", [])}
    bid = int(args.behavior) if args.behavior.isdigit() else bmap.get(args.behavior.strip().lower())
    if bid is None:
        sys.exit(f"behavior {args.behavior!r} not found; have {list(bmap)}")

    store = ProjectStore(proj_dir.parent)
    labels = LabelStore(store)
    features = FeatureStore(store, None)          # vm only needed to COMPUTE features; we only READ cached ones
    trainer = Trainer(store, labels, features)
    got = trainer.gather(proj_dir.name, bid)
    if got is None:
        sys.exit("gather returned nothing — no labeled+feature-ready clips. Run one native Train first to cache features.")
    X, y, groups, rows, used, skipped, bouts = got
    X = np.asarray(X, dtype="float32"); y = np.asarray(y); groups = np.asarray(groups)
    if skipped:
        print("  note: skipped (features not cached — run a native Train to cache them): "
              + ", ".join(s["video_id"][:20] for s in skipped))
    pos_groups = np.unique(groups[y == 1]); neg_groups = np.unique(groups[y == 0])
    nP, nN = len(pos_groups), len(neg_groups)
    print(f"pooled: {len(y)} frames · {nP} positive bouts · {nN} negative bouts · {len(used)} clips")
    if nP < 4:
        sys.exit(f"only {nP} positive bouts — too few for a learning curve. Label/train more first.")

    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(",")]
    else:                                         # geometric-ish sweep up to the pos-bout count
        sizes = sorted({int(round(v)) for v in np.linspace(4, nP, 6)})
    sizes = [s for s in sizes if 4 <= s <= nP]

    all_groups_by_class = (pos_groups, neg_groups)
    rng = np.random.RandomState(0)
    pts = []
    for k in sizes:
        f1s, aps = [], []
        # keep the pos:neg bout ratio of the full set when subsampling
        kneg = max(2, int(round(k * nN / max(nP, 1))))
        for rep in range(args.repeats):
            gp = rng.RandomState(rep) if False else np.random.RandomState(1000 + rep + k)
            keepP = set(gp.choice(pos_groups, size=min(k, nP), replace=False).tolist())
            keepN = set(gp.choice(neg_groups, size=min(kneg, nN), replace=False).tolist())
            keep = keepP | keepN
            mask = np.array([g in keep for g in groups])
            f1, apv = _grouped_f1_ap(X[mask], y[mask], groups[mask], seed=rep)
            if f1 is not None:
                f1s.append(f1); aps.append(apv)
        if f1s:
            pts.append({"pos_bouts": k, "neg_bouts": kneg, "n_runs": len(f1s),
                        "f1_mean": round(float(np.mean(f1s)), 3), "f1_sd": round(float(np.std(f1s)), 3),
                        "ap_mean": round(float(np.mean(aps)), 3), "ap_sd": round(float(np.std(aps)), 3)})
            print(f"  {k:>3} pos bouts → F1 {pts[-1]['f1_mean']:.3f} ± {pts[-1]['f1_sd']:.3f} · AP {pts[-1]['ap_mean']:.3f}")

    if not pts:
        sys.exit("no usable subsample sizes (too few bouts per class).")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(pts[0])); w.writeheader(); w.writerows(pts)
        print(f"wrote {args.csv}")

    beh_disp = next((b.get("name") for b in man["behaviors"] if int(b["id"]) == bid), args.behavior)
    data = json.dumps(pts)
    html = """<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RF Learning Curve</title><style>
:root{--s1:#fcfcfb;--s2:#f2f1ec;--ink:#0b0b0b;--muted:#8a8983;--grid:#e8e7e0;--f1:#2a78d6;--ap:#eb6834;--band:#2a78d633;}
@media(prefers-color-scheme:dark){:root{--s1:#1a1a19;--s2:#242422;--ink:#fff;--muted:#8f8e86;--grid:#2b2b28;--f1:#3987e5;--ap:#d95926;--band:#3987e544;}}
body{margin:0;background:var(--s1);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:22px 16px;max-width:820px;margin:0 auto;}
h1{font-size:19px;margin:0 0 3px;} .sub{color:var(--muted);font-size:12.5px;margin:0 0 6px;}
.legend{display:flex;gap:16px;font-size:12.5px;margin:8px 0 6px;} .legend span{display:inline-flex;align-items:center;gap:6px;}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block;} svg{width:100%;height:auto;}
.ax{fill:var(--muted);font-size:10.5px;font-variant-numeric:tabular-nums;} .axt{fill:var(--muted);font-size:12px;}
</style></head><body>
<h1>Random-forest learning curve — __BEH__</h1>
<p class="sub" id="sub"></p>
<div class="legend"><span><span class="dot" style="background:var(--f1)"></span>F1 (grouped CV)</span>
<span><span class="dot" style="background:var(--ap)"></span>AP</span><span style="color:var(--muted)">shaded = ±1 SD over subsamples</span></div>
<svg id="plot" xmlns="http://www.w3.org/2000/svg"></svg>
<script>
const D=__DATA__;
const NS="http://www.w3.org/2000/svg",W=820,H=360,PL=48,PR=16,PT=12,PB=42;
const xmax=Math.max(...D.map(d=>d.pos_bouts));
const x=v=>PL+v/xmax*(W-PL-PR),y=v=>PT+(1-v)*(H-PT-PB);
const el=(n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;};
const svg=document.getElementById("plot");svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
for(let v=0;v<=1.001;v+=0.2){svg.appendChild(el("line",{x1:PL,y1:y(v),x2:W-PR,y2:y(v),stroke:"var(--grid)","stroke-width":1}));
 const t=el("text",{x:PL-6,y:y(v)+3,"text-anchor":"end",class:"ax"});t.textContent=v.toFixed(1);svg.appendChild(t);}
const step=xmax<=20?5:xmax<=60?10:20;
for(let v=0;v<=xmax+0.01;v+=step){const t=el("text",{x:x(v),y:H-PB+18,"text-anchor":"middle",class:"ax"});t.textContent=v;svg.appendChild(t);}
svg.appendChild(Object.assign(el("text",{x:(PL+W-PR)/2,y:H-6,"text-anchor":"middle",class:"axt"}),{textContent:"number of labeled positive bouts"}));
function band(mkey,skey){let up="",dn="";D.forEach(p=>{up+=" "+x(p.pos_bouts).toFixed(1)+","+y(Math.min(1,p[mkey]+p[skey])).toFixed(1);});
 for(let i=D.length-1;i>=0;i--){const p=D[i];dn+=" "+x(p.pos_bouts).toFixed(1)+","+y(Math.max(0,p[mkey]-p[skey])).toFixed(1);}
 svg.appendChild(el("polygon",{points:up+dn,fill:"var(--band)",stroke:"none"}));}
function line(mkey,col){let d="";D.forEach((p,i)=>{d+=(i?"L":"M")+x(p.pos_bouts).toFixed(1)+" "+y(p[mkey]).toFixed(1)+" ";});
 svg.appendChild(el("path",{d,fill:"none",stroke:col,"stroke-width":2.5}));
 D.forEach(p=>svg.appendChild(el("circle",{cx:x(p.pos_bouts),cy:y(p[mkey]),r:3.5,fill:col})));}
band("f1_mean","f1_sd");line("ap_mean","var(--ap)");line("f1_mean","var(--f1)");
const L=D[D.length-1];document.getElementById("sub").textContent=`grouped-CV · up to ${L.pos_bouts} positive bouts → F1 ${L.f1_mean} · AP ${L.ap_mean}`;
</script></body></html>""".replace("__DATA__", data).replace("__BEH__", str(beh_disp))
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out} — open or screenshot it.")


if __name__ == "__main__":
    main()
