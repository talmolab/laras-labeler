#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Does each mouse have a behavioral signature? Two tests, inside each clip, from the per-bout table
`bout_kinematics_cv.py --out-bouts` writes.

WHAT IT DOES -- rates. How many bouts of each behavior each mouse has, against equal shares
    (chi-square; exact when the counts are small). Only meaningful if every mouse's track was
    labeled (or predicted) equally thoroughly: a mouse whose lane nobody reviewed has few bouts
    for reasons that have nothing to do with the mouse. Predicted bouts (export_bouts.py ->
    bout_kinematics_cv.py --bouts) cover every track alike.

HOW IT DOES IT -- identifiability. Given only a bout's kinematics (all of them at once), can a
    classifier say which mouse made it, better than chance (1 / number of mice)?
      accuracy      cross-validated balanced accuracy, with a permutation p (mouse labels shuffled)
      early->late   trained on each mouse's EARLIER half of bouts, tested on its LATER half: the
                    signature has to hold across time, not just within a burst of similar bouts
      across        trained on one behavior's bouts, tested on another's (each standardized on
                    its own): does a mouse's style carry ACROSS behaviors? Not a pooled classifier,
                    which would let a mouse that does one behavior more often be recognised by it

Inside a clip, because a mouse is a (clip, track): tracks are not linked across clips, so a
signature cannot yet be checked across days -- that needs identities linked across recordings.

    uv run scripts/mouse_signature.py bouts_rel.csv --baseline
    uv run scripts/mouse_signature.py bouts.csv --min-bouts 4 --permutations 500 --csv signature.csv

--baseline uses posture relative to each mouse's own baseline (*_rel), so a mouse that is simply
built differently -- or whose keypoints sit differently -- is not counted as a behavioral
signature. Without it, build counts too.
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

MOVEMENT = ["duration_s", "centroid_speed", "nose_speed", "tail_speed", "abs_turn", "accel_mag"]
POSTURE = ["body_length", "body_bend", "ego_len", "ego_width", "elongation", "spread_area", "spine_len",
           "haunch_width", "shoulder_width", "ear_width"]


# ----------------------------------------------------------------------------- rates
def rate_test(counts: np.ndarray) -> float:
    """p that bouts are spread over the mice in equal shares. Exact multinomial for small totals."""
    from scipy.stats import chisquare, multinomial
    n, k = int(counts.sum()), len(counts)
    if n == 0 or k < 2:
        return np.nan
    if n > 30:
        return float(chisquare(counts).pvalue)
    # exact: total probability of every split no more likely than the one observed
    from itertools import product
    probs = np.full(k, 1 / k)
    p_obs = multinomial.pmf(counts, n, probs)
    total = 0.0
    for head in product(range(n + 1), repeat=k - 1):
        last = n - sum(head)
        if last < 0:
            continue
        p = multinomial.pmf(list(head) + [last], n, probs)
        if p <= p_obs * (1 + 1e-9):
            total += p
    return float(min(total, 1.0))


# ----------------------------------------------------------------------------- identifiability
def _model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # Regularized: tens of bouts, ~16 kinematics. Linear on purpose -- a signature that needs a
    # flexible model to find in this little data is not one to believe.
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))


def _xy(d: pd.DataFrame, cols: list[str]):
    X = d[cols].to_numpy(float)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isfinite(X), X, med)                     # a bout missing one kinematic keeps the rest
    return X, d["mouse"].to_numpy()


def identifiability(d: pd.DataFrame, cols: list[str], n_perm: int, seed: int) -> dict:
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, permutation_test_score
    X, y = _xy(d, cols)
    mice, counts = np.unique(y, return_counts=True)
    out = {"mice": len(mice), "bouts": len(y), "chance": 1 / len(mice), "accuracy": np.nan, "p": np.nan,
           "early_late": np.nan, "p_early_late": np.nan}
    folds = int(min(5, counts.min()))
    if folds >= 2:
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acc, _, p = permutation_test_score(_model(), X, y, cv=cv, scoring="balanced_accuracy",
                                               n_permutations=n_perm, random_state=seed)
        out["accuracy"], out["p"] = float(acc), float(p)

    # early -> late: each mouse's earlier half of bouts trains, its later half tests
    order = d.assign(_i=np.arange(len(d))).sort_values(["mouse", "start_frame"])
    rank = order.groupby("mouse").cumcount().to_numpy()
    size = order.groupby("mouse")["mouse"].transform("size").to_numpy()
    train_idx = order["_i"].to_numpy()[rank < np.ceil(size / 2)]
    test_idx = order["_i"].to_numpy()[rank >= np.ceil(size / 2)]
    if len(np.unique(y[test_idx])) == len(mice):
        rng = np.random.default_rng(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = _model().fit(X[train_idx], y[train_idx])
            obs = balanced_accuracy_score(y[test_idx], m.predict(X[test_idx]))
            null = [balanced_accuracy_score(y[test_idx], _model().fit(X[train_idx], rng.permutation(y[train_idx]))
                                            .predict(X[test_idx])) for _ in range(n_perm)]
        out["early_late"] = float(obs)
        out["p_early_late"] = (1 + sum(v >= obs for v in null)) / (n_perm + 1)
    return out


def transfer(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], n_perm: int, seed: int) -> dict:
    """Fit mouse identity on one behavior's bouts, score it on another's. Each side standardized on
    its own, so only the mouse-specific pattern -- not the behaviors' different means -- can carry."""
    from sklearn.metrics import balanced_accuracy_score

    def z(d):
        X, y = _xy(d, cols)
        sd = X.std(axis=0, ddof=1)
        return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1), y

    (Xa, ya), (Xb, yb) = z(train), z(test)
    mice = np.unique(ya)
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs = balanced_accuracy_score(yb, _model().fit(Xa, ya).predict(Xb))
        null = [balanced_accuracy_score(yb, _model().fit(Xa, rng.permutation(ya)).predict(Xb))
                for _ in range(n_perm)]
    return {"mice": len(mice), "bouts": len(yb), "chance": 1 / len(mice), "accuracy": float(obs),
            "p": (1 + sum(v >= obs for v in null)) / (n_perm + 1)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Per-mouse behavioral signature: rates and identifiability.")
    ap.add_argument("bouts", help="per-bout CSV from bout_kinematics_cv.py --out-bouts")
    ap.add_argument("--baseline", action="store_true", help="posture relative to each mouse's own baseline (*_rel)")
    ap.add_argument("--min-bouts", type=int, default=4, help="bouts a mouse needs in a test (default 4)")
    ap.add_argument("--permutations", type=int, default=500, help="for each p (default 500)")
    ap.add_argument("--csv", help="write both tables here (one CSV, a `test` column tells them apart)")
    args = ap.parse_args(argv)

    b = pd.read_csv(args.bouts)
    b["behavior"] = b["behavior"].astype(str).str.strip()
    if args.baseline and not any(c.endswith("_rel") for c in b.columns):
        sys.exit(f"{args.bouts} has no *_rel columns: run bout_kinematics_cv.py with --baseline --out-bouts")
    cols = [f"{k}_rel" if args.baseline and k in POSTURE and f"{k}_rel" in b.columns else k
            for k in MOVEMENT + POSTURE]
    cols = [c for c in cols if c in b.columns and b[c].notna().any()]
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    short = lambda v: v.split(".predictions")[0][:40]              # noqa: E731
    out = []

    # --- rates ---------------------------------------------------------------------------------
    print("# Does each mouse have a behavioral signature?\n\n## What it does: bouts per mouse\n")
    print("Tracks that appear in any behavior of a clip count as that clip's mice; a mouse with no bouts "
          "of a behavior counts 0. **Only fair if every track was labeled equally thoroughly.**\n")
    print("| clip | behavior | bouts per mouse (track: n) | p (equal shares) |\n|---|---|---|---|")
    for vid, c in b.groupby("video_id"):
        tracks = sorted(c["track"].unique())
        for beh, g in c.groupby("behavior"):
            n = g.groupby("track").size().reindex(tracks, fill_value=0).to_numpy()
            p = rate_test(n)
            print(f"| {short(vid)} | {beh} | " + " · ".join(f"{t + 1}: {k}" for t, k in zip(tracks, n))
                  + f" | {p:.3f} |")
            out.append({"test": "rate", "clip": vid, "behavior": beh, "counts": " ".join(map(str, n)), "p": p})

    # --- identifiability -----------------------------------------------------------------------
    print("\n## How it does it: can a bout be attributed to its mouse?\n")
    print(f"From {len(cols)} kinematics" + (" (posture ÷ each mouse's own baseline)" if args.baseline else "")
          + f"; mice with {args.min_bouts}+ bouts.\n")
    print("| clip | behavior | mice | bouts | chance | accuracy (CV) | p | early→late | p |\n"
          "|---|---|---|---|---|---|---|---|---|")
    enough: dict[str, dict[str, pd.DataFrame]] = {}
    for vid, c in b.groupby("video_id"):
        for beh, g in c.groupby("behavior"):
            n = g.groupby("mouse").size()
            g = g[g["mouse"].isin(n[n >= args.min_bouts].index)]
            if g["mouse"].nunique() < 2:
                continue
            enough.setdefault(vid, {})[beh] = g
            r = identifiability(g, cols, args.permutations, seed=0)
            print(f"| {short(vid)} | {beh} | {r['mice']} | {r['bouts']} | {r['chance']:.2f} | "
                  f"{r['accuracy']:.2f} | {r['p']:.3f} | "
                  + ("— | — |" if not np.isfinite(r["early_late"]) else
                     f"{r['early_late']:.2f} | {r['p_early_late']:.3f} |"))
            out.append({"test": "identifiability", "clip": vid, "behavior": beh, **r})

    # --- across behaviors ----------------------------------------------------------------------
    # Train on one behavior, test on another, for the mice with enough bouts of both. Not a pooled
    # classifier: pooling lets a mouse that simply does one behavior more often be recognised by
    # what is left of that behavior's look after normalizing -- its RATE leaking into its style.
    rows = []
    for vid, behs in enough.items():
        for a, ga in behs.items():
            for bb, gb in behs.items():
                common = sorted(set(ga["mouse"]) & set(gb["mouse"]))
                if a == bb or len(common) < 2:
                    continue
                r = transfer(ga[ga["mouse"].isin(common)], gb[gb["mouse"].isin(common)], cols,
                             args.permutations, seed=0)
                rows.append((vid, f"{a} → {bb}", r))
                out.append({"test": "transfer", "clip": vid, "behavior": f"{a} -> {bb}", **r})
    if rows:
        print("\n## Across behaviors: does a mouse's style in one behavior identify it in another?\n")
        print("Trained on every bout of the first behavior, tested on every bout of the second (each "
              "kinematic standardized within its behavior).\n")
        print("| clip | trained → tested | mice | test bouts | chance | accuracy | p |\n|---|---|---|---|---|---|---|")
        for vid, name, r in rows:
            print(f"| {short(vid)} | {name} | {r['mice']} | {r['bouts']} | {r['chance']:.2f} | "
                  f"{r['accuracy']:.2f} | {r['p']:.3f} |")

    print("\nA signature shows up as accuracy well above chance with a small p — and, to be a signature "
          "rather than a passing state, early→late above chance too. Rates: a small p means the mice of "
          "a clip do that behavior unequally often (check labeling coverage first).")
    if args.csv:
        pd.DataFrame(out).to_csv(args.csv, index=False)
        print(f"wrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
