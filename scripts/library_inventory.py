#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn"]
# ///
"""Inventory of every behavior classifier in a projects folder — what the library has so far.

One row per (project, behavior):

  * what predicts it now: a HiDRA head (zero-shot or fine-tuned, with its lab/action/collapse and
    fine-tune runs), this project's own random forest (with its feature set), or nothing yet
  * its labels: positive bouts split by where they came from (painted by hand, accepted in
    review, imported), negative bouts, and how far they spread over clips and animals
  * the native model's own grouped-CV AP, when one has been trained
  * when the project holds a clip out as its TEST clip (clip bar → test), every stored model's
    AP / AUROC on that clip's reviewed frames: zero-shot head, each fine-tune, each native version.
    A model that trained on the test clip is marked in-sample, not hidden.
  * a suggested starting point from the behavior's name: a HiDRA head, a bake-off between a head
    and the native model, or the native model on landmark features. A first guess to argue with.

    uv run scripts/library_inventory.py ~/laras-projects --md library.md --csv library.csv
    uv run scripts/library_inventory.py ~/laras-projects --projects doom,hcm

Standalone: `uv run` installs its few dependencies from the header above, so it runs from any
directory, without the labeler installed and without a checkout (pass the raw GitHub URL in place
of the path). Read-only: it opens project.json, labels/*.parquet, model/<bid>/meta.json and
predictions/, and writes nothing into the projects. Under a plain python it needs pandas + pyarrow,
and scikit-learn for the test-clip scores (skipped without it).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:                                   # scores are reported as blank, never guessed
    average_precision_score = roc_auc_score = None

MIN_PER_CLASS = 5          # as the Model comparison panel: fewer reviewed frames per class is "thin"
SOURCES = {"manual": "hand", "candidate": "review", "imported": "import"}


# ----------------------------------------------------------------------------- where to start
# HiDRA sees poses only -- no spout, hopper or cage walls -- and knows only its own action
# vocabulary. So a behavior defined by a place in the cage starts native on landmark features, one
# about a partner starts from a head, and a solo action HiDRA also has gets a bake-off on a test clip.
_LOCATION = ("drink", "lick", "spout", "water", "eat", "feed", "hopper", "food", "forag", "nest")
_SOLO_HEADS = ("groom", "rear", "climb", "dig", "rest", "sleep", "freez", "immobil", "jump", "run")
# The partner keywords of hidra.default_collapse, copied rather than imported so this script runs
# without the labeler (whose install pulls in the whole pose stack) -- keep the two in step.
_DIRECTED = ("allo", "sniff", "attack", "mount", "chase", "approach", "follow", "dominan", "escape",
             "avoid", "flee", "defend", "intromi", "ejacul", "anogenital", "partner", "social",
             "huddle", "nose", "bite")
TIERS = {
    "head first": "partner-directed: HiDRA's strength, and the native social features are crude",
    "bake-off": "a solo action HiDRA also has, but your definition may differ: compare on a test clip",
    "native + landmarks": "defined by a place in the cage, which HiDRA cannot see",
    "native": "not in HiDRA's vocabulary, or simple kinematics the native model already handles",
}


def suggest(name: str, action: str | None = None) -> str:
    text = f"{name} {action or ''}".lower()
    if any(k in text for k in _LOCATION):
        return "native + landmarks"
    toward_object = "object" in text or "cage" in text or "wall" in text   # an act on the cage is solo
    if not toward_object and any(k in text for k in _DIRECTED):
        return "head first"
    if any(k in text for k in _SOLO_HEADS):
        return "bake-off"
    return "native"


# ----------------------------------------------------------------------------- labels
def _read_labels(proj: Path, vids: list[str]) -> pd.DataFrame:
    """Every label row of the project's current clips, with the columns old parquets lack filled in
    the way LabelStore fills them (track 0, source manual, target -1)."""
    parts = []
    for vid in vids:
        p = proj / "labels" / f"{vid}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        if "track" not in df.columns:
            df["track"] = 0
        df["source"] = df["source"].fillna("manual") if "source" in df.columns else "manual"
        df["target"] = df["target"].fillna(-1) if "target" in df.columns else -1
        parts.append(df.assign(video_id=vid)[["video_id", "frame", "track", "behavior_id", "value",
                                              "source", "target"]])
    if not parts:
        return pd.DataFrame(columns=["video_id", "frame", "track", "behavior_id", "value", "source", "target"])
    return pd.concat(parts, ignore_index=True)


def _bouts(df: pd.DataFrame) -> pd.DataFrame:
    """Contiguous same-value runs per (clip, animal), one row per bout. A bout trimmed in review
    can mix sources, so each is credited to the source of most of its frames."""
    cols = ["video_id", "track", "value", "n", "source", "directed"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    d = df.sort_values(["video_id", "track", "frame"]).reset_index(drop=True)
    new = (d["video_id"].ne(d["video_id"].shift()) | d["track"].ne(d["track"].shift())
           | d["frame"].diff().ne(1) | d["value"].ne(d["value"].shift()))
    d["bout"] = new.cumsum()
    return d.groupby("bout").agg(
        video_id=("video_id", "first"), track=("track", "first"), value=("value", "first"),
        n=("frame", "size"), source=("source", lambda s: s.value_counts().idxmax()),
        directed=("target", lambda t: bool((t >= 0).mean() > 0.5)))[cols]


# ----------------------------------------------------------------------------- test-clip scores
def _score(y: np.ndarray, p: np.ndarray) -> dict:
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    out = {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()), "ap": None, "auroc": None}
    if out["n_pos"] and out["n_neg"] and average_precision_score is not None:
        out["ap"] = float(average_precision_score(y, p))
        out["auroc"] = float(roc_auc_score(y, p))
    return out


def _lookup(p: np.ndarray, frames: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    """Scores at (frame, track); a whole-frame (F,) lane from before per-track models serves every
    track; anything out of range is NaN, so it drops out of the score instead of shifting it."""
    out = np.full(len(frames), np.nan)
    if p.ndim == 1:
        ok = frames < p.shape[0]
        out[ok] = p[frames[ok]]
    else:
        ok = (frames < p.shape[0]) & (tracks < p.shape[1])
        out[ok] = p[frames[ok], tracks[ok]]
    return out


def _models_on_clip(proj: Path, vid: str, bid: int, beh: dict) -> dict[str, tuple[np.ndarray, str, bool]]:
    """{model key: (scores, label, trained on this clip?)} for every model stored for this clip.
    Falls back to the timeline lane alone for projects from before per-model storage."""
    runs = {r.get("run_id"): r for r in beh.get("hidra_runs", [])}
    found = {}
    store = proj / "predictions" / vid / f"{bid}.models"
    for npy in sorted(store.glob("*.npy")) if store.is_dir() else []:
        key = npy.stem
        if key.endswith(".tmp"):
            continue
        rec = {}
        if npy.with_suffix(".json").exists():
            try:
                rec = json.loads(npy.with_suffix(".json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        if key == "zs":
            label, trained = "zero-shot", []
        elif key.startswith("ft-"):
            run = runs.get(key) or runs.get(key[3:]) or {}
            label = f"fine-tune {key[3:]}" + (f" ({run['mode']})" if run.get("mode") else "")
            trained = run.get("trained_videos") or []
        else:
            label = f"native {str(rec.get('version') or key.removeprefix('native-'))[:8]}"
            trained = rec.get("trained_videos") or []
        found[key] = (np.load(npy, mmap_mode="r"), label, vid in trained)
    if not found:
        lane = proj / "predictions" / vid / f"{bid}.npy"
        if lane.exists():
            side = lane.with_suffix(".meta.json")
            rec = {}
            if side.exists():
                try:
                    rec = json.loads(side.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
            if rec.get("engine") == "hidra":
                label = "HiDRA " + ("fine-tuned" if rec.get("weights") else "zero-shot") + " (timeline lane)"
            elif rec.get("engine") == "project":
                label = f"native {str(rec.get('version') or '')[:8]} (timeline lane)"
            else:
                label = "timeline lane"
            # no record of what it trained on: a lane from before clip roles may have seen this clip
            found["lane"] = (np.load(lane, mmap_mode="r"), label, not rec)
    return found


def _test_scores(proj: Path, tests: list[str], labels: pd.DataFrame, bid: int, beh: dict) -> list[dict]:
    """Each stored model's AP/AUROC on the reviewed (Happening vs Not-happening) frames of the
    project's test clips, pooled across them."""
    pooled: dict[str, dict] = {}
    for vid in tests:
        lab = labels[(labels["video_id"] == vid) & (labels["behavior_id"] == bid) & labels["value"].isin([0, 1])]
        if lab.empty:
            continue
        frames = lab["frame"].to_numpy(np.int64)
        tracks = lab["track"].to_numpy(np.int64)
        y = lab["value"].to_numpy(np.int8)
        for key, (p, label, in_sample) in _models_on_clip(proj, vid, bid, beh).items():
            e = pooled.setdefault(key, {"label": label, "in_sample": False, "y": [], "p": []})
            e["in_sample"] |= in_sample
            e["y"].append(y)
            e["p"].append(_lookup(np.asarray(p), frames, tracks))
    out = []
    for key, e in pooled.items():
        s = _score(np.concatenate(e["y"]), np.concatenate(e["p"]))
        out.append({"model": e["label"], "in_sample": e["in_sample"], **s,
                    "thin": s["n_pos"] < MIN_PER_CLASS or s["n_neg"] < MIN_PER_CLASS})
    return sorted(out, key=lambda r: (r["in_sample"], -(r["ap"] if r["ap"] is not None else -1)))


# ----------------------------------------------------------------------------- one project
def _json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None


def project_rows(proj: Path) -> tuple[dict, list[dict]]:
    m = json.loads((proj / "project.json").read_text(encoding="utf-8"))
    videos = m.get("videos", [])
    vids = [v["video_id"] for v in videos]
    fps = {v["video_id"]: float(v.get("fps") or 30.0) for v in videos}
    tests = [v["video_id"] for v in videos if v.get("role") == "test"]
    labels = _read_labels(proj, vids)
    fc = m.get("feature_config") or {}

    def roi_coverage(field: str) -> str:
        """A clip's own ROI (auto-filled from the HCM database) overrides the project-wide one, so
        coverage is counted per clip."""
        if fc.get(field):
            return "all (project-wide)"
        return f"{sum(1 for v in videos if v.get(field))}/{len(videos)} clips" if videos else "—"

    info = {"project": proj.name, "name": m.get("name", proj.name), "clips": len(vids), "test_clips": tests,
            "spout_roi": roi_coverage("spout_roi"), "cage_roi": roi_coverage("cage_roi")}

    rows = []
    for b in m.get("behaviors", []):
        bid = int(b["id"])
        bouts = _bouts(labels[labels["behavior_id"] == bid])
        pos, neg = bouts[bouts["value"] == 1], bouts[bouts["value"] == 0]
        h = b.get("hidra") or {}
        runs = b.get("hidra_runs") or []
        meta = _json(proj / "model" / str(bid) / "meta.json") or {}
        metrics = meta.get("metrics") or {}
        notes = []

        if h.get("action"):
            engine = "HiDRA"
            tuned = f"fine-tuned, {h.get('finetune_mode') or 'head'}" if h.get("weights") else "zero-shot"
            predicted_by = f"HiDRA {h.get('lab')}/{h['action']} ({h.get('collapse', 'scene')}) · {tuned}"
            if runs:
                predicted_by += f" · {len(runs)} fine-tune run{'s' * (len(runs) != 1)}"
            if meta:
                notes.append("also has a native model from before the head was bound")
        elif meta:
            engine = "native"
            predicted_by = f"native RF · {meta.get('feature_set') or b.get('feature_set') or 'all'} features"
        else:
            engine = "none"
            predicted_by = "— (labels only)"
        if metrics.get("warning"):
            notes.append("native CV: too few bouts per class to score")
        if meta.get("feature_set_note"):
            notes.append(meta["feature_set_note"])

        reviewed_on_test = bool(tests) and bool(
            (labels["video_id"].isin(tests) & (labels["behavior_id"] == bid) & labels["value"].isin([0, 1])).any())
        tests_scored = _test_scores(proj, tests, labels, bid, b) if reviewed_on_test else []
        fair = [t for t in tests_scored if not t["in_sample"] and t["ap"] is not None]
        best = fair[0] if fair else None

        by_src = pos["source"].map(lambda s: SOURCES.get(s, "other")).value_counts()
        pos_s = sum(n / fps.get(v, 30.0) for v, n in zip(pos["video_id"], pos["n"]))
        rows.append({
            "behavior": b.get("name", f"behavior {bid}"), "project": proj.name, "behavior_id": bid,
            "engine": engine, "predicted_by": predicted_by,
            "hidra_lab": h.get("lab", ""), "hidra_action": h.get("action", ""),
            "hidra_collapse": h.get("collapse", ""), "hidra_finetuned": bool(h.get("weights")),
            "hidra_finetune_mode": h.get("finetune_mode", ""), "hidra_runs": len(runs),
            "last_run_steps": runs[-1].get("steps") if runs else "", "last_run_bouts": runs[-1].get("bouts") if runs else "",
            "feature_set": meta.get("feature_set") or b.get("feature_set") or "",
            "native_version": meta.get("version", ""), "native_trained_at": meta.get("trained_at", ""),
            "native_cv_ap": metrics.get("average_precision"), "native_cv_f1": metrics.get("f1"),
            "pos_bouts": len(pos), "pos_hand": int(by_src.get("hand", 0)), "pos_review": int(by_src.get("review", 0)),
            "pos_import": int(by_src.get("import", 0)), "pos_other": int(by_src.get("other", 0)),
            "neg_bouts": len(neg), "pos_frames": int(pos["n"].sum()), "pos_minutes": round(pos_s / 60, 2),
            "directed_bouts": int(pos["directed"].sum()),
            "clips_with_pos": int(pos["video_id"].nunique()), "clips": len(vids),
            "lanes_with_pos": int(pos[["video_id", "track"]].drop_duplicates().shape[0]),
            "test_clips": ";".join(tests), "reviewed_on_test": reviewed_on_test, "test_scores": tests_scored,
            "test_best_model": best["model"] if best else "", "test_best_ap": best["ap"] if best else None,
            "test_best_auroc": best["auroc"] if best else None,
            "suggested": suggest(b.get("name", ""), h.get("action")), "notes": "; ".join(notes),
        })
    return info, rows


# ----------------------------------------------------------------------------- report
def _f(x, nd: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _cell(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _test_cell(r: dict) -> str:
    if not r["test_clips"]:
        return "no test clip"
    if not r["reviewed_on_test"]:
        return "not reviewed on it"
    if not r["test_scores"]:
        return "no stored predictions on it"
    parts = []
    for t in r["test_scores"]:
        tag = " in-sample" if t["in_sample"] else ""
        tag += " thin" if t["thin"] else ""
        parts.append(f"{t['model']} AP {_f(t['ap'])} / AUROC {_f(t['auroc'])}" + (f" ({tag.strip()})" if tag else ""))
    return "<br>".join(parts)


def markdown(root: Path, infos: list[dict], rows: list[dict]) -> str:
    n_names = len({"".join(c for c in r["behavior"].lower() if c.isalnum()) for r in rows})
    eng = pd.Series([r["engine"] for r in rows]).value_counts() if rows else pd.Series(dtype=int)
    out = ["# Classifier library inventory",
           "",
           f"`{root}` · {datetime.now():%Y-%m-%d %H:%M} · {len(infos)} project(s), {len(rows)} behavior(s) "
           f"across {n_names} distinct name(s): {int(eng.get('HiDRA', 0))} on a HiDRA head, "
           f"{int(eng.get('native', 0))} on the native model, {int(eng.get('none', 0))} with labels only.",
           "",
           "## Behaviors",
           "",
           "| Behavior | Project | Predicted by | + bouts (hand / review / import) | − bouts | Clips · lanes | Native CV AP | Test clip | Suggested start |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        src = f"{r['pos_bouts']} ({r['pos_hand']} / {r['pos_review']} / {r['pos_import']})"
        if r["pos_other"]:
            src += f" +{r['pos_other']} other"
        out.append("| " + " | ".join(_cell(x) for x in (
            r["behavior"], r["project"], r["predicted_by"], src, r["neg_bouts"],
            f"{r['clips_with_pos']}/{r['clips']} · {r['lanes_with_pos']}",
            _f(r["native_cv_ap"]))) + f" | {_test_cell(r)} | {r['suggested']} |")
    notes = [r for r in rows if r["notes"]]
    if notes:
        out += ["", "**Notes**", ""] + [f"- {r['behavior']} ({r['project']}): {r['notes']}" for r in notes]
    out += ["", "## Projects", "",
            "| Project | Name | Clips | Test clip(s) | Spout ROI | Cage ROI |", "|---|---|---|---|---|---|"]
    for i in infos:
        out.append("| " + " | ".join(_cell(x) for x in (
            i["project"], i["name"], i["clips"], ", ".join(i["test_clips"]) or "none",
            i["spout_roi"], i["cage_roi"])) + " |")
    out += ["", "## Reading this", "",
            "- **+ bouts** are contiguous Happening runs per clip and animal, credited to where most of "
            "their frames came from: painted by hand, accepted in review, or imported. **Clips · lanes** is "
            "clips with any positive / distinct (clip, animal) pairs with one.",
            "- **Native CV AP** is the random forest's grouped cross-validation over its own labeled bouts. "
            "It is not comparable to a head's score: only the **Test clip** column puts models on the same "
            "frames. That column scores every stored model on the reviewed frames of the project's held-out "
            "clip; *in-sample* means that model trained on the clip, *thin* means fewer than "
            f"{MIN_PER_CLASS} frames of a class. Review only visits proposed frames, so positives there "
            "lean toward what the proposing model already ranked high.",
            "- **Suggested start** is a guess from the name (and bound action), not a measurement:"]
    out += [f"  - *{k}*: {v}" for k, v in TIERS.items()]
    return "\n".join(out) + "\n"


def write_csv(path: Path, rows: list[dict]) -> None:
    flat = []
    for r in rows:
        f = {k: v for k, v in r.items() if k != "test_scores"}
        f["test_scores"] = "; ".join(
            f"{t['model']}: AP {_f(t['ap'])} AUROC {_f(t['auroc'])} pos {t['n_pos']} neg {t['n_neg']}"
            + (" in-sample" if t["in_sample"] else "") + (" thin" if t["thin"] else "")
            for t in r["test_scores"])
        flat.append(f)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0]) if flat else ["behavior"])
        w.writeheader()
        w.writerows(flat)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Inventory every behavior classifier in a projects folder.")
    ap.add_argument("root", help="projects folder (what you pass to laras-labeler), or one project")
    ap.add_argument("--projects", help="comma-separated substrings of project folder names (default: all)")
    ap.add_argument("--md", help="also write the Markdown report to this file")
    ap.add_argument("--csv", help="also write one row per (project, behavior) to this CSV")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser()
    dirs = [root] if (root / "project.json").exists() else sorted(
        d for d in root.iterdir() if (d / "project.json").exists()) if root.is_dir() else []
    if args.projects:
        want = [s.strip() for s in args.projects.split(",") if s.strip()]
        dirs = [d for d in dirs if any(w in d.name for w in want)]
    if not dirs:
        sys.exit(f"no labeler projects (folders with project.json) under {root}")

    infos, rows = [], []
    for d in dirs:
        try:
            info, rs = project_rows(d)
        except (OSError, ValueError, KeyError) as e:
            print(f"  skipped {d.name}: {e}", file=sys.stderr)
            continue
        infos.append(info)
        rows += rs
    rows.sort(key=lambda r: ("".join(c for c in r["behavior"].lower() if c.isalnum()), r["project"]))

    md = markdown(root, infos, rows)
    if hasattr(sys.stdout, "reconfigure"):   # `> out.md` on Windows is cp1252, which has no "−"
        sys.stdout.reconfigure(encoding="utf-8")
    print(md)
    if args.md:
        Path(args.md).write_text(md, encoding="utf-8")
        print(f"wrote {args.md}", file=sys.stderr)
    if args.csv:
        write_csv(Path(args.csv), rows)
        print(f"wrote {args.csv}", file=sys.stderr)
    if average_precision_score is None and any(i["test_clips"] for i in infos):
        print("scikit-learn is not importable, so test-clip scores are blank", file=sys.stderr)


if __name__ == "__main__":
    main()
