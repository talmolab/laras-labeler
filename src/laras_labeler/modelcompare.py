"""Score every model that has predicted a behavior against the project's labels — the zero-shot
HiDRA head, each fine-tune, each native version — for the Stats panel's model comparison.

HONESTY RULES this module encodes (see the GUI copy too):

* Only REVIEWED frames count: Happening = positive, Not-happening = negative; Unknown and unlabeled
  frames are excluded. (HiDRA's own fine-tune treats every unlabeled frame of a staged clip as a
  negative; that is a training choice, not ground truth.)
* A fine-tune trains on EVERY frame of every clip it staged (bouts as positives, everything else as
  implicit negatives), and HiDRA's internal validation falls back to the training clip when a
  project has one clip. So the only fair number is on a TEST clip the project holds out from all
  training (``video["role"] == "test"``). Scores on other clips are reported too, tagged in-sample
  per model and clip.
* Metrics are rank-based or best-threshold (AUROC, AP, best-F1), so they are identical on the raw
  scores and on the timeline's rate-rescaled lanes (a monotone map). F1 at the GUI's fixed 0.6 cut
  is deliberately NOT reported: the rescale puts ~`rate` of frames above it for every model, so it
  cannot tell models apart.
* Reviewed frames are not a random sample: review visits frames a model proposed. Positives are
  enriched for what the head that proposed them already ranked high. The GUI says so.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .predict import ft_model_key, model_pred_dir, weights_mtime

MIN_PER_CLASS = 5          # below this, a score is reported but flagged too thin to trust


def score(y: np.ndarray, p: np.ndarray) -> dict:
    """Frame metrics for binary labels `y` (1/0) and scores `p` (higher = more likely)."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p, dtype="float64").ravel()
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    out = {"n_pos": n_pos, "n_neg": n_neg, "prevalence": (n_pos / len(y)) if len(y) else None,
           "auroc": None, "ap": None, "best_f1": None, "precision": None, "recall": None,
           "thin": n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS}
    if n_pos == 0 or n_neg == 0:
        return out
    out["auroc"] = float(roc_auc_score(y, p))
    out["ap"] = float(average_precision_score(y, p))
    prec, rec, _ = precision_recall_curve(y, p)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    i = int(np.argmax(f1))
    out["best_f1"], out["precision"], out["recall"] = float(f1[i]), float(prec[i]), float(rec[i])
    return out


def label_grid(labels, pid: str, vid: str, bid: int, n_frames: int, n_tracks: int) -> np.ndarray:
    """(F, T) int8: 1 Happening, 0 Not-happening, -1 Unknown / unlabeled."""
    g = np.full((n_frames, n_tracks), -1, dtype="int8")
    try:
        df = labels.rows_for_behavior(pid, vid, bid)
    except (KeyError, FileNotFoundError):
        return g
    if df is None or df.empty:
        return g
    f = df["frame"].to_numpy(); t = df["track"].to_numpy(); v = df["value"].to_numpy()
    keep = (f >= 0) & (f < n_frames) & (t >= 0) & (t < n_tracks) & ((v == 0) | (v == 1))
    g[f[keep], t[keep]] = v[keep]
    return g


def _active_as_model(proj_path: Path, vid: str, bid: int) -> tuple[str, dict, np.ndarray] | None:
    """The timeline lane written before per-model storage existed, as a comparison entry, from its
    provenance sidecar. Rank metrics are unchanged by the lane's rescale, so it scores like raw."""
    lane = Path(proj_path) / "predictions" / vid / f"{bid}.npy"
    side = lane.with_suffix(".meta.json")
    if not lane.exists() or not side.exists():
        return None
    try:
        rec = json.loads(side.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict):
        return None
    if rec.get("engine") == "hidra":
        used, run_id = rec.get("weights"), rec.get("run_id")
        key = (ft_model_key(run_id) if used else "zs") + f"__{rec.get('collapse', 'scene')}"
        info = {"engine": "hidra", "kind": "finetune" if used else "zeroshot", "run_id": run_id if used else None,
                "lab": rec.get("lab"), "action": rec.get("action"), "collapse": rec.get("collapse"), "weights": used}
    elif rec.get("version"):
        key = f"native-{rec['version']}"
        info = {"engine": "project", "kind": "native", "version": rec["version"]}
    else:
        return None
    from datetime import datetime
    info["predicted_at"] = datetime.fromtimestamp(lane.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    return key, info, np.load(lane)


def compare(store, labels, pid: str, bid: int) -> dict:
    """Every stored model for this behavior, scored per scope. See the module docstring."""
    proj = store.get(pid)
    beh = next((b for b in proj.behaviors if b["id"] == bid), None)
    if beh is None:
        raise KeyError(bid)
    h = beh.get("hidra") or {}
    runs = {r["run_id"]: r for r in beh.get("hidra_runs", [])}
    test_vids = [v["video_id"] for v in proj.videos if v.get("role") == "test"]

    models: dict[str, dict] = {}
    acc: dict[tuple[str, str], list] = {}          # (model_key, scope) -> [(y, p), ...]
    coverage = {}
    for v in proj.videos:
        vid = v["video_id"]
        entries: dict[str, tuple[dict, np.ndarray]] = {}
        d = model_pred_dir(proj.path, vid, bid)
        if d.exists():
            for j in sorted(d.glob("*.json")):
                try:
                    info = json.loads(j.read_text())
                    arr = np.load(d / f"{j.stem}.npy")
                except (OSError, ValueError):
                    continue
                entries[j.stem] = (info, arr)
        act = _active_as_model(proj.path, vid, bid)
        if act and act[0] not in entries:
            entries[act[0]] = (act[1], act[2])
        if not entries:
            continue
        n_frames, n_tracks = next(iter(entries.values()))[1].shape
        grid = label_grid(labels, pid, vid, bid, n_frames, n_tracks)
        lab = grid >= 0
        if vid in test_vids:
            coverage[vid] = {"labeled_frac": float(lab.mean()), "n_pos": int((grid == 1).sum()), "n_neg": int((grid == 0).sum())}
        for key, (info, arr) in entries.items():
            if arr.shape != grid.shape:
                continue
            if info.get("engine") == "hidra" and (info.get("lab"), info.get("action")) != (h.get("lab"), h.get("action")):
                continue                                   # another head's output (keys don't name the head)
            m = models.setdefault(key, {"key": key, **{k: info.get(k) for k in ("engine", "kind", "run_id", "lab", "action", "collapse", "version")},
                                        "predicted_at": {}, "clips": []})
            m["predicted_at"][vid] = info.get("predicted_at")
            m["clips"].append(vid)
            if not lab.any():
                continue
            # Did THIS model train on this clip? Decided before the test scope, so a model trained on a
            # clip before it was marked test is never shown as a fair score on it. Unknown = trained.
            kind = info.get("kind")
            if kind == "finetune":
                trained = vid in (runs.get(info.get("run_id")) or {}).get("trained_videos", [vid])
            elif kind == "native":
                tv = info.get("trained_videos")
                trained = vid in tv if isinstance(tv, list) else True
            else:
                trained = False                            # zero-shot never saw this project's labels
            scope = "train_in" if trained else ("test" if vid in test_vids else "train_out")
            if vid in test_vids:
                (m.setdefault("test_clips", []) if scope == "test" else m.setdefault("test_in_sample", [])).append(vid)
            acc.setdefault((key, scope), []).append((grid[lab], arr[lab]))

    for (key, scope), parts in acc.items():
        y = np.concatenate([a for a, _ in parts]); p = np.concatenate([b for _, b in parts])
        models[key].setdefault("scores", {})[scope] = score(y, p)

    # order + human labels: zero-shot, fine-tunes oldest->newest, then native versions
    ft_order = sorted(runs.values(), key=lambda r: r.get("created_at", ""))
    ft_rank = {r["run_id"]: i + 1 for i, r in enumerate(ft_order)}
    for m in models.values():
        r = runs.get(m.get("run_id")) or {}
        if m["kind"] == "zeroshot":
            m["label"], m["order"] = f"zero-shot · {m.get('lab')} {m.get('action')}", (0, "")
        elif m["kind"] == "finetune":
            n = ft_rank.get(m.get("run_id"))
            when = (r.get("created_at") or "")[:10]
            m["label"] = (f"fine-tune #{n} · {when} · {r.get('bouts')} bouts · {r.get('mode')}" if n
                          else "fine-tune (from before versioning)")
            m["order"] = (1, r.get("created_at") or "")
            m["models_kept"] = bool(r.get("models_kept")) if r else None
            m["trained_videos"] = r.get("trained_videos")
        else:
            m["label"], m["order"] = f"project model · {m.get('version')}", (2, m.get("version") or "")
        m["current"] = (m["kind"] == "zeroshot" and h.get("action") and not h.get("weights")) or \
                       (m["kind"] == "finetune" and h.get("run_id") and m.get("run_id") == h.get("run_id")) or \
                       (m["kind"] == "finetune" and not m.get("run_id") and h.get("weights") and not h.get("run_id"))
        m["comparable"] = m.get("collapse") in (None, h.get("collapse")) if h else True
    out_models = sorted(models.values(), key=lambda m: m["order"])
    for m in out_models:
        m.pop("order", None)

    # What could still be computed on the test clip(s): the zero-shot baseline, and every fine-tune
    # whose checkpoints are still on disk — each a predict job that leaves the timeline untouched.
    todo = []
    if h.get("action") and test_vids:
        col = h.get("collapse", "scene")
        have = lambda key: all(vid in (models.get(key) or {}).get("clips", []) for vid in test_vids)
        if not have(f"zs__{col}"):
            todo.append({"model": "zs", "label": "zero-shot baseline"})
        for r in ft_order:
            if (r.get("lab"), r.get("action")) != (h.get("lab"), h.get("action")) or not r.get("models_kept"):
                continue
            if weights_mtime(r.get("weights")) is None:    # checkpoints gone from disk (pruned / deleted by hand)
                continue
            if not have(f"{ft_model_key(r['run_id'])}__{col}"):
                todo.append({"model": ft_model_key(r["run_id"]), "label": f"fine-tune #{ft_rank[r['run_id']]}"})
    for m in out_models:                                 # C7: a test score over fewer clips than the others is flagged
        m["test_partial"] = bool(test_vids) and 0 < len(m.get("test_clips", [])) < len(test_vids)
    return {"behavior_id": bid, "test_videos": test_vids, "test_coverage": coverage,
            "collapse": h.get("collapse"), "models": out_models, "can_compute": todo,
            "min_per_class": MIN_PER_CLASS}
