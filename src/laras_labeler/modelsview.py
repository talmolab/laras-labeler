"""The 🧠 Models overview: for every behavior, every model it has had (which one is in use, when it was
trained, what it learned from) and, for every clip, whether the clip's predictions come from that model.

Read-only and cheap enough to open on a project of ~200 clips: it reads project.json, history.json,
meta.json and each lane's small <bid>.meta.json sidecar, and stats files. It never loads a prediction
array. Scores are NOT here (they need every stored array -- see modelcompare); the panel fetches
/behaviors/{bid}/compare per behavior and joins them on `compare_key`.

Clip status per behavior:
  current   the lane on disk is what a Predict would give now (Predictor.provenance)
  older     made by an earlier model, another head or other settings (`why` says which)
  none      never predicted for this behavior
  blocked   cannot be predicted at all (no tracking file; for HiDRA, no px/cm) -- not "out of date"
`fixable` lists the older/none clips that predicting this behavior again would bring up to date.
"""
from __future__ import annotations

import json

from . import hidra
from .predict import ft_model_key, weights_mtime


def _read_json(p):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _hidra_models(beh: dict) -> tuple[list[dict], dict, dict]:
    """Registry for a HiDRA-bound behavior. Ranks (#n) count every run of the behavior oldest-first,
    the same numbering as the model comparison, so "fine-tune #3" means the same thing in both."""
    h = beh.get("hidra") or {}
    col = h.get("collapse", "scene")
    runs = beh.get("hidra_runs") or []
    ordered = sorted(runs, key=lambda r: r.get("created_at") or "")
    rank = {r["run_id"]: i + 1 for i, r in enumerate(ordered) if r.get("run_id")}
    tuned_ok = bool(h.get("weights")) and weights_mtime(h.get("weights")) is not None
    same_head = lambda r: (r.get("lab"), r.get("action")) == (h.get("lab"), h.get("action"))
    models = [{
        "key": "zs", "compare_key": f"zs__{col}", "kind": "zeroshot", "label": "zero-shot",
        "detail": f"HiDRA's pretrained {h.get('lab')} · {h.get('action')} head, before any of your labels",
        "current": not h.get("weights"), "checkpoints": True,
    }]
    for r in ordered:
        if not same_head(r):
            continue
        n = rank.get(r.get("run_id"))
        kept = bool(r.get("models_kept")) and weights_mtime(r.get("weights")) is not None
        models.append({
            "key": ft_model_key(r.get("run_id")), "compare_key": f"{ft_model_key(r.get('run_id'))}__{col}",
            "kind": "finetune", "rank": n, "label": f"fine-tune #{n}" if n else "fine-tune",
            "when": r.get("created_at"), "train_on": r.get("train_on"),
            "train_on_label": hidra.TRAIN_ON.get(r.get("train_on") or "all", r.get("train_on")),
            "mode": r.get("mode"), "steps": r.get("steps"), "bouts": r.get("bouts"),
            "n_clips": len(r.get("trained_videos") or []), "checkpoints": kept,
            "current": bool(h.get("run_id")) and r.get("run_id") == h.get("run_id") and bool(h.get("weights")),
        })
    if h.get("weights") and not h.get("run_id"):        # bound before runs were versioned
        models.append({"key": "ft-legacy", "compare_key": f"ft-legacy__{col}", "kind": "finetune",
                       "label": "fine-tune (from before versioning)", "current": True, "checkpoints": tuned_ok})
    other_heads = sum(1 for r in runs if not same_head(r))
    info = {"engine": "hidra", "head": f"{h.get('lab')} · {h.get('action')}", "collapse": col,
            "checkpoints_missing": bool(h.get("weights")) and not tuned_ok, "other_head_runs": other_heads}
    return models, rank, info


def _native_models(trainer, pid: str, bid: int) -> tuple[list[dict], dict, dict]:
    """Registry for the project's own model: one row per Train (history.json), the latest row of a
    version wins when retraining reproduced the same model. Numbered #1.. in training order."""
    meta = trainer.model_meta(pid, bid) or {}
    hist = trainer.history(pid, bid) or []
    last = {}
    for i, row in enumerate(hist):
        if row.get("version"):
            last[row["version"]] = i
    order = sorted(last, key=lambda v: last[v])
    num = {v: i + 1 for i, v in enumerate(order)}
    models = []
    for v in order:
        row = hist[last[v]]
        models.append({
            "key": f"native-{v}", "compare_key": f"native-{v}", "kind": "native", "rank": num[v],
            "label": f"project model #{num[v]}", "version": v, "when": row.get("trained_at"),
            "n_clips": row.get("n_videos"), "bouts": row.get("n_pos_bouts"), "neg_bouts": row.get("n_neg_bouts"),
            "feature_set": row.get("feature_set"), "cv_ap": row.get("ap"), "cv_f1": row.get("f1"),
            "current": v == meta.get("version"), "checkpoints": True,
        })
    if meta.get("version") and meta["version"] not in num:   # trained before history.json existed
        models.append({"key": f"native-{meta['version']}", "compare_key": f"native-{meta['version']}",
                       "kind": "native", "label": "project model", "version": meta["version"],
                       "when": meta.get("trained_at"), "n_clips": len(meta.get("videos_used") or []),
                       "current": True, "checkpoints": True})
        num[meta["version"]] = None
    return models, num, {"engine": "project" if meta.get("trained_at") else "none"}


def _made_by(rec: dict | None, h: dict, rank: dict, num: dict, cur: dict | None) -> str:
    """Name of the model that wrote a lane, from its sidecar ('' when the lane predates sidecars)."""
    if not isinstance(rec, dict):
        return ""
    if rec.get("engine") == "hidra":
        if not h.get("action"):
            return f"HiDRA {rec.get('action')} ({'fine-tuned' if rec.get('weights') else 'zero-shot'})"
        if (rec.get("lab"), rec.get("action")) != (h.get("lab"), h.get("action")):
            return f"another HiDRA head ({rec.get('action')})"
        if not rec.get("weights"):
            return "zero-shot"
        n = rank.get(rec.get("run_id"))
        if n:
            return f"fine-tune #{n}"
        if not rec.get("run_id") and rec.get("weights") == h.get("weights") and cur:   # the legacy fine-tune in use
            return cur["label"]
        return "an earlier fine-tune"
    if rec.get("version") in num:
        n = num[rec["version"]]
        return f"project model #{n}" if n else "project model"
    return "an earlier project model"


def overview(store, predictor, trainer, pid: str, hidra_block) -> dict:
    """See the module docstring. `hidra_block(entry)` -> reason a clip can't run HiDRA, or None."""
    proj = store.get(pid)
    clips = []
    for v in proj.videos:
        entry = proj.video(v["video_id"]) or v
        clips.append({"video_id": v["video_id"], "role": v.get("role"), "has_poses": bool(v.get("has_poses")),
                      "hidra_block": None if not v.get("has_poses") else hidra_block(entry)})
    out = []
    for beh in proj.behaviors:
        bid = beh["id"]
        h = beh.get("hidra") or {}
        if h.get("action"):
            models, rank, info = _hidra_models(beh)
            num = {}
        else:
            models, num, info = _native_models(trainer, pid, bid)
            rank = {}
        cur = next((m for m in models if m.get("current")), None)
        ctx: dict = {}
        statuses, counts, fixable = {}, {"current": 0, "older": 0, "none": 0, "blocked": 0}, []
        can_fix_any = info["engine"] != "none" and not info.get("checkpoints_missing")
        for c in clips:
            vid = c["video_id"]
            lane = predictor._path(pid, vid, bid)
            prov = predictor.provenance(pid, vid, bid, ctx) if lane.exists() else None
            rec = _read_json(lane.with_suffix(".meta.json")) if prov else None
            st = {"by": _made_by(rec, h, rank, num, cur), "predicted_at": (prov or {}).get("predicted_at")}
            block = ("no tracking file" if not c["has_poses"]
                     else c["hidra_block"] if info["engine"] == "hidra" else None)
            if prov is None:
                st["status"] = "blocked" if block else "none"
                st["why"] = block or ("no model yet" if info["engine"] == "none" else "never predicted")
            elif prov.get("stale"):
                st["status"], st["why"] = ("blocked", block) if block else ("older", prov.get("why") or "")
            else:
                st["status"], st["why"] = "current", ""
            if st["status"] in ("older", "none") and can_fix_any and not prov_no_model(prov):
                fixable.append(vid)
            counts[st["status"]] += 1
            statuses[vid] = st
        # held-out test clips first: they are what gives the model in use a fair score
        role = {c["video_id"]: c["role"] for c in clips}
        fixable.sort(key=lambda v: role.get(v) != "test")
        out.append({
            "behavior_id": bid, "name": beh.get("name"), "color": beh.get("color"), **info,
            "current": cur, "models": models, "clips": statuses, "counts": counts, "fixable": fixable,
        })
    return {"videos": clips, "behaviors": out}


def prov_no_model(prov: dict | None) -> bool:
    return bool(prov and prov.get("no_model"))
