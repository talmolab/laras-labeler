"""🧠 Models overview: per-clip status (current / older / none / blocked), the model registry for HiDRA
and the project's own model, and the per-behavior "predict these clips" job.

    python -m pytest tests/test_models_overview.py
"""
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from laras_labeler.app import create_app
from laras_labeler.config import Settings
from laras_labeler.project import ProjectStore


def _setup():
    root = Path(tempfile.mkdtemp())
    store = ProjectStore(root)
    app = create_app(Settings(projects_root=root), store)
    c = TestClient(app)
    pid = c.post("/api/projects", json={"name": "models"}).json()["pid"]
    p = store.get(pid)
    for vid, poses, ppc, role in (("a", True, 10.0, None), ("b", True, 10.0, None), ("c", True, None, None),
                                  ("d", False, 10.0, None), ("t", True, 10.0, "test")):
        e = {"video_id": vid, "video_path": f"/x/{vid}.mp4", "n_frames": 50, "fps": 30.0, "width": 10,
             "height": 10, "has_poses": poses, "slp_path": f"/x/{vid}.slp" if poses else None}
        if ppc:
            e["pix_per_cm"] = ppc
        if role:
            e["role"] = role
        p.manifest.setdefault("videos", []).append(e)
    nat = p.add_behavior("rearing", None, "r")
    hid = p.add_behavior("drinking", None, "d")
    # HiDRA: two fine-tunes, the second in use, checkpoints on disk for both
    ft = root / pid / "hidra" / "_finetune" / "Lab__drinking"
    runs = []
    for i, rid in enumerate(("ft-20261101-000000", "ft-20261105-000000")):
        (ft / rid / "models").mkdir(parents=True)
        (ft / rid / "models" / "c1.pkl").write_bytes(b"x")
        runs.append({"run_id": rid, "created_at": f"2026-11-0{1 + 4 * i}T00:00:00+00:00", "lab": "Lab",
                     "action": "drinking", "mode": "head", "steps": 8000, "train_on": "all", "bouts": 10 + i,
                     "trained_videos": ["a", "b"], "weights": str(ft / rid / "models" / "{config}.pkl"),
                     "models_kept": True})
    hid["hidra"] = {"lab": "Lab", "action": "drinking", "collapse": "self", "rate": 0.15,
                    "weights": runs[1]["weights"], "run_id": runs[1]["run_id"]}
    hid["hidra_runs"] = runs
    p.save()
    # native: two versions trained, v2 in use
    md = root / pid / "model" / str(nat["id"])
    md.mkdir(parents=True)
    (md / "meta.json").write_text(json.dumps({"version": "v2", "trained_at": "2026-11-06T00:00:00+00:00",
                                              "videos_used": ["a", "b"]}))
    (md / "history.json").write_text(json.dumps([
        {"version": "v1", "trained_at": "2026-11-02T00:00:00+00:00", "n_videos": 1, "ap": 0.5, "f1": 0.4},
        {"version": "v2", "trained_at": "2026-11-06T00:00:00+00:00", "n_videos": 2, "ap": 0.7, "f1": 0.6}]))
    return c, app, store, pid, nat["id"], hid["id"], runs


def _lane(store, pid, vid, bid, side):
    d = store.get(pid).path / "predictions" / vid
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{bid}.npy", np.zeros((50, 1), "float32"))
    (d / f"{bid}.meta.json").write_text(json.dumps(side))


def test_overview_status_and_registry():
    c, _app, store, pid, nb, hb, runs = _setup()
    _lane(store, pid, "a", nb, {"engine": "project", "version": "v2"})
    _lane(store, pid, "b", nb, {"engine": "project", "version": "v1"})
    from laras_labeler.predict import weights_mtime
    wm = weights_mtime(runs[1]["weights"])
    h = {"engine": "hidra", "lab": "Lab", "action": "drinking", "collapse": "self", "rate": 0.15}
    _lane(store, pid, "a", hb, {**h, "weights": runs[1]["weights"], "weights_mtime": wm, "run_id": runs[1]["run_id"]})
    _lane(store, pid, "b", hb, {**h, "weights": None, "weights_mtime": None, "run_id": None})
    d = c.get(f"/api/projects/{pid}/models").json()
    B = {b["name"]: b for b in d["behaviors"]}

    r = B["rearing"]
    assert r["engine"] == "project"
    assert [(m["label"], m["current"]) for m in r["models"]] == [("project model #1", False), ("project model #2", True)]
    st = {v: (s["status"], s["by"]) for v, s in r["clips"].items()}
    assert st == {"a": ("current", "project model #2"), "b": ("older", "project model #1"), "c": ("none", ""),
                  "d": ("blocked", ""), "t": ("none", "")}
    assert r["counts"] == {"current": 1, "older": 1, "none": 2, "blocked": 1}
    assert r["fixable"] == ["t", "b", "c"]                  # the test clip first

    k = B["drinking"]
    assert k["engine"] == "hidra" and not k["checkpoints_missing"]
    assert [(m["label"], m["current"], m["compare_key"]) for m in k["models"]] == [
        ("zero-shot", False, "zs__self"),
        ("fine-tune #1", False, "ft-20261101-000000__self"),
        ("fine-tune #2", True, "ft-20261105-000000__self")]
    st = {v: (s["status"], s["by"]) for v, s in k["clips"].items()}
    assert st["a"] == ("current", "fine-tune #2") and st["b"] == ("older", "zero-shot")
    assert st["c"] == ("blocked", "") and k["clips"]["c"]["why"].startswith("no px/cm")
    assert st["d"] == ("blocked", "") and st["t"] == ("none", "")
    assert k["fixable"] == ["t", "b"]


def test_missing_checkpoints_are_not_fixable_by_predicting():
    c, _app, _store, pid, _nb, hb, runs = _setup()
    import shutil
    shutil.rmtree(Path(runs[1]["weights"]).parent)
    k = next(b for b in c.get(f"/api/projects/{pid}/models").json()["behaviors"] if b["behavior_id"] == hb)
    assert k["checkpoints_missing"] and k["fixable"] == []
    r = c.post(f"/api/projects/{pid}/behaviors/{hb}/predict")
    assert r.status_code == 409 and "checkpoints" in r.json()["detail"]


def test_hidra_bound_behavior_is_not_predicted_by_a_leftover_native_model():
    _c, app, store, pid, nb, hb, _runs = _setup()
    md = store.get(pid).path / "model" / str(hb)
    md.mkdir(parents=True)
    (md / "meta.json").write_text(json.dumps({"version": "old", "trained_at": "2026-10-01T00:00:00+00:00"}))
    assert app.state.predictor.trained_behaviors(pid) == [nb]


def _wait(c, pid, jid, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/projects/{pid}/jobs/{jid}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.05)
    raise TimeoutError


def test_predict_clips_native_uses_the_captured_model_and_reports_failures():
    c, app, _store, pid, nb, _hb, _runs = _setup()
    calls = []
    app.state.trainer.load_model_and_meta = lambda p, b: ("MODEL", {"version": "v2"})
    app.state.features.status = lambda p, v: {"status": "ready"}

    def predict_one(p, vid, bid, model=None, meta=None):
        calls.append((vid, model, meta["version"]))
        return "feature_mismatch: nope" if vid == "c" else "ok"
    app.state.predictor.predict_one = predict_one
    r = c.post(f"/api/projects/{pid}/behaviors/{nb}/predict")
    assert r.status_code == 200 and r.json()["videos"] == ["t", "a", "b", "c"]
    j = _wait(c, pid, r.json()["job_id"])
    assert j["status"] == "done"
    assert j["result"]["videos"] == ["t", "a", "b"]
    assert j["result"]["skipped"] == [{"video_id": "c", "reason": "feature_mismatch: nope"}]
    assert all(m == "MODEL" and v == "v2" for _, m, v in calls)


def test_predict_clips_explicit_list_and_stop():
    c, app, _store, pid, nb, _hb, _runs = _setup()
    app.state.trainer.load_model_and_meta = lambda p, b: ("MODEL", {"version": "v2"})
    app.state.features.status = lambda p, v: {"status": "ready"}

    def slow(p, vid, bid, model=None, meta=None):
        time.sleep(0.4)
        return "ok"
    app.state.predictor.predict_one = slow
    r = c.post(f"/api/projects/{pid}/behaviors/{nb}/predict?videos=b,a,nope,b")
    assert r.json()["videos"] == ["b", "a"]                 # unknown ids dropped, duplicates once, order kept
    jid = r.json()["job_id"]
    assert c.post(f"/api/projects/{pid}/behaviors/{nb}/predict?videos=a").status_code == 409   # one sweep at a time
    time.sleep(0.1)
    c.post(f"/api/projects/{pid}/jobs/{jid}/cancel")
    j = _wait(c, pid, jid)
    assert j["status"] == "cancelled" and j["result"]["stopped"] and len(j["result"]["videos"]) <= 1


def test_predict_clips_needs_a_model():
    c, _app, store, pid, nb, _hb, _runs = _setup()
    import shutil
    shutil.rmtree(store.get(pid).path / "model" / str(nb))
    r = c.post(f"/api/projects/{pid}/behaviors/{nb}/predict")
    assert r.status_code == 409 and "no model" in r.json()["detail"]
