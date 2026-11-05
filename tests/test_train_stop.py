"""■ Stop on Train (project's own model): stopped while fitting -> nothing kept; stopped while
re-predicting -> the model that was just saved is kept and the job says so.

    python -m pytest tests/test_train_stop.py
"""
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from laras_labeler.app import create_app
from laras_labeler.config import Settings
from laras_labeler.project import ProjectStore


def _client():
    root = Path(tempfile.mkdtemp())
    store = ProjectStore(root)
    app = create_app(Settings(projects_root=root), store)
    c = TestClient(app)
    pid = c.post("/api/projects", json={"name": "stop"}).json()["pid"]
    p = store.get(pid)
    p.add_behavior("rearing", None, "r")
    p.save()
    return c, app, pid


def _slow(progress, n=200):
    for i in range(n):
        progress(i % 100, "working")
        time.sleep(0.02)


def _run_and_stop(c, pid, after=0.3):
    jid = c.post(f"/api/projects/{pid}/behaviors/0/train").json()["job_id"]
    time.sleep(after)
    assert c.post(f"/api/projects/{pid}/jobs/{jid}/cancel").status_code == 200
    for _ in range(300):
        j = c.get(f"/api/projects/{pid}/jobs/{jid}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.02)
    raise TimeoutError


def test_stop_while_training_keeps_nothing():
    c, app, pid = _client()
    saved = []

    def train(pid_, bid, progress):
        _slow(progress)                     # a Stop lands at the next progress report
        saved.append(bid)
        return {"version": "v-new"}

    app.state.trainer.train = train
    app.state.predictor.predict_behavior = lambda *a, **k: {"videos": [], "skipped": []}
    j = _run_and_stop(c, pid)
    assert j["status"] == "cancelled" and not j["result"] and not saved


def test_stop_while_repredicting_keeps_the_new_model():
    c, app, pid = _client()
    app.state.trainer.train = lambda pid_, bid, progress: {"version": "v-new", "metrics": {}}

    def predict(pid_, bid, progress, videos=None):
        _slow(progress)
        return {"videos": ["never"], "skipped": []}

    app.state.predictor.predict_behavior = predict
    j = _run_and_stop(c, pid)
    assert j["status"] == "cancelled"
    assert j["result"]["version"] == "v-new" and j["result"]["predict"]["stopped"] is True
