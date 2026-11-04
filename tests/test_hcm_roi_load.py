"""Opening a project with many clips missing outlines must not flood the shared lab database: one
engine, one connection at a time, and stop after the first lookup that gets nothing back.
python -m pytest tests/test_hcm_roi_load.py"""
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

from laras_labeler import hcm_roi

CAGE = {"type": "Polygon", "coordinates": [[[0, 0], [9, 0], [9, 9], [0, 9], [0, 0]]]}


class FakeSA:
    """sqlalchemy stand-in: counts engines and the connections open at the same time."""
    def __init__(self, answer=True):
        self.engines = 0
        self.open = self.max_open = self.lookups = 0
        self.answer = answer
        self.lock = threading.Lock()
        self.mod = types.ModuleType("sqlalchemy")
        self.mod.create_engine = self.create_engine
        self.mod.text = lambda s: s

    def create_engine(self, url, connect_args=None, **kw):
        self.engines += 1
        fake = self

        class Conn:
            def __enter__(self):
                if not fake.answer:
                    raise OSError("connection refused")
                with fake.lock:
                    fake.open += 1
                    fake.lookups += 1
                    fake.max_open = max(fake.max_open, fake.open)
                time.sleep(0.005)                       # long enough for overlapping callers to show up
                return self

            def __exit__(self, *a):
                with fake.lock:
                    fake.open -= 1
                return False

            def rollback(self):
                pass

            def execute(self, sql, params):
                return types.SimpleNamespace(scalar=lambda: "valid-feeder-cage-with-side",
                                             first=lambda: (CAGE,))

        return types.SimpleNamespace(connect=lambda: Conn())


def _project_with_missing_outlines(n):
    from fastapi.testclient import TestClient
    from laras_labeler.app import create_app
    from laras_labeler.config import Settings
    from laras_labeler.project import ProjectStore
    root = Path(tempfile.mkdtemp())
    store = ProjectStore(root)
    c = TestClient(create_app(Settings(projects_root=root), store))
    pid = c.post("/api/projects", json={"name": "many"}).json()["pid"]
    proj = store.get(pid)
    for i in range(n):
        vid = f"clip_{i:03d}_2025_02_12_2025_09_{1 + i % 28:02d}_00_01_05_cam_06.11.predictions.slp_batch_10_f0-15000"
        proj.manifest.setdefault("videos", []).append({"video_id": vid, "video_path": f"/x/{vid}.mp4", "n_frames": 100,
                                                       "fps": 50.0, "width": 10, "height": 10, "has_poses": True})
    proj.save()
    return c, store, pid


def _wait_for(cond, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout and not cond():
        time.sleep(0.05)


def test_one_engine_one_connection_at_a_time(monkeypatch):
    fake = FakeSA()
    monkeypatch.setitem(sys.modules, "sqlalchemy", fake.mod)
    monkeypatch.setattr(hcm_roi, "_ENGINES", {}, raising=False)
    monkeypatch.setattr(hcm_roi, "DATABASE_URL", "postgresql://u:p@h/db")
    c, store, pid = _project_with_missing_outlines(60)
    monkeypatch.setattr(hcm_roi, "DATABASE_URL", "postgresql://u:p@h/db")   # create_app re-read it
    c.get(f"/api/projects/{pid}")                       # project open -> the background outline fetch
    _wait_for(lambda: all(v.get("cage_roi") for v in store.get(pid).videos))
    assert all(v.get("cage_roi") for v in store.get(pid).videos)
    assert fake.lookups == 60 and fake.max_open == 1 and fake.engines == 1, (fake.lookups, fake.max_open, fake.engines)


def test_stops_after_the_first_lookup_without_an_answer(monkeypatch):
    fake = FakeSA(answer=False)
    monkeypatch.setitem(sys.modules, "sqlalchemy", fake.mod)
    monkeypatch.setattr(hcm_roi, "_ENGINES", {}, raising=False)
    c, store, pid = _project_with_missing_outlines(40)
    monkeypatch.setattr(hcm_roi, "DATABASE_URL", "postgresql://u:p@h/db")
    calls = []
    real = hcm_roi.fetch_rois
    monkeypatch.setattr(hcm_roi, "fetch_rois", lambda *a, **k: calls.append(1) or real(*a, **k))
    c.get(f"/api/projects/{pid}")
    time.sleep(1.0)
    assert len(calls) == 1                               # not 40 tries against a database that isn't answering


def test_real_engine_keeps_no_idle_connections(monkeypatch):
    import sqlalchemy
    from sqlalchemy.pool import NullPool
    monkeypatch.setattr(hcm_roi, "_ENGINES", {}, raising=False)
    monkeypatch.setattr(hcm_roi, "DATABASE_URL", "postgresql://u:p@127.0.0.1:1/db")
    e1 = hcm_roi._engine(sqlalchemy.create_engine, 8)
    assert e1 is hcm_roi._engine(sqlalchemy.create_engine, 8) and isinstance(e1.pool, NullPool)
