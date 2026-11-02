"""Object-directed bouts: the object list, the nearest-object suggestion, and label storage.

    python -m pytest tests/test_objects.py
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from laras_labeler import objects
from laras_labeler.labels import LabelStore
from laras_labeler.project import ProjectStore

SQUARE = [[100, 100], [200, 100], [200, 200], [100, 200]]
CAGE = [[0, 0], [400, 0], [400, 300], [0, 300]]


class FakeProj:
    def __init__(self, manifest=None):
        self.manifest = manifest or {"feature_config": {}}
        self.saved = 0

    def save(self):
        self.saved += 1


def poses_at(xy, n_frames=20):
    """(F, 1 track, 1 node, 3) with the nose fixed at xy."""
    p = np.zeros((n_frames, 1, 1, 3), dtype="float32")
    p[..., 0], p[..., 1], p[..., 2] = xy[0], xy[1], 1.0
    return p


def test_defaults_until_changed():
    proj = FakeProj()
    assert [o["id"] for o in objects.project_objects(proj)] == ["feeder_top", "feeder_side", "cage", "spout"]
    assert "objects" not in proj.manifest


def test_add_object_slug_unique_and_no_duplicate_names():
    proj = FakeProj()
    a = objects.add_object(proj, "  Running   wheel ")
    assert a == {"id": "running_wheel", "name": "Running wheel", "roi": None, "measure": "region"}
    b = objects.add_object(proj, "running-wheel 2")
    assert b["id"] == "running_wheel_2"
    with pytest.raises(ValueError):
        objects.add_object(proj, "RUNNING WHEEL")
    with pytest.raises(ValueError):
        objects.add_object(proj, "   ")
    assert [o["id"] for o in proj.manifest["objects"]][-2:] == ["running_wheel", "running_wheel_2"]


def test_suggest_nearest_outlined_object():
    proj = FakeProj()
    entry = {"feeder_top_roi": SQUARE, "cage_roi": CAGE, "spout": [390, 150]}
    objects.add_object(proj, "toy")                                  # no outline: listed, never suggested
    r = objects.suggest(proj, entry, poses_at((150, 150)), 0, 0, 0, 20)
    assert r["suggested"] == "feeder_top"                            # nose inside the feeder outline
    by = {o["id"]: o for o in r["objects"]}
    assert by["feeder_top"]["distance_px"] == 0.0
    assert by["cage"]["distance_px"] == 150.0                        # nearest wall, not 0 for being inside
    assert by["spout"]["distance_px"] == 240.0                       # spout point fallback
    assert by["toy"]["outline"] is False and by["toy"]["distance_px"] is None
    assert by["feeder_side"]["outline"] is False
    assert r["objects"][-1]["distance_px"] is None                   # outline-less objects sort last


def test_suggest_wall_when_nose_at_the_rim():
    proj = FakeProj()
    entry = {"feeder_top_roi": SQUARE, "cage_roi": CAGE}
    r = objects.suggest(proj, entry, poses_at((3, 150)), 0, 0, 0, 20)
    assert r["suggested"] == "cage"


def test_suggest_without_poses_or_visible_nose():
    proj = FakeProj()
    entry = {"feeder_top_roi": SQUARE}
    assert objects.suggest(proj, entry, None, None, 0, 0, 20)["suggested"] is None
    p = poses_at((150, 150)); p[..., :2] = np.nan
    assert objects.suggest(proj, entry, p, 0, 0, 0, 20)["suggested"] is None


def test_project_wide_roi_used_when_clip_has_none():
    proj = FakeProj({"feature_config": {"cage_roi": CAGE}})
    assert "cage" in objects.geometry(proj, {})


def _store():
    root = Path(tempfile.mkdtemp())
    store = ProjectStore(root)
    p = store.create("t")
    (p.path / "labels").mkdir(parents=True, exist_ok=True)
    return store, p.pid


def test_labels_store_object_and_split_runs():
    store, pid = _store()
    ls = LabelStore(store)
    ls.put_spans(pid, "v", [{"behavior_id": 0, "track": 0, "start": 10, "end": 20, "value": 1, "object": "feeder_top"},
                            {"behavior_id": 0, "track": 0, "start": 20, "end": 30, "value": 1, "object": "cage"},
                            {"behavior_id": 0, "track": 0, "start": 30, "end": 40, "value": 1}])
    runs = LabelStore(store).get_runs_src(pid, "v", 0, 0)[0]          # fresh store: read back from disk
    assert [(r[0], r[1], r[5]) for r in runs] == [(10, 20, "feeder_top"), (20, 30, "cage"), (30, 40, "")]
    assert LabelStore(store).stored_objects(pid, "v") == {"feeder_top", "cage"}


def test_old_parquet_without_obj_column_loads_as_no_object():
    store, pid = _store()
    path = store.get(pid).path / "labels" / "v.parquet"
    pd.DataFrame({"frame": np.arange(5, 8, dtype="int32"), "track": np.int16(0), "behavior_id": np.int16(0),
                  "value": np.int8(1), "source": "manual", "target": np.int16(-1)}).to_parquet(path, index=False)
    runs = LabelStore(store).get_runs_src(pid, "v", 0, 0)[0]
    assert runs == [[5, 8, 1, "manual", -1, ""]]


def test_export_has_object_column():
    store, pid = _store()
    proj = store.get(pid)
    proj.manifest.setdefault("videos", []).append({"video_id": "v", "video_path": "/x/v.mp4", "fps": 10.0})
    proj.save()
    ls = LabelStore(store)
    ls.put_spans(pid, "v", [{"behavior_id": 0, "track": 0, "start": 0, "end": 10, "value": 1, "object": "spout"}])
    rows = ls.export_bouts(pid)
    assert "object" in LabelStore.EXPORT_COLS and rows[0]["object"] == "spout"


def test_api_validates_objects_and_keeps_removed_ones_writable():
    from fastapi.testclient import TestClient
    from laras_labeler.app import create_app
    from laras_labeler.config import Settings
    root = Path(tempfile.mkdtemp())
    store = ProjectStore(root)
    c = TestClient(create_app(Settings(projects_root=root), store))
    pid = c.post("/api/projects", json={"name": "t"}).json()["pid"]
    proj = store.get(pid)
    proj.manifest.setdefault("videos", []).append({"video_id": "v", "video_path": "/x/v.mp4", "n_frames": 100,
                                                   "fps": 10.0, "width": 10, "height": 10, "has_poses": False})
    proj.save()
    span = lambda o: [{"behavior_id": 0, "track": 0, "start": 0, "end": 10, "value": 1, "object": o}]  # noqa: E731
    assert c.put(f"/api/projects/{pid}/labels/v", json=span("nest")).status_code == 400
    assert c.post(f"/api/projects/{pid}/objects", json={"name": "nest"}).json()["id"] == "nest"
    assert c.put(f"/api/projects/{pid}/labels/v", json=span("nest")).status_code == 200
    assert c.delete(f"/api/projects/{pid}/objects/cage").status_code == 409      # outlined: stays
    assert c.delete(f"/api/projects/{pid}/objects/nest").status_code == 200
    assert "nest" not in [o["id"] for o in c.get(f"/api/projects/{pid}/objects").json()]
    assert c.put(f"/api/projects/{pid}/labels/v", json=span("nest")).status_code == 200   # its old bouts re-save
    assert c.get(f"/api/projects/{pid}/labels/v?track=0&behavior=0").json()["0"][0][5] == "nest"
    sug = c.get(f"/api/projects/{pid}/videos/v/object-suggest?track=0&start=0&end=10").json()
    assert sug["suggested"] is None and {o["id"] for o in sug["objects"]} >= {"feeder_top", "cage", "spout"}
