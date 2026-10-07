#!/usr/bin/env python3
"""Verify the review queue's 'remaining' count — what tells a reviewer an animal's queue is finished.

The Review queue serves candN proposals at a time, per clip x behavior x animal, so a batch running
out does not mean the queue is empty. GET .../behaviors/<bid>/remaining/<vid> counts what each
animal's queue still holds. This checks, on a throwaway synthetic project with the real app:

  shape       {video_id, behavior_id, tracks: {"0": n, ...}, total, mode: "new"}
  counts      equal the full candidate list per track, including animals with no labels at all
  no preds    a clip that was never predicted gives an empty dict, total 0
  read-only   no event is logged and no project file changes
  decisions   accepting or rejecting a proposal (a label PUT) drops that animal's count to match
  404         an unknown behavior is refused

    PYTHONPATH=src python scripts/verify_review_queue.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from laras_labeler.config import Settings
from laras_labeler.labels import LabelStore
from laras_labeler.predict import DEFAULT_POSTPROC
from laras_labeler.project import ProjectStore

PID, VID, VID2 = "rq", "clip_001_test", "clip_002_test"
JUMP, F, T = 0, 800, 3
FAILED: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILED.append(what)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def build(root: Path) -> Path:
    """One predicted clip x 3 animals, one clip with no predictions. Lanes (lo == hi = 0.6):
    animal 1: two bouts [100,120) and [300,320)   animal 2: one bout [500,520)
    animal 3: one bout [600,620), already labeled positive -> nothing left to review."""
    d = root / PID
    (d / "labels").mkdir(parents=True)
    pp = {**DEFAULT_POSTPROC, "smooth": 1, "hi": 0.6, "lo": 0.6, "min_bout": 3, "min_cand": 3, "social_gate_s": 0}
    manifest = {
        "schema_version": 1, "name": PID, "created": "2026-10-07T00:00:00+00:00", "skeleton_roles": {},
        "feature_config": {}, "media_roots": [],
        "behaviors": [{"id": JUMP, "name": "jump down", "color": "#e8a33d", "key": None, "postproc": pp}],
        "videos": [{"video_id": v, "video_path": str(d / f"{v}.mp4"), "slp_path": None, "n_frames": F,
                    "fps": 50.0, "width": 64, "height": 48, "has_poses": False} for v in (VID, VID2)],
    }
    (d / "project.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lanes = np.full((F, T), 0.1, dtype="float32")
    lanes[100:120, 0] = lanes[300:320, 0] = 0.9
    lanes[500:520, 1] = 0.9
    lanes[600:620, 2] = 0.9
    (d / "predictions" / VID).mkdir(parents=True)
    np.save(d / "predictions" / VID / f"{JUMP}.npy", lanes)
    LabelStore(ProjectStore(root)).put_spans(PID, VID, [
        {"behavior_id": JUMP, "track": 2, "start": 600, "end": 620, "value": 1, "source": "manual"}])
    return d


def main() -> int:
    warnings.filterwarnings("ignore", message=r".*httpx.*")
    from fastapi.testclient import TestClient
    from laras_labeler.app import create_app

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        d = build(root)
        c = TestClient(create_app(Settings(projects_root=root), ProjectStore(root)))
        base = f"/api/projects/{PID}"

        print("GET remaining")
        ev = d / "events" / "server.jsonl"
        n_ev = len(ev.read_text(encoding="utf-8").splitlines()) if ev.exists() else 0
        before = {k: sha(d / k) for k in ("project.json", f"labels/{VID}.parquet")}
        r = c.get(f"{base}/behaviors/{JUMP}/remaining/{VID}")
        j = r.json()
        check(r.status_code == 200 and set(j) == {"video_id", "behavior_id", "tracks", "total", "mode"},
              "shape {video_id, behavior_id, tracks, total, mode}", r.text[:200])
        check(j.get("tracks") == {"0": 2, "1": 1, "2": 0} and j.get("total") == 3 and j.get("mode") == "new",
              "per-animal counts, animals with no labels included", json.dumps(j))
        full = [len(c.get(f"{base}/behaviors/{JUMP}/candidates/{VID}", params={"track": t, "n": 100000}).json())
                for t in range(T)]
        check(full == [j["tracks"][str(t)] for t in range(T)], "each count equals that animal's full queue", str(full))
        j2 = c.get(f"{base}/behaviors/{JUMP}/remaining/{VID2}").json()
        check(j2.get("tracks") == {} and j2.get("total") == 0, "a clip with no predictions -> {} and 0", json.dumps(j2))
        n_ev2 = len(ev.read_text(encoding="utf-8").splitlines()) if ev.exists() else 0
        check(n_ev2 == n_ev and all(sha(d / k) == h for k, h in before.items()), "read-only: no event, no file touched")
        check(c.get(f"{base}/behaviors/9/remaining/{VID}").status_code == 404, "unknown behavior -> 404")

        print("decisions drop the count")
        r = c.put(f"{base}/labels/{VID}", json=[{"behavior_id": JUMP, "track": 0, "start": 100, "end": 120,
                                                  "value": 1, "source": "candidate"}])           # accept
        j = c.get(f"{base}/behaviors/{JUMP}/remaining/{VID}").json()
        check(r.status_code == 200 and j["tracks"] == {"0": 1, "1": 1, "2": 0}, "accept: animal 1 drops to 1", json.dumps(j))
        c.put(f"{base}/labels/{VID}", json=[{"behavior_id": JUMP, "track": 0, "start": 300, "end": 320,
                                             "value": 0, "source": "candidate"}])                # reject
        c.put(f"{base}/labels/{VID}", json=[{"behavior_id": JUMP, "track": 1, "start": 500, "end": 520,
                                             "value": 0, "source": "candidate"}])
        j = c.get(f"{base}/behaviors/{JUMP}/remaining/{VID}").json()
        check(j["tracks"] == {"0": 0, "1": 0, "2": 0} and j["total"] == 0, "reject: every queue empty", json.dumps(j))

    print("\n" + ("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: " + "; ".join(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
