#!/usr/bin/env python3
"""Verify the training-data table: GET .../behaviors/<bid>/training-data, what the NEXT Train uses.

On a throwaway project (one 3-mouse clip with poses, one clip without), with the real app:

  native     the bouts listed as used are exactly Trainer.gather's training bouts; Unknown-only
             and unlabeled mice are not used; a clip without poses is not used
  hidra      the Happening spans listed as sent are exactly what export_bouts writes; a mouse with
             only Not-happening labels is not used; a clip whose .slp is missing is a blocker; "trains as Not-happening" = every other frame of
             a used mouse, and "never reviewed" = those minus its reviewed Not-happening frames
             (Unknown counts as never reviewed: the fine-tune sends Happening only)
  directed   a bout with a picked partner is sent to that partner; one without, in a 3-mouse clip,
             is listed as skipped
  queue      unreviewed proposals left on a used mouse are counted
  refactor   export_bouts writes byte-identical CSV to the pre-refactor version at git HEAD~
  read-only  no event logged, no project file changed; unknown behavior -> 404

    PYTHONPATH=src python scripts/verify_training_data.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from laras_labeler.config import Settings
from laras_labeler.labels import LabelStore
from laras_labeler.predict import DEFAULT_POSTPROC
from laras_labeler.project import ProjectStore

PID, A, B = "td", "clip_a_cam_01.10", "clip_b_cam_02.11"
JUMP, EAT, SNIFF = 0, 1, 2
F, T = 1200, 3
FAILED: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILED.append(what)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def write_slp(path: Path) -> None:
    """3 mice x 9 nodes walking around a 320x256 arena, no gaps."""
    import sleap_io as sio
    nodes = ["nose", "left_ear", "right_ear", "neck", "thorax", "hip_left", "hip_right", "tail_base", "tail_tip"]
    off = np.array([[20, 0], [12, -6], [12, 6], [10, 0], [0, 0], [-8, -6], [-8, 6], [-16, 0], [-34, 0]], float)
    rng = np.random.default_rng(0)
    skel = sio.Skeleton(nodes=nodes)
    tracks = [sio.Track(name=f"mouse{i + 1}") for i in range(T)]
    video = sio.Video.from_filename(str(path.with_suffix(".mp4")))
    cent = np.cumsum(rng.normal(0, 1.5, (F, T, 2)), axis=0) + np.array([[80, 80], [160, 128], [240, 170]])
    head = np.cumsum(rng.normal(0, 0.05, (F, T)), axis=0)
    frames = []
    for f in range(F):
        insts = []
        for t in range(T):
            c, s = np.cos(head[f, t]), np.sin(head[f, t])
            pts = cent[f, t] + off @ np.array([[c, s], [-s, c]])
            insts.append(sio.PredictedInstance.from_numpy(points_data=pts, skeleton=skel, track=tracks[t], score=1.0))
        frames.append(sio.LabeledFrame(video=video, frame_idx=f, instances=insts))
    sio.save_file(sio.Labels(labeled_frames=frames, videos=[video], skeletons=[skel], tracks=tracks), str(path))


def build(root: Path) -> Path:
    d = root / PID
    (d / "labels").mkdir(parents=True)
    write_slp(d / f"{A}.slp")
    pp = {**DEFAULT_POSTPROC, "smooth": 1, "hi": 0.6, "lo": 0.6, "min_bout": 3, "min_cand": 3, "social_gate_s": 0}
    manifest = {
        "schema_version": 1, "name": PID, "created": "2026-10-29T00:00:00+00:00", "skeleton_roles": {},
        "feature_config": {}, "media_roots": [],
        "behaviors": [
            {"id": JUMP, "name": "jump down", "color": "#e8a33d", "key": None},
            {"id": EAT, "name": "eating", "color": "#7fb069", "key": None, "postproc": pp,
             "hidra": {"lab": "LabX", "action": "biteobject", "collapse": "self", "rate": 0.15}},
            {"id": SNIFF, "name": "sniff", "color": "#b07cc6", "key": None,
             "hidra": {"lab": "LabX", "action": "sniff", "collapse": "directed", "rate": 0.15}}],
        "videos": [
            {"video_id": A, "video_path": str(d / f"{A}.mp4"), "slp_path": str(d / f"{A}.slp"), "n_frames": F,
             "fps": 30.0, "width": 320, "height": 256, "has_poses": True},
            {"video_id": B, "video_path": str(d / f"{B}.mp4"), "slp_path": None, "n_frames": F,
             "fps": 30.0, "width": 320, "height": 256, "has_poses": False}],
    }
    (d / "project.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ls = LabelStore(ProjectStore(root))
    span = lambda b, t, s, e, v, src="manual", tgt=None: {"behavior_id": b, "track": t, "start": s, "end": e,  # noqa: E731
                                                          "value": v, "source": src,
                                                          **({"target": tgt} if tgt is not None else {})}
    ls.put_spans(PID, A, [
        # native: mouse 1 one Happening run made of two sources (one bout to gather), one Not-happening;
        # mouse 2 Unknown only (not used); mouse 3 nothing
        span(JUMP, 0, 100, 120, 1), span(JUMP, 0, 120, 130, 1, "candidate"), span(JUMP, 0, 300, 350, 0),
        span(JUMP, 1, 400, 420, 2),
        # hidra self: mouse 1 Happening + Not-happening + Unknown; mouse 2 Not-happening only (not used)
        span(EAT, 0, 100, 150, 1, "candidate"), span(EAT, 0, 500, 600, 0, "candidate"), span(EAT, 0, 700, 710, 2),
        span(EAT, 1, 50, 60, 0),
        # hidra directed, 3 mice: one bout aimed at mouse 2 (sent), one with no partner (skipped)
        span(SNIFF, 0, 200, 220, 1, tgt=1), span(SNIFF, 0, 240, 260, 1)])
    ls.put_spans(PID, B, [span(JUMP, 2, 10, 40, 1), span(EAT, 2, 10, 40, 1)])
    # an eating prediction lane on clip A: one unlabeled proposal left on mouse 1
    lanes = np.full((F, T), 0.1, dtype="float32")
    lanes[800:820, 0] = 0.9
    lanes[100:150, 0] = 0.9                       # already labeled -> not a proposal
    (d / "predictions" / A).mkdir(parents=True)
    np.save(d / "predictions" / A / f"{EAT}.npy", lanes)
    return d


def old_export_bouts():
    """export_bouts as it was before the finetune_bouts refactor (git HEAD~ of this change)."""
    repo = Path(__file__).resolve().parent.parent
    for rev in ("HEAD", "HEAD~1", "HEAD~2"):
        src = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:src/laras_labeler/hidra.py"],
                             capture_output=True, text=True).stdout
        if src and "def finetune_bouts" not in src:
            break
    tmp = Path(tempfile.mkdtemp()) / "old_hidra.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("old_hidra", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["old_hidra"] = mod                     # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod.export_bouts


def main() -> int:
    warnings.filterwarnings("ignore")
    from fastapi.testclient import TestClient
    from laras_labeler import hidra
    from laras_labeler.app import create_app
    from laras_labeler.featurestore import FeatureStore
    from laras_labeler.training import Trainer
    from laras_labeler.video import VideoManager

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        d = build(root)
        store = ProjectStore(root)
        c = TestClient(create_app(Settings(projects_root=root), store))
        base = f"/api/projects/{PID}"
        row = lambda j, vid, t: next(x for cl in j["clips"] if cl["video_id"] == vid  # noqa: E731
                                     for x in cl["tracks"] if x["track"] == t)

        print("native (this project's model)")
        r = c.get(f"{base}/behaviors/{JUMP}/training-data")
        j = r.json()
        check(r.status_code == 200 and j["engine"] == "native" and j["head"] is None, "engine native", r.text[:200])
        m1, m2, m3, b3 = row(j, A, 0), row(j, A, 1), row(j, A, 2), row(j, B, 2)
        check(m1["used"] and m1["pos_bouts"] == 1 and m1["neg_bouts"] == 1 and m1["pos_frames"] == 30
              and m1["neg_frames"] == 50, "mouse 1: 1 Happening (two sources, one bout) + 1 Not-happening", json.dumps(m1))
        check(not m2["used"] and m2["unknown_frames"] == 20 and not m3["used"], "Unknown-only and unlabeled mice not used")
        check(not b3["used"] and not b3["blocker"] and "poses" in b3["note"] and j["totals"]["blockers"] == [],
              "clip without poses not used (and not a blocker: gather skips it)", json.dumps(b3))
        fs = FeatureStore(store, VideoManager(store))
        fs.compute(PID, A, lambda p, m: None)
        got = Trainer(store, LabelStore(store), fs).gather(PID, JUMP)
        gather_bouts = sorted((b["video_id"], b["track"], b["start"], b["end"], b["value"]) for b in got[6])
        table_bouts = sorted((cl["video_id"], t["track"], x["start"], x["end"], x["value"])
                             for cl in j["clips"] for t in cl["tracks"] if t["used"] for x in t["bouts"] if x["sent"])
        check(table_bouts == gather_bouts, "used bouts == Trainer.gather's training bouts", f"{table_bouts} vs {gather_bouts}")
        check(j["totals"]["pos_bouts"] == 1 and j["totals"]["neg_bouts"] == 1 and j["totals"]["never_reviewed_frames"] == 0,
              "native totals: 1+ / 1-, nothing trained unreviewed", json.dumps(j["totals"]))

        print("hidra (self-directed fine-tune)")
        j = c.get(f"{base}/behaviors/{EAT}/training-data").json()
        check(j["engine"] == "hidra" and j["head"] == {"lab": "LabX", "action": "biteobject", "collapse": "self"},
              "engine hidra + head", json.dumps(j["head"]))
        e1, e2, eb = row(j, A, 0), row(j, A, 1), row(j, B, 2)
        check(e1["used"] and e1["pos_bouts"] == 1 and e1["partners"] == ["self"], "mouse 1 used, 1 span sent to self", json.dumps(e1))
        check(e1["trained_neg_frames"] == F - 50, "every other frame trains as Not-happening", str(e1["trained_neg_frames"]))
        check(e1["never_reviewed_frames"] == F - 50 - 100, "never reviewed = those minus reviewed Not-happening (Unknown included)",
              str(e1["never_reviewed_frames"]))
        check([x["value"] for x in e1["bouts"] if not x["sent"]] == [0], "reviewed Not-happening listed, not sent")
        check(not e2["used"] and "no Happening" in e2["note"], "Not-happening-only mouse not used", json.dumps(e2))
        check(not eb["used"] and eb["blocker"] and ".slp" in eb["note"] and j["totals"]["blockers"] == [B],
              "clip without an .slp flagged as a blocker: the fine-tune stops on it", json.dumps(eb))
        check(j["totals"]["unknown_on_used"] == 10, "Unknown frames on a used mouse counted (they train as Not-happening)",
              json.dumps(j["totals"]))
        check(e1["remaining"] == 1 and j["totals"]["remaining_on_used"] == 1, "1 unreviewed proposal left on mouse 1",
              json.dumps(j["totals"]))
        n_by = {A: T}
        rows, _ = hidra.finetune_bouts(store, LabelStore(store), PID, EAT, j["head"], n_by)
        sent = sorted((x["video_id"], x["track"], x["start_frame"], x["stop_frame"]) for x in rows)
        shown = sorted((cl["video_id"], t["track"], x["start"], x["end"])
                       for cl in j["clips"] for t in cl["tracks"] for x in t["bouts"] if x["sent"])
        check(shown == sent, "sent spans == finetune_bouts (what export_bouts writes)", f"{shown} vs {sent}")

        print("hidra (directed, 3 mice)")
        j = c.get(f"{base}/behaviors/{SNIFF}/training-data").json()
        s1 = row(j, A, 0)
        check(s1["used"] and s1["partners"] == ["mouse2"] and s1["pos_bouts"] == 1 and s1["skipped_directed"] == 1,
              "bout with a picked partner sent; one without skipped", json.dumps(s1))
        check(s1["trained_neg_frames"] == F - 20 and s1["never_reviewed_frames"] == F - 40,
              "per-partner negatives; both Happening runs are reviewed frames", json.dumps(s1))
        check(any(x["sent"] is False and x.get("why") for x in s1["bouts"]), "the skipped bout is listed with why")

        print("export_bouts refactor")
        old = old_export_bouts()
        for bid in (EAT, SNIFF):
            head = next(b for b in store.get(PID).behaviors if b["id"] == bid)["hidra"]
            o, n = Path(tmp) / f"old_{bid}.csv", Path(tmp) / f"new_{bid}.csv"
            ro = old(store, LabelStore(store), PID, bid, head, o, n_by)
            rn = hidra.export_bouts(store, LabelStore(store), PID, bid, head, n, n_by)
            rn.pop("path"), ro.pop("path")
            check(o.read_bytes() == n.read_bytes() and ro == rn, f"behavior {bid}: identical CSV and counts",
                  f"{ro} vs {rn}")

        print("read-only")
        snap = lambda: {q.relative_to(d).as_posix(): sha(q) for q in d.rglob("*") if q.is_file()}  # noqa: E731
        s0 = snap()
        for bid in (JUMP, EAT, SNIFF):
            c.get(f"{base}/behaviors/{bid}/training-data")
        s1 = snap()
        changed = sorted(k for k in set(s0) | set(s1) if s0.get(k) != s1.get(k))
        check(not changed, "the endpoint writes nothing: no event, label, manifest or cache file", str(changed))
        check(c.get(f"{base}/behaviors/9/training-data").status_code == 404, "unknown behavior -> 404")

    print("\n" + ("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: " + "; ".join(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
