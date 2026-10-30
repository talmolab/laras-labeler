#!/usr/bin/env python3
"""Verify the feeder outlines and the re-ID clip -> recording link (hcm_roi + the ROI endpoints).

Against a stand-in for the HCM database (no VPN needed) and the real app:

  feeder     feeder-cage-top / -side come from the clip's own recording, a MultiPolygon's largest
             part; NEVER from the camera medoid -- not for a recording with no feeder
             ('valid-no-feeder-cage'), not for one whose segmentation isn't valid ('change-detected')
  wall/spout unchanged: recording first, camera medoid as the fallback
  isolation  one class's failing query costs only that class (rolled back), not the others
  re-ID      a clip named b21_20260613_h17.f37500-52500 (no cam_XX, no timestamp) is tied to its
             recording through the planner's clips.summary.json: next to the clip, or -- for a GUI
             upload, which copies the files into media/ -- next to the video its .slp names; the
             whole-hour .reid.slp too; a clip no summary lists is left alone
  endpoints  upload and add-by-path record source_video / source_frames; autoload-roi fills all four
             outlines and the recording status, and always returns the clip's saved outlines; the
             clip's meta carries them; a clip already looked up is not fetched again on project open,
             one that wasn't is (and the answer is saved)
  features   the feeder outlines are not a feature input: adding them leaves the clip's feature hash
             unchanged (no rebuild)

    PYTHONPATH=src python scripts/verify_feeder_roi.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import time
import types
import warnings
from pathlib import Path

import numpy as np

from laras_labeler import hcm_roi
from laras_labeler.config import Settings
from laras_labeler.project import ProjectStore

FAILED: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILED.append(what)


# ---------------------------------------------------------------- a stand-in HCM database
SQ = lambda x0, y0, s: [[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s], [x0, y0]]  # noqa: E731
POLY = lambda ring: {"type": "Polygon", "coordinates": [ring]}  # noqa: E731
SESSION = "carlos/HCM_Sync/talmolab/2025-02-12-AzimALS/cam_06"
REC_FEEDER = f"{SESSION}/2026-06-13-00-01-05/cam_06.17.mp4"     # feeder top + side
REC_NOFEED = f"{SESSION}/2026-06-21-00-01-05/cam_06.15.mp4"     # no feeder in this recording
REC_CHANGED = f"{SESSION}/2026-06-28-00-01-05/cam_06.15.mp4"    # segmentation not usable
CAGE, WATER, SIDE = SQ(100, 100, 800), SQ(400, 120, 40), SQ(300, 820, 200)
TOP_SMALL, TOP_BIG = SQ(10, 10, 5), SQ(300, 600, 200)
MEDOID_CAGE, MEDOID_WATER, MEDOID_FEEDER = SQ(90, 90, 820), SQ(410, 130, 40), SQ(0, 0, 999)


class FakeDB:
    def __init__(self):
        all_classes = {"cage-outer": POLY(CAGE), "water": POLY(WATER), "feeder-cage-side": POLY(SIDE),
                       "feeder-cage-top": {"type": "MultiPolygon", "coordinates": [[TOP_SMALL], [TOP_BIG]]}}
        self.recordings = {
            REC_FEEDER: ("valid-feeder-cage-with-side", all_classes),
            REC_NOFEED: ("valid-no-feeder-cage", {k: all_classes[k] for k in ("cage-outer", "water")}),
            REC_CHANGED: ("change-detected", all_classes)}
        self.medoid = {("cam_06", "cage-outer"): MEDOID_CAGE, ("cam_06", "water"): MEDOID_WATER,
                       ("cam_06", "feeder-cage-top"): MEDOID_FEEDER, ("cam_06", "feeder-cage-side"): MEDOID_FEEDER}
        self.calls: list[tuple] = []
        self.fail: set[tuple] = set()       # (kind, class) whose query raises
        self.rollbacks = 0

    def find(self, pat):
        rx = re.compile(".*".join(re.escape(p) for p in pat.split("%")) + "$")
        return next((v for k, v in self.recordings.items() if rx.match(k)), None)


DB = FakeDB()


class _Result:
    def __init__(self, value):
        self.value = value

    def first(self):
        return None if self.value is None else (self.value,)

    def scalar(self):
        return self.value


class _Conn:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def rollback(self):
        DB.rollbacks += 1

    def execute(self, sql, params):
        if sql == hcm_roi._STATUS_SQL:
            rec = DB.find(params["pat"])
            return _Result(rec[0] if rec else None)
        kind = "recording" if sql == hcm_roi._RECORDING_SQL else "medoid" if sql == hcm_roi._MEDOID_SQL else None
        assert kind, "unexpected SQL"
        DB.calls.append((kind, params["cls"]))
        if (kind, params["cls"]) in DB.fail:
            raise RuntimeError("bad row")
        if kind == "recording":
            rec = DB.find(params["pat"])
            ok = rec and rec[0].startswith("valid")
            return _Result(rec[1].get(params["cls"]) if ok else None)
        return _Result(DB.medoid.get((params["cam"], params["cls"])))


class _Engine:
    def connect(self):
        return _Conn()


fake_sa = types.ModuleType("sqlalchemy")
fake_sa.create_engine = lambda url, connect_args=None: _Engine()
fake_sa.text = lambda s: s
sys.modules["sqlalchemy"] = fake_sa
hcm_roi.DATABASE_URL = "postgresql://stand-in"

MAC = "/Volumes/talmo/" + REC_FEEDER


def ring(r):
    return hcm_roi._clean_ring(r)


# ---------------------------------------------------------------- fixtures: a re-ID planner folder
def write_clip(folder: Path, stem: str, n: int = 12) -> tuple[Path, Path]:
    """A tiny real mp4 and a 3-mouse .slp whose video is that mp4 (as the planner writes them)."""
    import imageio.v2 as iio
    import sleap_io as sio
    mp4, slp = folder / f"{stem}.mp4", folder / f"{stem}.reid.slp"
    with iio.get_writer(mp4, fps=50, codec="libx264", macro_block_size=16) as w:
        for _ in range(n):
            w.append_data(np.zeros((64, 80, 3), np.uint8))
    skel = sio.Skeleton(nodes=["nose", "tail_base"])
    tracks = [sio.Track(name=f"mouse_{i}") for i in range(3)]
    video = sio.Video.from_filename(str(mp4))
    frames = [sio.LabeledFrame(video=video, frame_idx=f, instances=[
        sio.PredictedInstance.from_numpy(points_data=np.array([[10. + 20 * t, 10.], [10. + 20 * t, 30.]]),
                                         skeleton=skel, track=tracks[t], score=1.0) for t in range(3)])
        for f in range(n)]
    sio.save_file(sio.Labels(labeled_frames=frames, videos=[video], skeletons=[skel], tracks=tracks), str(slp))
    return mp4, slp


def planner(root: Path) -> Path:
    d = root / "reid_b21_lifespan"
    d.mkdir()
    for stem in ("b21_20260613_h17.f37500-52500", "b21_20260621_h15.f88500-103500"):
        write_clip(d, stem)
    (d / "b21_20260613_h17.reid.slp").write_bytes((d / "b21_20260613_h17.f37500-52500.reid.slp").read_bytes())
    summary = [
        {"stem": "b21_20260613_h17", "start": 37500, "end": 52500, "video": MAC,
         "slp": str(d / "b21_20260613_h17.reid.slp"),
         "clip_video": str(d / "b21_20260613_h17.f37500-52500.mp4"),
         "clip_slp": str(d / "b21_20260613_h17.f37500-52500.reid.slp")},
        # written on Windows: matched by file name all the same
        {"stem": "b21_20260621_h15", "start": 88500, "end": 103500,
         "video": "V:\\" + REC_NOFEED.replace("/", "\\"),
         "slp": "C:\\Users\\TalmoLab\\reid\\b21_20260621_h15.reid.slp",
         "clip_video": "C:\\Users\\TalmoLab\\reid\\b21_20260621_h15.f88500-103500.mp4",
         "clip_slp": "C:\\Users\\TalmoLab\\reid\\b21_20260621_h15.f88500-103500.reid.slp"}]
    (d / hcm_roi.SUMMARY).write_text(json.dumps(summary, indent=2))
    return d


# ---------------------------------------------------------------- checks
def unit(root: Path) -> None:
    print("hcm_roi")
    check(hcm_roi.recording_like_path("/Volumes/talmo/" + REC_FEEDER) == "%/2026-06-13-00-01-05/cam_06.17.mp4"
          and hcm_roi.recording_like_path("V:\\" + REC_FEEDER.replace("/", "\\")) == "%/2026-06-13-00-01-05/cam_06.17.mp4"
          and hcm_roi.recording_like_path("/x/b21_20260613_h17.f37500-52500.mp4") is None,
          "a recording's own path (either separator) names it; a clip path doesn't")
    check(hcm_roi._outer_ring({"type": "MultiPolygon", "coordinates": [[TOP_SMALL], [TOP_BIG]]}) == ring(TOP_BIG)
          and hcm_roi._outer_ring(json.dumps(POLY(SIDE))) == ring(SIDE) and hcm_roi._outer_ring(CAGE) == ring(CAGE)
          and hcm_roi._outer_ring({"type": "Point", "coordinates": [1, 2]}) == []
          and hcm_roi._outer_ring([[1, 2], "x"]) == [],
          "outer ring: Polygon, largest MultiPolygon part, bare ring; [] for anything else")

    DB.calls.clear()
    g = hcm_roi.fetch_rois("b21_20260613_h17-f37500-52500", source_video=MAC)
    check(g.get("camera") == "cam_06" and g.get("recording_status") == "valid-feeder-cage-with-side"
          and g.get("cage_roi") == ring(CAGE) and g.get("spout_roi") == ring(WATER)
          and g.get("feeder_top_roi") == ring(TOP_BIG) and g.get("feeder_side_roi") == ring(SIDE)
          and set(g["source"].values()) == {"recording"},
          "re-ID clip with its source recording: all four outlines from that recording", str(g)[:300])
    check(not any(k == "medoid" for k, _ in DB.calls), "  ...and the camera medoid is never asked")

    DB.calls.clear()
    g = hcm_roi.fetch_rois("x", source_video="/Volumes/talmo/" + REC_NOFEED)
    check(g.get("recording_status") == "valid-no-feeder-cage" and g.get("cage_roi") == ring(CAGE)
          and "feeder_top_roi" not in g and "feeder_side_roi" not in g,
          "recording with no feeder: no feeder outline (status says why)", str(g)[:300])
    check(not any(k == "medoid" and c.startswith("feeder") for k, c in DB.calls),
          "  ...the feeder is not filled from the camera medoid")

    DB.calls.clear()
    g = hcm_roi.fetch_rois("x", source_video="/Volumes/talmo/" + REC_CHANGED)
    check(g.get("recording_status") == "change-detected" and g.get("cage_roi") == ring(MEDOID_CAGE)
          and g["source"] == {"cage_roi": "camera medoid", "spout_roi": "camera medoid"}
          and "feeder_top_roi" not in g and "feeder_side_roi" not in g,
          "segmentation not valid: wall/spout from the camera medoid, no feeder", str(g)[:300])

    g = hcm_roi.fetch_rois("clip_001_2025_02_12_2026_06_13_00_01_05_cam_06.17.predictions.slp_batch_21_f0-15000")
    check(g.get("feeder_top_roi") == ring(TOP_BIG) and g.get("source", {}).get("cage_roi") == "recording",
          "old-style clip id still names its recording (feeder included)")
    g = hcm_roi.fetch_rois("cam_06.17")
    check(g.get("recording") is None and g.get("cage_roi") == ring(MEDOID_CAGE) and "feeder_top_roi" not in g,
          "clip naming only a camera: medoid wall/spout, no feeder, recording None")
    check(hcm_roi.fetch_rois("b21_20260613_h17-f37500-52500") == {},
          "re-ID clip with NO source recording: nothing (it names no camera -- the old behavior)")

    DB.fail = {("recording", "water")}
    r0 = DB.rollbacks
    g = hcm_roi.fetch_rois("x", source_video=MAC)
    DB.fail = set()
    check(g.get("spout_roi") == ring(MEDOID_WATER) and g.get("feeder_top_roi") == ring(TOP_BIG)
          and g.get("cage_roi") == ring(CAGE) and DB.rollbacks == r0 + 1,
          "a failing query costs only its own class (rolled back; the spout falls back to the medoid)")

    url, hcm_roi.DATABASE_URL = hcm_roi.DATABASE_URL, ""
    check(hcm_roi.fetch_rois("x", source_video=MAC) == {}, "no HCM_DATABASE_URL: {} (unchanged)")
    hcm_roi.DATABASE_URL = url

    print("re-ID clip -> recording")
    d = planner(root)
    s = hcm_roi.source_recording({"video_path": str(d / "b21_20260613_h17.f37500-52500.mp4"),
                                  "slp_path": str(d / "b21_20260613_h17.f37500-52500.reid.slp")})
    check(s == {"video": MAC, "frames": [37500, 52500]}, "clip next to its summary", str(s))
    media = root / "elsewhere" / "media"
    media.mkdir(parents=True)
    for f in ("b21_20260613_h17.f37500-52500.mp4", "b21_20260613_h17.f37500-52500.reid.slp"):
        (media / f).write_bytes((d / f).read_bytes())
    s = hcm_roi.source_recording({"video_path": str(media / "b21_20260613_h17.f37500-52500.mp4"),
                                  "slp_path": str(media / "b21_20260613_h17.f37500-52500.reid.slp")})
    check(s == {"video": MAC, "frames": [37500, 52500]},
          "uploaded copy (in media/): found through the folder its .slp names", str(s))
    s = hcm_roi.source_recording({"video_path": "/Volumes/talmo/" + REC_FEEDER,
                                  "slp_path": str(d / "b21_20260613_h17.reid.slp")})
    check(s == {"video": MAC, "frames": None}, "whole-hour .reid.slp: its recording, no frame range", str(s))
    s = hcm_roi.source_recording({"video_path": str(d / "b21_20260621_h15.f88500-103500.mp4"),
                                  "slp_path": str(d / "b21_20260621_h15.f88500-103500.reid.slp")})
    check(s and hcm_roi.recording_like_path(s["video"]) == "%/2026-06-21-00-01-05/cam_06.15.mp4",
          "summary written on Windows: matched by file name, its path still names the recording", str(s))
    check(hcm_roi.source_recording({"video_path": str(root / "other.mp4"), "slp_path": None}) is None,
          "a clip no summary lists: None")


def app_level(root: Path) -> None:
    from fastapi.testclient import TestClient

    from laras_labeler.app import create_app
    from laras_labeler.featurestore import FeatureStore, feature_config_hash
    from laras_labeler.video import VideoManager

    print("endpoints")
    d = root / "reid_b21_lifespan"
    proot = root / "projects"
    store = ProjectStore(proot)
    app = create_app(Settings(projects_root=proot), store)
    c = TestClient(app)
    pid = c.post("/api/projects", json={"name": "feeder"}).json()["pid"]

    stem = "b21_20260613_h17.f37500-52500"
    with open(d / f"{stem}.mp4", "rb") as v, open(d / f"{stem}.reid.slp", "rb") as s:
        r = c.post(f"/api/projects/{pid}/videos/upload",
                   files={"video": (f"{stem}.mp4", v, "video/mp4"), "slp": (f"{stem}.reid.slp", s, "application/octet-stream")})
    e = r.json()
    check(r.status_code == 200 and e.get("source_video") == MAC and e.get("source_frames") == [37500, 52500]
          and "/media/" in e["video_path"], "upload: the copy in media/ is linked to its recording", str(e)[:300])
    vid = e["video_id"]

    stem2 = "b21_20260621_h15.f88500-103500"
    r = c.post(f"/api/projects/{pid}/videos", json={"video_path": str(d / f"{stem2}.mp4"),
                                                    "slp_path": str(d / f"{stem2}.reid.slp")})
    e2 = r.json()
    check(r.status_code == 200 and "2026-06-21-00-01-05" in (e2.get("source_video") or ""),
          "add by path: linked to its recording", str(e2)[:300])

    r = c.post(f"/api/projects/{pid}/videos/{vid}/autoload-roi").json()
    check(set(r["set"]) == set(hcm_roi.ROI_FIELDS) and r["checked"] and r["recording_status"] == "valid-feeder-cage-with-side"
          and r["camera"] == "cam_06", "autoload-roi: all four outlines, status, camera", str(r)[:300])
    r = c.post(f"/api/projects/{pid}/videos/{vid}/autoload-roi").json()
    check(r["set"] == {}, "a second autoload changes nothing (missing ROIs only)")
    check(r["rois"]["feeder_top_roi"] == ring(TOP_BIG) and r["rois"]["cage_roi"] == ring(CAGE),
          "  ...but still returns the clip's saved outlines (a background fetch that got there first is drawn)")

    # project open: e2 not yet looked up -> fetched in the background and saved; vid already settled
    calls: list[str] = []
    real = hcm_roi.fetch_rois
    hcm_roi.fetch_rois = lambda v, **kw: (calls.append(v), real(v, **kw))[1]

    def open_project() -> dict:
        body = c.get(f"/api/projects/{pid}").json()
        t0 = time.time()
        while time.time() - t0 < 20:
            if all(j.status in ("done", "error") for j in app.state.jobs._jobs.values() if j.kind == "roi-autoload"):
                break
            time.sleep(0.05)
        return body

    try:
        body = open_project()
        first = list(calls)
        open_project()
    finally:
        hcm_roi.fetch_rois = real
    check(first == [e2["video_id"]], "project open: only the clip not yet looked up is fetched", str(first))
    check(calls == first, "  ...and on the next open, nothing is (both settled)", str(calls))
    saved = json.loads((proot / pid / "project.json").read_text())
    ent2 = next(v for v in saved["videos"] if v["video_id"] == e2["video_id"])
    check(ent2.get("roi_recording_status") == "valid-no-feeder-cage" and ent2.get("cage_roi") == ring(CAGE)
          and "feeder_top_roi" not in ent2,
          "  ...its answer is saved: wall/spout set, no feeder, status recorded",
          str({k: ent2.get(k) for k in ("roi_recording_status", "feeder_top_roi")}))
    meta = next(v for v in body["videos"] if v["video_id"] == vid)
    check(meta["feeder_top_roi"] == ring(TOP_BIG) and meta["feeder_side_roi"] == ring(SIDE)
          and meta["cage_roi"] == ring(CAGE) and meta["roi_checked"] is True,
          "the clip's meta carries the feeder outlines and roi_checked")
    ent = next(v for v in saved["videos"] if v["video_id"] == vid)
    check(ent.get("roi_recording_status") == "valid-feeder-cage-with-side" and ent.get("feeder_side_roi"),
          "  ...and they are saved in project.json")

    fs = FeatureStore(store, VideoManager(store))
    proj = store.get(pid)
    entry = proj.video(vid)
    roles = proj.manifest.get("skeleton_roles", {})
    h1 = feature_config_hash(fs._clip_cfg(proj, entry), roles)
    h2 = feature_config_hash(fs._clip_cfg(proj, {k: v for k, v in entry.items() if k not in hcm_roi.FEEDER_FIELDS}), roles)
    check(h1 == h2, "feeder outlines don't change the clip's feature hash (no feature rebuild)")


def main() -> int:
    warnings.filterwarnings("ignore")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        unit(root)
        app_level(root)
    print("\nALL PASS" if not FAILED else f"\n{len(FAILED)} FAILED: " + "; ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
