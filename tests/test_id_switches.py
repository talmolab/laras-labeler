"""Tracking identity switches (idswitches.py): one list for the whole projects folder, logged on a clip,
edited, deleted and exported as CSV. Tiny real .slp files (3 tracks), no video.

    python -m pytest tests/test_id_switches.py
"""
import csv
import io
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio
from fastapi.testclient import TestClient

from laras_labeler import hcm_roi, idswitches
from laras_labeler.app import create_app
from laras_labeler.config import Settings
from laras_labeler.project import ProjectStore

REID = "b21_20260906_h16.f128500-143500"          # re-ID clip: frames 128500-143500 of its hour
OLD = "clip_001_2025_02_12_2025_09_02_00_01_05_cam_06.11.predictions.slp_batch_10_f0-15000"
PLAIN = "mouse_pair_3"                            # another lab's clip: nothing known about it
REC = "/mnt/vast/hcm/cam_07/2026-09-06-00-01-05/cam_07.16.mp4"


def _slp(path: Path, n_tracks=3, n_frames=10):
    skel = sio.Skeleton(nodes=["nose", "tail"], edges=[("nose", "tail")])
    video = sio.Video(filename=str(path.with_suffix(".mp4")))
    tracks = [sio.Track(name=f"track_{i}") for i in range(n_tracks)]
    frames = []
    for f in range(n_frames):
        insts = [sio.PredictedInstance.from_numpy(
            points_data=np.array([[10.0 + 5 * t, 10.0 + f, 1.0], [20.0 + 5 * t, 20.0 + f, 1.0]], "float32"),
            skeleton=skel, track=tr, score=1.0) for t, tr in enumerate(tracks)]
        frames.append(sio.LabeledFrame(video=video, frame_idx=f, instances=insts))
    sio.save_file(sio.Labels(labeled_frames=frames, videos=[video], skeletons=[skel], tracks=tracks), str(path))
    return path


def _clip(d: Path, vid: str, slp=True, n_frames=15000, fps=50.0):
    return {"video_id": vid, "video_path": str(d / f"{vid}.mp4"),
            "slp_path": str(_slp(d / f"{vid}.slp")) if slp else None, "n_frames": n_frames, "fps": fps,
            "width": 64, "height": 64, "has_poses": slp}


def _setup(projects=("a", "b")):
    """Projects holding the same clips (the .slp files live outside the projects root), and a client."""
    root, media = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    (media / hcm_roi.SUMMARY).write_text(json.dumps([{   # the clip planner's summary: the re-ID clip's recording
        "stem": "b21_20260906_h16", "day": "2026-09-06", "hour": 16, "age_days": 34, "start": 128500, "end": 143500,
        "video": REC, "clip_video": str(media / f"{REID}.mp4"), "clip_slp": str(media / f"{REID}.slp")}]))
    clips = [_clip(media, REID), _clip(media, OLD), _clip(media, PLAIN, fps=30.0),
             _clip(media, "no_tracking", slp=False)]
    store = ProjectStore(root)
    for pid in projects:
        p = store.create(pid, pid=pid)
        p.manifest["videos"] = [dict(c) for c in clips]
        p.save()
    return TestClient(create_app(Settings(projects_root=root), store)), root, store


def _add(c, vid=REID, pid="a", **body):
    return c.post(f"/api/projects/{pid}/videos/{vid}/id-switches", json=body)


def _rows(text):
    assert text.startswith("\ufeff")                       # a spreadsheet reads the ↔ as UTF-8
    return list(csv.DictReader(io.StringIO(text[1:])))


def test_add_list_edit_delete():
    c, root, store = _setup()
    before = (root / "a" / "project.json").read_bytes()
    r = _add(c, frame=3325, tracks=[2, 0], note="  lost them in the huddle ")
    assert r.status_code == 200, r.text
    s = r.json()
    assert (s["video_id"], s["frame"], s["tracks"], s["track_names"]) == (REID, 3325, [0, 2], ["track_0", "track_2"])
    assert (s["note"], s["status"], s["project"]) == ("lost them in the huddle", "open", "a")
    assert datetime.fromisoformat(s["logged_at"]).tzinfo is not None
    # the snapshot: the clip's files and timing, and where it was cut from
    assert s["slp_path"].endswith(f"{REID}.slp") and s["video_path"].endswith(f"{REID}.mp4")
    assert (s["fps"], s["n_frames"], s["n_tracks"]) == (50.0, 15000, 3)
    assert s["clip_track_names"] == ["track_0", "track_1", "track_2"]
    assert s["recording"] == "2026-09-06-00-01-05/cam_07.16.mp4" and s["source_frame"] == 128500 + 3325
    assert (s["cohort"], s["camera"], s["date"], s["hour"]) == (21, "cam_07", "2026-09-06", 16)
    # not project data: the project (and its labels) are untouched
    assert (root / "a" / "project.json").read_bytes() == before
    assert not list((root / "a" / "labels").iterdir())

    _add(c, frame=10, tracks=[])                           # not sure which tracks
    got = c.get("/api/id-switches").json()["switches"]
    assert [(x["frame"], x["tracks"]) for x in got] == [(10, []), (3325, [0, 2])]   # by clip, then frame
    assert c.get("/api/id-switches", params={"video_id": OLD}).json()["switches"] == []

    sid = s["id"]
    r = c.put(f"/api/id-switches/{sid}", json={"note": "huddle by the feeder"})
    assert r.status_code == 200 and r.json()["note"] == "huddle by the feeder" and r.json()["tracks"] == [0, 2]
    r = c.put(f"/api/id-switches/{sid}", json={"tracks": [1], "status": "fixed"})
    assert (r.json()["tracks"], r.json()["track_names"], r.json()["status"]) == ([1], ["track_1"], "fixed")
    assert c.put(f"/api/id-switches/{sid}", json={"status": "open", "tracks": []}).json()["track_names"] == []
    assert c.get("/api/id-switches").json()["switches"][1]["note"] == "huddle by the feeder"

    assert c.delete(f"/api/id-switches/{sid}").json() == {"ok": True, "id": sid}
    assert [x["frame"] for x in c.get("/api/id-switches").json()["switches"]] == [10]


def test_delete_or_edit_unknown_id_is_404():
    c, *_ = _setup()
    _add(c, frame=1)
    assert c.delete("/api/id-switches/nope").status_code == 404
    assert c.put("/api/id-switches/nope", json={"note": "x"}).status_code == 404
    assert len(c.get("/api/id-switches").json()["switches"]) == 1


@pytest.mark.parametrize("body", [
    {"frame": -1}, {"frame": 15000}, {"frame": "12"}, {"frame": 1.5}, {"frame": True}, {},
    {"frame": 5, "tracks": [3]}, {"frame": 5, "tracks": [-1]}, {"frame": 5, "tracks": "0"},
    {"frame": 5, "tracks": [True]}, {"frame": 5, "tracks": [0.0]}, {"frame": 5, "note": 7},
    {"frame": 5, "note": "x" * (idswitches.MAX_NOTE + 1)}])
def test_bad_frame_track_or_note_is_400(body):
    c, root, _ = _setup()
    r = _add(c, **body)
    assert r.status_code == 400, r.text
    assert not (root / idswitches.FILE).exists()           # nothing written


def test_unknown_clip_or_clip_without_tracking():
    c, root, _ = _setup()
    assert _add(c, pid="nope", frame=1).status_code == 404
    assert _add(c, vid="nope", frame=1).status_code == 404
    r = _add(c, vid="no_tracking", frame=1)
    assert r.status_code == 400 and "no tracking" in r.json()["detail"]
    assert not (root / idswitches.FILE).exists()


def test_bad_edits_are_400_and_change_nothing():
    c, *_ = _setup()
    sid = _add(c, frame=5, tracks=[0, 1], note="n").json()["id"]
    for body in ({"status": "done"}, {"tracks": [3]}, {"tracks": "1"}, {"note": 3}, {}, {"colour": "red"}):
        assert c.put(f"/api/id-switches/{sid}", json=body).status_code == 400, body
    s = c.get("/api/id-switches").json()["switches"][0]
    assert (s["tracks"], s["note"], s["status"]) == ([0, 1], "n", "open")


def test_persists_atomically_across_a_restart(monkeypatch):
    c, root, store = _setup()
    a = _add(c, frame=100, tracks=[0, 1]).json()
    b = _add(c, vid=OLD, frame=7, note="tail swap").json()
    data = json.loads((root / idswitches.FILE).read_text())
    assert data["version"] == 1 and {s["id"] for s in data["switches"]} == {a["id"], b["id"]}
    assert not [p.name for p in root.iterdir() if p.name.endswith(".tmp")]
    # a new server on the same projects folder sees the same list
    c2 = TestClient(create_app(Settings(projects_root=root), ProjectStore(root)))
    assert c2.get("/api/id-switches").json()["switches"] == c.get("/api/id-switches").json()["switches"]
    # a write that fails half way leaves the previous list whole
    saved = (root / idswitches.FILE).read_text()

    def broken_replace(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr("laras_labeler.project.os.replace", broken_replace)
    with pytest.raises(OSError):
        idswitches.IdSwitchStore(root).add({"id": "x", "video_id": PLAIN, "frame": 1})
    assert (root / idswitches.FILE).read_text() == saved


def test_unreadable_list_is_reported_and_never_overwritten():
    c, root, _ = _setup()
    (root / idswitches.FILE).write_text('{"switches": [ oops')
    r = c.get("/api/id-switches")
    assert r.status_code == 500 and "isn't valid JSON" in r.json()["detail"]
    assert _add(c, frame=1).status_code == 500
    assert (root / idswitches.FILE).read_text() == '{"switches": [ oops'


def test_one_list_for_every_project_holding_the_clip():
    c, root, store = _setup()
    _add(c, pid="a", frame=50, tracks=[0, 1])
    _add(c, pid="b", frame=900, tracks=[1, 2])
    got = c.get("/api/id-switches", params={"video_id": REID}).json()["switches"]
    assert [(s["project"], s["frame"]) for s in got] == [("a", 50), ("b", 900)]
    assert not (root / "a" / idswitches.FILE).exists()     # beside the projects, not inside one
    # the clip leaves project a: its switches stay, and still export from their snapshot
    assert c.delete(f"/api/projects/a/videos/{REID}").status_code == 200
    rows = _rows(c.get("/api/id-switches.csv").text)
    assert [(r["clip"], r["frame"], r["tracks"]) for r in rows] == [(REID, "50", "1 ↔ 2"), (REID, "900", "2 ↔ 3")]
    assert rows[0]["source_frame"] == str(128500 + 50)


def test_a_same_named_clip_of_another_length_is_told_apart():
    c, root, store = _setup(projects=("a",))
    p = store.create("other", pid="other")                 # another experiment's mouse_pair_3: 900 frames
    p.manifest["videos"] = [_clip(Path(tempfile.mkdtemp()), PLAIN, n_frames=900)]
    p.save()
    _add(c, vid=PLAIN, pid="a", frame=45, tracks=[1])
    _add(c, vid=PLAIN, pid="other", frame=12, tracks=[0])

    def frames(**q):
        return [s["frame"] for s in c.get("/api/id-switches", params=q).json()["switches"]]
    assert frames(video_id=PLAIN) == [12, 45]
    assert frames(video_id=PLAIN, n_frames=900) == [12] and frames(video_id=PLAIN, n_frames=15000) == [45]
    rows = _rows(c.get("/api/id-switches.csv", params={"video_id": PLAIN, "n_frames": 900}).text)
    assert [r["frame"] for r in rows] == ["12"]


def test_csv_columns_values_and_open_only():
    c, root, _ = _setup()
    a = _add(c, frame=3325, tracks=[0, 2], note='lost them, by the "feeder"').json()
    _add(c, vid=OLD, frame=100)
    _add(c, vid=PLAIN, frame=45, tracks=[1])
    fixed = _add(c, frame=20, tracks=[0, 1]).json()
    c.put(f"/api/id-switches/{fixed['id']}", json={"status": "fixed"})

    r = c.get("/api/id-switches.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="id_switches.csv"' == r.headers["content-disposition"]
    assert r.text[1:].splitlines()[0] == ",".join(idswitches.CSV_COLS)
    rows = _rows(r.text)
    assert [(x["clip"], x["frame"]) for x in rows] == [(REID, "3325"), (OLD, "100"), (PLAIN, "45")]   # open only
    reid, old, plain = rows
    assert reid == {
        "clip": REID, "frame": "3325", "time_s": "66.50", "tracks": "1 ↔ 3", "sleap_tracks": "track_0 ↔ track_2",
        "note": 'lost them, by the "feeder"', "status": "open",
        "logged": datetime.fromisoformat(a["logged_at"]).astimezone().strftime("%Y-%m-%d %H:%M"),
        "video_file": a["video_path"], "tracking_file": a["slp_path"],
        "source_recording": "2026-09-06-00-01-05/cam_07.16.mp4", "source_frame": "131825",
        "cohort": "21", "camera": "cam_07", "recording_date": "2026-09-06", "recording_hour": "16"}
    # an old-style HCM clip: its name says the recording and that the clip starts at its frame 0
    assert (old["tracks"], old["sleap_tracks"], old["time_s"]) == ("", "", "2.00")
    assert (old["source_recording"], old["source_frame"], old["cohort"], old["camera"]) == \
        ("2025-09-02-00-01-05/cam_06.11.mp4", "100", "10", "cam_06")
    # another lab's clip: nothing known, so blank (and its own frame rate)
    assert (plain["time_s"], plain["tracks"], plain["sleap_tracks"]) == ("1.50", "2", "track_1")
    assert all(plain[k] == "" for k in ("source_recording", "source_frame", "cohort", "camera",
                                       "recording_date", "recording_hour"))

    every = _rows(c.get("/api/id-switches.csv", params={"include_fixed": "true"}).text)
    assert [(x["frame"], x["status"]) for x in every if x["clip"] == REID] == [("20", "fixed"), ("3325", "open")]
    one = _rows(c.get("/api/id-switches.csv", params={"video_id": PLAIN}).text)
    assert [x["frame"] for x in one] == ["45"]
