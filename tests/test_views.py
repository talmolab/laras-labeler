"""Other camera views of a clip: add / replace / remove, served by ?view=, and display-only.

A view is another camera's video of the clip's frames (a multi-camera rig whose 3D poses were
projected into each camera). It must have the clip's frame count and fps, and the clip's own video
and poses stay what every feature, label and model uses.

    python -m pytest tests/test_views.py
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from laras_labeler.app import create_app
from laras_labeler.config import Settings
from laras_labeler.project import ProjectStore

N, FPS = 24, 40.0
NODES = ["nose", "neck", "tail_base"]


def _video(path: Path, n: int, w: int, h: int, level: int) -> None:
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h), isColor=False)
    assert vw.isOpened()
    for f in range(n):
        vw.write(np.full((h, w), (level + 3 * f) % 256, "uint8"))
    vw.release()


def _slp(path: Path, video_path: Path, n: int, x0: float) -> None:
    import sleap_io as sio
    skel = sio.Skeleton(nodes=list(NODES), edges=[("nose", "neck"), ("neck", "tail_base")])   # it converts its list in place
    video = sio.Video.from_filename(str(video_path))
    track = sio.Track(name="mouse1")
    frames = []
    for f in range(n):
        pts = np.array([[x0 + f, 10, 1], [x0 + f, 20, 1], [x0 + f, 30, 1]], "float32")
        inst = sio.PredictedInstance.from_numpy(points_data=pts, skeleton=skel, track=track, score=1.0)
        frames.append(sio.LabeledFrame(video=video, frame_idx=f, instances=[inst]))
    sio.save_file(sio.Labels(labeled_frames=frames, videos=[video], skeletons=[skel], tracks=[track]), str(path))


@pytest.fixture()
def clip():
    root = Path(tempfile.mkdtemp())
    media = root / "media"
    media.mkdir()
    _video(media / "a.mp4", N, 64, 48, 10)
    _slp(media / "a.slp", media / "a.mp4", N, 5.0)
    (media / "a.slp.scale.json").write_text(json.dumps({"pix_per_cm": 20.0, "image_size": [64, 48],
                                                        "camera": "camA"}))
    _video(media / "b.mp4", N, 80, 60, 100)                 # another camera: different size
    _slp(media / "b.slp", media / "b.mp4", N, 40.0)
    _video(media / "short.mp4", N - 4, 64, 48, 50)          # not the same frames
    store = ProjectStore(root / "projects")
    c = TestClient(create_app(Settings(projects_root=root / "projects"), store))
    pid = c.post("/api/projects", json={"name": "views"}).json()["pid"]
    vid = c.post(f"/api/projects/{pid}/videos",
                 json={"video_path": str(media / "a.mp4"), "slp_path": str(media / "a.slp")}).json()["video_id"]
    return c, pid, vid, media


def _poses(c, url):
    r = c.get(url)
    assert r.status_code == 200, r.text
    shape = tuple(int(x) for x in r.headers["X-Pose-Shape"].split(","))
    return np.frombuffer(r.content, "<f4").reshape(shape)


def test_add_view_and_serve_it(clip):
    c, pid, vid, media = clip
    meta = c.get(f"/api/projects/{pid}/videos/{vid}").json()
    assert meta["views"] == [] and meta["main_view"] == "camA"     # named from the scale sidecar

    r = c.post(f"/api/projects/{pid}/videos/{vid}/views",
               json={"name": "camB", "video_path": str(media / "b.mp4"), "slp_path": str(media / "b.slp")})
    assert r.status_code == 200, r.text
    assert r.json()["view"] == {"name": "camB", "width": 80, "height": 60, "has_poses": True}
    assert c.get(f"/api/projects/{pid}/videos/{vid}").json()["views"] == [r.json()["view"]]

    main = _poses(c, f"/api/projects/{pid}/poses/{vid}")
    other = _poses(c, f"/api/projects/{pid}/poses/{vid}?view=camB")
    assert main.shape[0] == other.shape[0] == N
    assert main[3, 0, 0, 0] == pytest.approx(8.0) and other[3, 0, 0, 0] == pytest.approx(43.0)

    assert c.get(f"/api/projects/{pid}/skeleton/{vid}?view=camB").json()["nodes"] == NODES
    assert c.get(f"/api/projects/{pid}/video/{vid}/stream.mp4?view=camB").content == (media / "b.mp4").read_bytes()
    assert c.get(f"/api/projects/{pid}/video/{vid}/stream.mp4").content == (media / "a.mp4").read_bytes()
    assert c.get(f"/api/projects/{pid}/frame/{vid}/5.jpg?view=camB").headers["content-type"] == "image/jpeg"

    # the clip's own poses (what features, training and HiDRA read) are untouched by the view
    again = _poses(c, f"/api/projects/{pid}/poses/{vid}")
    assert np.array_equal(main, again, equal_nan=True)


def test_refuses_views_that_would_not_line_up(clip):
    c, pid, vid, media = clip
    url = f"/api/projects/{pid}/videos/{vid}/views"
    r = c.post(url, json={"name": "camS", "video_path": str(media / "short.mp4")})
    assert r.status_code == 400 and "frames" in r.json()["detail"]
    assert c.post(url, json={"name": "camA", "video_path": str(media / "b.mp4")}).status_code == 400
    assert c.post(url, json={"name": "../etc", "video_path": str(media / "b.mp4")}).status_code == 400
    assert c.post(url, json={"name": "camX", "video_path": str(media / "nope.mp4")}).status_code == 404
    assert c.get(f"/api/projects/{pid}/poses/{vid}?view=nope").status_code == 404
    assert c.get(f"/api/projects/{pid}/video/{vid}/stream.mp4?view=nope").status_code == 404
    assert c.get(f"/api/projects/{pid}/videos/{vid}").json()["views"] == []


def test_replace_and_remove(clip):
    c, pid, vid, media = clip
    url = f"/api/projects/{pid}/videos/{vid}/views"
    c.post(url, json={"name": "camB", "video_path": str(media / "b.mp4"), "slp_path": str(media / "b.slp")})
    c.get(f"/api/projects/{pid}/poses/{vid}?view=camB")             # cache the old handle
    r = c.post(url, json={"name": "camB", "video_path": str(media / "b.mp4")})   # re-add without poses
    assert r.status_code == 200 and r.json()["video"]["views"][0]["has_poses"] is False
    assert len(c.get(f"/api/projects/{pid}/videos/{vid}").json()["views"]) == 1

    assert c.delete(f"{url}/camB").json()["views"] == []
    assert c.delete(f"{url}/camB").status_code == 404
    assert c.get(f"/api/projects/{pid}/poses/{vid}?view=camB").status_code == 404
