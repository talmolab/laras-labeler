"""Tests for scripts/slp_to_tailcycle.py (SLEAP .slp -> tailcyclenet dataset).

Needs tailcyclenet, which the labeler does not install, so this skips unless run in the
environment tailcyclenet lives in:

    pip install sleap-io pillow pytest
    pytest tests/test_slp_to_tailcycle.py

The video is synthetic: frame i is a uniform gray of brightness 5*i, so whether a written JPEG is
the source frame its group says it is can be checked from its mean alone.
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tailcyclenet.format")
sio = pytest.importorskip("sleap_io")
av = pytest.importorskip("av")
pq = pytest.importorskip("pyarrow.parquet")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "slp_to_tailcycle.py"
spec = importlib.util.spec_from_file_location("slp_to_tailcycle", SCRIPT)
conv = importlib.util.module_from_spec(spec)
sys.modules["slp_to_tailcycle"] = conv
spec.loader.exec_module(conv)

N_FRAMES, SIZE, STEP = 40, 64, 5
NAMES = ["nose", "earL", "earR", "tail"]


def _video(path: Path) -> None:
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=30, options={"crf": "0"})
        s.width = s.height = SIZE
        s.pix_fmt = "yuv420p"
        for i in range(N_FRAMES):
            img = np.full((SIZE, SIZE, 3), STEP * i, np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def _skeleton():
    nodes = [sio.Node(n) for n in NAMES]
    return sio.Skeleton(nodes=nodes,
                        edges=[sio.Edge(nodes[0], nodes[1]), sio.Edge(nodes[0], nodes[2])],
                        symmetries=[sio.Symmetry({nodes[1], nodes[2]})])


def _pts(x0, y0):
    return np.array([[x0, y0], [x0 - 4, y0 - 4], [x0 + 4, y0 - 4], [x0, y0 + 10]], float)


def _write_labels(tmp: Path) -> tuple[Path, Path]:
    video_path = tmp / "cage.mp4"
    _video(video_path)
    sk = _skeleton()
    m1, m2 = sio.Track("m1"), sio.Track("m2")
    # The .slp names the video by a path from another machine; the test rewrites the prefix.
    video = sio.Video(filename="/lab/share/cage.mp4", open_backend=False)

    user = []
    # Frame 10: two tracked mice. m1's earR toggled "not visible" (placed, so a judgement);
    # m2's tail NaN (never placed, so no determination).
    a = sio.Instance.from_numpy(_pts(20, 20), skeleton=sk, track=m1)
    a.points["visible"][2] = False
    b_pts = _pts(45, 40)
    b_pts[3] = np.nan
    b = sio.Instance.from_numpy(b_pts, skeleton=sk, track=m2)
    user.append(sio.LabeledFrame(video=video, frame_idx=10, instances=[a, b]))
    # Frame 12: m1 labelled; a prediction elsewhere nobody labelled (an unlabelled animal), and a
    # prediction under m1 that it was made from (the same animal, not a leftover).
    under = sio.PredictedInstance.from_numpy(_pts(21, 21), skeleton=sk, score=0.9)
    c = sio.Instance.from_numpy(_pts(21, 21), skeleton=sk, track=m1, from_predicted=under)
    other = sio.PredictedInstance.from_numpy(_pts(48, 44), skeleton=sk, score=0.8)
    user.append(sio.LabeledFrame(video=video, frame_idx=12, instances=[c, under, other]))
    # Frame 30: a placeholder (no visible points) only -> dropped.
    ph = sio.Instance.from_numpy(_pts(30, 30), skeleton=sk)
    ph.points["visible"][:] = False
    user.append(sio.LabeledFrame(video=video, frame_idx=30, instances=[ph]))
    # Frame 35: an untracked mouse near the end of the video.
    d = sio.Instance.from_numpy(_pts(30, 30), skeleton=sk)
    user.append(sio.LabeledFrame(video=video, frame_idx=35, instances=[d]))
    labels_path = tmp / "labels.slp"
    # A video the project lists but nobody labelled, and that isn't on this machine: ignored.
    ghost = sio.Video(filename="/nowhere/unlabelled.mp4", open_backend=False)
    sio.save_slp(sio.Labels(labeled_frames=user, videos=[video, ghost]), str(labels_path))

    tracked = []
    for fi in range(N_FRAMES):
        insts = [sio.PredictedInstance.from_numpy(_pts(20 + fi % 3, 20), skeleton=sk, score=0.9,
                                                  track=m1)]
        if fi % 2 == 0:
            insts.append(sio.PredictedInstance.from_numpy(_pts(45, 40), skeleton=sk, score=0.7,
                                                          track=m2))
        tracked.append(sio.LabeledFrame(video=video, frame_idx=fi, instances=insts))
    preds_path = tmp / "preds.slp"
    sio.save_slp(sio.Labels(labeled_frames=tracked), str(preds_path))
    return labels_path, preds_path


def _session(root: Path) -> Path:
    (s,) = list((root / "train").iterdir())
    return s


def test_human_labels():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        labels, _ = _write_labels(tmp)
        out = tmp / "ds"
        rc = conv.main([str(labels), "--out", str(out), "--context", "8",
                        "--replace", "/lab/share/", f"{tmp}/"])
        assert rc == 0  # includes tailcyclenet's own validator
        s = _session(out)
        cfg = (s / "session.toml").read_text()
        assert 'labels = "annotated"' in cfg and '"earL", "earR"' in cfg  # flip pair from symmetry

        groups = pq.read_table(s / "groups.pq").to_pandas().set_index("group_id")
        # [10, 12] share one clip, centred: 7..14. 35 is clamped to stay inside the video: 31..38.
        assert dict(groups.source_frame_start) == {"g0000007": 7, "g0000031": 31}
        assert (groups.n_frames == 8).all()

        # Each JPEG is the source frame it claims to be.
        from PIL import Image
        for gid, start in groups.source_frame_start.items():
            for t_ in range(8):
                img = np.asarray(Image.open(s / "groups" / gid / "cam0" / f"{t_:06d}.jpg"), float)
                assert abs(img.mean() - STEP * (start + t_)) < 1.5, (gid, t_)

        kp = pq.read_table(s / "keypoints.pq").to_pandas()
        kp["key"] = list(zip(kp.group_id, kp.frame, kp.animal_id, kp.bodypart))
        status = dict(zip(kp.key, kp.status))
        assert status[("g0000007", 3, "m1", "earR")] == "missing"   # toggled not visible
        assert ("g0000007", 3, "m2", "tail") not in status          # NaN: no row
        assert status[("g0000007", 5, "m1", "nose")] == "visible"   # frame 12, same track id
        assert status[("g0000031", 4, "f0000035_0", "nose")] == "visible"  # untracked
        assert not kp.animal_id.astype(str).str.startswith("f0000030").any()  # placeholder gone
        vis = kp[kp.status == "visible"]
        row = vis[(vis.group_id == "g0000007") & (vis.frame == 3) & (vis.animal_id == "m1")
                  & (vis.bodypart == "nose")].iloc[0]
        assert (row.x, row.y) == (20.0, 20.0)

        inst = pq.read_table(s / "instances.pq").to_pandas()
        f12 = inst[(inst.group_id == "g0000007") & (inst.frame == 5)]
        assert sorted(f12.status.astype(str)) == ["labeled", "present"]  # one leftover, not two
        leftover = f12[f12.status == "present"].iloc[0]
        assert leftover.x0 < 48 < leftover.x1 and leftover.y0 < 44 < leftover.y1
        assert (inst[inst.status == "labeled"].x1 > inst[inst.status == "labeled"].x0).all()


def test_tracked():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        _, preds = _write_labels(tmp)
        out = tmp / "ds"
        rc = conv.main([str(preds), "--out", str(out), "--tracked", "--video-dir", str(tmp)])
        assert rc == 0
        s = _session(out)
        assert 'labels = "tracked"' in (s / "session.toml").read_text()
        (gid,) = pq.read_table(s / "groups.pq").column("group_id").to_pylist()
        link = s / "groups" / gid / "cam0.mp4"
        assert link.is_symlink() and link.resolve() == (tmp / "cage.mp4").resolve()
        kp = pq.read_table(s / "keypoints.pq").to_pandas()
        assert set(kp.status.astype(str)) == {"visible"}
        per_track = kp.groupby(kp.animal_id.astype(str)).frame.nunique().to_dict()
        assert per_track == {"m1": N_FRAMES, "m2": N_FRAMES // 2}


def test_refuses_predictions_without_tracked_flag():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        _, preds = _write_labels(tmp)
        with pytest.raises(SystemExit, match="pass --tracked"):
            conv.main([str(preds), "--out", str(tmp / "ds"), "--video-dir", str(tmp)])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
