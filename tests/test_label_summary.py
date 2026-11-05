"""Per-behavior counts at the top of 🏷 Label sources: bouts, frames, seconds and clips, whatever
the source, for every behavior (labeled or not).

    python -m pytest tests/test_label_summary.py
"""
import tempfile
from pathlib import Path

from laras_labeler.labels import LabelStore
from laras_labeler.project import ProjectStore


def _project():
    root = Path(tempfile.mkdtemp())
    store = ProjectStore(root)
    p = store.create("t")
    for vid, fps in (("a", 30.0), ("b", 50.0), ("c", 30.0)):
        p.manifest.setdefault("videos", []).append({"video_id": vid, "video_path": f"/x/{vid}.mp4", "n_frames": 1000,
                                                    "fps": fps, "width": 10, "height": 10, "has_poses": True})
    p.add_behavior("drinking", None, "d")
    p.add_behavior("rearing", None, "r")
    p.save()
    (p.path / "labels").mkdir(parents=True, exist_ok=True)
    return store, p.pid


def test_summary_counts_bouts_frames_seconds_and_clips():
    store, pid = _project()
    ls = LabelStore(store)
    # clip a (30 fps): two Happening bouts on track 0 (one by hand, one from candidates; apart ->
    # separate bouts), one on track 1, one Not-happening
    ls.put_spans(pid, "a", [
        {"behavior_id": 0, "track": 0, "start": 0, "end": 30, "value": 1},
        {"behavior_id": 0, "track": 0, "start": 100, "end": 160, "value": 1, "source": "candidate"},
        {"behavior_id": 0, "track": 1, "start": 0, "end": 30, "value": 1},
        {"behavior_id": 0, "track": 0, "start": 200, "end": 260, "value": 0},
    ])
    # clip b (50 fps): one Happening bout of 100 frames = 2 s
    ls.put_spans(pid, "b", [{"behavior_id": 0, "track": 0, "start": 0, "end": 100, "value": 1}])
    # clip c: only Not-happening for drinking -> a labeled clip, but none with Happening
    ls.put_spans(pid, "c", [{"behavior_id": 0, "track": 0, "start": 0, "end": 15, "value": 0}])

    d = ls.source_stats(pid)
    s = {r["name"]: r for r in d["summary"]}
    dr = s["drinking"]
    assert (dr["pos_bouts"], dr["pos_frames"]) == (4, 30 + 60 + 30 + 100)
    assert dr["pos_seconds"] == 1 + 2 + 1 + 2                    # 30+60+30 frames at 30 fps, 100 at 50 fps
    assert (dr["neg_bouts"], dr["neg_frames"], dr["neg_seconds"]) == (2, 75, 2.5)
    assert (dr["clips"], dr["clips_with_pos"]) == (3, 2)
    # a behavior with no labels is still listed, with zeros
    re_ = s["rearing"]
    assert (re_["pos_bouts"], re_["neg_bouts"], re_["clips"], re_["clips_with_pos"]) == (0, 0, 0, 0)
    assert [r["name"] for r in d["summary"]] == ["drinking", "rearing"]
    # the per-clip tables are unchanged (rearing has none)
    assert [b["name"] for b in d["behaviors"]] == ["drinking"]


def test_summary_one_bout_split_by_source_counts_once():
    """A bout whose first half was painted by hand and second half accepted from candidates is one
    bout in the summary (the per-source table still shows it under both)."""
    store, pid = _project()
    ls = LabelStore(store)
    ls.put_spans(pid, "a", [
        {"behavior_id": 0, "track": 0, "start": 10, "end": 20, "value": 1},
        {"behavior_id": 0, "track": 0, "start": 20, "end": 40, "value": 1, "source": "candidate"},
    ])
    dr = ls.source_stats(pid)["summary"][0]
    assert (dr["pos_bouts"], dr["pos_frames"]) == (1, 30)


def test_summary_keeps_test_clip_out_of_the_totals():
    """The held-out test clip never trains a model: its bouts are reported on the side."""
    store, pid = _project()
    proj = store.get(pid)
    next(v for v in proj.manifest["videos"] if v["video_id"] == "b")["role"] = "test"
    proj.save()
    ls = LabelStore(store)
    ls.put_spans(pid, "a", [{"behavior_id": 0, "track": 0, "start": 0, "end": 30, "value": 1}])
    ls.put_spans(pid, "b", [{"behavior_id": 0, "track": 0, "start": 0, "end": 100, "value": 1},
                            {"behavior_id": 0, "track": 0, "start": 200, "end": 300, "value": 0}])
    dr = ls.source_stats(pid)["summary"][0]
    assert (dr["pos_bouts"], dr["pos_frames"], dr["neg_bouts"], dr["clips"], dr["clips_with_pos"]) == (1, 30, 0, 1, 1)
    assert (dr["test_pos_bouts"], dr["test_neg_bouts"]) == (1, 1)
