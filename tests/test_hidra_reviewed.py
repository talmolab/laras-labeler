"""Tests for reviewed-frames-only fine-tuning and held-out clips (hidra.export_reviewed, `only`).
Runs under pytest, or directly: python tests/test_hidra_reviewed.py
"""
import csv
import tempfile
from pathlib import Path

import pandas as pd

from laras_labeler import hidra

HEAD = {"lab": "LyricalHare", "action": "approach", "collapse": "directed"}
# Runs per clip on track 1 (GUI track 2): [start, stop, value, source, target]. 1 = yes, 0 = no,
# 2 = Unknown. Target -1 means none was picked, so the other rat is used.
RUNS = {
    "0-2": [[10, 20, 1, "candidate", -1], [30, 50, 0, "candidate", -1], [60, 65, 2, "manual", -1]],
    "0-11": [[5, 9, 1, "candidate", -1], [40, 44, 0, "candidate", -1]],
}


class _Proj:
    videos = [{"video_id": "0-2", "video_path": "/data/0-2/0.mp4", "n_animals": 2, "n_frames": 100,
               "social_start": 8, "social_end": 70},
              {"video_id": "0-11", "video_path": "/data/0-11/0.mp4", "n_animals": 2, "n_frames": 50}]


class _Store:
    def get(self, pid):
        return _Proj()


class _Labels:
    def rows_for_behavior(self, pid, vid, bid):
        return pd.DataFrame({"track": [1]})

    def get_runs_src(self, pid, vid, track, bid):
        return {bid: RUNS[vid]}


def _export(fn, only=None):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "spans.csv"
        res = fn(_Store(), _Labels(), "p", 0, HEAD, out, only=only)
        with out.open() as f:
            return res, list(csv.DictReader(f))


def test_reviewed_export_has_yes_and_no_but_not_unknown():
    res, rows = _export(hidra.export_reviewed)
    spans = {(r["file"], int(r["start_frame"]), int(r["stop_frame"])) for r in rows}
    assert spans == {("0-2_0.parquet", 10, 20), ("0-2_0.parquet", 30, 50),
                     ("0-11_0.parquet", 5, 9), ("0-11_0.parquet", 40, 44)}
    assert all(r["agent"] == "mouse2" and r["target"] == "mouse1" for r in rows)
    assert res["spans"] == 4


def test_bout_export_is_positives_only():
    _, rows = _export(hidra.export_bouts)
    assert [(r["file"], r["start_frame"]) for r in rows] == [("0-2_0.parquet", "10"), ("0-11_0.parquet", "5")]


def test_held_out_clip_sends_nothing():
    for fn in (hidra.export_bouts, hidra.export_reviewed):
        res, rows = _export(fn, only={"0-2"})
        assert {r["file"] for r in rows} == {"0-2_0.parquet"}
        assert res["video_ids"] == ["0-2"]


def test_window_export_is_the_social_window_or_the_whole_clip():
    res, rows = _export(hidra.export_window)
    spans = {(r["file"], int(r["start_frame"]), int(r["stop_frame"])) for r in rows}
    assert spans == {("0-2_0.parquet", 8, 70), ("0-11_0.parquet", 0, 50)}   # 0-11 has no window
    assert all(r["agent"] == "mouse2" and r["target"] == "mouse1" for r in rows)
    _, rows = _export(hidra.export_window, only={"0-2"})
    assert [r["file"] for r in rows] == ["0-2_0.parquet"]


def test_train_on_reads_old_and_new_settings():
    assert hidra.train_on({}) == "all"
    assert hidra.train_on({"reviewed_only": True}) == "reviewed"
    assert hidra.train_on({"reviewed_only": True, "train_on": "window"}) == "window"
    kept = hidra.rebind({**HEAD, "train_on": "window"}, "GroovyShrew", "approach", "directed", 0.15)
    assert kept["train_on"] == "window"


def test_rebind_keeps_reviewed_switch_and_tags_checkpoint():
    tuned = {**HEAD, "rate": 0.15, "weights": "/runs/x/{config}__t.pkl", "finetune_mode": "head",
             "weights_reviewed_only": True, "weights_manifest": "/runs/x/manifest.json",
             "reviewed_only": True}
    same = hidra.rebind(tuned, "LyricalHare", "approach", "directed", 0.05)
    assert same["reviewed_only"] and same["weights_reviewed_only"]
    assert same["weights_manifest"] == "/runs/x/manifest.json"
    other = hidra.rebind(tuned, "GroovyShrew", "approach", "directed", 0.15)
    assert other["reviewed_only"] and "weights" not in other and "weights_reviewed_only" not in other


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
