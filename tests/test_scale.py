"""Tests for laras_labeler.scale (the px/cm sidecar rules). Runs under pytest, or directly:
    python tests/test_scale.py
"""
import json
import tempfile
from pathlib import Path

from laras_labeler import scale


def _slp(tmp: Path, sidecar: dict | str | None) -> Path:
    slp = tmp / "poses.slp"
    slp.write_bytes(b"")
    if sidecar is not None:
        Path(str(slp) + scale.SIDECAR_SUFFIX).write_text(sidecar if isinstance(sidecar, str) else json.dumps(sidecar))
    return slp


GOOD = {"pix_per_cm": 15.2, "image_size": [1920, 1200], "iqr": [14.0, 16.4], "camera": "Camera1"}


def test_no_slp_or_no_sidecar_means_no_scale():
    with tempfile.TemporaryDirectory() as t:
        assert scale.auto_scale(None, 1920, 1200) is None
        assert scale.auto_scale(_slp(Path(t), None), 1920, 1200) is None


def test_same_size_uses_value_as_measured():
    with tempfile.TemporaryDirectory() as t:
        r = scale.auto_scale(_slp(Path(t), GOOD), 1920, 1200)
        assert r["usable"] and abs(r["pix_per_cm"] - 15.2) < 1e-9
        assert "Camera1" in r["note"] and "scaled" not in r["note"]
        assert r["iqr"] == [14.0, 16.4]


def test_half_resolution_video_gets_half_the_scale():
    with tempfile.TemporaryDirectory() as t:
        r = scale.auto_scale(_slp(Path(t), GOOD), 960, 600)
        assert r["usable"] and abs(r["pix_per_cm"] - 7.6) < 1e-9
        assert "scaled" in r["note"]
        assert abs(r["iqr"][0] - 7.0) < 1e-9


def test_different_aspect_ratio_is_refused_not_guessed():
    with tempfile.TemporaryDirectory() as t:
        r = scale.auto_scale(_slp(Path(t), GOOD), 1920, 1080)      # 16:9 vs 16:10: a crop or another camera mode
        assert r["usable"] is False and "by hand" in r["why"]
        assert "pix_per_cm" not in r


def test_unknown_video_size_is_refused():
    with tempfile.TemporaryDirectory() as t:
        assert scale.auto_scale(_slp(Path(t), GOOD), None, None)["usable"] is False


def test_broken_sidecars_never_produce_a_number():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for bad in ('{"pix_per_cm": 15.2}',                                          # no image_size
                    '{"pix_per_cm": "lots", "image_size": [1920, 1200]}',            # not a number
                    '{"pix_per_cm": -3, "image_size": [1920, 1200]}',                # negative
                    '{"pix_per_cm": 99999, "image_size": [1920, 1200]}',             # absurd
                    '{"pix_per_cm": 15.2, "image_size": [0, 1200]}',                 # zero-size grid
                    '[1, 2, 3]',                                                     # wrong JSON type
                    'not json at all {'):                                            # unreadable
            r = scale.auto_scale(_slp(tmp, bad), 1920, 1200)
            assert r is not None and r["usable"] is False and "pix_per_cm" not in r, bad   # reported, never hidden


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"\n{len(fns)} passed")
