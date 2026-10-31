"""Pixels-per-cm for HiDRA, read from a sidecar written by whatever produced the poses.

HiDRA's features are in centimetres and seconds, so it needs the camera's pixel scale, and a scale that
is merely plausible-looking silently ruins every prediction (see `set_video_scale` in app.py, which
deliberately never defaults). The labeler cannot know the scale from a .slp: that file only holds pixel
positions. The step that CAN know it is the pose converter, because it holds the 3D poses in millimetres
AND their pixel projection for one camera (lab_scripts/dannce_to_slp.py). That converter writes

    <poses>.slp.scale.json     {"pix_per_cm": 15.2, "image_size": [1920, 1200], "iqr": [...], ...}

and this module turns it into a value for a given video, under three rules:

1. A MEASURED value only. No sidecar -> no scale -> the caller still refuses to run, exactly as before.
2. Different sized video -> rescale, or refuse. The number is "pixels per cm on THAT pixel grid". A video
   whose grid is the sidecar's grid scaled uniformly (a half-resolution proxy) gets the value scaled by
   the same factor. One with a different aspect ratio (a crop, a different camera mode) gets nothing,
   because no single factor is right, and the user is told why.
3. A value typed by the user always wins over this one (that precedence lives in app.py).
"""

from __future__ import annotations

import json
from pathlib import Path

SIDECAR_SUFFIX = ".scale.json"
# Anything outside this range is a bug in the sidecar, not a camera: 0.5 px/cm would make a rat 10 px
# long, 1000 px/cm a 22 cm rat 22,000 px long.
MIN_PPC, MAX_PPC = 0.5, 1000.0
# Aspect ratios agreeing to within 1% count as "the same picture at a different size".
ASPECT_TOL = 0.01


def sidecar_path(slp_path: str | Path) -> Path:
    return Path(str(slp_path) + SIDECAR_SUFFIX)


def auto_scale(slp_path: str | Path | None, width: int | None, height: int | None) -> dict | None:
    """The scale a video can take from its poses' sidecar, or None when there is no sidecar at all.

    When a sidecar exists the result is always a dict so the GUI can say what happened — including
    when the file is broken, because a sidecar that silently vanished would leave the user wondering
    why no scale appeared:
      usable=True   -> {"usable": True,  "pix_per_cm": float, "note": str, "iqr": [lo, hi] | None}
      usable=False  -> {"usable": False, "why": str}     (found, but must not be applied to this video)
    """
    if not slp_path:
        return None
    p = sidecar_path(slp_path)
    if not p.exists():
        return None

    def bad(why: str) -> dict:
        return {"usable": False, "why": why}

    try:
        with open(p) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return bad(f"{p.name} exists but could not be read as JSON")
    if not isinstance(d, dict):
        return bad(f"{p.name} is malformed (expected a JSON object)")

    try:
        value = float(d["pix_per_cm"])
        sw, sh = (float(v) for v in d["image_size"])
    except (KeyError, TypeError, ValueError):
        return bad(f"{sidecar_path(slp_path).name} is malformed (needs pix_per_cm and image_size [w, h])")
    if not (MIN_PPC <= value <= MAX_PPC) or sw <= 0 or sh <= 0:
        return bad(f"{sidecar_path(slp_path).name} holds an implausible scale ({value:g} px/cm)")
    if not width or not height:
        return bad("this video's pixel size is unknown, so the scale cannot be matched to it")

    rw, rh = width / sw, height / sh
    if abs(rw - rh) > ASPECT_TOL * max(rw, rh):
        return bad(f"this video is {width}x{height} but the scale was measured on {int(sw)}x{int(sh)} "
                   "(a different shape, e.g. cropped) — set px/cm by hand")

    scaled = value * rw
    if not (MIN_PPC <= scaled <= MAX_PPC):
        return bad(f"rescaled value {scaled:g} px/cm is implausible")
    cam = d.get("camera")
    note = f"measured from the camera calibration{f' ({cam})' if cam else ''}"
    if abs(rw - 1.0) > 1e-6:
        note += f", scaled ×{rw:.3g} for a {width}x{height} video"
    iqr = d.get("iqr")
    if isinstance(iqr, (list, tuple)) and len(iqr) == 2:
        try:
            iqr = [float(iqr[0]) * rw, float(iqr[1]) * rw]
        except (TypeError, ValueError):
            iqr = None
    else:
        iqr = None
    return {"usable": True, "pix_per_cm": scaled, "note": note, "iqr": iqr}
