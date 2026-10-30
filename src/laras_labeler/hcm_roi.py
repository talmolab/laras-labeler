"""Fetch a clip's arena ROIs from the HCM database on demand (best-effort).

The HCM `recordings.rois` column holds per-recording GeoJSON polygons from UNet segmentation
(class_name 'water' = spout, 'cage-outer' = arena boundary, 'feeder-cage-top' / 'feeder-cage-side'
= the two visible faces of the feeder cage).

We prefer THIS recording's OWN polygon, so a clip is scored against the arena as it actually was —
the spout in particular does get nudged between recordings (~19 px on cam_07). When the clip can't be
matched to a recording, or that recording has no valid segmentation, we fall back per class to the
camera MEDOID polygon (the recording whose polygon center is closest to the camera's median center),
which is robust to a single-recording segmentation glitch — that is what happens for a recording whose
rois status is 'change-detected' rather than 'valid'. Same medoid idea as
scripts/import_hcm_spout_roi.py, but recording-first and callable in-process so the labeler can
auto-fill a newly-loaded clip's ROIs. The feeder never falls back (see RECORDING_ONLY).

A clip is tied to its recording by its id (old-style clip names embed it) or, for clips cut by the
re-ID clip planner (`b21_20260906_h16.f128500-143500.mp4`, whose names don't), by the
`clips.summary.json` the planner writes next to them -- see `source_recording`.

Everything here degrades gracefully: sqlalchemy is imported lazily and any DB error (not installed,
off-VPN, timeout, no ROI) returns {} rather than raising — so video loading never blocks on it.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

DATABASE_URL = os.environ.get("HCM_DATABASE_URL", "")  # set HCM_DATABASE_URL locally; do not commit credentials
CAM_RE = re.compile(r"cam[_-]?0*(\d+)")

# HCM segmentation class_name -> labeler ROI config field
ROI_CLASS = {"cage-outer": "cage_roi", "water": "spout_roi",
             "feeder-cage-top": "feeder_top_roi", "feeder-cage-side": "feeder_side_roi"}
ROI_FIELDS = tuple(ROI_CLASS.values())
FEEDER_FIELDS = ("feeder_top_roi", "feeder_side_roi")

# The feeder comes from the clip's OWN recording only, never the camera medoid. Unlike the wall and the
# spout it is not fixed per camera: it shifts, and a recording can have none -- before the feeder cages
# went in (2025-02-12), or with rois status 'valid-no-feeder-cage' / '-no-side'. The medoid of the
# camera's recent recordings would draw a feeder into those clips. No outline is the right answer there.
RECORDING_ONLY = {"feeder-cage-top", "feeder-cage-side"}

# clip ids embed the recording: ..._2025_06_16_00_01_05_cam_07.12.predictions... -> the directory
# 2025-06-16-00-01-05 and file cam_07.12.mp4 in recordings.video_path. Anchored on the cam token so
# the leading session date can't be mistaken for the recording start.
REC_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(cam_\d+)\.(\d+)")
# the same, read from a recording's own path: .../2026-06-13-00-01-05/cam_06.17.mp4 (either separator)
REC_PATH_RE = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})[/\\](cam_\d+)\.(\d+)\.(mp4|mkv)$")

# The whole geometry, not coordinates->0: the pipeline keeps the largest part of a feeder MultiPolygon
# today, but a row written before that rule would make coordinates->0 a list of rings, not a ring.
_RECORDING_SQL = """
SELECT feat->'geometry' AS geom
FROM recordings r
CROSS JOIN LATERAL jsonb_array_elements(r.rois->'features') AS feat
WHERE r.video_path LIKE :pat AND jsonb_typeof(r.rois) = 'object'
  AND r.rois->>'status' LIKE 'valid%'
  AND feat->'properties'->>'class_name' = :cls
LIMIT 1;
"""

# The recording's rois status ('valid-feeder-cage-with-side', 'valid-no-feeder-cage', 'change-detected',
# ...): says whether a missing feeder outline means "no feeder" or "segmentation not usable".
_STATUS_SQL = """
SELECT r.rois->>'status' FROM recordings r
WHERE r.video_path LIKE :pat AND jsonb_typeof(r.rois) = 'object'
LIMIT 1;
"""

# Medoid over the most recent MEDOID_SAMPLE recordings, not all of them: a camera can have >12k
# valid-ROI recordings (cam_07 did), and unnesting every polygon takes ~45 s. Sampling 300 picks a
# polygon with the same centroid to <0.1 px in ~0.2 s, because a camera's arena is fixed.
MEDOID_SAMPLE = 300

_MEDOID_SQL = """
WITH recs AS (
  SELECT r.id, r.rois
  FROM recordings r JOIN batches b ON r.batch_id = b.batch_id
  WHERE b.cage = :cam AND jsonb_typeof(r.rois) = 'object' AND r.rois->>'status' LIKE 'valid%'
  ORDER BY r.id DESC LIMIT :lim
),
cand AS (
  SELECT recs.id, feat->'geometry'->'coordinates'->0 AS ring
  FROM recs CROSS JOIN LATERAL jsonb_array_elements(recs.rois->'features') AS feat
  WHERE feat->'properties'->>'class_name' = :cls AND feat->'geometry'->>'type' = 'Polygon'
),
ctr AS (
  SELECT c.id, (MIN((pt->>0)::float)+MAX((pt->>0)::float))/2 AS cx,
               (MIN((pt->>1)::float)+MAX((pt->>1)::float))/2 AS cy
  FROM cand c CROSS JOIN LATERAL jsonb_array_elements(c.ring) AS pt GROUP BY c.id
),
med AS (
  SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY cx) AS mx,
         percentile_cont(0.5) WITHIN GROUP (ORDER BY cy) AS my FROM ctr
)
SELECT c.ring FROM ctr JOIN cand c USING (id) CROSS JOIN med
ORDER BY sqrt(power(ctr.cx-med.mx,2)+power(ctr.cy-med.my,2)) ASC LIMIT 1;
"""


def camera_of(video_id: str) -> str | None:
    m = CAM_RE.search(video_id or "")
    return f"cam_{int(m.group(1)):02d}" if m else None


def recording_like(video_id: str) -> str | None:
    """LIKE pattern selecting the ONE recording a clip was cut from, or None if unparseable."""
    m = REC_RE.search(video_id or "")
    if not m:
        return None
    y, mo, d, hh, mm, ss, cam, hour = m.groups()
    return f"%/{y}-{mo}-{d}-{hh}-{mm}-{ss}/{cam}.{hour}.mp4"


def recording_like_path(path: str | None) -> str | None:
    """`recording_like` for a recording's own path (.../2026-06-13-00-01-05/cam_06.17.mp4)."""
    m = REC_PATH_RE.search(str(path or ""))
    return f"%/{m.group(1)}/{m.group(2)}.{m.group(3)}.{m.group(4)}" if m else None


def _basename(p) -> str:
    """File name of a path written on any OS (a summary written on Windows, read on a Mac)."""
    return re.split(r"[/\\]", str(p))[-1]


def _slp_video_path(slp) -> str | None:
    """The video path recorded INSIDE a .slp. For a clip added through the GUI's upload this is the only
    trace of the folder it came from: the upload copies the files into the project's media/."""
    try:
        import h5py
        with h5py.File(str(slp), "r") as f:
            v = json.loads(f["videos_json"][0])
        return v.get("filename") or (v.get("backend") or {}).get("filename")
    except Exception:
        return None


SUMMARY = "clips.summary.json"


def source_recording(entry: dict) -> dict | None:
    """The HCM recording a re-ID clip was cut from, read from the clip planner's `clips.summary.json`:
    {"video": <the hour's recording path>, "frames": [start, end) of that hour, or None for a whole-hour
    .reid.slp}. Looked for next to the clip's video and .slp, and next to the video its .slp names (an
    uploaded clip's only link back to its folder). None when no summary lists the clip."""
    vp, sp = entry.get("video_path"), entry.get("slp_path")
    names = {_basename(p) for p in (vp, sp) if p}
    dirs: list[Path] = []
    for p in (vp, sp, _slp_video_path(sp) if sp else None):
        if p and Path(p).parent not in dirs:
            dirs.append(Path(p).parent)
    for d in dirs:
        try:
            items = json.loads((d / SUMMARY).read_text())
        except Exception:
            continue
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict) or not it.get("video"):
                continue
            if names & {_basename(it[k]) for k in ("clip_video", "clip_slp") if it.get(k)}:
                return {"video": str(it["video"]), "frames": [it.get("start"), it.get("end")]}
            if it.get("slp") and _basename(it["slp"]) in names:      # the hour-level .reid.slp
                return {"video": str(it["video"]), "frames": None}
    return None


def _clean_ring(ring) -> list[list[float]]:
    pts: list[list[float]] = []
    for p in ring:
        xy = [round(float(p[0]), 1), round(float(p[1]), 1)]
        if not pts or pts[-1] != xy:
            pts.append(xy)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    return pts


def _area(ring) -> float:
    xy = [(float(p[0]), float(p[1])) for p in ring]
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(xy, xy[1:] + xy[:1]))) / 2


def _outer_ring(g) -> list[list[float]]:
    """A polygon's outer ring, cleaned, from a GeoJSON geometry (a Polygon, or the largest part of a
    MultiPolygon -- the pipeline's own rule) or from a bare ring; [] if it is none of those."""
    try:
        if isinstance(g, str):
            g = json.loads(g)
        if isinstance(g, dict):
            coords = g.get("coordinates") or []
            if g.get("type") == "Polygon":
                g = coords[0] if coords else []
            elif g.get("type") == "MultiPolygon":
                g = max((p[0] for p in coords if p), key=_area, default=[])
            else:
                return []
        return _clean_ring(g)
    except (TypeError, ValueError, IndexError, KeyError):
        return []


def fetch_rois(video_id: str, classes=tuple(ROI_CLASS), timeout: int = 8,
               source_video: str | None = None) -> dict:
    """Best-effort: {roi_field: polygon} for a clip, e.g. {'cage_roi': [[x,y],...]}, plus
      camera            cam_XX
      recording         the LIKE pattern naming the clip's own recording, or None
      recording_status  that recording's rois status, or None (no recording found)
      source            per field: "recording" or "camera medoid"
    `source_video` is the recording the clip was cut from, when its id doesn't say (`source_recording`).
    Returns {} (never raises) if sqlalchemy is missing, the DB is unreachable or the camera can't be
    parsed — so callers can treat it as an optional enrichment. Otherwise the DB answered, and a field
    that is absent was looked for and not found."""
    cam = camera_of(source_video or "") or camera_of(video_id)
    if not cam:
        return {}
    try:
        from sqlalchemy import create_engine, text
    except Exception:
        return {}
    if not DATABASE_URL:
        # No credential configured. Returning early rather than letting create_engine("") raise into
        # the catch-all below keeps "nobody set HCM_DATABASE_URL" distinguishable from "the DB is down"
        # for anyone reading a log — both end in {}, but only one is a misconfiguration.
        return {}
    out: dict = {}
    source: dict = {}
    pat = recording_like(video_id) or recording_like_path(source_video)
    try:
        engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": timeout})
        with engine.connect() as conn:
            status = conn.execute(text(_STATUS_SQL), {"pat": pat}).scalar() if pat else None
            for cls in classes:
                field = ROI_CLASS.get(cls)
                if not field:
                    continue
                # this recording's own polygon first, camera medoid only as a fallback (never the feeder's)
                attempts = []
                if pat:   # only attempt the per-recording lookup when the clip names a recording
                    attempts.append((_RECORDING_SQL, {"pat": pat, "cls": cls}, "recording"))
                if cls not in RECORDING_ONLY:
                    attempts.append((_MEDOID_SQL, {"cam": cam, "cls": cls, "lim": MEDOID_SAMPLE}, "camera medoid"))
                for sql, params, src in attempts:
                    try:
                        row = conn.execute(text(sql), params).first()
                    except Exception:
                        conn.rollback()   # one class's bad row must not cost the others their outlines
                        continue
                    poly = _outer_ring(row[0]) if row and row[0] else []
                    if len(poly) >= 3:
                        out[field], source[field] = poly, src
                        break
    except Exception:
        return {}
    return {"camera": cam, "recording": pat, "recording_status": status, "source": source, **out}
