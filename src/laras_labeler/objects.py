"""Object-directed bouts: which arena object a bout is on (climbing -> feeder, rearing -> cage wall).

A behavior marked `object_directed` records, per Happening bout, the objects it is on: any number from
the PROJECT's object list (`project_objects`), stored as one sorted "+"-joined value ("cage+feeder_top",
see labels._obj). Like behaviors, the list is the project's own: it starts with the arena
objects the HCM database outlines, and the annotator can add more ("nest", "toy", ...):

    feeder_top   feeder cage, top outline        clip `feeder_top_roi`
    feeder_side  feeder cage, side outline       clip `feeder_side_roi`
    cage         the cage wall                    `cage_roi` (clip, else project-wide)
    spout        the water spout                  `spout_roi` polygon, else the `spout` point

The ROIs are the ones the HCM auto-fill (`hcm_roi.py`) pulls from the database on clip open, or that
were drawn by hand. An added object has no ROI field: it can be picked, but not suggested.

`suggest` ranks the objects with an outline on the clip by how close the acting animal's nose stays
to each: the median over the bout of the nose's distance to the object, in pixels. For a "region"
object (feeder, spout, anything added) that is 0 while the nose is inside the outline; for the cage
("wall") it is the distance to the wall itself, since the animal is always inside the cage. The GUI
pre-fills the nearest and the annotator confirms or changes it. A label attribute only: no model
trains on it.
"""

from __future__ import annotations

import re

import numpy as np

DEFAULT_OBJECTS = [
    {"id": "feeder_top", "name": "feeder top", "roi": "feeder_top_roi", "measure": "region"},
    {"id": "feeder_side", "name": "feeder side", "roi": "feeder_side_roi", "measure": "region"},
    {"id": "cage", "name": "cage wall", "roi": "cage_roi", "measure": "wall"},
    {"id": "spout", "name": "spout", "roi": "spout_roi", "measure": "region"},
]


def project_objects(proj) -> list[dict]:
    """The project's object list: what it saved (even an empty list), else the defaults, which are
    not written until changed. The defaults are the HCM arena; another experiment removes them."""
    objs = proj.manifest.get("objects")
    return [dict(o) for o in (DEFAULT_OBJECTS if objs is None else objs)]


def object_ids(proj) -> set[str]:
    return {o["id"] for o in project_objects(proj)}


def add_object(proj, name: str) -> dict:
    """Add an object to the project by name; its id is a slug of the name, made unique. Saves."""
    name = " ".join(str(name or "").split())
    if not name:
        raise ValueError("an object needs a name")
    objs = project_objects(proj)
    if any(o["name"].lower() == name.lower() for o in objs):
        raise ValueError(f"there is already an object called {name!r}")
    # adding back a removed default ("spout", "cage wall") restores it with its outline, so it is
    # suggested again, rather than making a new outline-less object of the same name
    back = next((d for d in DEFAULT_OBJECTS if name.lower() in (d["name"], d["id"].replace("_", " "))), None)
    if back and not any(o["id"] == back["id"] for o in objs):
        proj.manifest["objects"] = objs + [dict(back)]
        proj.save()
        return dict(back)
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "object"
    oid, i = base, 2
    while any(o["id"] == oid for o in objs):
        oid, i = f"{base}_{i}", i + 1
    obj = {"id": oid, "name": name, "roi": None, "measure": "region"}
    proj.manifest["objects"] = objs + [obj]
    proj.save()
    return obj


def geometry(proj, entry: dict) -> dict:
    """object id -> ("polygon", [[x, y], ...]) or ("point", [x, y]), for the objects with an outline
    on this clip. A clip's own ROI wins over the project-wide one, as for features
    (FeatureStore._clip_cfg); the spout falls back to its point."""
    fc = proj.manifest.get("feature_config") or {}
    out = {}
    for o in project_objects(proj):
        field = o.get("roi")
        if not field:
            continue
        poly = entry.get(field) if entry.get(field) is not None else fc.get(field)
        if poly and len(poly) >= 3:
            out[o["id"]] = ("polygon", [[float(x), float(y)] for x, y in poly])
        elif field == "spout_roi":
            pt = entry.get("spout") if entry.get("spout") is not None else fc.get("spout")
            if pt and len(pt) == 2:
                out[o["id"]] = ("point", [float(pt[0]), float(pt[1])])
    return out


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point in p (K, 2) to the segment a-b."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _inside(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon for each point in p (K, 2)."""
    x, y = p[:, 0], p[:, 1]
    inside = np.zeros(len(p), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cross = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= cross & (x < xint)
        j = i
    return inside


def distances(points: np.ndarray, kind: str, shape, measure: str = "region") -> np.ndarray:
    """Per-point distance (pixels) from each nose position (K, 2) to one object."""
    if kind == "point":
        return np.linalg.norm(points - np.asarray(shape, dtype=float), axis=1)
    poly = np.asarray(shape, dtype=float)
    edge = np.min([_seg_dist(points, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))], axis=0)
    if measure == "wall":                 # always inside the cage: what matters is how near the wall
        return edge
    return np.where(_inside(points, poly), 0.0, edge)


def suggest(proj, entry: dict, poses: np.ndarray | None, nose: int | None, track: int,
            start: int, end: int, max_frames: int = 300) -> dict:
    """Rank the project's objects for one bout. `poses` is (F, T, N, 3) [x, y, score]; `nose` the
    nose node index. Returns {"objects": [{id, name, outline, distance_px}], "suggested": id | None}:
    every project object, those with an outline on this clip first, nearest first. distance_px is
    None without an outline or when the nose was never visible during the bout."""
    geo = geometry(proj, entry)
    pts = np.empty((0, 2))
    if poses is not None and nose is not None and 0 <= track < poses.shape[1]:
        s, e = max(0, int(start)), min(int(end), poses.shape[0])
        if e > s:
            idx = np.unique(np.linspace(s, e - 1, num=min(max_frames, e - s)).astype(int))
            pts = poses[idx, track, nose, :2].astype(float)
            pts = pts[np.isfinite(pts).all(axis=1)]
    rows = []
    for o in project_objects(proj):
        g = geo.get(o["id"])
        d = (float(np.median(distances(pts, g[0], g[1], o.get("measure", "region"))))
             if g is not None and len(pts) else None)
        rows.append({"id": o["id"], "name": o["name"], "outline": g is not None,
                     "distance_px": None if d is None else round(d, 1)})
    rows.sort(key=lambda r: (r["distance_px"] is None, r["distance_px"] or 0.0))
    best = rows[0]["id"] if rows and rows[0]["distance_px"] is not None else None
    return {"objects": rows, "suggested": best}
