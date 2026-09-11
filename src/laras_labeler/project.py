"""On-disk project store (PLAN.md §9).

A project is a directory under the projects root:

    <projects_root>/<pid>/
      ├── project.json        manifest (name, behaviors, videos, feature_config, roles)
      └── labels/<video_id>.parquet

Videos and .slp files are referenced by path, never copied.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import sleap_io as sio

SCHEMA_VERSION = 1

DEFAULT_FEATURE_CONFIG = {
    "wradius_seconds": 0.5,
    "stats": ["mean", "std", "min", "max", "change"],
    "radii_seconds": [0.07, 0.25, 0.5],
    "offsets": [0],
    "pool_aggregates": ["min", "mean", "max"],
    "confidence_threshold": 0.0,
    "per_slot": False,
    "egocentric": True,        # posture extents in the heading frame (rotation-invariant)
    "normalize_scale": True,   # speeds/accel in body-lengths per second (size- & zoom-invariant)
    "spout": None,             # [x, y] arena landmark (px, shared across clips); enables distance-to-spout features
    "spout_zone_bl": 2.0,      # "in zone" radius around the spout, in body lengths
    "spout_roi": None,         # [[x,y], ...] polygon region (px, shared across clips); enables spout-ROI features
    "cage_roi": None,          # [[x,y], ...] arena-boundary polygon (px); enables cage-ROI (wall-distance) features
    "pairwise_distances": False,  # add ALL C(N,2) inter-keypoint distances (JABS-style, body-length-norm).
                                  # OFF by default: 15 nodes -> 105 pairs x window = ~1680 extra columns, which
                                  # OVERFIT on the small (10s of bouts) label sets this tool targets. Enable only
                                  # with lots of labels / for exploratory feature-richness.
    "feature_code_version": 2,
}

# non-blue/red so behavior chips don't collide with Happening (blue) / Not-happening (red) label colors
_PALETTE = ["#e8a33d", "#7fb069", "#b07cc6", "#5fb0a8", "#d4c256",
            "#c98a5e", "#8fd0c4", "#a0d468", "#c9a0dc", "#b8a88a"]


def slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s.strip()).strip("-._").lower()
    return s or "item"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Project:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.manifest: dict = json.loads((path / "project.json").read_text())

    # --- identity ---
    @property
    def pid(self) -> str:
        return self.path.name

    @property
    def name(self) -> str:
        return self.manifest.get("name", self.pid)

    @property
    def behaviors(self) -> list[dict]:
        return self.manifest.setdefault("behaviors", [])

    @property
    def videos(self) -> list[dict]:
        return self.manifest.setdefault("videos", [])

    def video(self, video_id: str) -> dict | None:
        return next((v for v in self.videos if v["video_id"] == video_id), None)

    def save(self) -> None:
        (self.path / "project.json").write_text(json.dumps(self.manifest, indent=2))

    # --- videos ---
    def add_video(self, video_path: str | Path, slp_path: str | Path | None = None) -> dict:
        video_path = str(Path(video_path))
        if any(v["video_path"] == video_path for v in self.videos):
            raise ValueError("video already added")
        stem = slugify(Path(video_path).stem)
        existing = {v["video_id"] for v in self.videos}
        vid, i = stem, 2
        while vid in existing:
            vid, i = f"{stem}-{i}", i + 1
        labels = sio.load_file(str(slp_path or video_path))
        v = labels.videos[0]
        # the .slp may reference the video by a path that doesn't exist on this machine
        # (e.g. an uploaded file) — repoint it at the actual video so shape/frames resolve.
        if slp_path and video_path and Path(video_path).exists():
            try:
                v.replace_filename(str(video_path))
            except Exception:
                pass
        entry = {
            "video_id": vid,
            "video_path": video_path,
            "slp_path": str(slp_path) if slp_path else None,
            "n_frames": int(v.shape[0]),
            "fps": float(v.fps or 30.0),
            "width": int(v.shape[2]),
            "height": int(v.shape[1]),
            "has_poses": slp_path is not None,
        }
        self.videos.append(entry)
        self.save()
        return entry

    # --- media relocation -------------------------------------------------------------------
    # A project references its video/.slp by absolute path and never copies them (see the module
    # docstring). That is right for a 40GB share, and wrong the moment the project moves: carry a
    # project to another machine — a different OS, a different mount point, a colleague's laptop —
    # and every one of those paths is a dead absolute path from someone else's filesystem. The
    # labels, features and models are all still perfectly good; only the pointer to the pixels is
    # stale. So the fix is to re-point, not to re-import: match the files by name under a root the
    # user names, and rewrite the manifest.
    MEDIA_KEYS = ("video_path", "slp_path", "playback_path")
    _MEDIA_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm", ".slp", ".h5", ".hdf5"}
    _SCAN_LIMIT = 300_000   # files; a lab share can be enormous and we must not hang on it
    _SCAN_DEPTH = 8

    @property
    def media_roots(self) -> list[str]:
        """Folders to search for a clip's files, newest first. Persisted, so a project only has to be
        pointed at its media once per machine."""
        return self.manifest.setdefault("media_roots", [])

    def media_status(self) -> dict:
        """Which clips can find their files on THIS machine and which cannot."""
        missing = []
        for v in self.videos:
            gone = sorted({Path(v[k]).name for k in self.MEDIA_KEYS
                           if v.get(k) and not Path(v[k]).exists()})
            if gone:
                missing.append({"video_id": v["video_id"], "files": gone})
        return {"total": len(self.videos), "missing": missing,
                "n_missing": len(missing), "media_roots": list(self.media_roots)}

    def _index_media(self, root: Path) -> dict[str, Path]:
        """basename -> path for every media-ish file under `root`. Shallow files win over deep ones,
        so a top-level `media/` beats a stray copy in some archive subfolder."""
        found: dict[str, Path] = {}
        n = 0
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop(0)
            try:
                entries = list(d.iterdir())
            except OSError:
                continue
            for e in entries:
                if n >= self._SCAN_LIMIT:
                    return found
                try:
                    if e.is_dir():
                        if depth < self._SCAN_DEPTH and not e.name.startswith("."):
                            stack.append((e, depth + 1))
                        continue
                except OSError:
                    continue
                if e.suffix.lower() in self._MEDIA_EXTS:
                    n += 1
                    found.setdefault(e.name.lower(), e)   # case-insensitive: SMB/exFAT shares are
        return found

    def relink_media(self, root: str | Path | None = None) -> dict:
        """Re-point clips whose files are missing at copies found under `root` (and under any root
        already remembered, plus this project's own media/ dir).

        Matching is by exact filename — the same clip, wherever it now lives. A path that still
        resolves is left alone, so this is safe to call on every project open and safe to re-run.
        """
        roots: list[Path] = []
        if root:
            roots.append(Path(root).expanduser())
        roots.append(self.path / "media")
        roots += [Path(r) for r in self.media_roots]
        seen, ordered = set(), []
        for r in roots:
            rs = str(r)
            if rs not in seen and r.is_dir():
                seen.add(rs)
                ordered.append(r)

        before = self.media_status()
        if not before["missing"]:
            # nothing to do, but still remember an explicitly-given root for next time
            if root and Path(root).expanduser().is_dir():
                self._remember_root(root)
                self.save()
            return {**before, "relinked": [], "still_missing": [], "searched": [str(r) for r in ordered]}

        index: dict[str, Path] = {}
        for r in ordered:
            for name, p in self._index_media(r).items():
                index.setdefault(name, p)

        relinked, hit_root = [], False
        for v in self.videos:
            fixed = {}
            for k in self.MEDIA_KEYS:
                old = v.get(k)
                if not old or Path(old).exists():
                    continue
                hit = index.get(Path(old).name.lower())
                if hit is not None:
                    v[k] = str(hit)
                    fixed[k] = str(hit)
            if fixed:
                hit_root = True
                relinked.append({"video_id": v["video_id"], **fixed})

        if root and hit_root:
            self._remember_root(root)
        if relinked or (root and hit_root):
            self.save()
        after = self.media_status()
        return {**after, "relinked": relinked, "still_missing": after["missing"],
                "searched": [str(r) for r in ordered]}

    def _remember_root(self, root: str | Path) -> None:
        r = str(Path(root).expanduser())
        roots = self.media_roots
        if r in roots:
            roots.remove(r)
        roots.insert(0, r)
        del roots[8:]

    def remove_video(self, video_id: str) -> bool:
        """Drop a clip from the project: manifest entry + its derived data (labels, feature cache,
        predictions). Uploaded media under this project is removed too, but a source video/slp that
        lives OUTSIDE the project (e.g. on an SMB share) is never touched. Models/trainset snapshots
        keep their historical `videos_used` — they're frozen provenance."""
        v = self.video(video_id)
        if v is None:
            return False
        self.manifest["videos"] = [x for x in self.videos if x["video_id"] != video_id]
        self.save()
        rm = lambda p: p.unlink() if p.exists() and p.is_file() else None
        rm(self.path / "labels" / f"{video_id}.parquet")
        rm(self.path / "features" / f"{video_id}.npy")
        rm(self.path / "features" / f"{video_id}.meta.json")
        preds = self.path / "predictions" / video_id
        if preds.exists():
            shutil.rmtree(preds, ignore_errors=True)
        # remove uploaded media only if it's inside this project AND no remaining clip references it
        for key in ("video_path", "slp_path"):
            p = v.get(key)
            if not p:
                continue
            pp = Path(p)
            inside = self.path == pp.parent or self.path in pp.parents
            still_used = any(x.get("video_path") == p or x.get("slp_path") == p for x in self.videos)
            if inside and not still_used:
                rm(pp)
        return True

    # --- behaviors (stable id; labels keyed by id survive rename/recolor) ---
    def _next_behavior_id(self) -> int:
        return max((b["id"] for b in self.behaviors), default=-1) + 1

    def add_behavior(self, name: str, color: str | None = None, key: str | None = None) -> dict:
        if key and any(b.get("key") == key for b in self.behaviors):
            raise ValueError("duplicate hotkey")
        b = {
            "id": self._next_behavior_id(),
            "name": name,
            "color": color or _PALETTE[len(self.behaviors) % len(_PALETTE)],
            "key": key,
        }
        self.behaviors.append(b)
        self.save()
        return b

    def update_behavior(self, bid: int, **fields) -> dict:
        b = next((x for x in self.behaviors if x["id"] == bid), None)
        if b is None:
            raise KeyError(bid)
        key = fields.get("key")
        if key and any(x.get("key") == key and x["id"] != bid for x in self.behaviors):
            raise ValueError("duplicate hotkey")
        for k, v in fields.items():
            if v is not None:
                b[k] = v
        self.save()
        return b

    def delete_behavior(self, bid: int) -> None:
        self.behaviors[:] = [b for b in self.behaviors if b["id"] != bid]
        self.save()

    def clone_behavior(self, bid: int) -> dict:
        """Create a parallel copy of a behavior — same feature_set/postproc, a distinct name '(vN)' and
        color, but ZERO labels and no model. For A/B experiments: freeze the original as a reference and
        relabel the clone from scratch to compare how many labels reach the same performance. The source
        (its labels + model) is untouched."""
        import re
        src = next((x for x in self.behaviors if x["id"] == bid), None)
        if src is None:
            raise KeyError(bid)
        root = re.sub(r"\s*\(v\d+\)$", "", src["name"])
        existing = {x["name"] for x in self.behaviors}
        n = 2
        while f"{root} (v{n})" in existing:
            n += 1
        b = self.add_behavior(f"{root} (v{n})")   # add_behavior saves + picks a fresh palette color; key=None (no dup hotkey)
        for k in ("feature_set", "postproc"):     # copy config so it's a fair comparison; NOT labels/model
            if src.get(k) is not None:
                b[k] = src[k]
        self.save()
        return b


class ProjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Project] = {}
        for d in sorted(root.iterdir()):
            if (d / "project.json").exists():
                self._cache[d.name] = Project(d)

    def _rescan(self) -> None:
        """Pick up project folders that appeared since startup. Dropping a project folder into the
        projects root is how a project moves between machines, and having to restart the server
        before the app would admit it existed made that look like it had not worked."""
        try:
            entries = sorted(self.root.iterdir())
        except OSError:
            return
        for d in entries:
            if d.name in self._cache:
                continue
            try:
                if (d / "project.json").exists():
                    self._cache[d.name] = Project(d)
            except (OSError, ValueError):
                continue        # a half-copied folder: ignore it now, pick it up next time

    def list(self) -> list[Project]:
        self._rescan()
        return list(self._cache.values())

    def get(self, pid: str) -> Project | None:
        p = self._cache.get(pid)
        if p is None:
            self._rescan()
            p = self._cache.get(pid)
        return p

    def create(self, name: str, pid: str | None = None) -> Project:
        base = pid or slugify(name)
        pid, i = base, 2
        while pid in self._cache or (self.root / pid).exists():
            pid, i = f"{base}-{i}", i + 1
        path = self.root / pid
        (path / "labels").mkdir(parents=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "created": _now(),
            "skeleton_roles": {},
            "feature_config": dict(DEFAULT_FEATURE_CONFIG),
            "behaviors": [],
            "videos": [],
        }
        (path / "project.json").write_text(json.dumps(manifest, indent=2))
        proj = Project(path)
        self._cache[pid] = proj
        return proj

    def ensure_dev(self, sample_slp: Path) -> None:
        """First-run convenience: a persistent 'dev' project seeded with the mice sample."""
        if self._cache or not sample_slp.exists():
            return
        proj = self.create("Dev (mice sample)", pid="dev")
        proj.add_video(sample_slp, sample_slp)
        proj.add_behavior("behavior", key="1")
