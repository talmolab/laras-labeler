"""FastAPI app factory (v0a): project-scoped frames, poses, behaviors, labels (PLAN.md §8).

Routes are inlined for now; they move to routers/ as the surface grows (training, predict).
"""

from __future__ import annotations

from pathlib import Path

import json
import shutil
import threading
import time

import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from fastapi.staticfiles import StaticFiles

from . import config, hidra, poseio
from .config import Settings
from .events import EventLog, rounds_csv
from .features import quick_series
from .featurestore import FeatureStore
from .importers import Importer
from .jobs import JobManager
from .labels import LabelStore
from .predict import CANDIDATE_ORDERS, Predictor
from .project import DEFAULT_FEATURE_CONFIG, ProjectStore
from .training import Trainer
from .video import VideoManager

_IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


class NewProject(BaseModel):
    name: str


class EventBatch(BaseModel):
    """A flush from the browser's annotation-event queue (events.py). `events` stays untyped: the
    log is append-only and read by tolerant consumers, so a new event field must never need a
    server change to be recorded."""
    session: str
    events: list[dict] = Field(default_factory=list)


class NewVideo(BaseModel):
    video_path: str
    slp_path: str | None = None


class NewBehavior(BaseModel):
    name: str
    color: str | None = None
    key: str | None = None
    feature_set: str | None = None   # explicit choice; if omitted, one is auto-suggested from the name


class SetSpout(BaseModel):
    x: float | None = None   # video pixel coords; both None clears the spout
    y: float | None = None


class SetSpoutRoi(BaseModel):
    points: list[list[float]] | None = None   # [[x,y], ...] polygon vertices; None/empty clears the ROI

    @field_validator("points")
    @classmethod
    def _xy_pairs(cls, v):
        if v is not None and any(len(p) != 2 for p in v):
            raise ValueError("each ROI point must be exactly [x, y]")
        return v


class EditBehavior(BaseModel):
    name: str | None = None
    color: str | None = None
    key: str | None = None
    feature_set: str | None = None   # None keeps current; 'all'|'spout'|'cage'|'spout_cage'|'no_social'|'pose'|'social'|'social_pose'


class ReviewedBout(BaseModel):
    vid: str
    track: int
    start: int
    end: int
    label_value: int | None = None
    action: str = "reviewed"   # 'add' -> mark reviewed; 'remove' -> un-review


class LabelSpan(BaseModel):
    behavior_id: int
    track: int = 0
    start: int
    end: int
    value: int = Field(ge=0, le=2)  # 1 = Happening, 0 = Not-happening, 2 = Unknown (saved, excluded from training)
    source: str = "manual"          # provenance: 'manual' (painted) | 'candidate' (accepted from review) | 'imported'


class ImportRequest(BaseModel):
    csv_path: str
    behavior_col: str = "behavior"
    start_col: str = "start"
    end_col: str = "end"
    units: str = "frames"          # 'frames' | 'seconds'
    coverage: str = "complete"     # 'complete' (sample negatives) | 'partial'
    neg_ratio: float = 1.0


class ImportEventsRequest(BaseModel):
    json_path: str
    track: int | None = None       # None = all tracks (whole-frame); int = one track


def create_app(settings: Settings, store: ProjectStore) -> FastAPI:
    app = FastAPI(title="laras-labeler", version="0.1.0")

    # index.html is served by StaticFiles; without this the browser heuristically caches it and can
    # keep running stale UI code after an edit. no-cache forces revalidation (ETag 304 keeps it cheap).
    @app.middleware("http")
    async def _no_cache_html(request, call_next):
        resp = await call_next(request)
        ctype = resp.headers.get("content-type", "")
        if ctype.startswith("text/html") and "cache-control" not in resp.headers:
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    vm = VideoManager(store, settings.frame_cache_size, settings.jpeg_quality)
    labels = LabelStore(store)
    features = FeatureStore(store, vm)
    jobs = JobManager()
    trainer = Trainer(store, labels, features)
    predictor = Predictor(store, features, trainer, labels)
    importer = Importer(store, labels, vm)
    elog = EventLog(store)

    # Persisted HiDRA paths (set from the GUI) take effect before the first probe, so a restart
    # comes back configured instead of looking unconfigured until someone re-enters them.
    _saved = config.load_app_settings(settings.projects_root)
    if _saved.get("hidra_home") or _saved.get("hidra_python"):
        hidra.configure(_saved.get("hidra_home"), _saved.get("hidra_python"))
    app.state.settings = settings
    app.state.store = store
    app.state.vm = vm
    app.state.labels = labels
    app.state.features = features
    app.state.jobs = jobs
    app.state.trainer = trainer
    app.state.predictor = predictor
    app.state.events = elog

    def _behavior(pid: str, bid: int):
        proj = _proj(pid)
        if not any(b["id"] == bid for b in proj.behaviors):
            raise HTTPException(404, "behavior not found")
        return proj

    def _proj(pid: str):
        proj = store.get(pid)
        if proj is None:
            raise HTTPException(404, "project not found")
        return proj

    def _video(pid: str, vid: str):
        proj = _proj(pid)
        if proj.video(vid) is None:
            raise HTTPException(404, "video not found")
        return proj

    def _meta(pid: str, vid: str) -> dict:
        proj = store.get(pid)
        entry = proj.video(vid) or {}
        fc = proj.manifest.get("feature_config", {})
        # effective arena landmarks for THIS clip: a per-clip value overrides the project-wide default
        eff = {k: (entry[k] if entry.get(k) is not None else fc.get(k)) for k in ("spout", "spout_roi", "cage_roi")}
        # pix_per_cm is per-clip only (never a project default -- see set_video_scale). It has to be
        # in this payload because the GUI's HiDRA field reads it from here: without it the field was
        # always blank and the "set px/cm" warning never cleared, so Predict looked permanently
        # gated even on a clip whose scale was set and which Predict would have run.
        return {**vm.meta(pid, vid), "features": features.status(pid, vid),
                "pix_per_cm": entry.get("pix_per_cm"), **eff}

    # Every background job is timed into the annotation event log (events.py). Round timings need to
    # separate the human's time from the machine's, and the machine's time has to be recorded where the
    # human can't lose it — the browser is closed at the end of a session, the log is not.
    def _timed_job(kind: str, pid: str, fn, meta: dict, summary=lambda r: {}, job_kind: str | None = None):
        ctx = {k: v for k, v in meta.items() if k != "pid"}
        elog.log(pid, "job_start", kind=kind, **ctx)

        def wrapped(progress):
            t0 = time.perf_counter()
            try:
                r = fn(progress) or {}
            except Exception as e:  # noqa: BLE001
                elog.log(pid, "job_done", kind=kind, status="error", error=str(e)[:300],
                         seconds=round(time.perf_counter() - t0, 2), **ctx)
                raise
            # merged, not two ** expansions: a summary key that collided with a ctx key would raise
            # TypeError here and fail a job whose work had already succeeded
            elog.log(pid, "job_done", kind=kind, status="done",
                     seconds=round(time.perf_counter() - t0, 2), **{**ctx, **summary(r)})
            return r

        return jobs.start(job_kind or kind, wrapped, meta=meta)

    def _train_summary(r: dict) -> dict:
        m = r.get("metrics") or {}
        return {"version": r.get("version"), "trained_at": r.get("trained_at"),
                "ap": m.get("average_precision"), "f1": m.get("f1"),
                "precision": m.get("precision"), "recall": m.get("recall"),
                "n_pos": r.get("n_pos"), "n_neg": r.get("n_neg"),
                "n_pos_bouts": r.get("n_pos_bouts"), "n_neg_bouts": r.get("n_neg_bouts"),
                "n_seed_bouts": r.get("n_seed_bouts"), "n_candidate_bouts": r.get("n_candidate_bouts"),
                "train_seconds": r.get("train_seconds"), "feature_seconds": r.get("feature_seconds"),
                "n_videos": len(r.get("videos_used") or []),
                "predicted_videos": len((r.get("predict") or {}).get("videos") or [])}

    def _hidra_train_summary(r: dict) -> dict:
        """What a LABTAIL fine-tune produced, for the round it closes.

        Deliberately no `ap`: HiDRA's fine-tune does not report one back through this path, and the
        rounds table showing a blank there is honest, where borrowing the project model's AP would
        not be. `engine` is what tells the two apart when a project mixes them.

        `version` carries the checkpoint, because the round record has a `version` column that the
        native path fills and this one otherwise would not: a round you cannot trace to the model
        it produced is a row you cannot check anything against later."""
        return {"engine": "hidra", "lab": r.get("lab"), "action": r.get("action"),
                "version": r.get("checkpoint"), "backend": r.get("backend"),
                "n_pos_bouts": r.get("bouts"), "n_spans": r.get("spans"),
                "n_videos": r.get("videos")}

    # Feature pre-warm: the first Train computes any missing feature cache lazily (a multi-minute cold
    # cost that lands inside the human-in-the-loop window). Instead we fire that same background compute
    # as soon as a clip is added or the project is opened, so by train time the cache is already warm.
    # Fire-and-forget + de-duped against in-flight warms; repeated calls (e.g. project-state polls) are
    # cheap no-ops once a clip is ready or already warming.
    #
    # BOUNDED, because a big project makes the naive version dangerous. Opening a 99-clip longitudinal
    # project with evicted caches used to start 99 jobs at once — 99 OS threads, each a single-threaded
    # build peaking around 700 MB RSS, all racing to write 104.4 MiB of features, i.e. ~10 GiB onto a
    # volume that may have less than that free. Three guards, in order of what they protect:
    #   PREWARM_MAX_INFLIGHT  how many actually compute at once (the RAM/CPU bound)
    #   PREWARM_MAX_PER_CALL  how many get queued per call (the thread-count bound); the rest stay cold
    #                         and warm on the next project-open poll or lazily at train time
    #   PREWARM_MIN_FREE_GIB  refuse entirely when the disk is nearly full — filling the boot volume is
    #                         far worse than a slow first Train
    PREWARM_MAX_INFLIGHT = 2
    PREWARM_MAX_PER_CALL = 12
    PREWARM_MIN_FREE_GIB = 3.0
    _prewarming: set = set()
    _prewarm_lock = threading.Lock()
    _prewarm_slots = threading.Semaphore(PREWARM_MAX_INFLIGHT)

    def _free_gib(path) -> float:
        # shutil.disk_usage is cross-platform (os.statvfs is POSIX-only, absent on Windows);
        # .free is the same "available to this user" figure statvfs reports as f_bavail.
        try:
            return shutil.disk_usage(path).free / 2 ** 30
        except OSError:
            return float("inf")

    def _prewarm_features(pid: str, vids=None) -> list[str]:
        proj = store.get(pid)
        if proj is None:
            return []
        explicit = vids is not None      # an explicit add/upload is always worth one warm
        if vids is None:
            vids = [v["video_id"] for v in proj.videos]
        free = _free_gib(proj.path)
        if free < PREWARM_MIN_FREE_GIB:
            return []
        started = []
        for vid in vids:
            if not explicit and len(started) >= PREWARM_MAX_PER_CALL:
                break
            entry = proj.video(vid)
            if not entry or not entry.get("has_poses"):
                continue
            if features.status(pid, vid).get("status") == "ready":   # warm already (incl. source_missing)
                continue
            with _prewarm_lock:
                if (pid, vid) in _prewarming:
                    continue
                _prewarming.add((pid, vid))

            def _fn(progress, _pid=pid, _vid=vid):
                try:
                    with _prewarm_slots:
                        # re-check under the slot: the cache may have been built while we queued, and
                        # the disk may have filled up behind us.
                        if features.status(_pid, _vid).get("status") == "ready":
                            return {"skipped": "already ready"}
                        if _free_gib(store.get(_pid).path) < PREWARM_MIN_FREE_GIB:
                            return {"skipped": "low disk"}
                        return features.compute(_pid, _vid, progress)
                finally:
                    with _prewarm_lock:
                        _prewarming.discard((_pid, _vid))

            _timed_job("features", pid, _fn, {"pid": pid, "video_id": vid, "reason": "prewarm"},
                       job_kind="prewarm")
            started.append(vid)
        return started

    # Self-heal ROI auto-load: the per-clip DB fetch is best-effort and gives up silently on any hiccup
    # (a transient DB/VPN blip leaves a clip without ROIs). On project open, retry the fetch in the
    # BACKGROUND (never blocks the request on the DB) for any pose clip still missing an ROI — capped at
    # a few tries per clip so a camera that genuinely has no ROI isn't re-queried forever.
    _roi_loading: set = set()
    _roi_attempts: dict = {}
    _MAX_ROI_TRIES = 3

    def _autoload_missing_rois(pid: str) -> None:
        from . import hcm_roi
        proj = store.get(pid)
        if proj is None:
            return
        for v in proj.videos:
            if not v.get("has_poses"):
                continue
            vid = v["video_id"]
            if v.get("cage_roi") and v.get("spout_roi"):    # nothing missing
                continue
            key = (pid, vid)
            with _prewarm_lock:
                if key in _roi_loading or _roi_attempts.get(key, 0) >= _MAX_ROI_TRIES:
                    continue
                _roi_loading.add(key)
                _roi_attempts[key] = _roi_attempts.get(key, 0) + 1

            def _fn(progress, _pid=pid, _vid=vid, _key=key):
                try:
                    got = hcm_roi.fetch_rois(_vid)
                    entry = store.get(_pid).video(_vid)
                    changed = False
                    for field in ("cage_roi", "spout_roi"):
                        if got.get(field) and entry is not None and not entry.get(field):
                            entry[field] = got[field]; changed = True
                    if changed:
                        store.get(_pid).save()
                finally:
                    with _prewarm_lock:
                        _roi_loading.discard(_key)

            jobs.start("roi-autoload", _fn, meta={"pid": pid, "video_id": vid})

    # ---- projects & videos ----
    @app.get("/api/projects")
    def list_projects():
        return [{"pid": p.pid, "name": p.name, "videos": [v["video_id"] for v in p.videos]}
                for p in store.list()]

    @app.post("/api/projects")
    def create_project(body: NewProject):
        p = store.create(body.name)
        return {"pid": p.pid, "name": p.name}

    @app.get("/api/projects/{pid}")
    def get_project(pid: str):
        p = _proj(pid)
        _prewarm_features(pid)              # warm feature caches on project open so the first Train is fast
        _autoload_missing_rois(pid)         # background-retry ROI fetch for clips still missing one (self-heals transient DB blips)
        return {
            "pid": p.pid, "name": p.name,
            "behaviors": p.behaviors,
            "videos": [_meta(pid, v["video_id"]) for v in p.videos],
            "spout": p.manifest.get("feature_config", {}).get("spout"),   # [x,y] arena landmark, shared across clips
            "spout_roi": p.manifest.get("feature_config", {}).get("spout_roi"),   # [[x,y],...] polygon region
            "cage_roi": p.manifest.get("feature_config", {}).get("cage_roi"),   # [[x,y],...] cage-boundary polygon
        }

    @app.get("/api/projects/{pid}/feature-sets")
    def feature_sets_info(pid: str):
        """Exactly what each feature-set option feeds the model: the base signals (and column count) it
        selects, computed live from this project's real feature names — so it reflects the ROIs/features
        actually available and can't drift from the selection logic. Signals grouped by family."""
        from .featurestore import select_feature_cols
        p = _proj(pid)
        fnames = []
        for v in p.videos:                                    # first clip with computed features defines the families
            try:
                fn = features.meta(pid, v["video_id"]).get("feature_names")
            except (FileNotFoundError, ValueError):
                fn = None
            if fn:
                fnames = fn; break
        def family(b):
            if b.startswith("spout_roi"): return "spout ROI"
            if b.startswith("spout"): return "spout point"
            if b.startswith("cage"): return "cage ROI"
            if b.startswith("social"): return "social"
            return "pose/kinematics"
        sets = {}
        for s in ["all", "spout", "spout_cage", "cage", "no_social", "pose", "social", "social_pose"]:
            cols = select_feature_cols(fnames, s) if fnames else []
            bases, seen = [], set()
            for c in cols:
                b = fnames[c].split("__")[0]
                if b not in seen:
                    seen.add(b); bases.append(b)
            groups: dict[str, list] = {}
            for b in bases:
                groups.setdefault(family(b), []).append(b)
            sets[s] = {"n_cols": len(cols), "n_signals": len(bases), "by_family": groups}
        return {"have_features": bool(fnames), "sets": sets}

    @app.put("/api/projects/{pid}/spout")
    def set_spout(pid: str, body: SetSpout):
        """Set (or clear) the shared arena-landmark point used by the distance-to-spout features.
        Lives in feature_config, so changing it invalidates the feature cache -> recompute + retrain."""
        p = _proj(pid)
        fc = p.manifest.setdefault("feature_config", dict(DEFAULT_FEATURE_CONFIG))
        fc["spout"] = None if body.x is None or body.y is None else [float(body.x), float(body.y)]
        p.save()
        return {"spout": fc["spout"]}

    @app.put("/api/projects/{pid}/spout-roi")
    def set_spout_roi(pid: str, body: SetSpoutRoi):
        """Set (or clear) the shared spout ROI polygon used by the spout-ROI features.
        Lives in feature_config, so changing it invalidates the feature cache -> recompute + retrain."""
        p = _proj(pid)
        fc = p.manifest.setdefault("feature_config", dict(DEFAULT_FEATURE_CONFIG))
        pts = body.points or []
        if pts and len(pts) < 3:
            raise HTTPException(400, "a spout ROI needs at least 3 points")
        fc["spout_roi"] = [[float(x), float(y)] for x, y in pts] if pts else None
        p.save()
        return {"spout_roi": fc["spout_roi"]}

    @app.put("/api/projects/{pid}/videos/{vid}/scale")
    def set_video_scale(pid: str, vid: str, body: dict = Body(...)):
        """Set (or clear) this clip's pixels-per-cm.

        Required before HiDRA can run: its features are in centimetres and seconds, so the scale is
        a model input, not a display setting. Deliberately never defaulted — a plausible-looking
        wrong scale silently changes every distance and speed the classifier sees, and the result
        looks like a bad head rather than a bad number. Set per clip because one project can hold
        recordings from cameras at different heights."""
        proj = _video(pid, vid)
        entry = proj.video(vid)
        v = body.get("pix_per_cm")
        if v in (None, ""):
            entry.pop("pix_per_cm", None)
        else:
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise HTTPException(400, "pix_per_cm must be a number")
            if not 0 < v < 10000:
                raise HTTPException(400, "pix_per_cm out of range")
            entry["pix_per_cm"] = v
        proj.save()
        return {"pix_per_cm": entry.get("pix_per_cm")}

    @app.put("/api/projects/{pid}/videos/{vid}/spout")
    def set_video_spout(pid: str, vid: str, body: SetSpout):
        """Set (or clear) THIS clip's spout point — overrides the project-wide default for this clip
        only (clips from different cameras carry their own landmark). Invalidates just this clip's cache."""
        p = _video(pid, vid)
        entry = p.video(vid)
        entry["spout"] = None if body.x is None or body.y is None else [float(body.x), float(body.y)]
        p.save()
        return {"spout": entry["spout"]}

    @app.put("/api/projects/{pid}/videos/{vid}/spout-roi")
    def set_video_spout_roi(pid: str, vid: str, body: SetSpoutRoi):
        """Set (or clear) THIS clip's spout ROI polygon — overrides the project-wide default for this
        clip only. Invalidates just this clip's feature cache (per-clip feature hash)."""
        p = _video(pid, vid)
        entry = p.video(vid)
        pts = body.points or []
        if pts and len(pts) < 3:
            raise HTTPException(400, "a spout ROI needs at least 3 points")
        entry["spout_roi"] = [[float(x), float(y)] for x, y in pts] if pts else None
        p.save()
        return {"spout_roi": entry["spout_roi"]}

    @app.put("/api/projects/{pid}/cage-roi")
    def set_cage_roi(pid: str, body: SetSpoutRoi):
        """Set (or clear) the shared cage-boundary ROI polygon used by the cage-ROI features
        (wall distance + normalized cage position). Lives in feature_config -> invalidates the
        feature cache -> recompute + retrain."""
        p = _proj(pid)
        fc = p.manifest.setdefault("feature_config", dict(DEFAULT_FEATURE_CONFIG))
        pts = body.points or []
        if pts and len(pts) < 3:
            raise HTTPException(400, "a cage ROI needs at least 3 points")
        fc["cage_roi"] = [[float(x), float(y)] for x, y in pts] if pts else None
        p.save()
        return {"cage_roi": fc["cage_roi"]}

    @app.put("/api/projects/{pid}/videos/{vid}/cage-roi")
    def set_video_cage_roi(pid: str, vid: str, body: SetSpoutRoi):
        """Set (or clear) THIS clip's cage-boundary ROI polygon — overrides the project-wide default
        for this clip only (clips from different cameras carry their own cage). Invalidates just this
        clip's feature cache (per-clip feature hash)."""
        p = _video(pid, vid)
        entry = p.video(vid)
        pts = body.points or []
        if pts and len(pts) < 3:
            raise HTTPException(400, "a cage ROI needs at least 3 points")
        entry["cage_roi"] = [[float(x), float(y)] for x, y in pts] if pts else None
        p.save()
        return {"cage_roi": entry["cage_roi"]}

    @app.post("/api/projects/{pid}/videos/{vid}/autoload-roi")
    def autoload_roi(pid: str, vid: str, overwrite: bool = False):
        """Best-effort: fill THIS clip's cage (and spout) ROI from the HCM database, preferring the
        clip's own recording and falling back per class to the camera's medoid polygon. Only fills a
        MISSING ROI unless overwrite=true. Silently no-ops (set: {}) if the DB is unreachable (off-VPN),
        sqlalchemy isn't installed, the camera can't be parsed, or no valid polygon exists — so video
        loading never depends on it. `source` reports which polygon each field came from."""
        from . import hcm_roi
        p = _video(pid, vid)
        entry = p.video(vid)
        got = hcm_roi.fetch_rois(vid)
        set_fields = {}
        for field in ("cage_roi", "spout_roi"):
            if got.get(field) and (overwrite or not entry.get(field)):
                entry[field] = got[field]
                set_fields[field] = got[field]
        if set_fields:
            p.save()
        return {"camera": got.get("camera"), "source": got.get("source", {}), "set": set_fields}

    @app.post("/api/projects/{pid}/videos")
    def add_video(pid: str, body: NewVideo):
        p = _proj(pid)
        try:
            entry = p.add_video(body.video_path, body.slp_path)
        except ValueError as e:
            raise HTTPException(409, str(e))
        _prewarm_features(pid, [entry["video_id"]])   # start computing features immediately in the background
        return entry

    @app.post("/api/projects/{pid}/videos/upload")
    async def upload_video(pid: str, video: UploadFile = File(...), slp: UploadFile | None = File(None)):
        proj = _proj(pid)
        media = proj.path / "media"
        media.mkdir(exist_ok=True)
        vpath = media / (video.filename or "video.mp4")
        with open(vpath, "wb") as f:
            shutil.copyfileobj(video.file, f)
        spath = None
        if slp is not None and slp.filename:
            spath = media / slp.filename
            with open(spath, "wb") as f:
                shutil.copyfileobj(slp.file, f)
        try:
            entry = proj.add_video(str(vpath), str(spath) if spath else None)
            _prewarm_features(pid, [entry["video_id"]])   # background feature compute right after upload
            return entry
        except ValueError as e:
            if "already added" in str(e):   # re-loading the same file -> just reopen it
                existing = next((v for v in proj.videos if v["video_path"] == str(vpath)), None)
                if existing:
                    _prewarm_features(pid, [existing["video_id"]])   # warm in case its cache is stale/missing
                    return existing
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(400, f"could not register uploaded video: {e}")

    @app.post("/api/projects/{pid}/videos/{vid}/import")
    def import_annotations(pid: str, vid: str, body: ImportRequest):
        _video(pid, vid)
        try:
            return importer.import_csv(pid, vid, body.csv_path, body.behavior_col,
                                       body.start_col, body.end_col, body.units,
                                       body.coverage, body.neg_ratio)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/projects/{pid}/videos/{vid}/import-events")
    def import_events(pid: str, vid: str, body: ImportEventsRequest):
        _video(pid, vid)
        try:
            return importer.import_event_annotator(pid, vid, body.json_path, body.track)
        except (ValueError, FileNotFoundError, KeyError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/projects/{pid}/videos/{vid}")
    def get_video(pid: str, vid: str):
        _video(pid, vid)
        return _meta(pid, vid)

    @app.delete("/api/projects/{pid}/videos/{vid}")
    def remove_video(pid: str, vid: str):
        """Remove a clip and its derived data (labels/features/predictions) from the project."""
        proj = _video(pid, vid)
        proj.remove_video(vid)            # deletes labels/features/predictions files + manifest entry
        labels.forget(pid, vid)           # evict cached label DataFrame
        vm.forget(pid, vid)               # evict cached video handle + frames
        return {"ok": True, "videos": [v["video_id"] for v in proj.videos]}

    # ---- features & jobs ----
    @app.post("/api/projects/{pid}/videos/{vid}/features")
    def compute_features(pid: str, vid: str):
        entry = _video(pid, vid).video(vid)
        if not entry.get("has_poses"):
            raise HTTPException(409, "video has no poses")
        if features.status(pid, vid)["status"] == "ready":
            return {"status": "ready"}
        job = _timed_job("features", pid, lambda p: features.compute(pid, vid, p),
                         {"pid": pid, "video_id": vid})
        return {"job_id": job.id, "status": "pending"}

    @app.get("/api/projects/{pid}/jobs/{jid}")
    def get_job(pid: str, jid: str):
        job = jobs.get(jid)
        if job is None:
            raise HTTPException(404, "job not found")
        return job.snapshot()

    @app.get("/api/projects/{pid}/jobs/{jid}/events")
    def job_events(pid: str, jid: str):
        if jobs.get(jid) is None:
            raise HTTPException(404, "job not found")
        def gen():
            for ev in jobs.stream(jid):
                yield f"data: {json.dumps(ev)}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    # ---- training + prediction (the human-in-the-loop) ----
    @app.post("/api/projects/{pid}/behaviors/{bid}/train")
    def train_behavior(pid: str, bid: int, predict_videos: str | None = None):
        """Fit this behavior's model, then apply it. `predict_videos` scopes that second step:
        omitted = every clip in the project (what the UI wants, so its timeline refreshes);
        empty (`?predict_videos=`) = train only, no prediction; a comma-separated list of video_ids =
        just those. Worth scoping — on a project with many or long clips the sweep costs far more than
        the fit itself."""
        _behavior(pid, bid)
        scope = None if predict_videos is None else [s for s in (x.strip() for x in predict_videos.split(",")) if s]

        beh = next((b for b in store.get(pid).behaviors if b["id"] == bid), {})
        if beh.get("hidra", {}).get("action"):
            def hjob(progress):
                return _hidra_train(pid, bid, beh, progress)
            # Through _timed_job like the native path, not jobs.start directly: a Train is what CLOSES
            # a round in the annotation event log (events.py), so a fine-tune that skipped the log
            # would leave the behavior's rounds open forever -- one endless round, no per-round split,
            # no compute time. A HiDRA-bound behavior is still a behavior being annotated.
            j = _timed_job("train", pid, hjob, {"pid": pid, "behavior_id": bid, "hidra": True},
                           _hidra_train_summary)
            return {"job_id": j.id}

        def job(progress):
            r = trainer.train(pid, bid, lambda p, m: progress(int(p * 0.7), m))
            step = lambda p, m: progress(70 + int(p * 0.3), m)
            if scope is None:
                r["predict"] = predictor.predict_behavior(pid, bid, step)
            elif scope:
                r["predict"] = predictor.predict_behavior(pid, bid, step, videos=scope)
            else:
                r["predict"] = {"videos": [], "skipped": [], "note": "no prediction — predict_videos was empty"}
            return r

        j = _timed_job("train", pid, job, {"pid": pid, "behavior_id": bid}, _train_summary)
        return {"job_id": j.id}

    @app.get("/api/projects/{pid}/behaviors/{bid}/model")
    def get_model(pid: str, bid: int):
        _behavior(pid, bid)
        meta = trainer.model_meta(pid, bid)
        if meta is None:
            raise HTTPException(404, "no trained model")
        return meta

    @app.get("/api/projects/{pid}/behaviors/{bid}/history")
    def get_history(pid: str, bid: int):
        _behavior(pid, bid)
        return {"history": trainer.history(pid, bid), "meta": trainer.model_meta(pid, bid)}

    @app.get("/api/projects/{pid}/behaviors/{bid}/feature-diff")
    def get_feature_diff(pid: str, bid: int, top_n: int = 12):
        """Per-feature positive-vs-negative separation (AUC + histograms) for this behavior."""
        _behavior(pid, bid)
        return trainer.feature_diff(pid, bid, top_n=top_n)

    @app.get("/api/projects/{pid}/behaviors/{bid}/trainset")
    def get_trainset(pid: str, bid: int):
        """Frozen provenance of the current model — which videos/tracks/bouts it was trained on."""
        proj = _behavior(pid, bid)
        ts = trainer.trainset(pid, bid)
        if ts is not None:
            return ts
        # A model trained before per-bout provenance existed has meta.json but no trainset.json.
        # It still predicts fine — degrade gracefully to the video list + totals meta.json does have,
        # marked `stale` so the UI can say "retrain to record per-bout detail" instead of "no model".
        meta = trainer.model_meta(pid, bid)
        if meta is None:
            raise HTTPException(404, "no trained model")
        return {
            "version": meta.get("version"), "trained_at": meta.get("trained_at"),
            "n_videos": len(meta.get("videos_used", [])),
            "n_pos": meta.get("n_pos", 0), "n_neg": meta.get("n_neg", 0),
            "n_pos_bouts": meta.get("n_pos_bouts", 0), "n_neg_bouts": meta.get("n_neg_bouts", 0),
            "videos": [{"video_id": v, "name": Path(str((proj.video(v) or {}).get("video_path", ""))).name or v,
                        "pos_bouts": None, "neg_bouts": None, "pos_frames": None, "neg_frames": None, "tracks": []}
                       for v in meta.get("videos_used", [])],
            "bouts": [], "stale": True,
        }

    # ------------------------------------------------------------------------------------------
    # HiDRA: pretrained heads as an alternative to this project's own models
    # ------------------------------------------------------------------------------------------
    def _hidra_predict(pid: str, vid: str, behaviors: list[dict], progress) -> dict:
        """Run each bound head over one video and write predictions/<vid>/<bid>.npy.

        Writing the same artifact the project's own predictor writes is the whole integration: the
        timeline, candidate queue and review keys never learn where a lane came from."""
        proj = store.get(pid)
        entry = proj.video(vid)
        slp = entry.get("slp_path")
        if not slp:
            raise RuntimeError(f"{vid} has no tracking file")

        header = poseio.read_header(slp)
        poses = poseio.read_poses(slp, n_frames_hint=entry.get("n_frames"), header=header)
        if poses is None:
            import sleap_io as sio
            poses = sio.load_slp(slp).numpy(return_confidence=True)
        # A .slp often covers a longer recording than the clip the project holds (this project's
        # tracking spans 54k frames for a 10.8k-frame clip). Trim to the clip: predicting past its
        # end costs proportionally more CPU and would misalign the lanes against the timeline.
        n_frames = int(entry.get("n_frames") or poses.shape[0])
        if poses.shape[0] > n_frames:
            poses = poses[:n_frames]
        n_frames, n_animals = poses.shape[0], poses.shape[1]

        # fps and scale are inputs to the model, not cosmetics: HiDRA's features are in cm and
        # seconds. A container's declared fps can disagree with the rig's true rate (ours declares
        # 30 for 50 fps video), so the project's recorded value wins and is reported back.
        fps = float(entry.get("fps") or 30.0)
        ppc = float(entry.get("pix_per_cm") or proj.meta.get("pix_per_cm") or 0) or None
        if ppc is None:
            raise RuntimeError(
                "no pixels-per-cm for this video. HiDRA's features are in centimetres, so a scale "
                "is required — set pix_per_cm on the video or the project.")

        work = proj.path / "hidra" / vid / "_work"
        stem = Path(str(entry.get("video_path") or vid)).stem or vid
        progress(3, "exporting tracking")
        exported = hidra.export_tracking(poses, header.node_names, fps, ppc, stem, work)

        done = []
        for i, b in enumerate(behaviors):
            h = b["hidra"]
            lo = 5 + int(90 * i / max(len(behaviors), 1))
            span = int(90 / max(len(behaviors), 1))
            out = proj.path / "hidra" / vid / f"{h['lab']}__{h['action']}"
            fp = hidra.infer(work, out, h["lab"], h["action"], fps, ppc,
                             lambda p, m, lo=lo, span=span: progress(lo + int(p * span / 100), m))
            lanes = hidra.to_lanes(fp, h["lab"], h["action"], h.get("collapse", "scene"),
                                   n_frames, n_animals, h.get("rate", 0.15))
            dest = proj.path / "predictions" / vid / f"{b['id']}.npy"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.save(dest, lanes)
            done.append({"behavior_id": b["id"], "name": b.get("name"),
                         "lab": h["lab"], "action": h["action"],
                         "collapse": h.get("collapse", "scene"), "rate": h.get("rate", 0.15),
                         "above_cut": int((lanes >= 0.6).sum()), "frames": n_frames})
        progress(100, "done")
        return {"behaviors": done, "fps": fps, "pix_per_cm": ppc, "export": exported}

    def _hidra_train(pid: str, bid: int, beh: dict, progress) -> dict:
        """Fine-tune a bound head on this project's reviewed labels (HiDRA's LABTAIL adaptation).

        LABTAIL trains the lab embedding and the tail blocks only, leaving the SSL trunk frozen —
        which is why it is viable on a few hundred reviewed bouts instead of a full corpus."""
        rt = hidra.runtime()
        if not rt["can_infer"]:
            raise RuntimeError(rt["why"])
        script = Path(rt["home"]) / "train_perlab_heads.py"
        if not script.exists():
            raise RuntimeError(f"no train_perlab_heads.py under {rt['home']}")

        h = beh["hidra"]
        proj = store.get(pid)
        work = proj.path / "hidra" / "_labtail" / f"{h['lab']}__{h['action']}"
        work.mkdir(parents=True, exist_ok=True)
        progress(5, f"exporting labels for {h['action']}")
        n = hidra.export_labels(store, pid, bid, h, work)
        if n["bouts"] == 0:
            raise RuntimeError("no reviewed labels for this behavior yet — review some of the "
                               "head's proposals first, then train")

        progress(15, f"fine-tuning on {n['bouts']} bouts ({rt['backend']})")
        return hidra.finetune(script, rt, h, work, n,
                              lambda p, m: progress(15 + int(p * 0.85), m))

    @app.post("/api/projects/{pid}/videos/{vid}/predict")
    def predict_video(pid: str, vid: str):
        """Apply already-trained behavior models to one video (e.g. a newly loaded clip) without retraining."""
        proj = _video(pid, vid)
        entry = proj.video(vid)
        if not entry.get("has_poses"):
            raise HTTPException(409, "video has no tracking — load a .slp first")

        # Behaviors bound to a HiDRA head are answered by that head; the rest by this project's own
        # trained models. Both write predictions/<vid>/<bid>.npy, so everything downstream — the
        # timeline, the candidate queue, the review keys — is identical either way.
        bound = [b for b in proj.behaviors if b.get("hidra", {}).get("action")]
        if not bound and not predictor.trained_behaviors(pid):
            raise HTTPException(409, "nothing to predict with — pick a HiDRA classifier for a "
                                     "behavior, or train one of this project's own models first")

        def job(progress):
            out = {}
            if bound:
                out["hidra"] = _hidra_predict(pid, vid, bound,
                                              lambda p, m: progress(int(p * (70 if predictor.trained_behaviors(pid) else 100) / 100), m))
            if predictor.trained_behaviors(pid):
                base = 70 if bound else 0
                span = 30 if bound else 100
                if features.status(pid, vid)["status"] != "ready":
                    features.compute(pid, vid, lambda p, m: progress(base + int(p * span * 0.6 / 100), f"features: {m}"))
                out["own"] = predictor.predict_video(
                    pid, vid, lambda p, m: progress(base + int(span * 0.6) + int(p * span * 0.4 / 100), m))
            return out

        j = _timed_job("predict", pid, job, {"pid": pid, "video_id": vid},
                       lambda r: {"n_behaviors": len(r.get("behaviors") or []),
                                  "n_skipped": len(r.get("skipped") or [])})
        return {"job_id": j.id}

    @app.get("/api/projects/{pid}/predict/{vid}")
    def get_predict(pid: str, vid: str, behavior: int, track: int = 0):
        _video(pid, vid)
        proba = predictor.get_proba(pid, vid, behavior, track)
        if proba is None:
            raise HTTPException(404, "no predictions")
        blob = proba.astype("<f4").tobytes()
        return Response(content=blob, media_type="application/octet-stream",
                        headers={"X-Shape": str(len(proba)), "Cache-Control": "no-cache"})

    @app.get("/api/projects/{pid}/behaviors/{bid}/candidates/{vid}")
    def get_candidates(pid: str, vid: str, bid: int, track: int = 0, n: int = 12, mode: str = "new",
                       order: str = "uncertain"):
        _behavior(pid, bid)
        _video(pid, vid)
        if order not in CANDIDATE_ORDERS:
            raise HTTPException(400, f"order must be one of {', '.join(CANDIDATE_ORDERS)}")
        return predictor.candidates(pid, vid, bid, track, n, mode, order)

    @app.post("/api/projects/{pid}/behaviors/{bid}/reviewed")
    def set_reviewed(pid: str, bid: int, body: ReviewedBout):
        """Mark (or un-mark) a bout as reviewed-for-mislabels, so it drops out of the mislabel queue
        and appears in the 'Reviewed' tab. Persisted per behavior."""
        _behavior(pid, bid)
        if body.action == "remove":
            return predictor.unmark_reviewed(pid, bid, body.vid, body.track, body.start, body.end)
        return predictor.mark_reviewed(pid, bid, body.vid, body.track, body.start, body.end,
                                       body.label_value, "reviewed")

    _SRC_MISSING = "video/pose source file missing on disk (moved, deleted, or an ephemeral scratch dir was cleaned up) — playback unavailable for this clip; labels/features/model are unaffected"

    @app.get("/api/projects/{pid}/skeleton/{vid}")
    def get_skeleton(pid: str, vid: str):
        try:
            return vm.skeleton(pid, vid)
        except KeyError:
            raise HTTPException(404, "video not found")
        except FileNotFoundError:
            raise HTTPException(404, _SRC_MISSING)

    @app.get("/api/projects/{pid}/features/{vid}/series")
    def feature_series(pid: str, vid: str, track: int = 0):
        proj = _video(pid, vid)
        try:
            ov = vm._get(pid, vid)
        except FileNotFoundError:
            raise HTTPException(404, _SRC_MISSING)
        pose = ov.poses()                                    # (F, T, N, 3)
        nodes = ov.node_names
        roles = proj.manifest.get("skeleton_roles", {})
        series = quick_series(pose, nodes, float(vm.meta(pid, vid)["fps"]), roles)
        F, T = pose.shape[0], pose.shape[1]
        t = max(0, min(T - 1, track))
        names = list(series)
        blob = b"".join(series[n][:, t].astype("<f4").tobytes() for n in names)
        return Response(content=blob, media_type="application/octet-stream",
                        headers={"X-Series": ",".join(names), "X-Frames": str(F),
                                 "Cache-Control": "no-cache"})

    @app.get("/api/projects/{pid}/poses/{vid}")
    def get_poses(pid: str, vid: str):
        try:
            shape, blob = vm.poses_blob(pid, vid)
        except KeyError:
            raise HTTPException(404, "video not found")
        except FileNotFoundError:
            raise HTTPException(404, _SRC_MISSING)
        headers = {"X-Pose-Shape": ",".join(map(str, shape)), "Cache-Control": "no-cache"}
        return Response(content=blob, media_type="application/octet-stream", headers=headers)

    @app.get("/api/projects/{pid}/frame/{vid}/{idx}.jpg")
    def get_frame(pid: str, vid: str, idx: int, gray: int = 1):
        try:
            data = vm.frame_jpeg(pid, vid, idx, gray=bool(gray))
        except KeyError:
            raise HTTPException(404, "video not found")
        except IndexError:
            raise HTTPException(404, "frame out of range")
        except FileNotFoundError:
            raise HTTPException(404, _SRC_MISSING)
        return Response(content=data, media_type="image/jpeg", headers=_IMMUTABLE)

    @app.get("/api/projects/{pid}/video/{vid}/stream.mp4")
    def stream_video(pid: str, vid: str):
        proj = _video(pid, vid)
        entry = proj.video(vid)
        path = entry.get("playback_path") or entry["video_path"]
        if not Path(path).exists():
            raise HTTPException(404, _SRC_MISSING)
        # FileResponse serves HTTP range requests -> the browser <video> element streams it natively.
        # Cache-Control: this Starlette version's FileResponse sets ETag/Last-Modified but does NOT
        # honor conditional requests (verified: a matching If-None-Match still gets a full 200, not
        # 304) — so without an explicit max-age, every page refresh re-fetches the whole file (often
        # 40-50MB) with zero caching benefit, which is the actual cause of "videos take a long time to
        # load on refresh". The video file for a given vid is effectively immutable once imported, so a
        # day-long cache is safe; the rare case of replacing a clip's media (e.g. a VAST recovery) needs
        # a hard-refresh (⌘⇧R) or cache-clear to pick up, same as any long-lived static asset.
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, max-age=86400"})

    # ---- behaviors ----
    @app.get("/api/projects/{pid}/behaviors")
    def list_behaviors(pid: str):
        return _proj(pid).behaviors

    @app.post("/api/projects/{pid}/behaviors")
    def add_behavior(pid: str, body: NewBehavior):
        from .featurestore import suggest_feature_set
        try:
            beh = _proj(pid).add_behavior(body.name, body.color, body.key)
        except ValueError as e:
            raise HTTPException(409, str(e))
        # never start a behavior on 'all' by accident: use the caller's choice, else auto-suggest from the name.
        fset, reason = (body.feature_set, "explicit choice") if body.feature_set else suggest_feature_set(body.name)
        if fset:
            beh = _proj(pid).update_behavior(beh["id"], feature_set=fset)
        beh["suggested_feature_set"] = fset
        beh["suggest_reason"] = reason
        return beh

    @app.get("/api/projects/{pid}/suggest-feature-set")
    def suggest_fset(pid: str, name: str):
        """Preview the auto-suggested feature set for a behavior name (so the create dialog can pre-select
        it and show why), grounded in the empirical axis rule."""
        from .featurestore import suggest_feature_set
        fset, reason = suggest_feature_set(name)
        return {"feature_set": fset, "reason": reason}

    @app.put("/api/projects/{pid}/behaviors/{bid}")
    def edit_behavior(pid: str, bid: int, body: EditBehavior):
        try:
            return _proj(pid).update_behavior(bid, **body.model_dump())
        except KeyError:
            raise HTTPException(404, "behavior not found")
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/projects/{pid}/behaviors/{bid}/clone")
    def clone_behavior(pid: str, bid: int, with_labels: bool = False):
        """Create a parallel behavior for A/B comparison. with_labels=False -> empty (relabel from
        scratch, e.g. to compare how many labels are needed). with_labels=True -> copy the source's
        INITIAL/seed labels only (hand-painted 'manual' + 'imported', NOT the candidate-review labels)
        so the clone shares the same starting point but runs its OWN candidate process + further
        labeling independently on the new model."""
        _behavior(pid, bid)
        try:
            nb = _proj(pid).clone_behavior(bid)
        except KeyError:
            raise HTTPException(404, "behavior not found")
        if with_labels:
            nb["labels_copied"] = labels.copy_labels(pid, bid, nb["id"], initial_only=True)["frames_copied"]
        return nb

    @app.post("/api/projects/{pid}/behaviors/{bid}/copy-labels-from/{src_bid}")
    def copy_labels(pid: str, bid: int, src_bid: int, initial_only: bool = True):
        """Overwrite this behavior's labels with a copy of another behavior's. initial_only=True (default)
        copies only the seed labels (excludes candidate-review labels); pass initial_only=false for a
        fully-identical copy."""
        _behavior(pid, bid); _behavior(pid, src_bid)
        return labels.copy_labels(pid, src_bid, bid, initial_only=initial_only)

    @app.delete("/api/projects/{pid}/behaviors/{bid}")
    def delete_behavior(pid: str, bid: int):
        _proj(pid).delete_behavior(bid)
        labels.delete_behavior(pid, bid)  # cascade
        return {"ok": True}

    # ---- labels (half-open [start, end); PUT overwrites, DELETE -> unlabeled) ----
    @app.get("/api/projects/{pid}/labels/{vid}")
    def get_labels(pid: str, vid: str, track: int = 0, behavior: int | None = None):
        _video(pid, vid)
        return labels.get_runs_src(pid, vid, track, behavior)   # [start, end, value, source] per run

    @app.put("/api/projects/{pid}/labels/{vid}")
    def put_labels(pid: str, vid: str, spans: list[LabelSpan]):
        _video(pid, vid)
        rows = [s.model_dump() for s in spans]
        labels.put_spans(pid, vid, rows)
        # Server-side backstop for the event log: the browser's semantic events (which paint, which
        # accept) are richer, but they live in a queue that a closed tab or a crash can lose. Every
        # label that reaches disk is on record here regardless.
        elog.log(pid, "label_write", video_id=vid, n_spans=len(rows),
                 frames=sum(max(0, r["end"] - r["start"]) for r in rows),
                 behaviors=sorted({r["behavior_id"] for r in rows}),
                 tracks=sorted({r["track"] for r in rows}),
                 sources=sorted({r.get("source") or "manual" for r in rows}))
        return {"ok": True}

    @app.delete("/api/projects/{pid}/labels/{vid}")
    def delete_labels(pid: str, vid: str, behavior: int, track: int, start: int, end: int):
        _video(pid, vid)
        labels.delete_range(pid, vid, behavior, track, start, end)
        elog.log(pid, "label_clear", video_id=vid, behavior_id=behavior, track=track,
                 start=start, end=end, frames=max(0, end - start))
        return {"ok": True}

    @app.get("/api/projects/{pid}/label-stats")
    def label_stats(pid: str):
        """Provenance of the current labels — per behavior/track/source bout+frame counts."""
        _proj(pid)
        return labels.source_stats(pid)

    # ---- annotation event log + timing rollup (events.py) ----
    # How long does a round of annotation actually take, and is the human-in-the-loop loop cheaper
    # than painting labels by hand? Nothing else on disk can answer that: labels record WHAT was
    # annotated, history.json records accuracy per bout — neither records the clock.
    @app.post("/api/projects/{pid}/events")
    def post_events(pid: str, body: EventBatch):
        """Flush of the browser's event queue. Fire-and-forget from the client's point of view."""
        _proj(pid)
        try:
            return elog.append(pid, body.session, body.events)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/projects/{pid}/events/sessions")
    def list_event_sessions(pid: str):
        _proj(pid)
        return {"sessions": elog.sessions(pid)}

    @app.get("/api/projects/{pid}/events.jsonl")
    def download_events(pid: str, session: str | None = None):
        """The raw log, newline-delimited JSON — for pandas/duckdb, or for archiving with the project."""
        _proj(pid)
        recs = elog.read(pid, [session] if session else None)
        body = "\n".join(json.dumps(r, separators=(",", ":"), default=str) for r in recs) + "\n"
        return Response(content=body, media_type="application/x-ndjson",
                        headers={"Content-Disposition": f'attachment; filename="{pid}-events.jsonl"',
                                 "Cache-Control": "no-cache"})

    @app.get("/api/projects/{pid}/timing")
    def timing(pid: str, behavior: int | None = None, session: str | None = None,
               gap_cap_s: float = 15.0, idle_break_s: float = 120.0):
        """Per-round human time (labeling vs reviewing), compute time, what was produced, and the
        model that came out — plus the manual-vs-review headline. `gap_cap_s`/`idle_break_s` tune
        what counts as working time; the defaults are deliberately conservative (see events.py)."""
        _proj(pid)
        return elog.summarize(pid, [session] if session else None, gap_cap_s=gap_cap_s,
                              idle_break_s=idle_break_s, behavior_id=behavior)

    @app.get("/api/projects/{pid}/timing.csv")
    def timing_csv(pid: str, behavior: int | None = None, session: str | None = None,
                   gap_cap_s: float = 15.0, idle_break_s: float = 120.0):
        _proj(pid)
        summary = elog.summarize(pid, [session] if session else None, gap_cap_s=gap_cap_s,
                                 idle_break_s=idle_break_s, behavior_id=behavior)
        return Response(content=rounds_csv(summary), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{pid}-rounds.csv"',
                                 "Cache-Control": "no-cache"})

    web = Path(__file__).parent / "web"
    # --- HiDRA support endpoints ---------------------------------------------------------------
    @app.get("/api/hidra/status")
    def hidra_status():
        """Feature detection for the GUI. Present always; `can_infer` says whether Predict will work.

        `heads` is 0 until a checkout is found, which is what used to make the GUI hide the whole
        feature with no explanation. It now shows the setup row instead, driven by `why` and by
        `home_source`/`python_source` (gui / env / default), so a new user can see that HiDRA exists,
        what is missing, and where the current paths came from."""
        rt = hidra.runtime()
        return {**rt, "heads": len(hidra.catalog())}

    @app.put("/api/hidra/config")
    def set_hidra_config(body: dict = Body(...)):
        """Point the integration at a checkout, from the GUI, and re-probe.

        Persisted to <projects_root>/settings.json so it survives a restart. Returns the same shape
        as /status, already re-probed, so the GUI can show the outcome of the change immediately
        rather than asking the user to reload and guess.

        Note what this does NOT do: it never installs anything and never runs the checkout. The
        probe executes the interpreter once as `<python> -c "import jax"` -- the same thing
        HIDRA_PYTHON already caused before this endpoint existed. The server binds 127.0.0.1 for a
        single local user, the same trust level under which it already accepts server-side video
        paths."""
        home, py = body.get("home"), body.get("python")
        for label, val in (("home", home), ("python", py)):
            if val is not None and not isinstance(val, str):
                raise HTTPException(400, f"{label} must be a string path")
        config.save_app_settings(settings.projects_root,
                                 {"hidra_home": (home or "").strip(),
                                  "hidra_python": (py or "").strip()})
        hidra.configure(home, py)
        rt = hidra.runtime()
        return {**rt, "heads": len(hidra.catalog())}

    @app.get("/api/hidra/heads")
    def hidra_heads():
        """The shipped (lab, action) heads, for the behavior->classifier picker."""
        return hidra.catalog()

    @app.put("/api/projects/{pid}/behaviors/{bid}/hidra")
    def set_hidra_head(pid: str, bid: int, body: dict = Body(...)):
        """Bind a behavior to a head — or unbind it by posting an empty action.

        Bound is what makes Predict and Train use HiDRA for this behavior; unbound falls back to
        this project's own model, so the two can coexist in one project."""
        _behavior(pid, bid)
        proj = store.get(pid)
        beh = next(b for b in proj.behaviors if b["id"] == bid)
        action = (body.get("action") or "").strip()
        if not action:
            beh.pop("hidra", None)
        else:
            known = {(h["lab"], h["action"]) for h in hidra.catalog()}
            lab = (body.get("lab") or "").strip()
            if known and (lab, action) not in known:
                raise HTTPException(400, f"no such head: ({lab}, {action})")
            collapse = (body.get("collapse") or "scene").strip()
            if collapse not in {c.value for c in hidra.Collapse}:
                raise HTTPException(400, "collapse must be self, scene or directed")
            rate = float(body.get("rate", 0.15))
            if not 0 < rate < 1:
                raise HTTPException(400, "rate must be between 0 and 1")
            beh["hidra"] = {"lab": lab, "action": action, "collapse": collapse, "rate": rate}
        proj.save()
        return beh.get("hidra", {})

    app.mount("/", StaticFiles(directory=str(web), html=True), name="web")
    return app
