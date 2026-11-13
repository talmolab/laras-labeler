# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tailcyclenet>=0.4.2", "sleap-io>=0.8", "pillow", "toml",
#   # tracktail (under tailcyclenet) pins numpy 1.26, and pyarrow from 24 on refuses to import
#   # under numpy 1.x. 23.0.1 is what tailcyclenet's own pixi.lock uses.
#   "pyarrow<24",
# ]
# ///
"""Convert SLEAP `.slp` files into a tailcyclenet dataset root.

tailcyclenet (https://github.com/AI-HHMI/tailcyclenet) reads one format, specified in its
`docs/annotation_format.md`. This script writes that format from SLEAP files, in one of two ways:

**Human labels (the default).** Every frame with user instances is placed in a short clip of real
video around it (`--context`, default 24 frames). The clip's frames are decoded once and written as
JPEGs, so training never seeks in the source video, which may be on a slow share. Labelled frames
that fall in the same clip share it. The session says `labels = "annotated"`. This is the data
`tailcyclenet train` and `tailcyclenet train-detector` learn from.

**Machine tracking (`--tracked`).** Each video's tracked instances become one dense group spanning
the whole recording. The video is symlinked, not decoded or copied. The session says
`labels = "tracked"`. Use it to QC SLEAP tracking with `tailcyclenet score`, or as extra
(down-weighted) training data next to a human-labelled root.

One session per source video: a 2D, one-camera session (`cam0`) with a nominal calibration. The
split is per video: `--val-fraction` of the videos go to `val/`, picked by a stable hash of the path.

What is carried over, and how:

- Keypoint names, edges and **left/right symmetries** come from the SLEAP skeleton. Symmetries
  become `flip_pairs` (which is what lets `[data].flip_2d_prob` be turned on). A skeleton with no
  symmetries gets no `flip_pairs` key, so flip augmentation is refused rather than run with the
  left and right labels unswapped. `--flip-pair A B` declares pairs by hand instead.
- A placed user point toggled "not visible" is written as `missing`, which tailcyclenet reads as
  "assessed and occluded". A NaN point (never placed, e.g. a node the prediction a user instance
  was made from had missed) is written as no row at all, "no determination".
- A user instance with no visible points is a SLEAP placeholder and is dropped.
- Identity: a user instance's SLEAP track becomes its `animal_id`, so labels on nearby frames of
  the same clip are one animal. Untracked user instances get a per-frame id. `--ignore-tracks`
  gives every instance a per-frame id.
- **Unlabelled animals.** tailcyclenet's detector treats everything outside the labelled animals as
  background, unless the frame says it is incompletely labelled. A predicted instance left on a
  labelled frame (not one a user instance was made from, and not overlapping one) is an animal
  nobody labelled, so it becomes a `present` row with its box. `--animals-per-frame N` also marks a
  frame incomplete when it has fewer than N labelled animals.

Run it in the environment tailcyclenet is installed in (it needs `tailcyclenet.format` to write and
validate the tables; it does not use a GPU):

    pip install sleap-io pillow
    python scripts/slp_to_tailcycle.py labels.v001.slp --out ~/tailcycle/hcm-mice

or with uv, which builds a throwaway environment from the header above:

    uv run scripts/slp_to_tailcycle.py labels.v001.slp --out ~/tailcycle/hcm-mice

Video paths stored in the `.slp` are usually from another machine. Rewrite a prefix with
`--replace /old/prefix /new/prefix`, or name folders to look in by file name with `--video-dir`.

Then, on a GPU machine:

    tailcyclenet train --data ~/tailcycle/hcm-mice --out runs/hcm-pose
    tailcyclenet train-detector --config hcm-detector.toml     # [data].path / [training].out
    tailcyclenet infer --pose runs/hcm-pose --detector runs/hcm-detector \\
        --videos new_videos/ --out pred/ --max-animals 3
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

CAM = "cam0"
# The crop rule tailcyclenet applies to a keypoint extent (data/crop.py: 5% of the longer side).
# A box written here is padded the same way, so a session with instances.pq gives the detector
# the box it would have derived from the keypoints.
BOX_PAD_FRAC = 0.05
# A predicted instance this much IoU over a user instance is the same animal, not an unlabelled one.
SAME_ANIMAL_IOU = 0.3
INCOMPLETE_ID = "_unlabelled"


def _need(module: str, hint: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as e:  # pragma: no cover - environment problem, not logic
        sys.exit(f"{module} is not importable ({e}). {hint}")


# ------------------------------------------------------------------------------------------------
# small helpers


def _slug(text: str) -> str:
    """A folder-safe id: letters, digits, `-`, `_`, `.`; everything else becomes `_`."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "x"


def _name_of(path_text: str) -> str:
    """The file name of a path written on any OS."""
    return (PureWindowsPath if "\\" in path_text else PurePosixPath)(path_text).name


def _stable_unit(text: str) -> float:
    """A number in [0, 1) that depends only on `text` (the val split must not move between runs)."""
    return int(hashlib.sha1(text.encode()).hexdigest()[:12], 16) / float(16**12)


def resolve_video(filename: str, slp_path: Path, replace: list[tuple[str, str]],
                  search: list[Path]) -> Path | None:
    """Find the video a `.slp` names on this machine, or None."""
    text = filename
    for old, new in replace:
        if text.startswith(old):
            text = new + text[len(old):]
            break
    p = Path(text)
    candidates = [p] if p.is_absolute() else [slp_path.parent / p, p]
    name = _name_of(text)
    candidates += [d / name for d in search] + [slp_path.parent / name]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def _xy_vis(inst) -> tuple[np.ndarray, np.ndarray]:
    """(K,2) float coordinates and (K,) visibility flags of one SLEAP instance."""
    pts = inst.points
    xy = np.asarray(pts["xy"], dtype=np.float64).reshape(-1, 2)
    vis = np.asarray(pts["visible"], dtype=bool).reshape(-1)
    return xy, vis


def _extent(xy: np.ndarray) -> np.ndarray | None:
    """Padded [x0, y0, x1, y1] around the finite points, or None if there are none."""
    ok = np.isfinite(xy).all(axis=1)
    if not ok.any():
        return None
    lo, hi = xy[ok].min(axis=0), xy[ok].max(axis=0)
    pad = BOX_PAD_FRAC * float((hi - lo).max())
    return np.array([lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad], dtype=np.float32)


def _iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _labelled_points(inst) -> tuple[np.ndarray, np.ndarray]:
    """A user instance's (K,2) xy (NaN where not visible) and (K,) status codes."""
    from tailcyclenet import format as fmt

    xy, vis = _xy_vis(inst)
    finite = np.isfinite(xy).all(axis=1)
    status = np.full(len(vis), fmt.UNLABELED, dtype=np.int8)
    status[vis & finite] = fmt.VISIBLE
    # A placed point the annotator toggled "not visible" is a judgement that it is occluded; its
    # stored position is stale and is not written. A NaN point was never placed (typically a node
    # the prediction a user instance was made from had missed): no determination, so no row.
    status[~vis & finite] = fmt.MISSING
    out = np.where((status == fmt.VISIBLE)[:, None], xy, np.nan).astype(np.float32)
    return out, status


# ------------------------------------------------------------------------------------------------
# reading the .slp files


class Source:
    """Everything converted from one video: its frames' instances, grouped by frame index."""

    def __init__(self, video_path: Path, slp_path: Path):
        self.video_path = video_path
        self.slp_paths = [slp_path]
        self.frames: dict[int, dict] = {}


def collect(slp_paths: list[Path], replace, search, tracked: bool, ignore_tracks: bool):
    """Read every `.slp`; return (sources by video, names, edges, symmetries, report lines)."""
    sio = _need("sleap_io", "Install it with `pip install sleap-io`.")
    names = edges = syms = None
    sources: dict[Path, Source] = {}
    report: list[str] = []
    for slp in slp_paths:
        labels = sio.load_slp(str(slp))
        used = {id(inst.skeleton) for lf in labels for inst in lf.instances}
        skels = [s for s in labels.skeletons if id(s) in used] or labels.skeletons[:1]
        if len(skels) != 1:
            sys.exit(f"{slp}: {len(skels)} skeletons are in use; convert one skeleton at a time.")
        sk = skels[0]
        sk_names = [n.name for n in sk.nodes]
        if names is None:
            names = sk_names
            edges = [list(e) for e in sk.edge_names]
            syms = [sorted(s) for s in getattr(sk, "symmetry_names", [])]
        elif sk_names != names:
            sys.exit(f"{slp}: skeleton nodes {sk_names} differ from the first file's {names}.")

        # Only videos with frames to convert are looked for: a SLEAP project often lists videos
        # nobody labelled, and those may not exist on this machine. Each is resolved once.
        by_video: dict[int, tuple] = {}
        for lf in labels:
            by_video.setdefault(id(lf.video), (lf.video, []))[1].append(lf)
        n_placeholder = 0
        for video, lfs in by_video.values():
            if not any(lf.user_instances for lf in lfs) and not tracked:
                continue
            fn = video.filename
            if isinstance(fn, list) or str(fn).endswith((".slp", ".h5", ".hdf5", ".pkg")):
                report.append(f"skipped {slp.name}: video {str(fn)[:80]!r} is an image list or "
                              f"embedded frames, not a video file")
                continue
            path = resolve_video(str(fn), slp, replace, search)
            if path is None:
                sys.exit(f"{slp}: cannot find video {fn!r}. Pass --replace OLD NEW to rewrite its "
                         f"prefix, or --video-dir DIR to look for {_name_of(str(fn))!r} there.")
            if path not in sources:
                sources[path] = Source(path, slp)
            elif slp not in sources[path].slp_paths:
                sources[path].slp_paths.append(slp)
            src = sources[path]
            for lf in lfs:
                n_placeholder += _add_frame(src, lf, path, tracked, ignore_tracks)
        if n_placeholder:
            report.append(f"{slp.name}: dropped {n_placeholder} placeholder user instance(s) "
                          f"with no visible points")
    return sources, names, edges, syms, report


def _add_frame(src: Source, lf, path: Path, tracked: bool, ignore_tracks: bool) -> int:
    """Record one labelled frame's instances on its Source. Returns placeholders dropped."""
    fi = int(lf.frame_idx)
    if tracked:
        insts = [i for i in lf.instances if i.track is not None
                 and np.isfinite(_xy_vis(i)[0]).all(axis=1).any()]
        if insts:
            src.frames.setdefault(fi, {"tracked": []})["tracked"].extend(insts)
        return 0
    users, n_placeholder = [], 0
    for inst in lf.user_instances:
        xy, vis = _xy_vis(inst)
        if not (vis & np.isfinite(xy).all(axis=1)).any():
            n_placeholder += 1
            continue
        users.append(inst)
    if not users:
        return n_placeholder
    if fi in src.frames:
        sys.exit(f"{path.name} frame {fi} is labelled in more than one .slp "
                 f"({', '.join(p.name for p in src.slp_paths)}); merge them first.")
    # Boxes from the visible points only: an occluded point's stored position is stale.
    boxes = [_extent(_labelled_points(u)[0]) for u in users]
    made_from = {id(u.from_predicted) for u in users if u.from_predicted is not None}
    leftover = []
    for p in lf.predicted_instances:
        if id(p) in made_from:
            continue
        pb = _extent(_xy_vis(p)[0])
        if any(_iou(pb, b) >= SAME_ANIMAL_IOU for b in boxes):
            continue
        leftover.append(pb)
    ids = _user_ids(users, fi, ignore_tracks)
    src.frames[fi] = {"users": users, "ids": ids, "leftover": leftover}
    return n_placeholder


def _user_ids(users, fi: int, ignore_tracks: bool) -> list[str]:
    """animal_ids for one frame's user instances: track names when usable, else per-frame ids."""
    per_frame = [f"f{fi:07d}_{j}" for j in range(len(users))]
    if ignore_tracks:
        return per_frame
    ids = [u.track.name if u.track is not None and u.track.name else None for u in users]
    out, seen = [], set()
    for j, t in enumerate(ids):
        # Untracked, or a track used twice in one frame: that instance gets a per-frame id.
        # (A track named like a per-frame id would collide with one, so it is not used either.)
        if t is None or t in seen or re.fullmatch(r"f\d{7}_p?\d+", t):
            out.append(per_frame[j])
        else:
            out.append(t)
            seen.add(t)
    return out


# ------------------------------------------------------------------------------------------------
# writing sessions


def _probe(path: Path):
    """(reader, n_frames, (width, height), fps) for one video, via tailcyclenet's own decoder."""
    from tailcyclenet.data.video import PyAVReader

    reader = PyAVReader(str(path))
    h, w = reader.frame_shape()[:2]
    return reader, len(reader), (int(w), int(h)), float(reader.fps)


def _clusters(frames: list[int], window: int) -> list[list[int]]:
    """Greedily pack labelled frames whose inclusive span fits in one window."""
    out: list[list[int]] = []
    for f in sorted(frames):
        if not out or f - out[-1][0] >= window:
            out.append([f])
        else:
            out[-1].append(f)
    return out


def _write_jpegs(reader, start: int, n: int, dest: Path, quality: int) -> None:
    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)
    frames = reader.get_batch(range(start, start + n))
    for i, rgb in enumerate(frames):
        Image.fromarray(np.ascontiguousarray(rgb)).save(dest / f"{i:06d}.jpg", quality=quality)


def convert_annotated(src: Source, session_dir: Path, names, args, provenance) -> dict:
    """One video's user labels -> one session of short JPEG clips. Returns counts."""
    from tailcyclenet import format as fmt

    reader, n_video, size, fps = _probe(src.video_path)
    K = len(names)
    labelled = sorted(src.frames)
    bad = [f for f in labelled if f >= n_video]
    if bad:
        sys.exit(f"{src.video_path}: labelled frame(s) {bad[:5]} are past the decoded length "
                 f"{n_video}. The .slp and this video disagree.")
    W = min(args.context, n_video)

    # Does this video need instances.pq (an unlabelled animal on some labelled frame)?
    def incomplete(fi):
        fr = src.frames[fi]
        n = len(fr["users"]) + len(fr["leftover"])
        return bool(fr["leftover"]) or (args.animals_per_frame and n < args.animals_per_frame)

    need_instances = any(incomplete(f) for f in labelled)

    groups, group_labels = {}, {}
    n_rows = 0
    for cluster in _clusters(labelled, W):
        # Centred on the cluster, but always covering all of it (a cluster spans up to W frames,
        # and centre - W//2 alone can leave its last frame out), and inside the video.
        centre = (cluster[0] + cluster[-1]) // 2
        start = min(max(centre - W // 2, cluster[-1] - W + 1), cluster[0])
        start = int(min(max(0, start), n_video - W))
        inside = [f for f in labelled if start <= f < start + W]
        gid = f"g{start:07d}"
        animals: list[str] = []
        for f in inside:
            fr = src.frames[f]
            animals += [a for a in fr["ids"] if a not in animals]
            animals += [f"f{f:07d}_p{j}" for j in range(len(fr["leftover"]))]
            if args.animals_per_frame and incomplete(f) and INCOMPLETE_ID not in animals:
                animals.append(INCOMPLETE_ID)
        S = len(animals)
        aidx = {a: i for i, a in enumerate(animals)}
        p2d = np.full((S, W, K, 1, 2), np.nan, dtype=np.float32)
        v2d = np.full((S, W, K, 1), fmt.UNLABELED, dtype=np.int8)
        boxes = np.full((S, W, 1, 4), np.nan, dtype=np.float32) if need_instances else None
        inst = np.full((S, W, 1), fmt.INST_NONE, dtype=np.int8) if need_instances else None
        for f in inside:
            t = f - start
            fr = src.frames[f]
            for u, a in zip(fr["users"], fr["ids"]):
                xy, status = _labelled_points(u)
                s = aidx[a]
                p2d[s, t, :, 0] = xy
                v2d[s, t, :, 0] = status
                n_rows += int((status != fmt.UNLABELED).sum())
                if need_instances:
                    box = _extent(xy)
                    ok = box is not None and box[2] > box[0] and box[3] > box[1]
                    inst[s, t, 0] = fmt.INST_LABELED if ok else fmt.INST_PRESENT
                    if box is not None:
                        boxes[s, t, 0] = box
            for j, pb in enumerate(fr["leftover"]):
                s = aidx[f"f{f:07d}_p{j}"]
                inst[s, t, 0] = fmt.INST_PRESENT
                if pb is not None:
                    boxes[s, t, 0] = pb
            if args.animals_per_frame and incomplete(f) and not fr["leftover"]:
                inst[aidx[INCOMPLETE_ID], t, 0] = fmt.INST_PRESENT
        groups[gid] = fmt.Group(
            gid, W, fps=fps, source_video=str(src.video_path), source_frame_start=start,
            source_frame_step=1,
            notes=f"{W} frames of context; labelled source frames {', '.join(map(str, inside))}")
        group_labels[gid] = fmt.Labels(animal_ids=animals, points3d=None, vis3d=None,
                                       points2d=p2d, vis2d=v2d, boxes=boxes, instance=inst)
        _write_jpegs(reader, start, W, session_dir / "groups" / gid / CAM, args.jpeg_quality)
    reader.close()

    fmt.write_session(
        session_dir, mode="2d", units="px", label_source="annotated", names=names,
        rig=fmt.nominal_rig(CAM, size), groups=groups, labels=group_labels,
        skeleton=provenance.pop("_edges"), flip_pairs=provenance.pop("_flip_pairs"),
        provenance=provenance)
    return {"frames": len(labelled), "groups": len(groups), "rows": n_rows,
            "incomplete": sum(bool(incomplete(f)) for f in labelled)}


def convert_tracked(src: Source, session_dir: Path, names, args, provenance) -> dict:
    """One video's tracked instances -> one dense group over the whole (symlinked) video."""
    from tailcyclenet import format as fmt

    ext = src.video_path.suffix.lower()
    if ext not in fmt.VIDEO_EXTS:
        sys.exit(f"{src.video_path}: tailcyclenet reads {', '.join(fmt.VIDEO_EXTS)} videos, not "
                 f"{ext}. Transcode it (e.g. ffmpeg -i in{ext} -c:v libx264 -crf 18 out.mp4).")
    reader, T, size, fps = _probe(src.video_path)
    reader.close()
    K = len(names)
    tracks: list[str] = []
    for fi in sorted(src.frames):
        for inst in src.frames[fi]["tracked"]:
            if inst.track.name not in tracks:
                tracks.append(inst.track.name)
    S = len(tracks)
    gb = S * T * K * 9 / 1e9
    if gb > 4:
        print(f"  note: {S} tracks x {T} frames x {K} keypoints is ~{gb:.1f} GB in memory")
    aidx = {a: i for i, a in enumerate(tracks)}
    p2d = np.full((S, T, K, 1, 2), np.nan, dtype=np.float32)
    v2d = np.full((S, T, K, 1), fmt.UNLABELED, dtype=np.int8)
    best = np.full((S, T), -np.inf)
    past_end = 0
    for fi, fr in src.frames.items():
        if fi >= T:
            past_end += 1
            continue
        for inst in fr["tracked"]:
            s = aidx[inst.track.name]
            # A user instance beats a prediction; between predictions the higher score wins.
            rank = np.inf if not hasattr(inst, "score") else float(inst.score or 0.0)
            if rank <= best[s, fi]:
                continue
            best[s, fi] = rank
            xy, _ = _xy_vis(inst)
            ok = np.isfinite(xy).all(axis=1)
            p2d[s, fi, :, 0] = np.where(ok[:, None], xy, np.nan)
            # A machine point is a position, not an occlusion judgement: NaN -> no row.
            v2d[s, fi, :, 0] = np.where(ok, fmt.VISIBLE, fmt.UNLABELED)
    if past_end:
        print(f"  warning: {past_end} tracked frame(s) are past the decoded length {T}; dropped")
    gid = _slug(src.video_path.stem)
    groups = {gid: fmt.Group(gid, T, fps=fps, source_video=str(src.video_path),
                             notes="whole recording; SLEAP tracking")}
    labels = {gid: fmt.Labels(animal_ids=tracks, points3d=None, vis3d=None, points2d=p2d,
                              vis2d=v2d, boxes=None, instance=None)}
    gdir = session_dir / "groups" / gid
    gdir.mkdir(parents=True, exist_ok=True)
    fmt.link(gdir / f"{CAM}{ext}", src.video_path)
    fmt.write_session(
        session_dir, mode="2d", units="px", label_source="tracked", names=names,
        rig=fmt.nominal_rig(CAM, size), groups=groups, labels=labels,
        skeleton=provenance.pop("_edges"), flip_pairs=provenance.pop("_flip_pairs"),
        provenance=provenance)
    return {"frames": len(src.frames), "groups": 1, "rows": int((v2d != fmt.UNLABELED).sum()),
            "tracks": S}


# ------------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slp", type=Path, nargs="+", help="one or more .slp files")
    ap.add_argument("--out", type=Path, required=True, help="dataset root to write")
    ap.add_argument("--tracked", action="store_true",
                    help="convert tracked instances over whole videos instead of user labels")
    ap.add_argument("--context", type=int, default=24,
                    help="frames per clip around labelled frames (default 24; tailcyclenet trains "
                         "on windows of up to 12, placed around the labels)")
    ap.add_argument("--val-fraction", type=float, default=None,
                    help="fraction of videos put in val/ (default 0.1 for labels, 0 for --tracked)")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"],
                    help="split for the videos not chosen for val (default train)")
    ap.add_argument("--animals-per-frame", type=int, default=0,
                    help="if set, a labelled frame with fewer labelled animals than this is "
                         "marked incompletely labelled")
    ap.add_argument("--ignore-tracks", action="store_true",
                    help="never use SLEAP tracks as identities for user labels")
    ap.add_argument("--flip-pair", nargs=2, action="append", metavar=("LEFT", "RIGHT"),
                    help="declare a left/right keypoint pair (overrides the skeleton symmetries)")
    ap.add_argument("--replace", nargs=2, action="append", default=[], metavar=("OLD", "NEW"),
                    help="rewrite a video path prefix stored in the .slp")
    ap.add_argument("--video-dir", type=Path, action="append", default=[],
                    help="folder to look in for a video, by file name")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--clean", action="store_true", help="delete --out first if it exists")
    args = ap.parse_args(argv)

    _need("tailcyclenet.format", "Run this in the environment tailcyclenet is installed in.")
    from tailcyclenet import format as fmt

    for p in args.slp:
        if not p.is_file():
            sys.exit(f"{p}: no such file")
    if args.out.exists() and any(args.out.iterdir()):
        if not args.clean:
            sys.exit(f"{args.out} exists and is not empty; pass --clean to replace it")
        shutil.rmtree(args.out)
    val_fraction = args.val_fraction if args.val_fraction is not None else (
        0.0 if args.tracked else 0.1)

    sources, names, edges, syms, report = collect(
        [p.resolve() for p in args.slp], args.replace, [d.resolve() for d in args.video_dir],
        args.tracked, args.ignore_tracks)
    for line in report:
        print(line)
    sources = {k: v for k, v in sources.items() if v.frames}
    if not sources:
        what = "tracked instances" if args.tracked else "user-labelled frames"
        sys.exit(f"no {what} found in {', '.join(map(str, args.slp))}"
                 + ("" if args.tracked else " (for SLEAP predictions, pass --tracked)"))

    flip_pairs = [list(p) for p in args.flip_pair] if args.flip_pair else (syms or None)
    if flip_pairs:
        unknown = sorted({n for p in flip_pairs for n in p} - set(names))
        if unknown:
            sys.exit(f"--flip-pair names not in the skeleton: {unknown}")

    # Val: the videos with the smallest stable hash, at least one if any are asked for.
    order = sorted(sources, key=lambda p: _stable_unit(str(p)))
    n_val = int(round(val_fraction * len(order)))
    if val_fraction > 0 and len(order) > 1:
        n_val = min(max(1, n_val), len(order) - 1)
    elif len(order) == 1:
        n_val = 0
    val = set(order[:n_val])

    used_ids: set[str] = set()
    totals = defaultdict(int)
    for i, path in enumerate(sorted(sources)):
        src = sources[path]
        sid = f"v{i:03d}_{_slug(path.stem)}"
        assert sid not in used_ids
        used_ids.add(sid)
        split = "val" if path in val else args.split
        session_dir = args.out / split / sid
        provenance = {
            "source": ", ".join(str(p) for p in src.slp_paths),
            "source_video": str(path),
            "annotator": "",
            "annotator_tool": "SLEAP" + (" (tracking)" if args.tracked else ""),
            "converter": "laras-labeler scripts/slp_to_tailcycle.py",
            "created": _dt.date.today().isoformat(),
            "_edges": edges,
            "_flip_pairs": flip_pairs,
        }
        convert = convert_tracked if args.tracked else convert_annotated
        counts = convert(src, session_dir, names, args, provenance)
        for k, v in counts.items():
            totals[k] += v
        extra = (f", {counts['tracks']} tracks" if args.tracked
                 else f", {counts['incomplete']} incompletely labelled")
        print(f"{split}/{sid}: {counts['frames']} frames -> {counts['groups']} group(s), "
              f"{counts['rows']} keypoint rows{extra}")

    print(f"\n{len(sources)} video(s), {totals['groups']} group(s), {totals['rows']} keypoint rows "
          f"-> {args.out}  ({len(val)} video(s) in val/)")
    print(f"keypoints: {names}")
    print(f"flip_pairs: {flip_pairs if flip_pairs else 'none declared (flip augmentation refused)'}")
    problems = fmt.validate_dataset(fmt.load_dataset(args.out))
    warnings = [p for p in problems if "WARNING" in p]
    errors = [p for p in problems if "WARNING" not in p]
    if warnings:
        print("\ntailcyclenet validation warnings:\n  " + "\n  ".join(warnings))
    if errors:
        print("\ntailcyclenet validation FAILED:\n  " + "\n  ".join(errors))
        return 1
    print("tailcyclenet validation: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
