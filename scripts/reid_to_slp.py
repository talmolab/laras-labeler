#!/usr/bin/env python3
"""Turn a Tailprint re-ID identity table into a .slp whose tracks are the same mouse in every file.

The re-ID pipeline (talmolab/tailreid, Tom Han's `26-09-reid-v2`) does not write new .slp files. For
every recording hour it stores `<stem>.ident.npz` in a batch's `ident_tables/` folder:

    ident       (n_frames, n_slp_tracks) int8   global mouse id of that SLP track in that frame, -1 = none
    lane_ident  (n_frames, n_lanes) int8        the decoder's per-lane identities (not used here)
    tracks      (n_slp_tracks,) str             SLP track names, in the column order of `ident`
    meta        JSON: stem, video, tracked_predictions (.slp), fps, age_days, batch, n_mice, ...

This script reads the original tracked .slp named in `meta`, re-assigns every instance to the track
`mouse_<id>` given by `ident[frame, column of its old track]`, and writes `<stem>.reid.slp`. Tracks are
written in the order mouse_0, mouse_1, ... so the labeler's track 1 is mouse_0 in every file, and the
same number means the same animal across hours and days. Poses are untouched; only `.track` changes.
An instance whose identity is -1 is written untracked, which the labeler does not display (it reads
tracked instances only), so the summary reports how many there are.

Optionally `--clip START END` also cuts frames [START, END) into `<stem>.f<START>-<END>.mp4` plus a
matching `.reid.slp` with frame indices shifted to 0, which is how the labeler's existing clips are
made. The cut re-encodes (frame-exact select filter), so it needs ffmpeg on PATH.

Paths inside `meta` are the server's (`/mnt/vast/...`); `--path-map` rewrites the prefix for wherever
the share is mounted. On a Mac with the talmo share connected that is the default:
`/mnt/vast=/Volumes/talmo` (so `/mnt/vast/than/...` -> `/Volumes/talmo/than/...`).

    # one hour
    uv run --with sleap-io python scripts/reid_to_slp.py \\
        /Volumes/talmo/than/26-09-reid-v2/studies/reid_remaining_20261007/b21/ident_tables/b21_20260608_h16.ident.npz \\
        --out ~/reid_b21

    # every hour of one day, plus a 5-min clip from each
    uv run --with sleap-io python scripts/reid_to_slp.py \\
        /Volumes/talmo/than/26-09-reid-v2/studies/reid_remaining_20261007/b21/ident_tables \\
        --day 2026-06-08 --out ~/reid_b21 --clip 0 15000
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def remap(path: str, path_map: list[tuple[str, str]]) -> str:
    for old, new in path_map:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def load_table(npz: Path) -> dict:
    d = np.load(npz, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    return {"ident": d["ident"], "tracks": [str(t) for t in d["tracks"]], "meta": meta}


def relabel(labels, table: dict, frames: range | None = None):
    """Re-assign tracks in place from the identity table. Returns counts for the summary."""
    import sleap_io as sio

    ident, col_names = table["ident"], table["tracks"]
    n_mice = int(table["meta"].get("n_mice") or (int(ident.max()) + 1))
    new_tracks = [sio.Track(name=f"mouse_{k}") for k in range(n_mice)]
    col_of = {name: j for j, name in enumerate(col_names)}
    missing = sorted({t.name for t in labels.tracks} - set(col_of))
    if missing:
        raise SystemExit(f"{table['meta']['stem']}: SLP tracks {missing} are not in the identity table "
                         f"{col_names} - wrong .slp for this table?")

    n = {"instances": 0, "assigned": 0, "unresolved": 0, "untracked_before": 0, "beyond_table": 0}
    for lf in labels.labeled_frames:
        f = int(lf.frame_idx)
        if frames is not None and f not in frames:
            continue
        for inst in lf.instances:
            n["instances"] += 1
            if inst.track is None:
                n["untracked_before"] += 1
                continue
            if f >= ident.shape[0]:
                n["beyond_table"] += 1
                inst.track = None
                continue
            k = int(ident[f, col_of[inst.track.name]])
            if 0 <= k < n_mice:
                inst.track = new_tracks[k]
                n["assigned"] += 1
            else:
                inst.track = None
                n["unresolved"] += 1
    labels.tracks = new_tracks
    return n


def check_unique(labels) -> int:
    """Frames where two instances got the same mouse (should be 0: ident is a permutation)."""
    bad = 0
    for lf in labels.labeled_frames:
        names = [i.track.name for i in lf.instances if i.track is not None]
        bad += len(names) != len(set(names))
    return bad


def cut_video(src: str, dst: Path, start: int, end: int, fps: str) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("--clip needs ffmpeg on PATH (Mac: `brew install ffmpeg`)")
    vf = f"select=between(n\\,{start}\\,{end - 1}),setpts=N/FRAME_RATE/TB"
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-vf", vf, "-r", fps,
           "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True)


def process(npz: Path, args, path_map) -> dict:
    import sleap_io as sio

    table = load_table(npz)
    meta = table["meta"]
    stem = meta["stem"]
    slp_src = remap(meta["tracked_predictions"], path_map)
    video_src = remap(meta["video"], path_map)
    for p in (slp_src, video_src):
        if not Path(p).exists():
            raise SystemExit(f"{stem}: not found: {p}\n  (is the share mounted? check --path-map)")

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    row = {"stem": stem, "age_days": meta.get("age_days"), "video": video_src}

    labels = sio.load_slp(slp_src)
    labels.replace_filenames(new_filenames=[video_src])
    n = relabel(labels, table)
    dup = check_unique(labels)
    full = out / f"{stem}.reid.slp"
    sio.save_slp(labels, str(full))
    row.update(n, duplicate_frames=dup, slp=str(full))

    if args.clip:
        start, end = args.clip
        fps = str(meta.get("fps") or "50/1")
        clip_mp4 = out / f"{stem}.f{start}-{end}.mp4"
        clip_slp = out / f"{stem}.f{start}-{end}.reid.slp"
        cut_video(video_src, clip_mp4, start, end, fps)
        clip = sio.load_slp(str(full))
        video = sio.Video.from_filename(str(clip_mp4))
        frames = []
        for lf in clip.labeled_frames:
            if start <= lf.frame_idx < end:
                frames.append(sio.LabeledFrame(video=video, frame_idx=int(lf.frame_idx) - start,
                                               instances=lf.instances))
        clip_labels = sio.Labels(labeled_frames=frames, videos=[video], skeletons=clip.skeletons,
                                 tracks=clip.tracks)
        sio.save_slp(clip_labels, str(clip_slp))
        row.update(clip_video=str(clip_mp4), clip_slp=str(clip_slp))
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", nargs="+", help=".ident.npz files, or ident_tables/ folders")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--day", action="append", default=[], help="only this day (YYYY-MM-DD); repeatable")
    ap.add_argument("--stem", action="append", default=[], help="only this stem, e.g. b21_20260608_h16")
    ap.add_argument("--clip", nargs=2, type=int, metavar=("START", "END"),
                    help="also cut frames [START, END) into an mp4 + matching .reid.slp")
    ap.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                    help="rewrite a path prefix from the table's meta (default /mnt/vast=/Volumes/talmo "
                         "when /Volumes/talmo exists)")
    args = ap.parse_args(argv)

    path_map = [tuple(m.split("=", 1)) for m in args.path_map]
    if not path_map and Path("/Volumes/talmo").exists():
        path_map = [("/mnt/vast", "/Volumes/talmo"), ("/root/vast", "/Volumes/talmo")]

    files: list[Path] = []
    for t in args.tables:
        p = Path(t).expanduser()
        files += sorted(p.glob("*.ident.npz")) if p.is_dir() else [p]
    days = {d.replace("-", "") for d in args.day}
    if days:
        files = [f for f in files if f.name.split("_")[1] in days]
    if args.stem:
        files = [f for f in files if f.name.split(".")[0] in set(args.stem)]
    if not files:
        print("no identity tables matched", file=sys.stderr)
        return 1

    rows = []
    for i, f in enumerate(files, 1):
        row = process(f, args, path_map)
        rows.append(row)
        tot = max(row["instances"], 1)
        print(f"[{i}/{len(files)}] {row['stem']}  P{row['age_days']}  "
              f"{row['assigned'] / tot:6.1%} assigned  {row['unresolved'] / tot:5.1%} no identity  "
              f"{'DUPLICATES ' + str(row['duplicate_frames']) if row['duplicate_frames'] else ''}")
    summary = Path(args.out).expanduser() / "reid_to_slp.summary.json"
    summary.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} file(s) to {Path(args.out).expanduser()} (summary: {summary.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
