#!/usr/bin/env python3
"""Copy each clip's .slp pose file into the project so labeling works with NO server/network drive.

Opening a clip reads its poses/skeleton from ``slp_path`` (video.py: OpenVideo.source = slp_path or
video_path); playback reads ``playback_path`` (the local proxy from make_playback_proxies.py); clip
metadata (fps/width/height/n_frames) is already in project.json; and your labels + timing live in the
project. So the ONLY thing still read from the server while hand-labeling is the .slp. This copies
each .slp next to the project and re-points ``slp_path`` at the local copy.

Safe: the copy is byte-identical, and your labels are stored in the project keyed by clip (not in the
source .slp), so nothing is lost. ``video_path`` is left as-is — it isn't read while labeling (only
HiDRA/feature jobs use it, and those you run on the server anyway).

    # Run this WHILE you still have the server connection (it needs to read the .slp to copy it).
    # STOP the labeler first (it rewrites project.json on save), then:
    python scripts/localize_poses.py C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-test
    # ...restart the labeler. You can then label with the network drive disconnected.

Re-runnable: a clip already pointing at a local copy is skipped unless --force. A clip whose .slp
isn't reachable right now is skipped with a note. project.json is backed up to project.json.bak.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Localize each clip's .slp so labeling works offline.")
    ap.add_argument("project", help="the project directory (contains project.json)")
    ap.add_argument("--dest", default=None, help="output dir (default: <project>/poses)")
    ap.add_argument("--force", action="store_true", help="re-copy even if slp_path is already local")
    args = ap.parse_args(argv)

    proj = Path(args.project).expanduser()
    pj = proj / "project.json"
    if not pj.exists():
        sys.exit(f"no project.json in {proj} (point me at the project directory)")
    manifest = json.loads(pj.read_text(encoding="utf-8"))
    videos = manifest.get("videos", [])
    if not videos:
        sys.exit("this project has no videos")

    dest = Path(args.dest).expanduser() if args.dest else proj / "poses"
    dest.mkdir(parents=True, exist_ok=True)

    bak = pj.with_suffix(".json.bak")
    if not bak.exists():
        shutil.copy2(pj, bak)
        print(f"backed up project.json -> {bak.name}")

    changed = 0
    for v in videos:
        vid = v.get("video_id", "?")
        src = v.get("slp_path")
        if not src:
            print(f"SKIP {vid}: no slp_path (no poses recorded for this clip)")
            continue
        srcp = Path(src)
        out = dest / srcp.name

        # already local under this project (and present)? then it's done.
        already_local = False
        try:
            already_local = out.exists() and Path(v["slp_path"]).resolve() == out.resolve()
        except OSError:
            already_local = False
        if already_local and not args.force:
            print(f"ok   {vid}: poses already local")
            continue

        if not srcp.exists():
            print(f"SKIP {vid}: .slp not reachable right now ({src}) — run this while connected")
            continue

        try:
            shutil.copy2(srcp, out)
        except OSError as e:
            print(f"  FAILED to copy {vid}: {e} — left on the original slp_path")
            continue

        # sanity: same size as the source
        if out.stat().st_size != srcp.stat().st_size:
            print(f"  SIZE MISMATCH copying {vid} — NOT re-pointing (left on original)")
            try:
                out.unlink()
            except OSError:
                pass
            continue

        v["slp_path"] = str(out)
        changed += 1
        print(f"  poses -> {out}  ({out.stat().st_size // 1024} KB)")

    if changed:
        pj.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nupdated {pj} for {changed} clip(s).")
        print("RESTART the labeler. You can now label with the server/network drive disconnected.")
    else:
        print("\nno changes written.")


if __name__ == "__main__":
    main()
