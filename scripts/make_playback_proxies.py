#!/usr/bin/env python3
"""Give each clip a smooth all-intra PLAYBACK PROXY, without touching the analysis video.

The clips are encoded H.264 with very long gaps between keyframes (low bitrate, heavy inter-frame
compression). The browser decodes each frame from the previous keyframe, so at high fps playback
can't keep up and skips ~half a second at a time — which also inflates hand-labeling time. Re-encoding
to all-intra (every frame a keyframe) makes each frame decode independently and instantly.

The labeler serves playback from ``playback_path or video_path`` (app.py: stream_video). So this
writes an all-intra copy next to the project and sets each clip's ``playback_path`` to it. Nothing
else changes: ``video_path`` / ``slp_path`` — what poses, features, HiDRA and the .slp all read — are
untouched, and the proxy is frame-exact (same count, same order), so every label stays aligned.

    # STOP the labeler first (it rewrites project.json on save and would clobber these edits), then:
    python scripts/make_playback_proxies.py C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-test
    # ...restart the labeler and hard-refresh the tab (Ctrl+F5).

Re-runnable: a clip that already has a valid proxy is skipped unless --force. A clip whose source
isn't reachable right now (e.g. on a disconnected network drive) is skipped with a note — re-run
when it's back. project.json is backed up to project.json.bak on the first run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _nb_packets(path: str) -> int | None:
    """Frame count via demux only (fast; for all-intra, packets == frames)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
            capture_output=True, text=True)
        return int(out.stdout.strip())
    except (ValueError, OSError):
        return None


def _encode(src: str, out: Path, crf: int, encoder: str) -> bool:
    nvenc = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-an",
             "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "constqp", "-qp", str(crf),
             "-g", "1", "-bf", "0",                       # all-intra: keyframe every frame, no B-frames
             "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", str(out)]
    x264 = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-x264-params", "keyint=1:min-keyint=1:scenecut=0",
            "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", str(out)]
    chain = {"auto": [nvenc, x264], "nvenc": [nvenc], "libx264": [x264]}[encoder]
    for i, cmd in enumerate(chain):
        r = subprocess.run(cmd)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return True
        if i + 1 < len(chain):
            print("    encoder failed — trying the next one…")
    return False


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build all-intra playback proxies for a project's clips.")
    ap.add_argument("project", help="the project directory (contains project.json)")
    ap.add_argument("--proxies", default=None, help="output dir (default: <project>/proxies)")
    ap.add_argument("--crf", type=int, default=20, help="quality, lower=better/bigger (default 20)")
    ap.add_argument("--encoder", choices=["auto", "nvenc", "libx264"], default="auto",
                    help="auto tries GPU (nvenc) then CPU (libx264)")
    ap.add_argument("--force", action="store_true", help="re-encode even if a valid proxy exists")
    args = ap.parse_args(argv)

    proj = Path(args.project).expanduser()
    pj = proj / "project.json"
    if not pj.exists():
        sys.exit(f"no project.json in {proj} (point me at the project directory)")
    manifest = json.loads(pj.read_text(encoding="utf-8"))
    videos = manifest.get("videos", [])
    if not videos:
        sys.exit("this project has no videos")

    proxies = Path(args.proxies).expanduser() if args.proxies else proj / "proxies"
    proxies.mkdir(parents=True, exist_ok=True)

    bak = pj.with_suffix(".json.bak")
    if not bak.exists():
        shutil.copy2(pj, bak)
        print(f"backed up project.json -> {bak.name}")

    changed = 0
    for v in videos:
        vid = v.get("video_id", "?")
        src = v.get("video_path")
        if not src or not Path(src).exists():
            print(f"SKIP {vid}: source not reachable right now ({src}) — re-run when it's available")
            continue
        out = proxies / (Path(src).stem + ".intra.mp4")

        if not args.force and v.get("playback_path") == str(out) and out.exists():
            print(f"ok   {vid}: proxy already set")
            continue

        print(f"encoding {vid} …")
        if not _encode(src, out, args.crf, args.encoder):
            print(f"  FAILED to encode {vid} — left on the original video (playback unchanged)")
            continue

        # Frame-exactness guard: a proxy with a different frame count would misalign every label,
        # so refuse to point at it. (Demux-only count; fast.)
        ns, no = _nb_packets(src), _nb_packets(str(out))
        if ns is not None and no is not None and ns != no:
            print(f"  FRAME COUNT MISMATCH ({ns} src vs {no} proxy) — NOT using this proxy for {vid}")
            continue

        v["playback_path"] = str(out)
        changed += 1
        print(f"  proxy -> {out}  ({no if no is not None else '?'} frames)")

    if changed:
        pj.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nupdated {pj} for {changed} clip(s).")
        print("RESTART the labeler, then hard-refresh the tab (Ctrl+F5). Playback now uses the proxies;")
        print("poses, features, HiDRA and the .slp still read the original video_path.")
    else:
        print("\nno changes written.")


if __name__ == "__main__":
    main()
