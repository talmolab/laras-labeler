#!/usr/bin/env python3
"""Why a clip has no wall / spout / feeder outlines: walk the HCM ROI lookup one step at a time.

Every step of the lookup fails silently by design (video loading must never wait on the DB), so when
outlines don't appear nothing says why. This prints each step for the clips whose id contains --clip:

  setup     the labeler code, HCM_DATABASE_URL, the sqlalchemy + Postgres driver the lookup needs
  saved     the outlines the clip has, and whether its recording was already looked up
  link      a re-ID clip's entry in the planner's clips.summary.json (its name doesn't say)
  camera    the camera and recording pattern the lookup will use
  database  for the first clip: connect, the recording's ROI status, recordings with a similar path
            (when the pattern finds none), and the lookup's own result

Read-only, except --repair: a project.json torn by two saves at once (project.save is atomic now)
still holds the newest complete manifest at its start; --repair backs the file up and writes that.

    uv run --with sqlalchemy --with "psycopg[binary]" python scripts/check_hcm_roi.py --clip b21
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from laras_labeler import hcm_roi
from laras_labeler.project import write_json_atomic


def say(k: str, v) -> None:
    print(f"{k:<28} {v}")


def load(pj: Path, repair: bool) -> dict | None:
    text = pj.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        manifest, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as e:
        say(f"{pj.parent.name}", f"project.json does NOT parse and can't be recovered here ({e})")
        return None
    if not repair:
        say(f"{pj.parent.name}", f"project.json is TORN ({len(text) - end} stray bytes after the manifest) -- "
                                 "the labeler can't open it; rerun with --repair")
        return manifest
    backup = pj.with_name(f"project.json.torn-{time.strftime('%Y%m%d-%H%M%S')}")
    backup.write_text(text)
    write_json_atomic(pj, manifest)
    say(f"{pj.parent.name}", f"project.json REPAIRED (the torn file is kept as {backup.name})")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", nargs="?", default="~/laras-projects", help="projects folder")
    ap.add_argument("--clip", default="", help="only clips whose id contains this (e.g. b21)")
    ap.add_argument("--repair", action="store_true", help="repair a torn project.json (keeps a backup)")
    a = ap.parse_args()

    say("labeler has feeder code", hasattr(hcm_roi, "source_recording"))
    url = os.environ.get("HCM_DATABASE_URL", "")
    say("HCM_DATABASE_URL set", bool(url))
    have = {}
    for mod in ("sqlalchemy", "psycopg", "psycopg2"):
        try:
            __import__(mod)
            have[mod] = True
        except Exception:  # noqa: BLE001
            have[mod] = False
    say("sqlalchemy", "installed" if have["sqlalchemy"] else "MISSING")
    say("postgres driver", "psycopg" if have["psycopg"] else "psycopg2" if have["psycopg2"] else "MISSING")
    if not (have["sqlalchemy"] and (have["psycopg"] or have["psycopg2"])):
        say("", 'start with: uv run --with sqlalchemy --with "psycopg[binary]" ...')

    clips = []
    for pj in sorted(Path(a.root).expanduser().glob("*/project.json")):
        m = load(pj, a.repair)
        clips += [(pj.parent.name, v) for v in (m or {}).get("videos", []) if a.clip in v["video_id"]]
    say("matching clips", len(clips))

    for name, v in clips[:3]:
        print(f"\n{name} / {v['video_id']}")
        say("  video", v.get("video_path"))
        say("  slp", v.get("slp_path"))
        say("  saved outlines", [k for k in hcm_roi.ROI_FIELDS if v.get(k)] or "none")
        say("  recording looked up", repr(v["roi_recording_status"]) if "roi_recording_status" in v else "not yet")
        say("  .slp's own video path", hcm_roi._slp_video_path(v["slp_path"]) if v.get("slp_path") else None)
        src = hcm_roi.source_recording(v)
        say("  clips.summary.json", src or ("not needed (id names its recording)" if hcm_roi.recording_like(v["video_id"])
                                            else "NOT FOUND next to the clip, its .slp, or the video its .slp names"))
        sv = (src or {}).get("video") or v.get("source_video")
        say("  camera", hcm_roi.camera_of(sv or "") or hcm_roi.camera_of(v["video_id"]))
        say("  recording pattern", hcm_roi.recording_like(v["video_id"]) or hcm_roi.recording_like_path(sv))
    if len(clips) > 3:
        print(f"\n  ... {len(clips) - 3} more")

    if not clips or not url:
        return
    try:
        from sqlalchemy import create_engine, text
    except Exception:  # noqa: BLE001
        return
    _, v = clips[0]
    sv = (hcm_roi.source_recording(v) or {}).get("video") or v.get("source_video")
    pat = hcm_roi.recording_like(v["video_id"]) or hcm_roi.recording_like_path(sv)
    print(f"\ndatabase ({v['video_id']})")
    try:
        with create_engine(hcm_roi.engine_url(url), connect_args={"connect_timeout": 8}).connect() as c:
            say("  connected", True)
            status = c.execute(text(hcm_roi._STATUS_SQL), {"pat": pat}).scalar() if pat else None
            say("  recording ROI status", status if pat else "no recording pattern")
            m = re.search(r"(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}-\d{2}/(cam_\d+)\.(\d+)", pat or "")
            if m and status is None:
                rows = c.execute(text("SELECT video_path, rois->>'status' FROM recordings WHERE video_path LIKE :p "
                                      "LIMIT 3"), {"p": f"%{m.group(1)}%{m.group(2)}.{m.group(3)}.%"}).all()
                say("  similar recordings", [tuple(r) for r in rows] or "none")
            say("  cameras in batches", c.execute(text("SELECT DISTINCT cage FROM batches ORDER BY cage")).scalars().all())
        got = hcm_roi.fetch_rois(v["video_id"], source_video=sv)
        say("  lookup result", {k: (f"{len(x)} pts" if isinstance(x, list) else x) for k, x in got.items()}
            or "{} (nothing)")
    except Exception as e:  # noqa: BLE001
        say("  DB error", f"{e.__class__.__name__}: {str(e).splitlines()[0][:200]}")


if __name__ == "__main__":
    main()
