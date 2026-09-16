#!/usr/bin/env python3
"""Clone a project's clips + behaviors into another project, WITHOUT its labels — for the HITL arm.

The HITL arm needs the SAME clips as the hand arm but a clean slate (no labels), so HiDRA's
predictions are genuine and review time is measured against unlabeled data. This copies the source
project's manifest (its 6 clips — with the already-localized proxies/poses — plus behavior
definitions, fps, px/cm, feature config, media roots) into the destination project.json, and moves
the destination's existing labels/ and events/ aside so it starts empty.

    # STOP the labeler first (it rewrites project.json on save), then:
    python scripts/clone_project.py --from C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-test \\
                                    --to   C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-hitl
    # ...restart the labeler; the HITL project now has the same 6 clips and NO labels.

Labels live in labels/<vid>.parquet (not in project.json), so copying the manifest never copies
labels. The destination's old labels/events are renamed to *.bak-N, not deleted. Pure stdlib.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path


def _rename_aside(d: Path) -> str | None:
    if not d.exists():
        return None
    for n in range(1, 100):
        dst = d.with_name(f"{d.name}.bak-{n}")
        if not dst.exists():
            d.rename(dst)
            return dst.name
    raise RuntimeError(f"too many backups of {d}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Clone clips+behaviors into another project, minus labels.")
    ap.add_argument("--from", dest="src", required=True, help="source project dir (the hand arm)")
    ap.add_argument("--to", dest="dst", required=True, help="destination project dir (the HITL arm)")
    ap.add_argument("--name", default=None, help="display name for the destination (default: keep its own)")
    ap.add_argument("--keep-behaviors", action="store_true",
                    help="keep the destination's own behavior definitions instead of copying the source's")
    args = ap.parse_args(argv)

    src, dst = Path(args.src).expanduser(), Path(args.dst).expanduser()
    src_pj, dst_pj = src / "project.json", dst / "project.json"
    if not src_pj.exists():
        sys.exit(f"no project.json in source {src}")
    if not dst.exists():
        sys.exit(f"destination {dst} does not exist — create the project in the labeler first, then re-run")

    src_m = json.loads(src_pj.read_text(encoding="utf-8"))
    dst_m = json.loads(dst_pj.read_text(encoding="utf-8")) if dst_pj.exists() else {}

    new = copy.deepcopy(src_m)
    # keep the destination's identity
    new["name"] = args.name or dst_m.get("name") or dst.name
    if args.keep_behaviors and dst_m.get("behaviors"):
        new["behaviors"] = dst_m["behaviors"]

    # back up destination project.json, then write the cloned manifest
    if dst_pj.exists():
        shutil.copy2(dst_pj, dst_pj.with_suffix(".json.bak"))
    dst_pj.write_text(json.dumps(new, indent=2), encoding="utf-8")

    # move any existing labels/events aside so the HITL arm starts clean
    moved = []
    for sub in ("labels", "events"):
        r = _rename_aside(dst / sub)
        if r:
            moved.append(r)

    vids = [v.get("video_id", "?") for v in new.get("videos", [])]
    behs = [b.get("name") for b in new.get("behaviors", [])]
    print(f"cloned into {dst_pj}")
    print(f"  clips ({len(vids)}): " + ", ".join(v[:24] for v in vids))
    print(f"  behaviors: " + (", ".join(str(b) for b in behs) or "(none)"))
    if moved:
        print(f"  moved aside (destination started clean): {', '.join(moved)}")
    print("\nRESTART the labeler. The HITL project now has the same clips, no labels — ready for "
          "zero-shot Predict + review. (Media proxies/poses are shared with the source project.)")


if __name__ == "__main__":
    main()
