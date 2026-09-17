#!/usr/bin/env python3
"""Bundle a project's (or several projects') LABELS into one clean, shareable zip.

For each arm you name, it: (1) reconstructs bouts from the CURRENT label parquets — read via
project.json's video list, so the `.parquet.bak` backups left by clear_labels/repair are never
included (you always get the clean, latest labels); (2) writes a tidy per-bout CSV; (3) copies the
clean parquets + project.json (behavior + clip definitions) into the bundle; (4) writes a README.
Everything is zipped into --out.

    python scripts/bundle_labels.py --out doom_labels_bundle.zip ^
        "hand=C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-test" ^
        "hitl=C:\\Users\\TalmoLab\\laras-projects\\doom-annotation-time-hitl"

Each arm is `name=path`. `--value pos` (default) keeps Happening bouts; `all` includes not-happening/
unknown. Needs the labeler env python (pandas):
    & "C:\\Users\\TalmoLab\\AppData\\Roaming\\uv\\tools\\laras-labeler\\Scripts\\python.exe" scripts\\bundle_labels.py ...
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

_VALUE_NAME = {1: "happening", 0: "not-happening", 2: "unknown"}


def _bouts(df: pd.DataFrame) -> list[dict]:
    """RLE per-frame rows into bouts: a maximal run of consecutive frames sharing (track, behavior_id,
    value, source, target)."""
    out: list[dict] = []
    if df.empty:
        return out
    if "target" not in df.columns:
        df = df.assign(target=np.int16(-1))
    if "source" not in df.columns:
        df = df.assign(source="manual")
    for (track, bid), g in df.groupby(["track", "behavior_id"], sort=True):
        g = g.sort_values("frame")
        fr = g["frame"].to_numpy(); val = g["value"].to_numpy()
        src = g["source"].astype("object").to_numpy(); tgt = g["target"].to_numpy()
        brk = np.ones(len(fr), dtype=bool)
        brk[1:] = (fr[1:] != fr[:-1] + 1) | (val[1:] != val[:-1]) | (src[1:] != src[:-1]) | (tgt[1:] != tgt[:-1])
        starts = np.flatnonzero(brk); ends = np.append(starts[1:], len(fr))
        for s, e in zip(starts, ends):
            out.append({"track": int(track), "behavior_id": int(bid), "value": int(val[s]),
                        "source": str(src[s]), "target": int(tgt[s]),
                        "start": int(fr[s]), "end": int(fr[e - 1]) + 1})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Bundle project labels (clean, current) into a shareable zip.")
    ap.add_argument("arms", nargs="+", help="one or more `name=path` project arms")
    ap.add_argument("--out", default="labels_bundle.zip", help="output zip path")
    ap.add_argument("--value", default="pos", choices=["pos", "all"], help="pos (default)=Happening only; all=include 0/2")
    args = ap.parse_args(argv)

    arms = []
    for a in args.arms:
        if "=" not in a:
            sys.exit(f"arm {a!r} must be name=path")
        nm, pth = a.split("=", 1)
        p = Path(pth).expanduser()
        if not (p / "project.json").exists():
            sys.exit(f"no project.json in {p}")
        arms.append((nm.strip(), p))

    tmp = Path(tempfile.mkdtemp(prefix="labels_bundle_"))
    all_rows: list[dict] = []
    summary_lines = []
    for nm, root in arms:
        man = json.loads((root / "project.json").read_text(encoding="utf-8"))
        names = {int(b["id"]): b.get("name") for b in man.get("behaviors", [])}
        default_fps = float(man.get("fps") or 50.0)
        armdir = tmp / nm / "labels"; armdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "project.json", tmp / nm / "project.json")
        n_bouts = 0
        for v in man.get("videos", []):
            vid = v.get("video_id")
            p = root / "labels" / f"{vid}.parquet"          # CURRENT parquet only — never *.bak
            if not p.exists():
                continue
            shutil.copy2(p, armdir / f"{vid}.parquet")       # clean copy into the bundle
            df = pd.read_parquet(p)
            fps = float(v.get("fps") or default_fps)
            for b in _bouts(df):
                if args.value == "pos" and b["value"] != 1:
                    continue
                n = b["end"] - b["start"]
                all_rows.append({"arm": nm, "clip": vid, "behavior": names.get(b["behavior_id"], "?"),
                                 "behavior_id": b["behavior_id"], "track": b["track"],
                                 "value": _VALUE_NAME.get(b["value"], b["value"]),
                                 "start": b["start"], "end": b["end"], "n_frames": n,
                                 "seconds": round(n / (fps or 50.0), 2),
                                 "source": b["source"], "target": b["target"] if b["target"] >= 0 else ""})
                if b["value"] == 1:
                    n_bouts += 1
        # per-arm per-behavior tally
        by: dict[str, int] = {}
        for r in all_rows:
            if r["arm"] == nm and r["value"] == "happening":
                by[r["behavior"]] = by.get(r["behavior"], 0) + 1
        summary_lines.append(f"{nm}: {n_bouts} positive bouts  ({', '.join(f'{k} {v}' for k, v in sorted(by.items()))})")

    if not all_rows:
        sys.exit("no bouts found in any arm.")

    all_rows.sort(key=lambda r: (r["arm"], r["behavior_id"], r["clip"], r["track"], r["start"]))
    cols = ["arm", "clip", "behavior", "behavior_id", "track", "value", "start", "end", "n_frames", "seconds", "source", "target"]
    csv_path = tmp / "bouts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wr = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); wr.writeheader(); wr.writerows(all_rows)

    readme = tmp / "README.md"
    readme.write_text(
        "# DooM behavior labels\n\n"
        "Hand-labeled and HITL-reviewed behavior annotations from the laras-labeler.\n\n"
        "## Contents\n"
        "- `bouts.csv` — every labeled bout, all arms, one row each. Columns: arm, clip, behavior, "
        "behavior_id, track (0-indexed animal), value (happening/not-happening/unknown), start, end "
        "(frame, end EXCLUSIVE), n_frames, seconds, source (`manual` = hand-drawn, `candidate` = "
        "HITL-accepted from a HiDRA proposal), target (recipient track for directed/social behaviors, "
        "blank otherwise).\n"
        "- `<arm>/labels/<clip>.parquet` — the raw per-frame label store (frame, track, behavior_id, "
        "value, source, target). These are the CURRENT/clean parquets (no `.bak` backups).\n"
        "- `<arm>/project.json` — behavior definitions, clip list, fps, and per-clip metadata.\n\n"
        "## Arms\n" + "\n".join(f"- **{nm}** — `{root.name}`" for nm, root in arms) + "\n\n"
        "## Summary (positive bouts)\n" + "\n".join(f"- {s}" for s in summary_lines) + "\n\n"
        "Bouts are run-length-encoded from the per-frame store: a maximal run of consecutive frames "
        "sharing (track, behavior, value, source, target).\n",
        encoding="utf-8")

    out = Path(args.out).expanduser()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(tmp.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(tmp))
    shutil.rmtree(tmp, ignore_errors=True)

    print("bundled clean labels ->", out)
    for s in summary_lines:
        print("  " + s)
    print(f"  {len(all_rows)} total rows in bouts.csv")


if __name__ == "__main__":
    main()
