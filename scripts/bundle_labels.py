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


_DEC = {"candidate_accept": "accept", "candidate_reject": "reject", "candidate_merge": "merge",
        "candidate_split": "split", "candidate_reclassify": "reclassify"}


def _decisions(evdir: Path, names: dict) -> list[dict]:
    """Per-decision timing from the event log: each review decision's dwell (dwell_ms) and each manual
    paint's draw time (paint_start -> paint_commit). Returns one row per action, in time order."""
    recs = []
    for p in sorted(evdir.glob("*.jsonl")):
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    recs.sort(key=lambda r: (float(r.get("t_ms") or 0), int(r.get("seq") or 0)))
    last_start: dict[tuple, float] = {}
    rows = []
    for ev in recs:
        typ = str(ev.get("type", ""))
        bid = ev.get("behavior_id")
        if bid is None:
            continue
        bid = int(bid); tr = int(ev.get("track") or 0); vid = ev.get("video_id")
        if typ == "paint_start":
            last_start[(bid, tr)] = float(ev.get("t_ms") or 0)
        elif typ == "paint_commit":
            t = float(ev.get("t_ms") or 0); st = last_start.pop((bid, tr), None)
            secs = round((t - st) / 1000, 2) if st else ""
            rows.append({"clip": vid, "behavior": names.get(bid, "?"), "behavior_id": bid, "track": tr,
                         "decision": "paint", "seconds": secs, "frame": ev.get("frame", ""),
                         "start": ev.get("start", ""), "end": ev.get("end", "")})
        elif typ in _DEC:
            dw = ev.get("dwell_ms")
            secs = round(float(dw) / 1000, 2) if isinstance(dw, (int, float)) else ""
            rows.append({"clip": vid, "behavior": names.get(bid, "?"), "behavior_id": bid, "track": tr,
                         "decision": _DEC[typ], "seconds": secs, "frame": ev.get("frame", ""),
                         "start": ev.get("start", ""), "end": ev.get("end", "")})
    return rows


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

    try:                                            # the labeler's own active-time model (15s gap cap, 120s idle break)
        from laras_labeler.events import summarize as _summarize
    except Exception:
        _summarize = None

    tmp = Path(tempfile.mkdtemp(prefix="labels_bundle_"))
    all_rows: list[dict] = []
    all_dec: list[dict] = []
    all_tim: list[dict] = []
    deccols = ["arm", "clip", "behavior", "behavior_id", "track", "decision", "seconds", "frame", "start", "end"]
    timcols = ["arm", "behavior", "label_active_min", "review_active_min", "draw_min", "finding_min", "watch_video_min"]
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
        # per-decision timing from the event log (dwell per review decision, draw time per manual paint)
        evdir = root / "events"
        if evdir.is_dir():
            dec = _decisions(evdir, names)
            for r in dec:
                r["arm"] = nm
            all_dec.extend(dec)
            with (tmp / f"{nm}_decisions.csv").open("w", newline="", encoding="utf-8") as f:
                wr = _csv.DictWriter(f, fieldnames=deccols, extrasaction="ignore"); wr.writeheader(); wr.writerows(dec)
            # per-behavior active/finding/draw time (finding = active labeling time - drawing gestures)
            if _summarize is not None:
                recs = []
                for p in sorted(evdir.glob("*.jsonl")):
                    for line in p.open(encoding="utf-8"):
                        line = line.strip()
                        if line:
                            try:
                                recs.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                recs.sort(key=lambda r: (float(r.get("t_ms") or 0), int(r.get("seq") or 0)))
                draw_by: dict[str, float] = {}
                for r in dec:
                    try:
                        sec = float(r["seconds"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if r["decision"] == "paint" and sec >= 0:
                        draw_by[r["behavior"]] = draw_by.get(r["behavior"], 0.0) + sec
                agg: dict[int, list] = {}                    # behavior_id -> [label_s, review_s, manual_video_s, cand_video_s]
                try:
                    for rnd in _summarize(recs).get("rounds", []):
                        bid2 = rnd.get("behavior_id")
                        if bid2 is None:
                            continue
                        a = agg.setdefault(int(bid2), [0.0, 0.0, 0.0, 0.0])
                        a[0] += float(rnd.get("label_s", 0) or 0)
                        a[1] += float(rnd.get("review_s", 0) or 0)
                        a[2] += float(rnd.get("manual_video_s", 0) or 0)
                        a[3] += float(rnd.get("candidate_video_s", 0) or 0)
                except Exception as e:
                    print(f"  (time_breakdown skipped for {nm}: {e})")
                for bid2, bnm in sorted(names.items()):
                    a = agg.get(bid2, [0.0, 0.0, 0.0, 0.0])
                    lab = a[0] / 60.0
                    rev = a[1] / 60.0
                    watch = (a[2] + a[3]) / 60.0         # video actually played (manual scan + candidate review)
                    draw = draw_by.get(bnm, 0.0) / 60.0
                    if lab > 0.05 or rev > 0.05:
                        all_tim.append({"arm": nm, "behavior": bnm, "label_active_min": round(lab, 1),
                                        "review_active_min": round(rev, 1), "draw_min": round(draw, 1),
                                        "finding_min": round(max(0.0, lab - draw), 1),
                                        "watch_video_min": round(watch, 1)})

    if not all_rows:
        sys.exit("no bouts found in any arm.")

    all_rows.sort(key=lambda r: (r["arm"], r["behavior_id"], r["clip"], r["track"], r["start"]))
    cols = ["arm", "clip", "behavior", "behavior_id", "track", "value", "start", "end", "n_frames", "seconds", "source", "target"]
    csv_path = tmp / "bouts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wr = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); wr.writeheader(); wr.writerows(all_rows)
    # also a separate CSV per arm, so each can be attached on its own
    for nm, _root in arms:
        arm_rows = [r for r in all_rows if r["arm"] == nm]
        with (tmp / f"{nm}_bouts.csv").open("w", newline="", encoding="utf-8") as f:
            wr = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); wr.writeheader(); wr.writerows(arm_rows)
    # combined per-decision timing across arms
    if all_dec:
        with (tmp / "decisions.csv").open("w", newline="", encoding="utf-8") as f:
            wr = _csv.DictWriter(f, fieldnames=deccols, extrasaction="ignore"); wr.writeheader(); wr.writerows(all_dec)
    # per-behavior time breakdown (active / drawing / finding-watching)
    if all_tim:
        with (tmp / "time_breakdown.csv").open("w", newline="", encoding="utf-8") as f:
            wr = _csv.DictWriter(f, fieldnames=timcols, extrasaction="ignore"); wr.writeheader(); wr.writerows(all_tim)

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
        "- `<arm>_bouts.csv` — the same, split per arm (e.g. `hand_bouts.csv`, `hitl_bouts.csv`) so "
        "each can be shared on its own.\n"
        "- `<arm>_decisions.csv` / `decisions.csv` — per-decision TIMING from the event log, one row "
        "per action in time order: arm, clip, behavior, track, decision (`accept`/`reject`/`merge`/"
        "`split` for HITL review; `paint` for hand-drawing), `seconds` (time spent on that decision — "
        "the review dwell for HITL, the draw time for a manual paint), frame, start, end.\n"
        "- `time_breakdown.csv` — active minutes per arm+behavior, split into `draw_min` (drawing "
        "gestures) and `finding_min` (scanning/watching to locate bouts = active labeling time minus "
        "drawing), plus `review_active_min`. Active time uses the labeler's model (gaps capped at 15 s, "
        "idle >120 s excluded). There is no discrete 'scrub' event, so finding_min is that derived "
        "watch/scan time.\n"
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
    if all_tim:
        print("  time breakdown (min): " + " · ".join(
            f"{t['arm']}/{t['behavior']} find {t['finding_min']}+draw {t['draw_min']} watch {t['watch_video_min']}" for t in all_tim))
    elif _summarize is None:
        print("  (time_breakdown.csv skipped — could not import laras_labeler.events; run with the labeler env python)")


if __name__ == "__main__":
    main()
