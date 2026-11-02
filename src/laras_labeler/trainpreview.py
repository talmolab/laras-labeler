"""What the NEXT Train of a behavior will learn from, clip by clip and mouse by mouse.

The two engines read the same labels very differently, and the difference is invisible from the
timeline:

  native  this project's model fits every Happening and Not-happening bout you labeled; unlabeled
          and Unknown frames are not used at all (`Trainer.gather`).
  hidra   depends on the behavior's "train on" setting (`hidra.train_on`):
          all       a fine-tune sends only Happening bouts. For every mouse with a Happening bout in
                    a clip, EVERY other frame of that clip trains as Not-happening -- including
                    frames nobody looked at (`hidra.export_bouts`, HiDRA's `finetune.py prepare`).
          window    the same, but only inside each clip's social window; the rest is left out.
          reviewed  only the Happening and Not-happening spans you answered are trained on; every
                    other frame (unreviewed, Unknown) is left out (`hidra.export_reviewed`).
          Clips are staged exactly as `_hidra_train` stages them.

`preview` reports exactly that per (clip, track), so a missed bout on a half-reviewed mouse shows up
as "never reviewed, trains as Not-happening" before the Train, not after. It reuses each engine's own
selection -- `hidra.finetune_bouts` for the fine-tune, the same labeled-run rule as `Trainer.gather`
for the native model -- so the table cannot drift from what training gets. Read-only.
"""

from __future__ import annotations

from pathlib import Path

from . import hidra, poseio


def _n_animals(entry: dict) -> int:
    """Tracks in the clip's .slp, from its header alone (no pose read); 0 if it can't be read."""
    slp = entry.get("slp_path")
    if not slp or not Path(slp).exists():
        return 0
    h = poseio.read_header(slp)
    return len(h.track_names) if h is not None else 0


def preview(store, labels, predictor, features, pid: str, bid: int) -> dict:
    proj = store.get(pid)
    beh = next(b for b in proj.behaviors if b["id"] == bid)
    head = beh.get("hidra") or {}
    engine = "hidra" if head.get("action") else "native"

    n_animals = {v["video_id"]: _n_animals(v) for v in proj.videos}
    train_on = hidra.train_on(head) if engine == "hidra" else None
    sent: dict[tuple, list[dict]] = {}       # Happening spans sent
    negs: dict[tuple, list[dict]] = {}       # reviewed: Not-happening spans sent
    wins: dict[tuple, list[dict]] = {}       # window: the window span sent per track
    skipped: dict[tuple, list[dict]] = {}
    if engine == "hidra":
        staged = _staged_clips(proj, labels, pid, bid, train_on == "reviewed")
        rows, skip = hidra.finetune_bouts(store, labels, pid, bid, head, n_animals, only=staged)
        for r in rows:
            sent.setdefault((r["video_id"], r["track"]), []).append(r)
        for r in skip:
            skipped.setdefault((r["video_id"], r["track"]), []).append(r)
        if train_on == "reviewed":
            for r in hidra.finetune_bouts(store, labels, pid, bid, head, n_animals, values={0}, only=staged)[0]:
                negs.setdefault((r["video_id"], r["track"]), []).append(r)
        elif train_on == "window":
            for r in hidra.finetune_bouts(store, labels, pid, bid, head, n_animals, values={0}, only=staged,
                                          window=True)[0]:
                wins.setdefault((r["video_id"], r["track"]), []).append(r)

    clips = []
    for v in proj.videos:
        vid = v["video_id"]
        F = int(v.get("n_frames") or 0)
        try:
            df = labels.rows_for_behavior(pid, vid, bid)
        except (KeyError, FileNotFoundError):
            df = None
        label_tracks = set() if df is None or df.empty else {int(t) for t in df["track"].unique()}
        n_tracks = max(n_animals[vid], predictor.n_tracks(pid, vid, bid), max(label_tracks, default=-1) + 1)
        left = predictor.remaining(pid, vid, bid)["tracks"] if predictor.n_tracks(pid, vid, bid) else {}
        slp_ok = bool(v.get("slp_path")) and Path(v["slp_path"]).exists()

        tracks = []
        for t in range(n_tracks):
            runs = labels.get_runs(pid, vid, t, bid).get(bid, []) if t in label_tracks else []
            frames = {0: 0, 1: 0, 2: 0}
            bouts = []
            for s, e, val in runs:
                frames[int(val)] = frames.get(int(val), 0) + int(e) - int(s)
                if int(val) in (0, 1):
                    bouts.append({"start": int(s), "end": int(e), "value": int(val)})
            row = {"track": t, "pos_bouts": sum(b["value"] == 1 for b in bouts),
                   "neg_bouts": sum(b["value"] == 0 for b in bouts), "pos_frames": frames[1],
                   "neg_frames": frames[0], "unknown_frames": frames.get(2, 0),
                   "remaining": left.get(str(t)) if left else None}

            if engine == "native":
                # Trainer.gather: a clip with poses; every labeled Happening / Not-happening run
                # of every track is one training bout; Unknown and unlabeled frames are dropped.
                row["used"] = bool(v.get("has_poses")) and bool(bouts)
                row["note"] = ("" if row["used"] else "no poses for this clip" if bouts
                               else "no Happening / Not-happening labels")
                row["blocker"] = bool(row["used"] and not slp_ok and features.status(pid, vid)["status"] != "ready")
                if row["blocker"]:
                    row["note"] = "clip's .slp not found and its features aren't cached — Train will stop on it"
                row["bouts"] = [{**b, "sent": row["used"]} for b in bouts]
                row["trained_neg_frames"] = frames[0] if row["used"] else 0
                row["never_reviewed_frames"] = 0
            else:
                mine, mneg, mwin = sent.get((vid, t), []), negs.get((vid, t), []), wins.get((vid, t), [])
                row["pos_bouts"] = len(mine)                  # the spans the fine-tune actually sends
                row["skipped_directed"] = len(skipped.get((vid, t), []))
                contributes = bool(mine or mneg)              # reviewed: Not-happening alone trains too
                # Staging needs the clip's .slp (its tracking is what HiDRA trains on); a clip
                # without one makes the fine-tune fail, so say so rather than count it.
                row["used"] = contributes and slp_ok
                row["blocker"] = contributes and not slp_ok
                if contributes and not slp_ok:
                    row["note"] = "clip's .slp not found — the fine-tune will stop on it"
                elif not contributes and vid not in staged and not v.get("slp_path") and (
                        frames[1] or (train_on == "reviewed" and frames[0])):
                    row["note"] = "no .slp for this clip — the fine-tune skips it"
                elif not contributes:
                    row["note"] = ("directed bouts with no partner picked — skipped" if row["skipped_directed"]
                                   else "no Happening or Not-happening label — not trained on" if train_on == "reviewed"
                                   else "no Happening bout — not trained on")
                else:
                    row["note"] = ""
                # One supervised (agent -> target) pair per partner. Which of that pair's frames
                # train, and as what, is the train-on setting.
                partners: dict[str, int] = {}
                for r in mine:
                    partners[r["target"]] = partners.get(r["target"], 0) + r["stop_frame"] - r["start_frame"]
                for r in mneg + mwin:
                    partners.setdefault(r["target"], 0)
                row["partners"] = sorted(partners)
                k = len(partners)
                if train_on == "reviewed":            # answered spans only: nothing unreviewed trains
                    trained_neg = sum(r["stop_frame"] - r["start_frame"] for r in mneg)
                    never, left_out = 0, k * F - sum(partners.values()) - trained_neg
                elif train_on == "window":            # every unmarked frame inside the window is a negative
                    ws, we = (mwin[0]["start_frame"], mwin[0]["stop_frame"]) if mwin else (0, F)
                    inside = {0: 0, 1: 0}
                    for s, e, val in runs:
                        if int(val) in inside:
                            inside[int(val)] += max(0, min(int(e), we) - max(int(s), ws))
                    W = we - ws
                    trained_neg = sum(max(W - n, 0) for n in partners.values())
                    never, left_out = k * max(W - inside[1] - inside[0], 0), k * (F - W)
                else:                                 # every non-bout frame of the clip is a negative
                    trained_neg = sum(F - n for n in partners.values())
                    never, left_out = k * max(F - frames[1] - frames[0], 0), 0
                row["trained_neg_frames"] = trained_neg if row["used"] else 0
                row["never_reviewed_frames"] = never if row["used"] else 0
                row["left_out_frames"] = max(left_out, 0) if row["used"] else 0
                # sent Happening spans, then what is NOT sent: Happening bouts skipped for lack of a
                # partner, and Not-happening bouts (already negatives under the every-other-frame rule)
                row["bouts"] = ([{"start": r["start_frame"], "end": r["stop_frame"], "value": 1, "sent": True,
                                  "target": r["target"]} for r in mine]
                                + [{"start": r["start_frame"], "end": r["stop_frame"], "value": 1, "sent": False,
                                    "why": "no partner picked"} for r in skipped.get((vid, t), [])]
                                + [{**b, "sent": train_on == "reviewed"} for b in bouts if b["value"] == 0])
            tracks.append(row)

        used = any(t["used"] for t in tracks)
        clips.append({"video_id": vid, "name": Path(str(v.get("video_path") or vid)).name or vid,
                      "n_frames": F, "used": used, "tracks": tracks})

    used_tracks = [t for c in clips for t in c["tracks"] if t["used"]]
    totals = {
        "clips_used": sum(c["used"] for c in clips),
        "tracks_used": len(used_tracks),
        "pos_bouts": sum(sum(b["value"] == 1 and b["sent"] for b in t["bouts"]) for t in used_tracks),
        "neg_bouts": (sum(t["neg_bouts"] for t in used_tracks) if engine == "native" else 0),
        "pos_frames": sum(t["pos_frames"] for t in used_tracks),
        "trained_neg_frames": sum(t["trained_neg_frames"] for t in used_tracks),
        "never_reviewed_frames": sum(t["never_reviewed_frames"] for t in used_tracks),
        "left_out_frames": sum(t.get("left_out_frames", 0) for t in used_tracks),
        # Unknown is not sent either, so on an every-frame fine-tune it trains as Not-happening too
        "unknown_on_used": (sum(t["unknown_frames"] for t in used_tracks) if train_on == "all" else 0),
        "remaining_on_used": sum(t["remaining"] or 0 for t in used_tracks),
        "skipped_directed": sum(len(s) for s in skipped.values()),
        # clips that will make the Train stop -- shown loudly, not as quietly 'not used'
        "blockers": sorted({c["video_id"] for c in clips for t in c["tracks"] if t.get("blocker")}),
    }
    return {"behavior_id": bid, "behavior": beh.get("name", ""), "engine": engine, "train_on": train_on,
            "head": ({k: head.get(k) for k in ("lab", "action", "collapse")} if engine == "hidra" else None),
            "clips": clips, "totals": totals}


def _staged_clips(proj, labels, pid: str, bid: int, reviewed_only: bool) -> set[str]:
    """The clips `_hidra_train` stages: labels for this behavior, a Happening frame (or, reviewed
    frames only, any answered frame), and a tracking file."""
    out = set()
    for v in proj.videos:
        vid = v["video_id"]
        try:
            df = labels.rows_for_behavior(pid, vid, bid)
        except (KeyError, FileNotFoundError):
            continue
        if df is None or df.empty or not v.get("slp_path"):
            continue
        runs = [r for t in {int(x) for x in df["track"].unique()}
                for r in labels.get_runs(pid, vid, t, bid).get(bid, [])]
        n_pos = sum(int(r[1]) - int(r[0]) for r in runs if int(r[2]) == 1)
        n_neg = sum(int(r[1]) - int(r[0]) for r in runs if int(r[2]) == 0)
        if n_pos or (reviewed_only and n_neg):
            out.add(vid)
    return out
