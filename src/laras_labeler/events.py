"""Annotation event log — timestamped record of what the annotator did, and when (PLAN.md §9.1).

The point of this file is one measurement: **does the human-in-the-loop loop actually cost less human
time than annotating by hand?** Nothing else in the project can answer that. `history.json` records
accuracy vs *bouts labeled*, which is the wrong x-axis: a bout accepted from candidate review and a
bout painted from scratch are one bout each, and take wildly different amounts of a person's time.
So we log the actions themselves, with wall-clock timestamps, and derive time from them.

On disk (append-only, one JSON object per line, never rewritten):

    <project>/events/<session_id>.jsonl

A record is flat so pandas/duckdb can read it directly:

    {"session": "s-20260908T2011-a1b2c3", "seq": 41, "src": "client",
     "t": "2026-09-08T20:11:03.412+00:00", "t_ms": 1789…, "dt_ms": 1840, "t_srv": "…",
     "type": "candidate_accept", "video_id": "cam0", "behavior_id": 3, "track": 0, "frame": 1204,
     "start": 1180, "end": 1240, "dwell_ms": 4120, ...}

`t`/`t_ms` are the CLIENT's clock (the instant the human acted); `t_srv` is when the server received
the batch. They differ by the flush interval, not by clock skew — both processes are on this machine.
Server-side records (job durations, label writes) set `src: "server"` and stamp `t` themselves.

`summarize()` turns the log into ROUNDS — the unit the question is actually about. One round is the
stretch of work between two Trains of the same behavior: label/review, Train, look, repeat. Per round
it reports human active time split into labeling vs reviewing, what was produced in each, the compute
time that the human waited through, and the model quality that came out the far end.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

SESSION_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

MAX_BATCH = 2000          # events accepted in one POST
MAX_FIELD_CHARS = 2000    # a single string value is truncated to this (a log line stays greppable)

# Gaps longer than this between consecutive events are NOT counted as working time. The client
# heartbeats every 10 s while its tab is visible, so a real working gap is ≤ ~10 s; anything longer
# means the tab was hidden, the browser was closed, or the person left. Deliberately tight: an
# over-count here would flatter whichever workflow involved more staring into space.
ACTIVE_GAP_CAP_S = 15.0
# A heartbeat whose "no pointer/key/playback in this long" exceeds this ends the active stretch, even
# though the tab is still open. Someone reading email in another window with the labeler visible on a
# second monitor is not annotating, and this is what catches them.
#
# Note it is idle time, NOT window focus, that decides. The client records `focused` too, but
# `document.hasFocus()` is false in plenty of situations where real annotation is happening (an
# embedded browser, a second monitor, devtools open), and breaking on it silently drops whole stretches
# of genuine work. Idle time measures the same thing without the false negatives — someone working in
# another window is not pressing keys in this one either, they just get a 2-minute grace period first.
# `focused` stays in the log, so a stricter analysis can still filter on it.
IDLE_BREAK_S = 120.0

_REVIEW_DECISIONS = ("accept", "reject", "merge", "split", "reclassify", "skip", "undo")
_REVIEW_TYPES = {f"candidate_{k}" for k in _REVIEW_DECISIONS}

# labelState in the browser: 1 = Happening, 2 = Not-happening, 3 = Unknown (see index.html modeVal).
_POS, _NEG = 1, 2


def _paint_state(ev: dict) -> int:
    """Which of the three states a paint_commit laid down.

    This matters for the headline and not just for the record: `manual_bouts` is the DENOMINATOR of
    seconds-per-hand-labeled-bout, and the review arm's numerator counts only the decisions that
    produced a positive bout (accept/merge/split/reclassify — never a reject). Counting a
    Not-happening paint as a "bout" on the manual side would therefore divide the same labeling
    seconds by a bigger number and make hand-labeling look cheaper than it is, biasing the one
    comparison this module exists to make. Negatives still cost time and that time still lands in
    `label_s`; they are simply not the product being priced. Same on the review side, where the
    seconds spent rejecting are charged but rejects buy no bout.

    `value` is what the client sends; `mode` is the same thing spelled out. A paint_commit carrying
    neither is read as Happening — a log old enough to lack both predates the distinction, and
    positives are the overwhelming majority of what gets painted."""
    v = ev.get("value")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(v)
    return {"pos": 1, "neg": 2, "unknown": 3}.get(str(ev.get("mode")), 1)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _iso(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000.0, timezone.utc).isoformat(timespec="milliseconds")


def _clean(v):
    """Keep a value JSON-safe and bounded — the log is written unattended for hours."""
    if isinstance(v, str):
        return v[:MAX_FIELD_CHARS]
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k)[:64]: _clean(x) for k, x in list(v.items())[:64]}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v[:64]]
    return str(v)[:MAX_FIELD_CHARS]


class EventLog:
    """Append-only per-session event log under each project."""

    def __init__(self, store) -> None:   # store: ProjectStore
        self.store = store
        self._lock = threading.Lock()

    # ---- write ----
    def _dir(self, pid: str) -> Path:
        proj = self.store.get(pid)
        if proj is None:
            raise KeyError(pid)
        return proj.path / "events"

    def path(self, pid: str, session: str) -> Path:
        if not SESSION_RE.match(session or ""):
            raise ValueError("bad session id")
        return self._dir(pid) / f"{session}.jsonl"

    def append(self, pid: str, session: str, events: list[dict], src: str = "client") -> dict:
        """Append a batch. Returns {written, session}. Bad individual events are dropped, not fatal —
        losing one line of telemetry must never break the annotation the telemetry is measuring."""
        p = self.path(pid, session)
        t_srv = _now()
        lines = []
        for ev in events[:MAX_BATCH]:
            if not isinstance(ev, dict) or not ev.get("type"):
                continue
            rec = {k: _clean(v) for k, v in ev.items() if k not in ("session", "src", "t_srv")}
            t_ms = rec.get("t_ms")
            if not isinstance(t_ms, (int, float)):
                t_ms = datetime.now(timezone.utc).timestamp() * 1000
                rec["t_ms"] = t_ms
            rec["t"] = rec.get("t") or _iso(float(t_ms))
            out = {"session": session, "src": src, "t_srv": t_srv, **rec}
            lines.append(json.dumps(out, separators=(",", ":"), default=str))
        if not lines:
            return {"written": 0, "session": session}
        with self._lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        return {"written": len(lines), "session": session}

    def log(self, pid: str, type: str, session: str = "server", **fields) -> None:
        """Server-side one-liner (job durations, label writes). Never raises — a telemetry failure
        must not take an endpoint down with it."""
        try:
            self.append(pid, session, [{"type": type, "t": _now(),
                                        "t_ms": datetime.now(timezone.utc).timestamp() * 1000,
                                        **fields}], src="server")
        except Exception:  # noqa: BLE001
            pass

    # ---- read ----
    def sessions(self, pid: str) -> list[dict]:
        d = self._dir(pid)
        if not d.exists():
            return []
        out = []
        for p in sorted(d.glob("*.jsonl")):
            recs = sorted(self._read_file(p), key=lambda r: float(r.get("t_ms") or 0))
            if not recs:
                continue
            client = [r for r in recs if r.get("src") == "client"]
            out.append({"session": p.stem, "n_events": len(recs),
                        "start": recs[0].get("t"), "end": recs[-1].get("t"),
                        "n_client": len(client), "bytes": p.stat().st_size})
        return out

    @staticmethod
    def _read_file(p: Path):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue          # a torn last line (killed mid-write) costs one event, not the file

    def read(self, pid: str, sessions: list[str] | None = None) -> list[dict]:
        """Every event across the project's sessions, ordered by client timestamp.

        Ordering is by `t_ms` GLOBALLY, not per file: server events (job durations) live in their own
        `server.jsonl` but interleave with the browser's events in real time, and the rollup below
        depends on that interleaving being right."""
        d = self._dir(pid)
        if not d.exists():
            return []
        files = [d / f"{s}.jsonl" for s in sessions] if sessions else sorted(d.glob("*.jsonl"))
        recs = [r for p in files if p.exists() for r in self._read_file(p)]
        recs.sort(key=lambda r: (float(r.get("t_ms") or 0), int(r.get("seq") or 0)))
        return recs

    # ---- rollup ----
    def summarize(self, pid: str, sessions: list[str] | None = None,
                  gap_cap_s: float = ACTIVE_GAP_CAP_S, idle_break_s: float = IDLE_BREAK_S,
                  behavior_id: int | None = None) -> dict:
        return summarize(self.read(pid, sessions), self.store.get(pid),
                         gap_cap_s=gap_cap_s, idle_break_s=idle_break_s, behavior_id=behavior_id)


# Which activity each event DECLARES. A phase changes only on an event that says what the person is
# doing; anything not listed inherits the phase already in effect.
#
# This split is the load-bearing part of the whole module, because `label_s` and `review_s` are the
# two numerators of the comparison. It used to be "review if a review event said so, otherwise
# label", with 'label' as the starting phase — which made the manual arm the catch-all: opening the
# app, picking a project, browsing the Stats panel, staring at fresh predictions after a Train, all
# of it accrued to hand-labeling. Review, bounded by explicit start/end events, had no such
# slack. Since `s_per_bout` (label_s / bouts) divided by `s_per_accepted_bout` IS the headline, every
# unattributed second inflated the by-hand arm and pushed the answer toward "review pays off" — the
# conclusion the measurement exists to test. Unclassified time now goes to `other_s`, which is
# reported, counted in `active_s`, and used as a denominator by nothing.
_PHASE_OF = {}
_PHASE_OF.update(dict.fromkeys(
    ("review_start", "bout_review_open", "candidate_show", "candidate_trim", "candidate_replay",
     *_REVIEW_TYPES), "review"))
_PHASE_OF.update(dict.fromkeys(("train_click", "predict_click"), "wait"))
# hand-labeling proper: producing, editing or undoing a label, and arming the tool to do so
_PHASE_OF.update(dict.fromkeys(
    ("paint_start", "paint_commit", "paint_cancel", "label_delete", "label_trim",
     "label_undo", "label_redo", "import_labels", "mode_change", "tool_change"), "label"))
# not annotation of either kind: setting up, navigating, or looking at what came back
_PHASE_OF.update(dict.fromkeys(
    ("session_start", "project_open", "video_open", "behavior_select", "track_select",
     "review_end", "review_close", "candidates_load", "hidra_configure",
     "train_result", "predict_result", "train_error", "predict_error"), "other"))

# Deliberately NOT in the map, so they inherit: `heartbeat`, `play_start`/`play_stop`,
# `tab_visible`/`tab_hidden`. Playback means "reviewing this proposal" inside review and "watching
# for the next bout" while labeling; the same event cannot be attributed without its context, and
# guessing one owner would systematically credit or charge one arm.
_PHASE_BUCKET = {"review": "review_s", "wait": "wait_s", "label": "label_s", "other": "other_s"}


def _phase(ev_type: str, cur: str) -> str:
    """The activity in effect after this event — see _PHASE_OF."""
    return _PHASE_OF.get(ev_type, cur)


def _blank_round(idx: int, bid, name, start) -> dict:
    return {
        "round": idx, "behavior_id": bid, "behavior": name, "start": start, "end": start,
        "wall_s": 0.0, "active_s": 0.0, "label_s": 0.0, "review_s": 0.0, "wait_s": 0.0,
        # work_s = the annotation itself (label + review). other_s = present but doing neither:
        # opening the project, picking a clip, reading Stats, looking at fresh predictions.
        "work_s": 0.0, "other_s": 0.0,
        "compute_s": 0.0, "train_s": 0.0, "predict_s": 0.0, "feature_s": 0.0,
        "manual_bouts": 0, "manual_frames": 0, "manual_video_s": 0.0,
        "manual_neg_bouts": 0, "manual_neg_frames": 0, "manual_unknown_bouts": 0,
        "deletes": 0, "trims": 0, "undos": 0,
        "candidates_shown": 0, "candidate_frames": 0, "candidate_video_s": 0.0,
        "decisions": {k: 0 for k in _REVIEW_DECISIONS},
        "decision_s": [], "median_decision_s": None,
        # how much fixing the model's proposals needed: bound edits made, decisions that ended up
        # with edited bounds, and re-watches. A model whose bounds are always trimmed is not costing
        # a decision, it is costing an edit — and that shows up here rather than in the dwell alone.
        "candidate_trims": 0, "decisions_trimmed": 0, "replays": 0,
        "n_sessions": 0,
        # filled in from the Train that closes the round
        "version": None, "ap": None, "f1": None, "n_pos_bouts": None, "n_neg_bouts": None,
        "n_seed_bouts": None, "n_candidate_bouts": None, "trained_at": None, "closed": False,
    }


def summarize(recs: list[dict], proj=None, gap_cap_s: float = ACTIVE_GAP_CAP_S,
              idle_break_s: float = IDLE_BREAK_S, behavior_id: int | None = None) -> dict:
    """Event stream -> per-round timings + the manual-vs-HITL headline.

    Active time is the sum of gaps between consecutive events, each capped at `gap_cap_s`, with the
    chain broken by an idle/hidden heartbeat. Each gap is charged to the phase in effect when it
    started, so 'how long did reviewing take' and 'how long did painting take' are separable — that
    separation IS the comparison the log exists to make.
    """
    names = {}
    fps = {}
    if proj is not None:
        names = {int(b["id"]): b.get("name") for b in proj.behaviors}
        fps = {v["video_id"]: float(v.get("fps") or 30.0) for v in proj.videos}

    rounds: list[dict] = []
    cur: dict | None = None
    phase = "label"
    prev_ms = None
    prev_round: dict | None = None     # a gap is charged to the round that was open when it STARTED,
    show_ms: dict[int, float] = {}     # behavior_id -> when the candidate under review was shown

    def _fps(ev) -> float:
        return fps.get(ev.get("video_id"), 30.0)

    def _round_for(bid) -> dict:
        """The open round for this behavior. Rounds are per behavior: two behaviors labeled in the
        same sitting are two independent loops, and merging them would make both look slower."""
        nonlocal cur
        if cur is None or (bid is not None and cur["behavior_id"] not in (None, bid)):
            cur = _blank_round(len(rounds) + 1, bid, names.get(bid), None)
            rounds.append(cur)
        if cur["behavior_id"] is None and bid is not None:
            cur["behavior_id"], cur["behavior"] = bid, names.get(bid)
        return cur

    for ev in recs:
        t = float(ev.get("t_ms") or 0)
        typ = str(ev.get("type"))
        bid = ev.get("behavior_id")
        bid = int(bid) if isinstance(bid, (int, float)) else None
        # `bid is None` events (session_start, project_open -- anything logged before a behavior was
        # selected) used to survive this filter, so booting the app and browsing were billed to
        # whichever behavior you filtered to. Asking for one behavior means only that behavior.
        if behavior_id is not None and bid != behavior_id:
            continue

        # --- time accounting (client events only; server records are instantaneous notes) ---
        if ev.get("src") != "server":
            if prev_ms is not None and t >= prev_ms:
                # ...so the wait after clicking Train lands in the round that clicked it, not the one
                # the finished Train opened.
                gap = (t - prev_ms) / 1000.0
                charged = min(gap, gap_cap_s)
                tgt = prev_round if prev_round is not None else _round_for(bid)
                if tgt["start"] is None:
                    tgt["start"] = ev.get("t")
                tgt["active_s"] += charged
                tgt[_PHASE_BUCKET.get(phase, "other_s")] += charged
                # The gap is charged to the round that was OPEN when it started, but that round's
                # `end` was already stamped at its own last event -- so a round could report more
                # active time than it had wall time (a 15 s gap after a 1 s round read "15.0s active
                # of 1.1s on the clock"). The round really did stay open until this moment, so its
                # clock runs to here. Rounds stay contiguous, never overlapping: this `end` is the
                # next round's `start`.
                tgt["end"] = ev.get("t") or tgt["end"]
            idle_ms = ev.get("idle_ms")
            hidden = ev.get("visible") is False or typ == "tab_hidden"
            broke = hidden or (isinstance(idle_ms, (int, float)) and idle_ms / 1000.0 > idle_break_s)
            prev_ms = None if broke else t

        r = _round_for(bid)
        if r["start"] is None:
            r["start"] = ev.get("t")
        r["end"] = ev.get("t") or r["end"]
        phase = _phase(typ, phase)
        if ev.get("src") != "server":
            prev_round = r

        # --- what was produced ---
        if typ == "paint_commit":
            n = int(ev.get("n_frames") or max(0, int(ev.get("end") or 0) - int(ev.get("start") or 0)))
            state = _paint_state(ev)
            if state == _POS:
                r["manual_bouts"] += 1
                r["manual_frames"] += n
                r["manual_video_s"] += n / _fps(ev)
            elif state == _NEG:
                r["manual_neg_bouts"] += 1
                r["manual_neg_frames"] += n
            else:
                r["manual_unknown_bouts"] += 1
        elif typ == "candidate_trim":
            r["candidate_trims"] += 1
        elif typ == "candidate_replay":
            r["replays"] += 1
        elif typ == "label_delete":
            r["deletes"] += 1
        elif typ == "label_trim":
            r["trims"] += 1
        elif typ in ("label_undo", "label_redo"):
            r["undos"] += 1
        elif typ == "candidate_show":
            n = int(ev.get("n_frames") or max(0, int(ev.get("end") or 0) - int(ev.get("start") or 0)))
            r["candidates_shown"] += 1
            r["candidate_frames"] += n
            r["candidate_video_s"] += n / _fps(ev)
            show_ms[bid] = t
        elif typ in _REVIEW_TYPES:
            kind = typ.split("_", 1)[1]
            r["decisions"][kind] = r["decisions"].get(kind, 0) + 1
            if ev.get("trimmed"):
                r["decisions_trimmed"] += 1
            dwell = ev.get("dwell_ms")
            if not isinstance(dwell, (int, float)) and bid in show_ms:
                dwell = t - show_ms[bid]      # client didn't send one — derive it from the log
            if isinstance(dwell, (int, float)) and 0 <= dwell <= gap_cap_s * 1000 * 40:
                r["decision_s"].append(dwell / 1000.0)
            show_ms.pop(bid, None)
        elif typ == "job_done":
            secs = float(ev.get("seconds") or 0.0)
            kind = ev.get("kind")
            if kind == "train":
                r["train_s"] += secs
                r["compute_s"] += secs      # compute_s = only the jobs a human sits and waits for;
            elif kind == "predict":
                r["predict_s"] += secs      # a background feature pre-warm costs machine time, not
                r["compute_s"] += secs      # human time, so it is tracked apart in feature_s
            elif kind == "features":
                r["feature_s"] += secs

        # --- a Train closes the round: stamp the model it produced, then start a fresh one ---
        if typ == "job_done" and ev.get("kind") == "train" and ev.get("status") == "done":
            r["behavior_id"] = r["behavior_id"] if r["behavior_id"] is not None else bid
            r["behavior"] = r["behavior"] or names.get(r["behavior_id"])
            for k in ("version", "ap", "f1", "n_pos_bouts", "n_neg_bouts",
                      "n_seed_bouts", "n_candidate_bouts", "trained_at"):
                if ev.get(k) is not None:
                    r[k] = ev.get(k)
            r["closed"] = True
            cur = None
            show_ms.clear()

    # finalize
    cum_active = cum_work = cum_manual = cum_accept = 0.0
    for r in rounds:
        r["median_decision_s"] = round(median(r["decision_s"]), 2) if r["decision_s"] else None
        r["decision_s"] = [round(x, 2) for x in r["decision_s"]]
        try:
            r["wall_s"] = round((datetime.fromisoformat(r["end"]) - datetime.fromisoformat(r["start"])).total_seconds(), 1)
        except Exception:  # noqa: BLE001
            r["wall_s"] = 0.0
        r["work_s"] = r["label_s"] + r["review_s"]
        for k in ("active_s", "label_s", "review_s", "wait_s", "work_s", "other_s", "compute_s",
                  "train_s", "predict_s", "feature_s", "manual_video_s", "candidate_video_s"):
            r[k] = round(float(r[k]), 1)
        r["accepted"] = r["decisions"].get("accept", 0) + r["decisions"].get("merge", 0) \
            + r["decisions"].get("split", 0) + r["decisions"].get("reclassify", 0)
        cum_active += r["active_s"]
        cum_work += r["work_s"]
        cum_manual += r["manual_bouts"]
        cum_accept += r["accepted"]
        r["cum_active_min"] = round(cum_active / 60.0, 2)
        # The axis a learning curve should be read against: annotation only. `cum_active_min` also
        # carries `wait_s` (watching a progress bar) and `other_s` (setting up, navigating), so
        # plotting accuracy against it would let a slower machine or a longer browse read as more
        # annotation effort.
        r["cum_work_min"] = round(cum_work / 60.0, 2)
        r["cum_manual_bouts"] = int(cum_manual)
        r["cum_accepted_bouts"] = int(cum_accept)
        # per-round productivity, the numbers the whole log exists to produce
        r["s_per_manual_bout"] = round(r["label_s"] / r["manual_bouts"], 1) if r["manual_bouts"] else None
        r["s_per_decision"] = round(r["review_s"] / sum(r["decisions"].values()), 1) if sum(r["decisions"].values()) else None

    label_s = sum(r["label_s"] for r in rounds)
    review_s = sum(r["review_s"] for r in rounds)
    manual_bouts = sum(r["manual_bouts"] for r in rounds)
    manual_frames = sum(r["manual_frames"] for r in rounds)
    manual_video_s = sum(r["manual_video_s"] for r in rounds)
    decisions = sum(sum(r["decisions"].values()) for r in rounds)
    accepted = sum(r["accepted"] for r in rounds)
    trimmed = sum(r["decisions_trimmed"] for r in rounds)
    cand_video_s = sum(r["candidate_video_s"] for r in rounds)
    all_dec = [x for r in rounds for x in r["decision_s"]]

    # time-to-quality: cumulative HUMAN minutes at which each accuracy level was first reached.
    # This is the y-vs-x the comparison needs — history.json plots accuracy against bouts, which
    # hides the very difference (a reviewed bout is cheaper than a painted one) being measured.
    curve = [{"round": r["round"], "behavior": r["behavior"],
              "cum_work_min": r["cum_work_min"], "cum_active_min": r["cum_active_min"],
              "ap": r["ap"], "f1": r["f1"], "n_pos_bouts": r["n_pos_bouts"],
              "n_seed_bouts": r["n_seed_bouts"], "n_candidate_bouts": r["n_candidate_bouts"]}
             for r in rounds if r["closed"]]
    milestones = []
    for target in (0.5, 0.7, 0.8, 0.9):
        hit = next((c for c in curve if (c["ap"] or 0) >= target), None)
        if hit:
            milestones.append({"ap": target, "cum_work_min": hit["cum_work_min"],
                               "cum_active_min": hit["cum_active_min"], "round": hit["round"]})

    return {
        "n_events": len(recs),
        "n_rounds": len(rounds),
        "rounds": rounds,
        "curve": curve,
        "milestones": milestones,
        "totals": {
            "active_s": round(label_s + review_s + sum(r["wait_s"] for r in rounds), 1),
            "label_s": round(label_s, 1), "review_s": round(review_s, 1),
            "wait_s": round(sum(r["wait_s"] for r in rounds), 1),
            "work_s": round(label_s + review_s, 1),
            "other_s": round(sum(r["other_s"] for r in rounds), 1),
            "compute_s": round(sum(r["compute_s"] for r in rounds), 1),
            "feature_s": round(sum(r["feature_s"] for r in rounds), 1),
            # Both arms are priced the same way: all the seconds the arm consumed, over the
            # POSITIVE bouts it produced. Painting negatives and rejecting candidates are real work
            # and are charged, but neither yields a bout, so neither lands in a denominator.
            "manual": {
                "bouts": manual_bouts, "frames": manual_frames,
                "video_s": round(manual_video_s, 1),
                "neg_bouts": sum(r["manual_neg_bouts"] for r in rounds),
                "neg_frames": sum(r["manual_neg_frames"] for r in rounds),
                "unknown_bouts": sum(r["manual_unknown_bouts"] for r in rounds),
                "s_per_bout": round(label_s / manual_bouts, 1) if manual_bouts else None,
                "s_per_video_s": round(label_s / manual_video_s, 2) if manual_video_s else None,
            },
            "review": {
                "decisions": decisions, "accepted": accepted,
                "shown_frames": sum(r["candidate_frames"] for r in rounds),
                "video_s": round(cand_video_s, 1),
                "s_per_decision": round(review_s / decisions, 1) if decisions else None,
                "s_per_accepted_bout": round(review_s / accepted, 1) if accepted else None,
                "s_per_video_s": round(review_s / cand_video_s, 2) if cand_video_s else None,
                "median_decision_s": round(median(all_dec), 2) if all_dec else None,
                # how often the model's bounds had to be fixed rather than taken as proposed
                "trimmed": trimmed, "replays": sum(r["replays"] for r in rounds),
                "candidate_trims": sum(r["candidate_trims"] for r in rounds),
                "frac_trimmed": round(trimmed / decisions, 2) if decisions else None,
            },
            # The headline: how much cheaper (or not) a bout is through review than by hand.
            # Derived from the two ROUNDED figures it is displayed beside, so a reader who divides
            # "5.2s by hand vs 5.0s accepted" gets the ratio actually printed. Dividing the
            # unrounded values instead made the sentence fail its own arithmetic by a hundredth.
            "speedup_per_bout": (round(round(label_s / manual_bouts, 1)
                                       / round(review_s / accepted, 1), 2)
                                 if manual_bouts and accepted and review_s else None),
        },
    }


ROUND_CSV_COLS = [
    "round", "behavior_id", "behavior", "start", "end", "wall_s", "active_s", "work_s", "label_s",
    "review_s", "wait_s", "other_s", "compute_s", "train_s", "predict_s", "feature_s",
    "manual_bouts", "manual_frames", "manual_video_s",
    "manual_neg_bouts", "manual_neg_frames", "manual_unknown_bouts",
    "deletes", "trims", "undos", "candidates_shown", "candidate_frames", "candidate_video_s",
    "accept", "reject", "merge", "split", "reclassify", "skip", "undo", "accepted",
    "candidate_trims", "decisions_trimmed", "replays",
    "median_decision_s", "s_per_manual_bout", "s_per_decision",
    "cum_work_min", "cum_active_min", "cum_manual_bouts", "cum_accepted_bouts",
    "version", "ap", "f1", "n_pos_bouts", "n_neg_bouts", "n_seed_bouts", "n_candidate_bouts", "trained_at",
]


def rounds_csv(summary: dict) -> str:
    """Per-round table as CSV — the artifact you paste into a paper/notebook."""
    def cell(v):
        s = "" if v is None else str(v)
        return '"' + s.replace('"', '""') + '"' if any(c in s for c in ',"\n') else s

    lines = [",".join(ROUND_CSV_COLS)]
    for r in summary["rounds"]:
        flat = {**r, **{k: v for k, v in r["decisions"].items()}}
        lines.append(",".join(cell(flat.get(c)) for c in ROUND_CSV_COLS))
    return "\n".join(lines) + "\n"
