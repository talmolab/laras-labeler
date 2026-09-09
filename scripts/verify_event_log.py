"""Drive the paint -> train -> review loop in a real browser and check the event log against it.

This is the end-to-end verification the annotation event log was missing. The rollup arithmetic can
be (and is) checked over a synthetic event stream, but that only tests `events.py` reading events
`events.py` was handed. What was never tested is the other half: whether the BROWSER emits the
events the rollup assumes, at the moments it assumes, with the fields it reads -- which is the half
that decides whether a published number is real.

So this script drives the actual UI in Chromium, doing what an annotator does (paint bouts by hand,
Train, review the model's proposals, Train again), keeps its OWN ledger of every action it took and
when, and then asserts the rollup at `GET /timing` matches that ledger.

    uv run python scripts/make_synthetic_clip.py /tmp/synth
    uv run python scripts/verify_event_log.py /tmp/synth

Needs `playwright` and a Chromium (`pip install playwright && playwright install chromium`, or set
CHROMIUM_PATH / PLAYWRIGHT_BROWSERS_PATH). Everything else is project dependencies.

The clip is synthetic, so the accuracy numbers mean nothing -- the planted behavior is trivially
separable on purpose. What is being checked is the bookkeeping: counts, phase attribution, round
boundaries, and that a break is not billed to the annotation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = "127.0.0.1"

# The break planted inside round 1. It is longer than idle_break_s (120 s by default) would need to
# be to matter... so the check below runs the rollup with a small --idle-break instead of making a
# verification script sit still for two minutes. What is being verified is the MECHANISM (a gap
# stops accruing active time), which is scale-free.
BREAK_S = 8.0
IDLE_BREAK_S = 4.0        # rollup param for the check, chosen so BREAK_S exceeds it
GAP_CAP_S = 3.0           # ditto: with sub-second actions, a 3 s cap still caps only real pauses


# ---------------------------------------------------------------- server + HTTP


def _req(url: str, method: str = "GET", body=None, raw: bool = False):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(r, timeout=120) as f:
        blob = f.read()
    return blob if raw else json.loads(blob or b"null")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, root: Path, port: int) -> None:
        self.base = f"http://{HOST}:{port}"
        self.log = open(root.parent / "server.log", "w")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "laras_labeler.cli", str(root), "--port", str(port),
             "--no-browser", "--host", HOST],
            stdout=self.log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"})

    def wait(self, timeout: float = 90.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with {self.proc.returncode}; see server.log")
            try:
                _req(self.base + "/api/projects")
                return
            except (urllib.error.URLError, OSError):
                time.sleep(0.25)
        raise TimeoutError("server did not come up")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


# ---------------------------------------------------------------- the ledger


class Ledger:
    """What the driver actually did, recorded independently of the app's own log.

    The whole point: the event log is only trustworthy if something that is not the event log agrees
    with it. Every entry is (wall-clock ms, kind, detail)."""

    def __init__(self) -> None:
        self.rows: list[tuple[float, str, dict]] = []

    def add(self, action: str, **detail) -> None:
        self.rows.append((time.time() * 1000.0, action, detail))

    def count(self, kind: str) -> int:
        return sum(1 for _, k, _ in self.rows if k == kind)

    def of(self, kind: str) -> list[dict]:
        return [d for _, k, d in self.rows if k == kind]

    def span_s(self, first: str, last: str) -> float:
        a = next((t for t, k, _ in self.rows if k == first), None)
        b = next((t for t, k, _ in reversed(self.rows) if k == last), None)
        return (b - a) / 1000.0 if a is not None and b is not None else 0.0


# ---------------------------------------------------------------- checks


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def __call__(self, ok: bool, name: str, detail: str = "") -> bool:
        self.rows.append((bool(ok), name, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""),
              flush=True)
        return bool(ok)

    @property
    def failed(self) -> list[str]:
        return [n for ok, n, _ in self.rows if not ok]


# ---------------------------------------------------------------- browser driving


def _chromium_path() -> str | None:
    if os.environ.get("CHROMIUM_PATH"):
        return os.environ["CHROMIUM_PATH"]
    root = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""))
    if root.is_dir():
        for p in sorted(root.glob("chromium-*/chrome-linux/chrome"), reverse=True):
            return str(p)
        for p in sorted(root.glob("chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium"),
                        reverse=True):
            return str(p)
    return None


class Ui:
    """Thin wrapper over the page: only interactions a person could actually perform."""

    def __init__(self, page, led: Ledger, rng: random.Random) -> None:
        self.page, self.led, self.rng = page, led, rng

    # -- pacing. A human does not act instantly, and the rollup's whole job is to measure the gaps,
    # so the driver has to produce realistic ones (and the ledger records what it produced).
    def beat(self, lo: float = 0.25, hi: float = 0.8) -> float:
        d = self.rng.uniform(lo, hi)
        time.sleep(d)
        return d

    def status(self) -> str:
        return self.page.inner_text("#loadStatus") or ""

    def goto_frame(self, f: int) -> None:
        self.page.fill("#frameInput", str(f))
        self.page.press("#frameInput", "Enter")
        self.page.wait_for_function("(f) => document.getElementById('scrub').value == String(f)",
                                    arg=f, timeout=15000)

    def select_behavior(self, name: str) -> None:
        # the chip is a <label> holding the checkbox that selects it -- click it like a person does
        self.page.click(f"#behaviorBar label.chip:has(.bname:text-is('{name}')) .behcheck")
        self.page.wait_for_selector(
            f"#behaviorBar label.chip.active:has(.bname:text-is('{name}'))", timeout=15000)
        self.beat(0.1, 0.3)

    def paint(self, start: int, end: int, kind: str = "pos") -> None:
        """Anchor-paint a bout the way the buttons do it: state button, scrub, state button."""
        btn = {"pos": "#mPos", "neg": "#mNeg", "unknown": "#mErase"}[kind]
        self.goto_frame(start)
        self.beat(0.15, 0.4)
        self.page.click(btn)                       # -> paint_start at the anchor
        self.beat(0.2, 0.6)
        self.goto_frame(end - 1)                   # inclusive playhead; commit makes it exclusive
        self.beat(0.2, 0.7)
        self.page.click(btn)                       # -> paint_commit
        self.led.add("paint", start=start, end=end, kind=kind)
        self.page.wait_for_timeout(150)

    def train(self, timeout_s: float = 900.0) -> dict:
        """Click Train and wait for the status line to report the finished model."""
        self.led.add("train_click")
        self.page.click("#train")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            now = (self.page.inner_text("#metrics") or "").strip()
            if now.startswith("Trained on"):
                self.led.add("train_done", metrics=now[:200])
                return {"metrics": now}
            if "error" in now.lower() or "failed" in now.lower():
                raise RuntimeError(f"train failed: {now}")
            self.page.wait_for_timeout(500)
        raise TimeoutError("train did not finish")

    # -- review
    def candidate(self) -> dict | None:
        """The bout under review, read off the review rail exactly as the annotator reads it."""
        if "on" not in (self.page.get_attribute("#reviewPanel", "class") or ""):
            return None
        txt = self.page.inner_text("#reviewQ")
        m = re.search(r"bout\s+(\d+)[–-](\d+)", txt)
        idx = re.search(r"(\d+)\s*/\s*(\d+)", txt)
        if not m:
            return None
        return {"start": int(m.group(1)), "end": int(m.group(2)),
                "idx": int(idx.group(1)) if idx else None,
                "queue_len": int(idx.group(2)) if idx else None, "text": txt}

    def decide(self, key: str, cand: dict, dwell_s: float, replays: int = 0,
               trim: int | None = None) -> None:
        """Look at the bout for `dwell_s`, optionally replay/trim, then call it."""
        time.sleep(dwell_s)
        for _ in range(replays):
            self.page.keyboard.press("r")
            self.led.add("replay")
            time.sleep(0.4)
        if trim is not None:
            self.goto_frame(trim)
            self.page.keyboard.press("[")
            self.led.add("trim", frame=trim)
            time.sleep(0.2)
        self.page.keyboard.press(key)
        self.led.add({"y": "accept", "n": "reject", "Enter": "skip"}[key],
                     start=cand["start"], end=cand["end"], dwell_s=round(dwell_s, 2),
                     replays=replays, trimmed=trim is not None)
        self.page.wait_for_timeout(400)


# ---------------------------------------------------------------- the session


def overlaps_truth(start: int, end: int, bouts: list[dict], frac: float = 0.4) -> bool:
    """Would an honest annotator accept this proposal? True if it mostly covers a real bout."""
    for b in bouts:
        ov = min(end, b["end"]) - max(start, b["start"])
        if ov > 0 and ov >= frac * (end - start):
            return True
    return False


def run_session(ui: Ui, led: Ledger, truth: dict, behavior: str, n_hand: int) -> None:
    bouts = truth["bouts"]
    rng = ui.rng

    # ---- round 1: label by hand ----
    print(f"\n[round 1] painting {n_hand} bouts by hand", flush=True)
    ui.select_behavior(behavior)
    for i, b in enumerate(bouts[:n_hand]):
        ui.paint(b["start"], b["end"], "pos")
        print(f"    + {behavior} {b['start']}-{b['end']}", flush=True)
        if i == n_hand // 2:
            # The break. Nothing happens for BREAK_S -- the rollup must not bill it to the labeling.
            print(f"    ... {BREAK_S}s break (must NOT count as active time)", flush=True)
            led.add("break_start")
            time.sleep(BREAK_S)
            led.add("break_end")

    # a few negatives, from the gaps between real bouts -- the model needs both sides
    gaps = [(bouts[i]["end"] + 25, bouts[i + 1]["start"] - 25) for i in range(len(bouts) - 1)]
    for (s, e) in [g for g in gaps if g[1] - g[0] > 60][:3]:
        ui.paint(s, min(e, s + 90), "neg")
        print(f"    - not-{behavior} {s}-{min(e, s + 90)}", flush=True)

    m1 = ui.train()
    print(f"    trained: {m1['metrics'][:110]}", flush=True)

    # ---- round 2: review what the model proposes ----
    print("\n[round 2] reviewing candidates", flush=True)
    ui.beat(0.4, 1.0)
    # Half-and-half: pure least-confident surfaces only the ramp frames either side of bouts the
    # annotator already painted, which an honest reviewer rejects every time -- so the run would
    # never exercise an accept. Mixing in random picks brings up the bouts the model found on its
    # own, which is what the workflow is actually for.
    ui.page.select_option("#candOrder", "mixed")
    ui.page.fill("#candN", "14")
    ui.page.dispatch_event("#candN", "change")
    ui.page.wait_for_timeout(1200)
    led.add("review_click")
    ui.page.click("#review")
    ui.page.wait_for_timeout(2500)

    n_dec = n_acc = 0
    while n_dec < 14:
        cand = ui.candidate()
        if cand is None:
            break
        good = overlaps_truth(cand["start"], cand["end"], bouts)
        # dwell is deliberately varied and always > 0.5 s: dwell_ms is read straight out of the log
        # by the rollup, so a zero-length decision would hide a broken measurement.
        dwell = rng.uniform(0.6, 1.8)
        replays = 1 if (n_dec % 4 == 1) else 0
        # Tighten the model's start on every other accepted bout -- trimming is only meaningful on a
        # bout you are keeping, and it has to happen at least once or the fields that record it
        # (candidate_trim, the decision's `trimmed`) go unexercised.
        trim = (cand["start"] + 5) if (good and n_acc % 2 == 1) else None
        key = "y" if good else "n"
        n_acc += 1 if good else 0
        print(f"    {cand['idx']}/{cand['queue_len']} bout {cand['start']}-{cand['end']} "
              f"-> {'accept' if good else 'reject'} (dwell {dwell:.1f}s"
              f"{', replay' if replays else ''}{', trim' if trim else ''})", flush=True)
        ui.decide(key, cand, dwell, replays, trim)
        n_dec += 1
        if ui.candidate() is None:
            break

    if "on" in (ui.page.get_attribute("#reviewPanel", "class") or ""):
        ui.page.click("#rvStop")
        led.add("review_end")
        ui.page.wait_for_timeout(300)

    # one more hand-painted bout after review, then the closing Train
    if len(bouts) > n_hand:
        b = bouts[n_hand]
        ui.paint(b["start"], b["end"], "pos")
        print(f"    + {behavior} {b['start']}-{b['end']} (post-review)", flush=True)
    m2 = ui.train()
    print(f"    trained: {m2['metrics'][:110]}", flush=True)


# ---------------------------------------------------------------- verification


def verify(base: str, pid: str, projects_root: Path, led: Ledger, events: list[dict],
           timing: dict, console_errors: list[str], http_errors: list[str],
           button_csv: str) -> Checks:
    ck = Checks()
    types = [e.get("type") for e in events]
    n = lambda t: types.count(t)
    client = [e for e in events if e.get("src") == "client"]
    rounds = timing["rounds"]

    print("\n--- the log recorded what the driver did ---", flush=True)
    ck(n("session_start") >= 1, "session_start logged")
    ck(n("session_end") >= 1, "session_end survived page close (sendBeacon)")
    ck(n("paint_commit") == led.count("paint"),
       "every painted bout logged", f"log {n('paint_commit')} vs driver {led.count('paint')}")
    ck(n("paint_start") >= led.count("paint"), "paint_start logged per paint",
       f"{n('paint_start')} starts")
    ck(n("train_click") == led.count("train_click"),
       "every Train click logged", f"log {n('train_click')} vs driver {led.count('train_click')}")
    ck(n("candidate_show") == led.count("accept") + led.count("reject") + led.count("skip"),
       "one candidate_show per decision",
       f"{n('candidate_show')} shown / {led.count('accept') + led.count('reject')} decided")
    ck(n("candidate_accept") == led.count("accept"),
       "accepts logged", f"log {n('candidate_accept')} vs driver {led.count('accept')}")
    ck(n("candidate_reject") == led.count("reject"),
       "rejects logged", f"log {n('candidate_reject')} vs driver {led.count('reject')}")
    ck(n("review_start") >= 1 and n("review_end") >= 1, "review session bracketed")
    ck(n("heartbeat") >= 1, "heartbeats present", f"{n('heartbeat')} beats")
    ck(all(e.get("t_ms") for e in client), "every client event carries t_ms")
    # Script errors are the ones that matter: a thrown exception mid-session silently stops the
    # instrumentation. "Failed to load resource" is Chromium narrating an HTTP status, which the
    # next two checks judge on their own terms rather than as JS breakage.
    script_errors = [c for c in console_errors if "Failed to load resource" not in c]
    ck(not script_errors, "no script errors", "; ".join(script_errors[:3]))
    # 404s the page provokes on purpose (no model yet, no predictions yet) are not failures; anything
    # else is. The event-log endpoints in particular must never be among them.
    unexpected = [h for h in http_errors
                  if not re.search(r"/(model|predict|history|candidates)/?", h)]
    ck(not unexpected, "no unexpected HTTP failures", "; ".join(unexpected[:3]))
    ck(not [h for h in http_errors if "/events" in h or "/timing" in h],
       "no failures on the event-log endpoints")

    print("\n--- server-side records survive the browser ---", flush=True)
    srv = [e for e in events if e.get("src") == "server"]
    jobs_done = [e for e in srv if e.get("type") == "job_done"]
    trains = [e for e in jobs_done if e.get("kind") == "train" and e.get("status") == "done"]
    ck(len(trains) == led.count("train_click"),
       "a server job_done per Train", f"{len(trains)} train jobs")
    ck(all(isinstance(e.get("seconds"), (int, float)) for e in jobs_done),
       "job durations recorded server-side")
    ck(all(t.get("ap") is not None for t in trains), "train job_done carries the model's AP",
       f"AP {[t.get('ap') for t in trains]}")
    ck(n("label_write") >= led.count("paint"), "label writes recorded server-side",
       f"{n('label_write')} writes")

    print("\n--- ordering ---", flush=True)
    ts = [float(e.get("t_ms") or 0) for e in events]
    ck(ts == sorted(ts), "events.jsonl is chronological across client + server files")

    print("\n--- rounds ---", flush=True)
    closed = [r for r in rounds if r["closed"]]
    ck(len(closed) == led.count("train_click"),
       "one closed round per Train", f"{len(closed)} closed of {len(rounds)}")
    ck(all(r["behavior"] for r in closed), "each round names its behavior")
    if len(closed) >= 2:
        r1, r2 = closed[0], closed[1]
        # everything the driver painted BEFORE the first Train belongs to round 1, and nothing else
        t_train = next(t for t, k, _ in led.rows if k == "train_click")
        pre = [d for t, k, d in led.rows if k == "paint" and t < t_train]
        ck(r1["manual_bouts"] == len([p for p in pre if p["kind"] == "pos"]),
           "round 1 credited exactly the bouts painted before the first Train",
           f"{r1['manual_bouts']} vs {len([p for p in pre if p['kind'] == 'pos'])}")
        ck(r1["manual_neg_bouts"] == len([p for p in pre if p["kind"] == "neg"]),
           "round 1 negatives credited to round 1", f"{r1['manual_neg_bouts']}")
        ck(r1["label_s"] > 0 and r1["review_s"] == 0,
           "round 1 is pure hand-labeling", f"label {r1['label_s']}s review {r1['review_s']}s")
        ck(r2["review_s"] > 0, "round 2 accrued review time", f"review {r2['review_s']}s")
        ck(r2["review_s"] > r2["label_s"], "round 2 is review-dominated",
           f"review {r2['review_s']}s vs label {r2['label_s']}s")
        painted = sum(r["manual_bouts"] + r["manual_neg_bouts"] + r["manual_unknown_bouts"]
                      for r in rounds)
        ck(painted == led.count("paint"), "every paint is accounted for in some round",
           f"{painted} vs {led.count('paint')}")
        ck(sum(sum(r["decisions"].values()) for r in rounds)
           == led.count("accept") + led.count("reject") + led.count("skip"),
           "decisions add up across rounds")
        ck(r1["train_s"] > 0 and r1["compute_s"] > 0,
           "the Train's compute landed in the round that clicked it",
           f"train_s {r1['train_s']}s")
        ck(r1["wait_s"] > 0, "time spent waiting on Train is its own phase",
           f"wait {r1['wait_s']}s")
        ck(r1["ap"] is not None, "round 1 carries the model it produced", f"AP {r1['ap']}")
        ck(r2["ap"] is not None, "round 2 carries the model it produced", f"AP {r2['ap']}")

    print("\n--- the break is not billed to the annotation ---", flush=True)
    r1 = closed[0] if closed else None
    if r1:
        ck(r1["active_s"] < r1["wall_s"],
           "active time is less than wall time", f"active {r1['active_s']}s of {r1['wall_s']}s wall")
        ck(r1["wall_s"] - r1["active_s"] >= BREAK_S * 0.6,
           f"the {BREAK_S}s break is excluded from active time",
           f"unbilled {round(r1['wall_s'] - r1['active_s'], 1)}s")

    print("\n--- the headline numbers exist and are consistent ---", flush=True)
    tot = timing["totals"]
    pos = [p for p in led.of("paint") if p["kind"] == "pos"]
    neg = [p for p in led.of("paint") if p["kind"] == "neg"]
    # The denominator of the headline. A Not-happening paint is work, not a bout: counting it here
    # would divide the same labeling seconds by a bigger number and flatter the by-hand arm.
    ck(tot["manual"]["bouts"] == len(pos), "totals: hand-painted POSITIVE bouts only",
       f"{tot['manual']['bouts']} vs {len(pos)} positive ({len(neg)} negative painted)")
    ck(tot["manual"]["neg_bouts"] == len(neg), "totals: negatives counted separately, not dropped",
       f"{tot['manual']['neg_bouts']}")
    ck(tot["manual"]["frames"] == sum(p["end"] - p["start"] for p in pos),
       "totals: positive frames match what was painted")
    ck(tot["review"]["decisions"] == led.count("accept") + led.count("reject") + led.count("skip"),
       "totals: decisions")
    ck(tot["review"]["accepted"] == led.count("accept"), "totals: accepted")
    ck(tot["manual"]["s_per_bout"] is not None, "seconds per hand-painted bout",
       f"{tot['manual']['s_per_bout']}s")
    ck(tot["review"]["s_per_decision"] is not None, "seconds per review decision",
       f"{tot['review']['s_per_decision']}s")
    ck(tot["review"]["median_decision_s"] is not None, "median decision dwell",
       f"{tot['review']['median_decision_s']}s")
    dwells = [e.get("dwell_ms") for e in events
              if e.get("type") in ("candidate_accept", "candidate_reject")]
    ck(all(isinstance(d, (int, float)) and d > 0 for d in dwells),
       "every decision carries a positive dwell_ms", f"{len(dwells)} decisions")
    driver_dwell = [d["dwell_s"] for d in led.of("accept") + led.of("reject")]
    if dwells and driver_dwell:
        logged = sorted(round(d / 1000.0, 1) for d in dwells)
        acted = sorted(round(d, 1) for d in driver_dwell)
        close = sum(1 for a, b in zip(logged, acted) if abs(a - b) < 1.5)
        ck(close >= len(acted) - 1, "logged dwell matches how long the driver actually looked",
           f"{close}/{len(acted)} within 1.5s")
    ck(tot["speedup_per_bout"] is not None, "manual-vs-review headline computed",
       f"{tot['speedup_per_bout']}x")

    print("\n--- how much fixing the proposals needed ---", flush=True)
    ck(tot["review"]["candidate_trims"] == led.count("trim"),
       "bound edits rolled up", f"log {tot['review']['candidate_trims']} vs driver {led.count('trim')}")
    ck(tot["review"]["trimmed"] == sum(1 for d in led.of("accept") + led.of("reject")
                                       if d["trimmed"]),
       "decisions that ended with edited bounds rolled up", f"{tot['review']['trimmed']}")
    ck(tot["review"]["replays"] == led.count("replay"),
       "replays rolled up", f"log {tot['review']['replays']} vs driver {led.count('replay')}")
    ck(tot["review"]["frac_trimmed"] is not None, "trim rate reported",
       f"{tot['review']['frac_trimmed']}")

    print("\n--- surfaces ---", flush=True)
    csv = _req(f"{base}/api/projects/{pid}/timing.csv", raw=True).decode()
    head, *body = [ln for ln in csv.splitlines() if ln.strip()]
    ck(head.startswith("round,behavior_id,behavior"), "timing.csv header")
    ck(button_csv.splitlines()[:1] == [head],
       "the header's \u2913 rounds button downloads that same table",
       f"{len(button_csv.splitlines())} lines")
    ck(len(body) == len(rounds), "timing.csv has a row per round", f"{len(body)} rows")
    sess = _req(f"{base}/api/projects/{pid}/events/sessions")["sessions"]
    ck(len(sess) >= 2, "sessions endpoint lists the client + server logs",
       f"{[s['session'] for s in sess]}")
    one = _req(f"{base}/api/projects/{pid}/timing?behavior=999")
    ck(one["n_rounds"] <= len(rounds), "?behavior= filters the rollup")
    try:
        _req(f"{base}/api/projects/{pid}/events", "POST",
             {"session": "../../evil", "events": [{"type": "x"}]})
        ck(False, "path traversal in session id is rejected")
    except urllib.error.HTTPError as e:
        ck(e.code == 400, "path traversal in session id is rejected", f"HTTP {e.code}")

    # the offline path: same rollup, straight off disk, with the server out of the picture
    script = Path(__file__).with_name("annotation_timing.py")
    out = subprocess.run([sys.executable, str(script), str(projects_root), "--pid", pid,
                          "--csv", str(projects_root.parent / "rounds.csv")],
                         capture_output=True, text=True, timeout=120)
    ck(out.returncode == 0, "scripts/annotation_timing.py runs", out.stderr.strip()[-160:])
    ck("BY HAND" in out.stdout and "IN REVIEW" in out.stdout,
       "annotation_timing.py prints both arms")
    ck(f"{len(rounds)} rounds" in out.stdout, "annotation_timing.py agrees on the round count",
       f"expected {len(rounds)}")
    ck((projects_root.parent / "rounds.csv").exists(), "annotation_timing.py --csv writes the table")

    return ck


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("synth", type=Path, help="directory from make_synthetic_clip.py")
    ap.add_argument("--work", type=Path, default=None, help="scratch dir (default: <synth>/verify)")
    ap.add_argument("--behavior", default="drinking")
    ap.add_argument("--hand-bouts", type=int, default=5, help="bouts to paint by hand in round 1")
    ap.add_argument("--headed", action="store_true", help="watch it happen")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("needs playwright:  uv pip install playwright && playwright install chromium",
              file=sys.stderr)
        return 2

    truth = json.loads((a.synth / "truth.json").read_text())
    work = a.work or (a.synth / "verify")
    root = work / "projects"
    root.mkdir(parents=True, exist_ok=True)

    port = _free_port()
    srv = Server(root, port)
    base = srv.base
    led, rng = Ledger(), random.Random(a.seed)
    console_errors: list[str] = []
    http_errors: list[str] = []
    button_csv = ""
    try:
        srv.wait()
        print(f"server {base}  (projects in {root})", flush=True)

        proj = _req(f"{base}/api/projects", "POST", {"name": "loop-verify"})
        pid = proj["pid"]
        entry = _req(f"{base}/api/projects/{pid}/videos", "POST",
                     {"video_path": truth["video"], "slp_path": truth["slp"]})
        print(f"clip {entry['video_id']}: {entry['n_frames']} frames @ {entry['fps']} fps", flush=True)
        _req(f"{base}/api/projects/{pid}/behaviors", "POST", {"name": a.behavior})

        print("waiting for the feature pre-warm…", flush=True)
        for _ in range(600):
            st = _req(f"{base}/api/projects/{pid}")
            fs = (st["videos"][0].get("features") or {}).get("status")
            if fs == "ready":
                print(f"features ready (D={st['videos'][0]['features'].get('D')})", flush=True)
                break
            time.sleep(1)
        else:
            print("features never became ready; training will compute them lazily", flush=True)

        exe = _chromium_path()
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not a.headed,
                                         **({"executable_path": exe} if exe else {}))
            page = browser.new_context(viewport={"width": 1600, "height": 1100}).new_page()
            page.on("console", lambda m: console_errors.append(m.text[:300])
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: console_errors.append(str(e)[:300]))
            page.on("response", lambda r: http_errors.append(f"{r.status} {r.url}")
                    if r.status >= 400 else None)

            # The page imports its video-decode library from a CDN. Serve it through the same egress
            # the rest of this script uses, so a locked-down network does not stop the module (and
            # with it the whole UI) from loading.
            def cdn(route):
                try:
                    with urllib.request.urlopen(route.request.url, timeout=60) as f:
                        route.fulfill(status=200, body=f.read(),
                                      headers={"content-type": f.headers.get_content_type(),
                                               "access-control-allow-origin": "*"})
                except Exception:
                    route.abort()
            page.route(re.compile(r"^https://esm\.sh/"), cdn)

            page.goto(f"{base}/?pid={pid}", wait_until="domcontentloaded")
            page.wait_for_function("() => /\\d+ \\/ [1-9]/.test(document.getElementById"
                                   "('counter').textContent)", timeout=120000)
            page.wait_for_timeout(1500)
            print(f"UI loaded: {page.inner_text('#meta')[:100]}", flush=True)

            ui = Ui(page, led, rng)
            run_session(ui, led, truth, a.behavior, a.hand_bouts)

            # the ⤓ rounds button: the only surface a user reaches without touching the API
            with page.expect_download(timeout=30000) as dl:
                page.click("#timingExport")
            download = dl.value
            csv_path = work / "rounds-from-button.csv"
            download.save_as(str(csv_path))
            button_csv = csv_path.read_text()

            page.wait_for_timeout(3500)      # let the last batch flush
            # Leaving the page is what a closing tab does: pagehide -> session_end via sendBeacon.
            page.goto("about:blank", wait_until="load")
            page.wait_for_timeout(1500)
            page.close()
            browser.close()
        time.sleep(1.5)

        events = [json.loads(ln) for ln in
                  _req(f"{base}/api/projects/{pid}/events.jsonl", raw=True).decode().splitlines()
                  if ln.strip()]
        timing = _req(f"{base}/api/projects/{pid}/timing"
                      f"?gap_cap_s={GAP_CAP_S}&idle_break_s={IDLE_BREAK_S}")
        (work / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n")
        (work / "timing.json").write_text(json.dumps(timing, indent=2))
        (work / "ledger.json").write_text(json.dumps(
            [{"t_ms": t, "kind": k, **d} for t, k, d in led.rows], indent=2))

        print(f"\n{len(events)} events, {timing['n_rounds']} rounds "
              f"-> {work}/events.jsonl, timing.json, ledger.json", flush=True)
        ck = verify(base, pid, root, led, events, timing, console_errors, http_errors,
                    button_csv)

        print("\n--- per-round table (the artifact) ---", flush=True)
        print(_req(f"{base}/api/projects/{pid}/timing.csv"
                   f"?gap_cap_s={GAP_CAP_S}&idle_break_s={IDLE_BREAK_S}", raw=True).decode(),
              flush=True)

        n_ok = sum(1 for ok, _, _ in ck.rows if ok)
        print(f"\n{n_ok}/{len(ck.rows)} checks passed", flush=True)
        if ck.failed:
            print("FAILED:\n  - " + "\n  - ".join(ck.failed), flush=True)
            return 1
        print("OK: paint -> train -> review verified end to end against the event log.", flush=True)
        return 0
    finally:
        srv.stop()


if __name__ == "__main__":
    sys.exit(main())
