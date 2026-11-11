#!/usr/bin/env python3
"""Spot-check a tailreid run by eye: is the same mouse called the same name?

A much lighter ground truth than labeling every switch of an hour. The tool picks random frames from
the decoded hours of a run in which both mice have a complete tail. For each one it shows the two tails
side by side, each turned so its base is on the left and blown up, and asks one question: which tail
carries the given tattoo (default: "singles")? You answer with a key. The page never shows the re-ID's
names, so the answers are blind to it.

Scoring then asks how often re-ID gave the tattoo's mouse the same name. Identities in a tailreid run
are one global frame (mouse_0 is the same animal in every file), so one best mapping is used for the
whole run: this tests switches within hours and consistency across hours and days at once. With two
mice, guessing scores 50%.

    # answer questions in the browser (answers are saved after every click; stop and resume anytime)
    uv run --no-project --python 3.12 --with numpy --with opencv-python-headless python reid_spotcheck.py \\
        ~/tailreid_runs/cam_05_b26

    # the result
    uv run --no-project --python 3.12 --with numpy python reid_spotcheck.py ~/tailreid_runs/cam_05_b26 --score

Reads the run's identities/<stem>.npz (lane_ident, video) and parent/kymos_b*/tails/<stem>.tails.npz
(tail points per lane), and the videos those name. Writes only <run>/spotcheck/ (plan.json, answers.csv).
The frames are drawn once with --seed and kept in plan.json, so restarting continues the same sample;
--new-plan draws again (keeping the old answers file aside). Only frames with both tails complete can be
judged by eye, so the sample is of those frames, which are also the ones re-ID can read.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ sampling


def load_hours(run: Path) -> list[dict]:
    """Decoded hours with identities and a tails cache, in stem order."""
    tails_dirs = sorted(run.glob("parent/kymos_b*/tails"))
    hours = []
    for p in sorted((run / "identities").glob("*.npz")):
        if p.name.endswith(".reid.npz"):
            continue
        z = np.load(p, allow_pickle=False)
        if "lane_ident" not in z.files or not bool(z["decoded"]) or z["lane_ident"].shape[0] == 0:
            continue
        stem = str(z["stem"])
        tp = next((d / f"{stem}.tails.npz" for d in tails_dirs if (d / f"{stem}.tails.npz").exists()), None)
        if tp is None:
            continue
        hours.append(dict(stem=stem, ident=p, tails=tp, video=str(z["video"]), lanes=[str(x) for x in z["lanes"]]))
    return hours


def candidates(h: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frames where every lane has a complete tail and its own identity; plus lane_ident and tails."""
    t = np.load(h["tails"], allow_pickle=False)
    tails, lanes_t = t["tails"], [str(x) for x in t["tracks"]]
    li = np.load(h["ident"], allow_pickle=False)["lane_ident"].astype(np.int16)
    if lanes_t != h["lanes"]:
        return np.zeros(0, int), li, tails
    n = min(len(li), len(tails))
    li, tails = li[:n], tails[:n]
    real = [i for i, nm in enumerate(lanes_t) if not nm.startswith("pad_")]
    if len(real) < 2:
        return np.zeros(0, int), li, tails
    full = np.isfinite(tails[:, real]).all(axis=(1, 2, 3))
    known = (li[:, real] >= 0).all(axis=1)
    distinct = (np.diff(np.sort(li[:, real], axis=1), axis=1) != 0).all(axis=1)
    return np.flatnonzero(full & known & distinct), li, tails


def make_plan(run: Path, n: int, seed: int) -> dict:
    rng = random.Random(seed)
    hours = load_hours(run)
    if not hours:
        sys.exit(f"no decoded hours with a tails cache under {run}")
    by_day = defaultdict(list)
    for h in hours:
        by_day[h["stem"].split("_")[1]].append(h)
    days = sorted(by_day)
    per_day = [n // len(days) + (1 if i < n % len(days) else 0) for i in range(len(days))]
    items, cache = [], {}
    print(f"drawing {n} frames from {len(hours)} decoded hours over {len(days)} days ...", flush=True)
    for d, k in zip(days, per_day):
        pool = by_day[d][:]
        rng.shuffle(pool)
        tries, limit = 0, 20 * max(k, 1)
        while k > 0 and tries < limit:
            tries += 1
            h = rng.choice(pool)
            if h["stem"] not in cache:
                cache[h["stem"]] = candidates(h)
            cand, li, tails = cache[h["stem"]]
            if len(cand) == 0:
                continue
            f = int(rng.choice(cand.tolist()))
            real = [i for i, nm in enumerate(h["lanes"]) if not nm.startswith("pad_")][:2]
            if rng.random() < 0.5:
                real = real[::-1]
            left, right = real
            items.append(dict(stem=h["stem"], frame=f, video=h["video"], left_lane=left, right_lane=right,
                              left_ident=int(li[f, left]), right_ident=int(li[f, right]),
                              left_tail=tails[f, left].round(1).tolist(), right_tail=tails[f, right].round(1).tolist()))
            k -= 1
        if len(cache) > 40:                                            # keep memory bounded
            cache.clear()
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["i"] = i
    return dict(run=str(run), seed=seed, n=len(items), created=time.strftime("%Y-%m-%d %H:%M"), items=items)


# ------------------------------------------------------------------ rendering


class Videos:
    """A few open captures, one lock each (seek + read must not interleave)."""

    def __init__(self, keep=4):
        import cv2
        self.cv2, self.keep, self.caps, self.lock = cv2, keep, {}, threading.Lock()

    def frame(self, path: str, f: int):
        with self.lock:
            ent = self.caps.get(path)
            if ent is None:
                cap = self.cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError(f"cannot open video: {path}")
                ent = self.caps[path] = [cap, threading.Lock(), time.time()]
                if len(self.caps) > self.keep:
                    old = min(self.caps, key=lambda k: self.caps[k][2])
                    self.caps.pop(old)[0].release()
            ent[2] = time.time()
        with ent[1]:
            ent[0].set(self.cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = ent[0].read()
        if not ok:
            raise RuntimeError(f"cannot read frame {f} of {path}")
        return img


def tail_crop(cv2, img, pts, out_w=720):
    """The tail turned so base -> tip points right, cropped around it and scaled to out_w pixels wide."""
    pts = np.asarray(pts, np.float64)
    base, tip = pts[0], pts[-1]
    v = tip - base
    ang = math.degrees(math.atan2(v[1], v[0]))
    c = pts.mean(axis=0)
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    u = v / (np.linalg.norm(v) + 1e-9)
    dev = float(np.max(np.abs((pts - c) @ np.array([-u[1], u[0]])))) if len(pts) else 0.0
    w = int(max(80, length * 1.25 + 30))
    h = int(max(48, 2 * dev + 36))
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, 1.0)
    M[:, 2] += (w / 2 - c[0], h / 2 - c[1])
    crop = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    s = out_w / w
    return cv2.resize(crop, (out_w, max(1, int(h * s))), interpolation=cv2.INTER_CUBIC)


def context(cv2, img, left, right, out_w=560):
    im = img.copy()
    for pts, col, name in ((left, (71, 181, 255), "L"), (right, (255, 201, 76), "R")):   # BGR of the page's orange / blue
        p = np.asarray(pts, np.int32)
        cv2.polylines(im, [p], False, col, 3, cv2.LINE_AA)
        cv2.putText(im, name, tuple(int(x) for x in p[0] + np.array([6, -6])), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3, cv2.LINE_AA)
    s = out_w / im.shape[1]
    return cv2.resize(im, (out_w, int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)


class Renderer:
    def __init__(self, items):
        self.items, self.vids = items, Videos()
        import cv2
        self.cv2, self.cache, self.lock = cv2, {}, threading.Lock()

    def images(self, i):
        with self.lock:
            if i in self.cache:
                return self.cache[i]
        it = self.items[i]
        try:
            img = self.vids.frame(it["video"], it["frame"])
            enc = lambda a: self.cv2.imencode(".jpg", a, [self.cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
            out = dict(left=enc(tail_crop(self.cv2, img, it["left_tail"])), right=enc(tail_crop(self.cv2, img, it["right_tail"])),
                       frame=enc(context(self.cv2, img, it["left_tail"], it["right_tail"])), error=None)
        except Exception as ex:                                        # noqa: BLE001
            out = dict(left=b"", right=b"", frame=b"", error=f"{type(ex).__name__}: {ex}")
        with self.lock:
            self.cache[i] = out
            for k in [k for k in self.cache if k < i - 3]:
                self.cache.pop(k)
        return out

    def prefetch(self, order):
        threading.Thread(target=lambda: [self.images(i) for i in order], daemon=True).start()


# ------------------------------------------------------------------ answers

FIELDS = ["i", "stem", "frame", "left_lane", "right_lane", "answer", "at"]


def read_answers(path: Path) -> dict[int, dict]:
    out = {}
    if path.exists():
        for r in csv.DictReader(path.open()):
            i = int(r["i"])
            out.pop(i, None)                                           # keep file order: last answered is last
            if r.get("answer") != "UNDO":
                out[i] = r
    return out


def append_answer(path: Path, row: dict):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


# ------------------------------------------------------------------ scoring


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def score(run: Path, tattoo: str):
    plan = json.loads((run / "spotcheck" / "plan.json").read_text())
    ans = read_answers(run / "spotcheck" / "answers.csv")
    items = {it["i"]: it for it in plan["items"]}
    # Your answers say which lane carried the tattoo; the names are read from the run's identities as
    # they are NOW, so the same answers score a later re-run of the same files.
    current, judged, unsure, gone = {}, [], 0, 0
    for i, a in ans.items():
        it = items[i]
        if a["answer"] == "?":
            unsure += 1
            continue
        lane = it["left_lane"] if a["answer"] == "L" else it["right_lane"]
        if it["stem"] not in current:
            p = run / "identities" / f"{it['stem']}.npz"
            z = np.load(p, allow_pickle=False) if p.exists() else None
            current[it["stem"]] = z["lane_ident"] if z is not None and bool(z["decoded"]) else None
        li = current[it["stem"]]
        ident = int(li[it["frame"], lane]) if li is not None and it["frame"] < len(li) else -1
        if ident < 0:
            gone += 1
            continue
        judged.append((it["stem"].split("_")[1], it["stem"], ident))
    print(f"run: {run}")
    print(f"answered {len(ans)} of {plan['n']} frames: {len(judged)} judged, {unsure} can't tell"
          + (f", {gone} no longer named by the run" if gone else ""))
    if not judged:
        return
    names = sorted({j[2] for j in judged})
    best = max(names, key=lambda k: sum(j[2] == k for j in judged))
    k = sum(j[2] == best for j in judged)
    lo, hi = wilson(k, len(judged))
    print(f"\nthe {tattoo} mouse was called mouse_{best} in {k}/{len(judged)} frames = {100 * k / len(judged):.1f}% "
          f"(95% CI {100 * lo:.0f}-{100 * hi:.0f}%; guessing = 50%)")
    print("\nday       judged   same name as overall   best name that day")
    by_day = defaultdict(list)
    for d, _, ident in judged:
        by_day[d].append(ident)
    for d in sorted(by_day):
        v = by_day[d]
        same = sum(x == best for x in v)
        day_best = max(set(v), key=v.count)
        print(f"{d}  {len(v):6d}   {100 * same / len(v):5.0f}%  ({same}/{len(v)})        mouse_{day_best} {100 * v.count(day_best) / len(v):4.0f}%")
    print("\nA day whose 'best name' differs from the overall one means the names swapped between days;"
          "\nlow agreement inside a day means switches within hours.")


# ------------------------------------------------------------------ web page

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Re-ID spot check</title>
<style>
:root{--bg:#111418;--fg:#e8eaed;--mut:#9aa0a6;--acc:#ffb547;--acc2:#4cc9ff;--ok:#3ddc84}
body{margin:0;background:var(--bg);color:var(--fg);font:16px system-ui,-apple-system,sans-serif}
main{max-width:1560px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 4px}.mut{color:var(--mut)}
.q{font-size:22px;margin:14px 0}.q b{color:#fff;background:#2a2f37;padding:2px 8px;border-radius:6px}
.row{display:flex;gap:18px;flex-wrap:wrap}.card{background:#1a1e24;border-radius:12px;padding:12px;flex:1;min-width:340px}
.card h2{margin:0 0 8px;font-size:18px}.L h2{color:var(--acc)}.R h2{color:var(--acc2)}
.card img{width:100%;border-radius:6px;background:#000;image-rendering:auto}
.ends{display:flex;justify-content:space-between;color:var(--mut);font-size:13px;margin-top:4px}
.btns{display:flex;gap:12px;margin:16px 0;flex-wrap:wrap}
button{font:600 17px system-ui;padding:14px 22px;border-radius:10px;border:0;cursor:pointer}
.bL{background:var(--acc);color:#111}.bR{background:var(--acc2);color:#111}.bQ{background:#3a404a;color:#fff}.bU{background:transparent;color:var(--mut);border:1px solid #444}
.ctx{max-width:560px}.ctx img{width:100%;border-radius:6px}
.bar{height:8px;background:#2a2f37;border-radius:4px;overflow:hidden;margin:6px 0 2px}.bar div{height:100%;background:var(--ok)}
.err{color:#ff6b6b}.done{font-size:22px;color:var(--ok);margin:30px 0}
kbd{background:#2a2f37;border-radius:4px;padding:1px 6px;font-size:13px}
</style></head><body><main>
<h1>Re-ID spot check</h1>
<div class="mut" id="prog"></div><div class="bar"><div id="pbar" style="width:0"></div></div>
<div id="body"></div>
</main><script>
let cur = null;
const $ = (id) => document.getElementById(id);
async function load() {
  const r = await (await fetch('api/next')).json();
  $('prog').textContent = `${r.answered} of ${r.n} answered` + (r.unsure ? ` · ${r.unsure} can't tell` : '');
  $('pbar').style.width = (100 * r.answered / r.n) + '%';
  if (r.done) { $('body').innerHTML = `<div class="done">All ${r.n} answered. Close this page and run the --score command.</div>`; cur = null; return; }
  cur = r.i; const t = Date.now();
  $('body').innerHTML = `
  <div class="q">Which tail is the <b>${r.tattoo.toUpperCase()}</b> mouse?</div>
  <div class="row">
    <div class="card L"><h2>LEFT tail</h2><img src="img/${r.i}/left?t=${t}"><div class="ends"><span>base</span><span>tip</span></div></div>
    <div class="card R"><h2>RIGHT tail</h2><img src="img/${r.i}/right?t=${t}"><div class="ends"><span>base</span><span>tip</span></div></div>
  </div>
  <div class="btns">
    <button class="bL" onclick="answer('L')">LEFT is ${r.tattoo} <kbd>1</kbd> <kbd>&larr;</kbd></button>
    <button class="bR" onclick="answer('R')">RIGHT is ${r.tattoo} <kbd>2</kbd> <kbd>&rarr;</kbd></button>
    <button class="bQ" onclick="answer('?')">Can't tell <kbd>0</kbd> <kbd>space</kbd></button>
    <button class="bU" onclick="undo()">Undo last <kbd>Backspace</kbd></button>
  </div>
  <div class="ctx"><div class="mut">Whole frame (L and R marked) · ${r.stem} frame ${r.frame}</div><img src="img/${r.i}/frame?t=${t}"></div>
  ${r.error ? `<p class="err">Could not load this frame: ${r.error}. Press "Can't tell" to skip it.</p>` : ''}`;
}
async function answer(a) { if (cur === null) return; const i = cur; cur = null;
  await fetch('api/answer', { method: 'POST', body: JSON.stringify({ i, answer: a }) }); load(); }
async function undo() { await fetch('api/undo', { method: 'POST' }); load(); }
document.addEventListener('keydown', (e) => {
  if (e.key === '1' || e.key === 'ArrowLeft') answer('L');
  else if (e.key === '2' || e.key === 'ArrowRight') answer('R');
  else if (e.key === '0' || e.key === ' ') { e.preventDefault(); answer('?'); }
  else if (e.key === 'Backspace') { e.preventDefault(); undo(); }
});
load();
</script></body></html>"""


def serve(run: Path, plan: dict, port: int, tattoo: str, open_browser: bool):
    sc = run / "spotcheck"
    ans_path = sc / "answers.csv"
    items = plan["items"]
    rend = Renderer(items)
    lock = threading.Lock()

    def next_index():
        done = read_answers(ans_path)
        for it in items:
            if it["i"] not in done:
                return it["i"], done
        return None, done

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            path = self.path.split("?")[0].strip("/")
            if path in ("", "index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "api/next":
                with lock:
                    i, done = next_index()
                base = dict(n=len(items), answered=len(done), unsure=sum(r["answer"] == "?" for r in done.values()), tattoo=tattoo)
                if i is None:
                    return self._json(dict(base, done=True))
                order = [it["i"] for it in items if it["i"] not in done][:4]
                rend.prefetch(order)
                err = rend.images(i)["error"]
                return self._json(dict(base, done=False, i=i, stem=items[i]["stem"], frame=items[i]["frame"], error=err))
            parts = path.split("/")
            if len(parts) == 3 and parts[0] == "img" and parts[2] in ("left", "right", "frame"):
                body = rend.images(int(parts[1]))[parts[2]]
                return self._send(200 if body else 404, body, "image/jpeg")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            path = self.path.strip("/")
            with lock:
                if path == "api/answer" and body.get("answer") in ("L", "R", "?"):
                    it = items[int(body["i"])]
                    append_answer(ans_path, dict(i=it["i"], stem=it["stem"], frame=it["frame"], left_lane=it["left_lane"],
                                                 right_lane=it["right_lane"], answer=body["answer"], at=time.strftime("%Y-%m-%dT%H:%M:%S")))
                    return self._json(dict(ok=True))
                if path == "api/undo":
                    done = read_answers(ans_path)
                    if done:
                        last = done[list(done)[-1]]
                        append_answer(ans_path, dict(last, answer="UNDO", at=time.strftime("%Y-%m-%dT%H:%M:%S")))
                    return self._json(dict(ok=True))
            self._json(dict(error="bad request"), 400)

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://localhost:{port}/"
    print(f"spot check: {len(items)} frames · answers -> {ans_path}\n  open {url}   (Ctrl+C to stop; answers are kept)", flush=True)
    if open_browser:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped; run again to continue where you left off")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default="~/tailreid_runs/cam_05_b26", help="tailreid output folder (its --out)")
    ap.add_argument("--n", type=int, default=150, help="frames to draw (default 150)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tattoo", default="singles", help="the tattoo you are asked to find (default: singles)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--score", action="store_true", help="print the result instead of asking")
    ap.add_argument("--new-plan", action="store_true", help="draw a new sample (moves the old answers aside)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    run = Path(a.run).expanduser()
    sc = run / "spotcheck"
    if a.score:
        return score(run, a.tattoo)
    sc.mkdir(parents=True, exist_ok=True)
    plan_p = sc / "plan.json"
    if a.new_plan and plan_p.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        plan_p.rename(sc / f"plan.{stamp}.json")
        if (sc / "answers.csv").exists():
            (sc / "answers.csv").rename(sc / f"answers.{stamp}.csv")
    if plan_p.exists():
        plan = json.loads(plan_p.read_text())
    else:
        plan = make_plan(run, a.n, a.seed)
        plan_p.write_text(json.dumps(plan))
    serve(run, plan, a.port, a.tattoo, not a.no_browser)


if __name__ == "__main__":
    main()
