"""Generate a synthetic clip (video + .slp) with a known, learnable behavior.

The point is to make the paint -> train -> review loop runnable on a machine that has no real
data. The mice sample this repo's other scripts want (`mice.tracked.slp`) is not in this
repository, so without something like this there is no way to exercise the loop end to end --
which is exactly the gap the annotation event log's verification was missing.

What it writes:

    <out>/synth.mp4    grayscale clip, two blobs moving
    <out>/synth.slp    two tracks x 9 nodes x N frames, no gaps
    <out>/truth.json   the ground-truth bouts of the planted behavior

The planted behavior is "drinking": animal 0 parks its nose at the spout (a fixed point) and
holds still for a while. It is deliberately easy -- nose-to-spout distance plus low speed
separate it almost perfectly -- because the thing under test is the instrumentation around the
loop, not the classifier. A behavior the model cannot learn would produce no candidates and so
no review phase to measure.

    uv run python scripts/make_synthetic_clip.py /tmp/synth

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NODES = ["nose", "left_ear", "right_ear", "neck", "thorax", "hip_left", "hip_right",
         "tail_base", "tail_tip"]
EDGES = [("nose", "neck"), ("left_ear", "neck"), ("right_ear", "neck"), ("neck", "thorax"),
         ("thorax", "hip_left"), ("thorax", "hip_right"), ("thorax", "tail_base"),
         ("tail_base", "tail_tip")]

# Offsets of each node from the animal's centroid, in its own heading frame (x = forward).
# Roughly mouse-shaped at ~40 px nose-to-tail, which is body-length scale for a 320x256 cage.
BODY = {
    "nose": (18.0, 0.0), "left_ear": (11.0, -5.0), "right_ear": (11.0, 5.0),
    "neck": (8.0, 0.0), "thorax": (0.0, 0.0), "hip_left": (-8.0, -4.0),
    "hip_right": (-8.0, 4.0), "tail_base": (-12.0, 0.0), "tail_tip": (-22.0, 0.0),
}

W, H = 320, 256
FPS = 30.0
SPOUT = (286.0, 128.0)      # on the right wall; animal 0 reaches it to "drink"


def _plan_bouts(n_frames: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Bouts of 40-90 frames with >=60-frame gaps, so postprocessing cannot merge two into one."""
    bouts, t = [], 120
    while t < n_frames - 150:
        dur = int(rng.integers(40, 91))
        bouts.append((t, t + dur))
        t += dur + int(rng.integers(60, 200))
    return bouts


def _walk(n: int, rng: np.random.Generator, box=(30, W - 40, 30, H - 30)) -> np.ndarray:
    """A smooth random walk of centroids inside `box`, as (n, 2)."""
    x0, x1, y0, y1 = box
    pos = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
    vel = rng.normal(0, 1.2, 2)
    out = np.empty((n, 2), "float64")
    for i in range(n):
        vel += rng.normal(0, 0.45, 2)
        vel *= 0.93
        pos = pos + vel
        for d, (lo, hi) in enumerate(((x0, x1), (y0, y1))):
            if pos[d] < lo or pos[d] > hi:
                pos[d] = np.clip(pos[d], lo, hi)
                vel[d] *= -0.6
        out[i] = pos
    return out


def _pose_from(centroids: np.ndarray, headings: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(F, 2) centroids + (F,) headings -> (F, N, 2) node positions, with per-node jitter."""
    c, s = np.cos(headings), np.sin(headings)
    out = np.empty((len(centroids), len(NODES), 2), "float64")
    for j, name in enumerate(NODES):
        fx, fy = BODY[name]
        out[:, j, 0] = centroids[:, 0] + fx * c - fy * s
        out[:, j, 1] = centroids[:, 1] + fx * s + fy * c
    return out + rng.normal(0, 0.35, out.shape)


def build_poses(n_frames: int, seed: int = 0):
    """(F, 2, N, 2) poses + the ground-truth bouts of animal 0's planted behavior."""
    rng = np.random.default_rng(seed)
    bouts = _plan_bouts(n_frames, rng)

    # --- animal 0: wanders, but is parked at the spout during each bout ---
    cent = _walk(n_frames, rng)
    for (s0, e0) in bouts:
        # approach over 12 frames, hold, then leave over 12 -- a plateau with soft edges, so the
        # model has to pick bounds rather than read a step function.
        park = np.array([SPOUT[0] - 19.0, SPOUT[1] + rng.uniform(-6, 6)])
        ramp = 12
        for i in range(max(0, s0 - ramp), min(n_frames, e0 + ramp)):
            if i < s0:
                w = (i - (s0 - ramp)) / ramp
            elif i >= e0:
                w = 1.0 - (i - e0) / ramp
            else:
                w = 1.0
            w = float(np.clip(w, 0, 1))
            cent[i] = (1 - w) * cent[i] + w * (park + rng.normal(0, 0.4, 2))

    # heading: face travel direction, except at the spout where it faces the spout and holds
    vel = np.gradient(cent, axis=0)
    head = np.arctan2(vel[:, 1], vel[:, 0])
    head = np.unwrap(head)
    for (s0, e0) in bouts:
        head[s0:e0] = np.arctan2(SPOUT[1] - cent[s0:e0, 1], SPOUT[0] - cent[s0:e0, 0])
    p0 = _pose_from(cent, head, rng)

    # --- animal 1: just wanders, never drinks (keeps the left half so it is a distinct track) ---
    cent1 = _walk(n_frames, rng, box=(30, W // 2, 30, H - 30))
    vel1 = np.gradient(cent1, axis=0)
    p1 = _pose_from(cent1, np.unwrap(np.arctan2(vel1[:, 1], vel1[:, 0])), rng)

    poses = np.stack([p0, p1], axis=1)                    # (F, 2, N, 2)
    return poses.astype("float32"), bouts


def write_video(path: Path, poses: np.ndarray) -> None:
    """Draw the poses as blobs so the clip is watchable in the UI (and decodes frame-exact)."""
    import cv2

    n_frames = poses.shape[0]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H), isColor=False)
    if not vw.isOpened():
        raise RuntimeError(f"could not open {path} for writing")
    yy, xx = np.mgrid[0:H, 0:W].astype("float32")
    for f in range(n_frames):
        img = np.full((H, W), 24, "uint8")
        cv2.circle(img, (int(SPOUT[0]), int(SPOUT[1])), 5, 200, -1)     # the spout, as a landmark
        for t in range(poses.shape[1]):
            pts = poses[f, t]
            # a soft body: sum of gaussians at the nodes, so it looks like an animal not a stick
            acc = np.zeros((H, W), "float32")
            for (x, y) in pts:
                acc += np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * 5.5 ** 2)))
            img = np.maximum(img, (np.clip(acc, 0, 1) * (215 if t == 0 else 150)).astype("uint8"))
        vw.write(img)
    vw.release()


def write_slp(path: Path, video_path: Path, poses: np.ndarray) -> None:
    import sleap_io as sio

    skel = sio.Skeleton(nodes=NODES, edges=EDGES)
    video = sio.Video.from_filename(str(video_path))
    tracks = [sio.Track(name="animal0"), sio.Track(name="animal1")]
    frames = []
    for f in range(poses.shape[0]):
        insts = []
        for t, track in enumerate(tracks):
            pts = np.concatenate([poses[f, t], np.ones((len(NODES), 1), "float32")], axis=1)
            insts.append(sio.PredictedInstance.from_numpy(
                points_data=pts, skeleton=skel, track=track, score=1.0))
        frames.append(sio.LabeledFrame(video=video, frame_idx=f, instances=insts))
    sio.save_file(sio.Labels(labeled_frames=frames, videos=[video], skeletons=[skel],
                             tracks=tracks), str(path))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="output directory (created if missing)")
    ap.add_argument("--frames", type=int, default=3000, help="clip length (default 3000 = 100 s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-video", action="store_true",
                    help="poses only -- enough to train, not enough for the UI to show frames")
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    poses, bouts = build_poses(a.frames, a.seed)
    mp4, slp = a.out / "synth.mp4", a.out / "synth.slp"

    if not a.no_video:
        print(f"video  {mp4}  ({a.frames} frames, {W}x{H})", flush=True)
        write_video(mp4, poses)
    print(f"poses  {slp}  ({poses.shape[0]} frames x {poses.shape[1]} animals x {len(NODES)} nodes)",
          flush=True)
    write_slp(slp, mp4, poses)

    truth = {"video": str(mp4), "slp": str(slp), "fps": FPS, "n_frames": a.frames,
             "spout": list(SPOUT), "behavior": "drinking", "track": 0,
             "bouts": [{"start": s, "end": e} for s, e in bouts]}
    (a.out / "truth.json").write_text(json.dumps(truth, indent=2))
    total = sum(e - s for s, e in bouts)
    print(f"truth  {a.out / 'truth.json'}  ({len(bouts)} bouts, {total} frames "
          f"= {total / FPS:.0f}s of behavior)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
