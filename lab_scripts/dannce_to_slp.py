"""Convert one camera view of an sDANNCE save_data_AVG0.mat 3D-pose file into a
per-camera 2D .slp that laras-labeler (sleap-io) can load alongside that camera's video.

DANNCE poses are triangulated 3D world coordinates (mm), not pixel coordinates, so they
are re-projected into a chosen camera's 2D image plane using that camera's calibration
(calibration.json, standard OpenCV convention: K, rotation_matrix, translation_vector,
r_distort=[k1,k2], t_distort=[p1,p2]). Verified by re-projecting frame 0 onto Camera1's
actual video frame and visually confirming the points land on the rat.

It also MEASURES the camera's pixel scale (px/cm) and writes it next to the .slp as
`<out>.slp.scale.json`. HiDRA's features are in centimetres, so it needs this number, and the
labeler reads the file automatically. Here both the true size (3D, mm) and the pixel size (2D
projection) of the same body segments are known, so the scale is measured, not guessed: the
median of pixels/mm over long head-and-trunk segments, over the whole session, both animals.
The file also records the spread (IQR) and the image size it was measured on, so the labeler can
rescale it for a resized video or refuse it for a cropped one.

    # convert + measure
    python dannce_to_slp.py .../day1/cage1RB --camera Camera1 --out ~/laras-poses/cage1RB_day1_camera1.slp
    # only (re)measure the scale for a .slp you already have (does not touch the .slp)
    python dannce_to_slp.py .../day1/cage1RB --camera Camera1 --out ~/laras-poses/cage1RB_day1_camera1.slp --scale-only
"""
import argparse, datetime, json
from pathlib import Path

import cv2
import numpy as np
import scipy.io
import sleap_io as sio

# Written next to the .slp; laras_labeler/scale.py reads exactly this name.
SIDECAR_SUFFIX = ".scale.json"

# Long, stable segments of the 23-keypoint rat skeleton, 0-based indices (order verified against the
# data and the s-DANNCE paper's rat23 definition): Snout-TailBase, Snout-SpineL, SpineF-TailBase,
# SpineF-SpineL. Limbs are excluded: they change length in the image far more as the animal moves.
RAT23_SEGMENTS = [(0, 6), (0, 5), (3, 6), (3, 5)]
RAT23_N_KEYPOINTS = 23

# rat23 node names in order. Without real names the labeler drops its heading and social features.
RAT23_NODE_NAMES = [
    "nose", "ear_left", "ear_right", "neck", "body_center", "spine_lower", "tail_base",
    "shoulder_left", "elbow_left", "wrist_left", "forepaw_left",
    "shoulder_right", "elbow_right", "wrist_right", "forepaw_right",
    "hip_left", "knee_left", "ankle_left", "hindpaw_left",
    "hip_right", "knee_right", "ankle_right", "hindpaw_right",
]


def project_camera(pts3d_flat: np.ndarray, cam_params: dict) -> np.ndarray:
    K = np.array(cam_params["camera_matrix"], dtype=np.float64)
    R = np.array(cam_params["rotation_matrix"], dtype=np.float64)
    t = np.array(cam_params["translation_vector"], dtype=np.float64).reshape(3)
    dist = np.array(cam_params["r_distort"][:2] + cam_params["t_distort"][:2], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    uv, _ = cv2.projectPoints(pts3d_flat, rvec, t, K, dist)
    return uv.reshape(-1, 2)


def estimate_px_per_cm(pred_mm: np.ndarray, uv_px: np.ndarray, step: int = 25) -> dict:
    """px/cm from the 3D poses (mm) and their pixel projection.

    pred_mm: (F, T, 3, N) world coordinates in mm;  uv_px: (F, T, N, 2) pixel coordinates.
    Refuses (ValueError) unless the skeleton is the 23-keypoint rat layout the segment indices assume.
    """
    F, T, _, N = pred_mm.shape
    if N != RAT23_N_KEYPOINTS:
        raise ValueError(f"expected the 23-keypoint rat skeleton, got {N} keypoints — "
                         "not guessing which segments to use; set px/cm by hand")
    p3 = pred_mm[::step].transpose(0, 1, 3, 2)          # (f, T, N, 3)
    p2 = uv_px[::step]                                  # (f, T, N, 2)
    per_segment, samples = {}, []
    for a, b in RAT23_SEGMENTS:
        d3 = np.linalg.norm(p3[:, :, a] - p3[:, :, b], axis=-1)          # mm, true length
        d2 = np.linalg.norm(p2[:, :, a] - p2[:, :, b], axis=-1)          # px, as seen by this camera
        ok = np.isfinite(d3) & np.isfinite(d2) & (d3 > 0)
        if not ok.any():
            continue
        med = np.median(d3[ok])
        ok &= np.abs(d3 - med) <= 0.2 * med                              # drop tracking glitches (bone length is ~constant)
        if ok.sum() < 100:
            continue
        r = d2[ok] / d3[ok] * 10.0                                       # px per mm -> px per cm
        per_segment[f"{a}-{b}"] = float(np.median(r))
        samples.append(r)
    if not samples:
        raise ValueError("no usable body segments to measure the scale from")
    r = np.concatenate(samples)
    med = float(np.median(r))
    spread = (max(per_segment.values()) - min(per_segment.values())) / med
    return {"pix_per_cm": round(med, 3), "iqr": [round(float(np.percentile(r, 25)), 3), round(float(np.percentile(r, 75)), 3)],
            "n_samples": int(r.size), "per_segment": {k: round(v, 3) for k, v in per_segment.items()},
            "segment_disagreement": round(float(spread), 4)}


def write_scale_sidecar(slp_out: Path, est: dict, width: int, height: int, camera: str, source_mat: str, calib: str) -> Path:
    path = Path(str(slp_out) + SIDECAR_SUFFIX)
    doc = {**est, "image_size": [int(width), int(height)], "camera": camera,
           "method": "median px/mm x 10 over long head/trunk segments (Snout-TailBase, Snout-SpineL, SpineF-TailBase, "
                     "SpineF-SpineL) of the 3D poses vs their camera projection, tracking glitches removed",
           "source_mat": source_mat, "calibration": calib,
           "made_by": "lab_scripts/dannce_to_slp.py", "made_on": datetime.date.today().isoformat()}
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def default_mat(cage_dir: Path) -> str:
    """The sDANNCE output to use: predict02 when it exists, else predict01.

    Some sessions have both. predict01 can stop partway (60000 of 105000 frames on the day 2-5
    sessions), while predict02 holds the same poses for the whole recording."""
    for name in ("predict02", "predict01"):
        rel = f"sDANNCE/{name}/save_data_AVG0.mat"
        if (cage_dir / rel).exists():
            return rel
    raise SystemExit(f"no sDANNCE/predict02 or predict01 save_data_AVG0.mat under {cage_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cage_dir", help="e.g. .../day1/cage1RB")
    ap.add_argument("--mat", default=None,
                    help="pose file relative to cage_dir (default: sDANNCE/predict02/save_data_AVG0.mat, "
                         "or predict01 when there is no predict02)")
    ap.add_argument("--calib", default="calibration/calibration.json")
    ap.add_argument("--camera", default="Camera1")
    ap.add_argument("--out", default=None, help="output .slp path")
    ap.add_argument("--max-frames", type=int, default=None, help="truncate for a quick test")
    ap.add_argument("--scale-only", action="store_true",
                    help="only (re)measure px/cm and write <out>.scale.json; do not write the .slp")
    ap.add_argument("--no-scale", action="store_true", help="convert without measuring/writing the scale sidecar")
    args = ap.parse_args()

    cage_dir = Path(args.cage_dir)
    args.mat = args.mat or default_mat(cage_dir)
    print(f"poses from {cage_dir / args.mat}")
    cal = json.load(open(cage_dir / args.calib))
    cam_idx = cal["camera_names"].index(args.camera)
    cam_params = cal["camera_params"][cam_idx]

    m = scipy.io.loadmat(cage_dir / args.mat)
    pred = m["pred"]  # (F, T, 3, N)
    p_max = m["p_max"]  # (F, T, N)
    if args.max_frames:
        pred = pred[: args.max_frames]
        p_max = p_max[: args.max_frames]
    F, T, _, N = pred.shape
    print(f"{F} frames, {T} animals, {N} keypoints")

    pts3d = pred.transpose(0, 1, 3, 2).reshape(-1, 3).astype(np.float64)  # (F*T*N, 3)
    uv = project_camera(pts3d, cam_params).reshape(F, T, N, 2)

    video_path = cage_dir / "videos" / args.camera / "0.mp4"
    video = sio.Video.from_filename(str(video_path))
    if video.shape[0] != F:
        print(f"WARNING: video has {video.shape[0]} frames, poses have {F} -- check alignment")
    height, width = int(video.shape[1]), int(video.shape[2])

    out = Path(args.out) if args.out else cage_dir / f"{args.camera.lower()}_poses.slp"

    if not args.no_scale:
        try:
            est = estimate_px_per_cm(pred.astype(np.float64), uv)
        except ValueError as e:
            print(f"NOT writing a scale file: {e}")
        else:
            side = write_scale_sidecar(out, est, width, height, args.camera, str(cage_dir / args.mat), str(cage_dir / args.calib))
            lo, hi = est["iqr"]
            print(f"scale: {est['pix_per_cm']} px/cm (middle half of measurements {lo}-{hi}; {est['n_samples']} samples; "
                  f"segments disagree by {est['segment_disagreement']:.1%}; image {width}x{height}) -> {side}")
            if est["segment_disagreement"] > 0.10:
                print("WARNING: the body segments give scales differing by more than 10% -- check the poses/calibration")

    if args.scale_only:
        return

    tracks_arr = np.concatenate([uv, p_max[..., None]], axis=-1).astype("float32")  # (F,T,N,3)
    node_names = RAT23_NODE_NAMES if N == RAT23_N_KEYPOINTS else [f"kp{i:02d}" for i in range(N)]
    skeleton = sio.Skeleton(nodes=node_names, name="dannce23")
    tracks = [sio.Track(name=f"rat{i}") for i in range(T)]

    labels = sio.Labels.from_numpy(
        tracks_arr, videos=[video], skeletons=[skeleton], tracks=tracks,
        first_frame=0, return_confidence=True,
    )

    labels.save(str(out))
    print(f"saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
