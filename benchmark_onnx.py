"""
Offline (no ROS) benchmark for FoundationPose with TAO ONNX refine/score nets.

Uses illustrations/objects_hackathon2.jpeg by default. Depth is optional: if omitted,
a planar fake depth is filled inside the detection mask so the pipeline can run for
timing/smoke tests (poses will not be metrically accurate without real depth).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
from ultralytics import YOLO

from scipy.spatial.transform import Rotation

from foundationpose.estimater import FoundationPose, set_logging_format, set_seed
from foundationpose.onnx_predictors import (
    DEFAULT_REFINER_ONNX,
    DEFAULT_SCORER_ONNX,
    PoseRefinePredictorOnnx,
    ScorePredictorOnnx,
)
from foundationpose.Utils import draw_posed_3d_box, draw_xyz_axis


def _parse_symmetry_angles(s: str | None):
    if s is None or str(s).strip() == "":
        return None
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]


def _symmetry_tfs_from_angles(x_angles=None, y_angles=None, z_angles=None):
    """Cartesian product of per-axis rotations (degrees). None => no symmetry."""
    xa, ya, za = x_angles, y_angles, z_angles
    if xa is None and ya is None and za is None:
        return None
    xs = xa if xa is not None else [0.0]
    ys = ya if ya is not None else [0.0]
    zs = za if za is not None else [0.0]
    tfs = []
    for x_angle in xs:
        for y_angle in ys:
            for z_angle in zs:
                r = Rotation.from_euler("xyz", [x_angle, y_angle, z_angle], degrees=True)
                tf = np.eye(4)
                tf[:3, :3] = r.as_matrix()
                tfs.append(tf)
    return np.array(tfs)


def _default_K(h: int, w: int) -> np.ndarray:
    """Reasonable pinhole intrinsics when no camera_info is available."""
    f = 0.9 * float(w)
    return np.array([[f, 0.0, w * 0.5], [0.0, f, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)


def _load_depth(path: str | None, h: int, w: int, depth_scale: float) -> np.ndarray:
    if path is None or path == "":
        return np.zeros((h, w), dtype=np.float32)
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.shape[0] != h or depth.shape[1] != w:
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    depth = depth.astype(np.float32)
    if depth.max() > 20:  # likely millimeters / uint16
        depth = depth * float(depth_scale)
    return depth


def _fill_fake_depth(depth: np.ndarray, mask: np.ndarray, z_m: float) -> np.ndarray:
    out = depth.copy()
    m = mask.astype(bool)
    if not m.any():
        return out
    out[m] = float(z_m)
    return out


def _detect_mask(rgb_bgr: np.ndarray, seg_model, target_object: str, conf: float):
    """Return binary uint8 mask (H,W) for the largest matching detection, or None."""
    results = seg_model.predict(rgb_bgr, conf=conf, verbose=False)
    if not results:
        return None, None
    r0 = results[0]
    if r0.masks is None or r0.boxes is None or len(r0.boxes) == 0:
        return None, None

    names = r0.names
    target = target_object.lower().strip()
    best_i = None
    best_area = -1
    for i in range(len(r0.boxes)):
        cls_id = int(r0.boxes.cls[i].item())
        cls_name = str(names.get(cls_id, cls_id)).lower()
        if target not in cls_name and cls_name not in target:
            # allow exact COCO class match or substring
            if target != cls_name:
                continue
        mask_i = r0.masks.data[i].cpu().numpy()
        area = float(mask_i.sum())
        if area > best_area:
            best_area = area
            best_i = i
    if best_i is None:
        # fallback: largest mask of any class
        for i in range(len(r0.boxes)):
            mask_i = r0.masks.data[i].cpu().numpy()
            area = float(mask_i.sum())
            if area > best_area:
                best_area = area
                best_i = i
                target = str(names.get(int(r0.boxes.cls[i].item()), "?"))
        logging.warning(f"No class match for '{target_object}', using largest mask ({target})")
    if best_i is None:
        return None, None

    mask = r0.masks.data[best_i].cpu().numpy()
    h, w = rgb_bgr.shape[:2]
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    mask_u8 = (mask > 0.5).astype(np.uint8)
    xyxy = r0.boxes.xyxy[best_i].cpu().numpy().astype(int).tolist()
    return mask_u8, xyxy


def _mask_from_bbox(h: int, w: int, bbox_xyxy) -> np.ndarray:
    umin, vmin, umax, vmax = [int(v) for v in bbox_xyxy]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[vmin:vmax, umin:umax] = 1
    return mask


def main(args):
    set_logging_format(level=getattr(logging, args.fp_verbosity.upper(), logging.INFO))
    set_seed(0)

    rgb_bgr = cv2.imread(args.rgb_file)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {args.rgb_file}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = rgb.shape[:2]
    logging.info(f"RGB {args.rgb_file}: {w0}x{h0}")

    # --- segmentation / mask at native resolution, then optional downscale ---
    t_det0 = time.perf_counter()
    if args.bbox:
        bbox = [float(x) for x in args.bbox.split(",")]
        assert len(bbox) == 4, "--bbox must be umin,vmin,umax,vmax"
        mask = _mask_from_bbox(h0, w0, bbox)
        xyxy = [int(v) for v in bbox]
        det_ms = (time.perf_counter() - t_det0) * 1000.0
        logging.info(f"Using --bbox {xyxy}")
    else:
        logging.info(f"Loading seg model {args.seg_model_name}")
        seg_model = YOLO(args.seg_model_name)
        mask, xyxy = _detect_mask(rgb_bgr, seg_model, args.target_object, args.det_conf)
        det_ms = (time.perf_counter() - t_det0) * 1000.0
        if mask is None:
            raise RuntimeError(
                f"No detection for target_object={args.target_object!r}. "
                "Pass --bbox umin,vmin,umax,vmax instead."
            )
        logging.info(f"Detection bbox={xyxy} mask_pixels={int(mask.sum())} ({det_ms:.1f} ms)")

    depth = _load_depth(args.depth_file, h0, w0, args.depth_scale)
    used_fake_depth = False
    if depth.max() < 1e-6:
        depth = _fill_fake_depth(depth, mask, args.fake_depth_m)
        used_fake_depth = True
        logging.warning(
            f"No real depth; filled mask with planar z={args.fake_depth_m}m "
            "(timing only — pose will not be accurate)."
        )

    if args.K_file:
        K = np.loadtxt(args.K_file).reshape(3, 3).astype(np.float64)
    else:
        K = _default_K(h0, w0)
        logging.warning(f"No --K_file; using default intrinsics:\n{K}")

    rf = max(1, int(args.resize_factor))
    if rf > 1:
        h, w = h0 // rf, w0 // rf
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        K = K.copy()
        K[:2] /= float(rf)
        if xyxy is not None:
            xyxy = [int(round(v / rf)) for v in xyxy]
        logging.info(f"Resized by {rf}: {w}x{h}")
    else:
        h, w = h0, w0

    # --- mesh + ONNX estimators ---
    assert os.path.isfile(args.mesh_file), args.mesh_file
    mesh = trimesh.load(args.mesh_file, force="mesh")
    mesh_name = os.path.splitext(os.path.basename(args.mesh_file))[0]
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox3d = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    sym = _symmetry_tfs_from_angles(
        _parse_symmetry_angles(args.symmetry_x_angles),
        _parse_symmetry_angles(args.symmetry_y_angles),
        _parse_symmetry_angles(args.symmetry_z_angles),
    )

    t_load0 = time.perf_counter()
    refiner = PoseRefinePredictorOnnx(
        onnx_path=args.refiner_onnx or DEFAULT_REFINER_ONNX,
        prefer_tensorrt=not args.no_prefer_tensorrt,
        infer_batch_size=args.infer_batch_size,
    )
    scorer = ScorePredictorOnnx(
        onnx_path=args.scorer_onnx or DEFAULT_SCORER_ONNX,
        prefer_tensorrt=not args.no_prefer_tensorrt,
        infer_batch_size=args.infer_batch_size,
    )
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=args.out_dir,
        debug=args.debug,
        glctx=glctx,
        symmetry_tfs=sym,
        meshname=mesh_name,
    )
    load_ms = (time.perf_counter() - t_load0) * 1000.0
    logging.info(f"ONNX predictors + FoundationPose ready in {load_ms:.1f} ms")

    os.makedirs(args.out_dir, exist_ok=True)

    # warmup
    if args.warmup > 0:
        logging.info(f"Warmup x{args.warmup}")
        for _ in range(args.warmup):
            _ = est.register(
                K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=args.est_refine_iter
            )
            est.pose_last = None
        torch.cuda.synchronize()

    times_ms = []
    pose = None
    for i in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pose = est.register(
            K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=args.est_refine_iter
        )
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt)
        logging.info(f"iter {i+1}/{args.iters}: register {dt:.1f} ms")
        est.pose_last = None

    times_ms = np.asarray(times_ms, dtype=np.float64)
    summary = {
        "rgb_file": os.path.abspath(args.rgb_file),
        "mesh_file": os.path.abspath(args.mesh_file),
        "refiner_onnx": os.path.abspath(args.refiner_onnx or DEFAULT_REFINER_ONNX),
        "scorer_onnx": os.path.abspath(args.scorer_onnx or DEFAULT_SCORER_ONNX),
        "providers_refiner": list(refiner.session.providers),
        "providers_scorer": list(scorer.session.providers),
        "image_hw": [h, w],
        "bbox_xyxy": xyxy,
        "used_fake_depth": used_fake_depth,
        "det_ms": det_ms,
        "load_ms": load_ms,
        "register_ms_mean": float(times_ms.mean()),
        "register_ms_std": float(times_ms.std()),
        "register_ms_min": float(times_ms.min()),
        "register_ms_max": float(times_ms.max()),
        "register_ms_all": times_ms.tolist(),
        "pose": pose.tolist() if pose is not None else None,
    }
    summary_path = os.path.join(args.out_dir, "benchmark_onnx_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(
        f"register mean={summary['register_ms_mean']:.1f} ms "
        f"±{summary['register_ms_std']:.1f} "
        f"(min={summary['register_ms_min']:.1f}, max={summary['register_ms_max']:.1f})"
    )
    logging.info(f"Wrote {summary_path}")

    # visualization
    center_pose = pose @ np.linalg.inv(to_origin)
    vis = rgb.copy()
    vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox3d)
    vis = draw_xyz_axis(
        vis, ob_in_cam=center_pose, scale=0.08, K=K, thickness=3, transparency=0, is_input_rgb=True
    )
    if xyxy is not None:
        u0, v0, u1, v1 = xyxy
        cv2.rectangle(vis, (u0, v0), (u1, v1), (0, 255, 0), 2)
    out_img = os.path.join(args.out_dir, "benchmark_onnx_vis.png")
    cv2.imwrite(out_img, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    mask_path = os.path.join(args.out_dir, "benchmark_onnx_mask.png")
    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
    logging.info(f"Wrote {out_img}")
    print(json.dumps({k: summary[k] for k in (
        "register_ms_mean", "register_ms_std", "register_ms_min", "register_ms_max",
        "providers_refiner", "providers_scorer", "used_fake_depth",
    )}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline FoundationPose ONNX benchmark (no ROS)."
    )
    parser.add_argument(
        "--rgb_file",
        type=str,
        default="./illustrations/objects_hackathon2.jpeg",
        help="RGB image path.",
    )
    parser.add_argument(
        "--depth_file",
        type=str,
        default="",
        help="Optional depth image (meters or mm). Empty => fake planar depth in mask.",
    )
    parser.add_argument(
        "--K_file",
        type=str,
        default="",
        help="Optional 3x3 intrinsics txt. Empty => default pinhole.",
    )
    parser.add_argument(
        "--mesh_file",
        type=str,
        default="./assets/hackathon2/milk/milk.obj",
        help="Object mesh (.obj).",
    )
    parser.add_argument(
        "--target_object",
        type=str,
        default="bottle",
        help="YOLO class / name substring for mask selection.",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default="",
        help="Optional umin,vmin,umax,vmax; skips YOLO if set.",
    )
    parser.add_argument(
        "--seg_model_name",
        type=str,
        default="yolo26n-seg.pt",
        help="Ultralytics YOLO-seg weights.",
    )
    parser.add_argument("--det_conf", type=float, default=0.25, help="Detection confidence.")
    parser.add_argument(
        "--fake_depth_m",
        type=float,
        default=0.55,
        help="Planar depth (meters) inside mask when no --depth_file.",
    )
    parser.add_argument("--depth_scale", type=float, default=0.001, help="Scale if depth is mm.")
    parser.add_argument("--est_refine_iter", type=int, default=3, help="Register refine iters.")
    parser.add_argument(
        "--resize_factor",
        type=int,
        default=2,
        help="Divide image / intrinsics by this factor (saves GPU memory).",
    )
    parser.add_argument("--iters", type=int, default=3, help="Timed register repetitions.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup register runs.")
    parser.add_argument("--debug", type=int, default=0, help="FoundationPose debug level.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./debug_node/benchmark_onnx",
        help="Output directory for vis + JSON summary.",
    )
    parser.add_argument("--refiner_onnx", type=str, default="", help="Path to refiner_net.onnx.")
    parser.add_argument("--scorer_onnx", type=str, default="", help="Path to score_net.onnx.")
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=64,
        help="ONNX forward micro-batch (lower if GPU OOM).",
    )
    parser.add_argument(
        "--no_prefer_tensorrt",
        action="store_true",
        default=False,
        help="Disable TensorRT EP.",
    )
    parser.add_argument(
        "--symmetry_z_angles",
        "-sza",
        type=str,
        default="0,90,180,270",
        help="Z symmetry angles (fewer => less GPU memory on first register).",
    )
    parser.add_argument("--symmetry_x_angles", "-sxa", type=str, default=None)
    parser.add_argument("--symmetry_y_angles", "-sya", type=str, default=None)
    parser.add_argument(
        "--fp_verbosity",
        "-v",
        type=str,
        default="warning",
        help="Logging verbosity.",
    )
    args = parser.parse_args()
    if not args.bbox:
        args.bbox = ""
    main(args)
