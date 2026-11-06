"""Multi-view camera-pose refinement from VGGT depth point clouds."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from .pointcloud import unproject_depth
from .pose_graph import PoseEdge, optimize_pose_graph
from .registration import global_register_fpfh


def _mask_to_vggt(mask_path, preprocess, depth_shape):
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    h, w = depth_shape
    yy, xx = np.mgrid[0:h, 0:w]
    u_original = (xx - preprocess["pad_left"] + 0.5) * (
        preprocess["crop_width"] / preprocess["resized_width"]
    ) - 0.5 + preprocess["crop_left"]
    v_original = (yy - preprocess["pad_top"] + 0.5) * (
        preprocess["crop_height"] / preprocess["resized_height"]
    ) - 0.5 + preprocess["crop_top"]
    sampled = map_coordinates(
        mask, [v_original.ravel(), u_original.ravel()],
        order=0, mode="constant", cval=0.0,
    ).reshape(h, w)
    valid = (
        (u_original >= preprocess["crop_left"])
        & (u_original < preprocess["crop_left"] + preprocess["crop_width"])
        & (v_original >= preprocess["crop_top"])
        & (v_original < preprocess["crop_top"] + preprocess["crop_height"])
    )
    return (sampled >= 0.5) & valid


def _camera_points(view, stride, mask_path=None):
    depth = np.asarray(np.load(view["depth_path"]), dtype=np.float32).squeeze()
    confidence = (
        np.asarray(np.load(view["confidence_path"]), dtype=np.float32).squeeze()
        if view.get("confidence_path") else None
    )
    mask = None
    if mask_path is not None:
        if not view.get("preprocess"):
            raise ValueError("VGGT prediction lacks preprocessing metadata for mask projection")
        mask = _mask_to_vggt(mask_path, view["preprocess"], depth.shape)
    return unproject_depth(
        depth,
        np.asarray(view["intrinsics"], dtype=np.float64),
        np.eye(4),
        confidence=confidence,
        stride=stride,
        mask=mask,
    )


def _candidate_pairs(poses, neighbors):
    centers = np.asarray([T[:3, 3] for T in poses])
    pairs = set()
    for i in range(len(poses)):
        order = np.argsort(np.linalg.norm(centers - centers[i], axis=1))
        for j in [int(k) for k in order if k != i][:neighbors]:
            pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def refine_prediction_file(input_json, output_json, stride=4, voxel_size=0.01,
                           neighbors=4, max_points=80000, masks=None):
    """Refine VGGT-Ω camera poses using pairwise depth-cloud registration."""
    payload = json.loads(Path(input_json).read_text(encoding="utf-8"))
    views = payload.get("views", [])
    if len(views) < 2:
        raise ValueError("At least two views are required for pose refinement")
    if masks is not None and len(masks) != len(views):
        raise ValueError("The number of masks must match the number of VGGT views")
    if stride < 1 or voxel_size <= 0 or neighbors < 1 or max_points < 100:
        raise ValueError("Invalid pose-refinement parameters")
    poses = [np.asarray(v["camera_to_world"], dtype=np.float64) for v in views]
    rng = np.random.default_rng(1234)
    clouds = []
    for i, view in enumerate(views):
        mask_path = masks[i] if masks is not None else None
        points = _camera_points(view, stride, mask_path)
        if len(points) > max_points:
            points = points[rng.choice(len(points), max_points, replace=False)]
        clouds.append(points)

    edges, diagnostics = [], []
    for i, j in _candidate_pairs(poses, neighbors):
        try:
            T_j_i, stats = global_register_fpfh(
                clouds[i], clouds[j], voxel_size=voxel_size
            )
            weight = max(0.1, stats["fitness"]) / max(stats["inlier_rmse"], 1e-3)
            edges.append(PoseEdge(i, j, T_j_i, float(weight), "fpfh_ransac_icp"))
            diagnostics.append({"i": i, "j": j, "accepted": True, **stats})
        except Exception as exc:
            diagnostics.append({"i": i, "j": j, "accepted": False, "reason": str(exc)})

    refined, graph_stats = optimize_pose_graph(poses, edges)
    result = dict(payload)
    result["views"] = []
    for i, view in enumerate(views):
        updated = dict(view)
        updated["camera_to_world_initial"] = view["camera_to_world"]
        updated["camera_to_world"] = refined[i].tolist()
        result["views"].append(updated)
    result["pose_graph"] = {
        **graph_stats,
        "method": "VGGT pose priors + masked FPFH RANSAC/point-to-plane ICP edges + robust SE3 optimization",
        "pairwise_registration": diagnostics,
        "depth_stride": stride,
        "voxel_size": voxel_size,
        "used_masks": masks is not None,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result["pose_graph"]
