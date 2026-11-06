"""Multi-view camera-pose refinement from VGGT depth point clouds."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .pointcloud import unproject_depth
from .pose_graph import PoseEdge, optimize_pose_graph
from .registration import global_register_fpfh


def _camera_points(view, stride):
    depth = np.load(view["depth_path"])
    confidence = np.load(view["confidence_path"]) if view.get("confidence_path") else None
    return unproject_depth(depth, np.asarray(view["intrinsics"], dtype=np.float64),
                           np.eye(4), confidence=confidence, stride=stride)


def _candidate_pairs(poses, neighbors):
    centers = np.asarray([T[:3, 3] for T in poses])
    pairs = set()
    for i in range(len(poses)):
        order = np.argsort(np.linalg.norm(centers - centers[i], axis=1))
        for j in [int(k) for k in order if k != i][:neighbors]:
            pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def refine_prediction_file(input_json, output_json, stride=4, voxel_size=0.01,
                           neighbors=4, max_points=80000):
    payload = json.loads(Path(input_json).read_text(encoding="utf-8"))
    views = payload.get("views", [])
    if len(views) < 2:
        raise ValueError("At least two views are required for pose refinement")
    poses = [np.asarray(v["camera_to_world"], dtype=np.float64) for v in views]
    rng = np.random.default_rng(1234)
    clouds = []
    for view in views:
        points = _camera_points(view, stride)
        if len(points) > max_points:
            points = points[rng.choice(len(points), max_points, replace=False)]
        clouds.append(points)

    edges, diagnostics = [], []
    for i, j in _candidate_pairs(poses, neighbors):
        try:
            T_j_i, stats = global_register_fpfh(clouds[i], clouds[j], voxel_size=voxel_size)
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
        "method": "VGGT pose priors + FPFH RANSAC/point-to-plane ICP edges + robust SE3 optimization",
        "pairwise_registration": diagnostics,
        "depth_stride": stride,
        "voxel_size": voxel_size,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result["pose_graph"]
