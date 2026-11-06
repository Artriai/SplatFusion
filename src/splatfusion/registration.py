"""Global and local point-cloud registration adapters."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .geometry import umeyama_similarity, transform_points


def trimmed_icp(source: np.ndarray, target: np.ndarray, max_iterations: int = 40,
                trim_fraction: float = 0.7, max_distance: float | None = None,
                estimate_scale: bool = False):
    """Register source to target using trimmed nearest-neighbour ICP.

    This is a local refinement method. Its initial transform must be close.
    """
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.ndim != 2 or dst.ndim != 2 or src.shape[1] != 3 or dst.shape[1] != 3:
        raise ValueError("point clouds must have shape (N,3)")
    src = src[np.isfinite(src).all(axis=1)]
    dst = dst[np.isfinite(dst).all(axis=1)]
    if len(src) < 10 or len(dst) < 10:
        raise ValueError("At least 10 finite points are required per cloud")
    if not 0 < trim_fraction <= 1:
        raise ValueError("trim_fraction must be in (0,1]")
    tree = cKDTree(dst)
    T = np.eye(4)
    previous = np.inf
    used = 0
    for _ in range(max_iterations):
        moved = transform_points(src, T)
        distances, indices = tree.query(moved, k=1)
        threshold = np.quantile(distances, trim_fraction)
        keep = distances <= threshold
        if max_distance is not None:
            keep &= distances <= max_distance
        if keep.sum() < 6:
            raise RuntimeError("ICP failed: too few inlier correspondences")
        delta = umeyama_similarity(moved[keep], dst[indices[keep]], with_scale=estimate_scale)
        T = delta @ T
        rmse = float(np.sqrt(np.mean(distances[keep] ** 2)))
        used = int(keep.sum())
        if abs(previous - rmse) < 1e-7:
            break
        previous = rmse
    return T, {"rmse": previous, "inliers": used, "source_points": len(src), "target_points": len(dst)}


def global_register_fpfh(source: np.ndarray, target: np.ndarray, voxel_size: float = 0.01,
                          ransac_iterations: int = 100000, confidence: float = 0.999):
    """Open3D FPFH + RANSAC global registration, followed by point-to-plane ICP.

    Returns a rigid transform mapping source camera-frame points to target
    camera-frame points. Open3D is an optional dependency.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Global FPFH registration requires Open3D: pip install 'splatfusion[registration]'") from exc

    def make_cloud(points):
        points = np.asarray(points, dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 100:
            raise ValueError("At least 100 finite points are required for FPFH registration")
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        return cloud

    src = make_cloud(source)
    dst = make_cloud(target)
    src_down = src.voxel_down_sample(voxel_size)
    dst_down = dst.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2.0
    radius_feature = voxel_size * 5.0
    for cloud in (src_down, dst_down):
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    dst_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        dst_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    distance = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, dst_down, src_fpfh, dst_fpfh, True, distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_iterations, confidence))
    if result.fitness < 0.02:
        raise RuntimeError(f"Global registration found insufficient overlap (fitness={result.fitness:.4f})")
    for cloud in (src, dst):
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    refined = o3d.pipelines.registration.registration_icp(
        src, dst, voxel_size, result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    if refined.fitness < 0.02:
        raise RuntimeError(f"ICP refinement found insufficient overlap (fitness={refined.fitness:.4f})")
    return refined.transformation, {
        "fitness": float(refined.fitness),
        "inlier_rmse": float(refined.inlier_rmse),
        "ransac_fitness": float(result.fitness),
        "correspondences": int(len(refined.correspondence_set)),
    }
