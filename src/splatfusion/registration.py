"""Robust point-cloud alignment for predicted multi-view geometry."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .geometry import umeyama_similarity, transform_points


def trimmed_icp(source: np.ndarray, target: np.ndarray, max_iterations: int = 40,
                trim_fraction: float = 0.7, max_distance: float | None = None,
                estimate_scale: bool = False):
    """Register source to target using trimmed nearest-neighbour ICP.

    Returns (4x4 source-to-target similarity transform, diagnostics).
    Initialization must be reasonably close; for large viewpoint changes use
    feature/RANSAC global registration upstream before this local refinement.
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
