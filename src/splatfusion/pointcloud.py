"""Depth-map unprojection utilities for multi-view geometry."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def unproject_depth(depth: np.ndarray, intrinsics: np.ndarray, camera_to_world: np.ndarray,
                    confidence: np.ndarray | None = None, confidence_quantile: float = 0.2,
                    stride: int = 2, mask: np.ndarray | None = None) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    T = np.asarray(camera_to_world, dtype=np.float64)
    if depth.ndim != 2 or K.shape != (3, 3) or T.shape != (4, 4):
        raise ValueError("Expected depth HxW, intrinsics 3x3, camera_to_world 4x4")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z > 0)
    if confidence is not None:
        conf = np.asarray(confidence)
        if conf.shape != depth.shape:
            raise ValueError("confidence shape must match depth")
        cs = conf[::stride, ::stride]
        vals = cs[np.isfinite(cs)]
        if vals.size:
            valid &= np.isfinite(cs) & (cs >= np.quantile(vals, confidence_quantile))
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError("mask shape must match depth")
        valid &= mask[::stride, ::stride]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    camera = np.stack([x[valid], y[valid], z[valid], np.ones(valid.sum())], axis=1)
    world = (T @ camera.T).T[:, :3]
    return world.astype(np.float32)


def save_point_cloud(points: np.ndarray, output: str | Path):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as exc:
        raise RuntimeError("Install plyfile to export point clouds") from exc
    vertex = np.empty(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex["x"], vertex["y"], vertex["z"] = points.T
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(output))
