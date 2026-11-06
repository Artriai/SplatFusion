"""Coordinate and similarity-transform utilities.

All transforms use column vectors. T_dst_src maps points in src coordinates
into dst coordinates. The utilities intentionally keep scale explicit because
monocular predictions can have different scale conventions.
"""
from __future__ import annotations

import numpy as np


def validate_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("Transform must be a finite 4x4 matrix")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid homogeneous transform bottom row")
    return T


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    T = validate_transform(T)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return points @ T[:3, :3].T + T[:3, 3]


def umeyama_similarity(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> np.ndarray:
    """Estimate T mapping src points to dst points using Umeyama least squares."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise ValueError("src and dst must be matching (N,3) arrays with N >= 3")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("Correspondences must be finite")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, singular, Vt = np.linalg.svd(cov)
    sign = np.ones(3)
    if np.linalg.det(U @ Vt) < 0:
        sign[-1] = -1
    R = U @ np.diag(sign) @ Vt
    variance = (xs * xs).sum() / len(src)
    if variance < 1e-12:
        raise ValueError("Degenerate source correspondences")
    scale = float((singular * sign).sum() / variance) if with_scale else 1.0
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Estimated invalid scale")
    t = mu_d - scale * (R @ mu_s)
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T


def transform_covariances(covariances: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply the linear part of a similarity transform to 3x3 covariances."""
    A = validate_transform(T)[:3, :3]
    covariances = np.asarray(covariances, dtype=np.float64)
    if covariances.shape[-2:] != (3, 3):
        raise ValueError("covariances must end in shape (3,3)")
    return A @ covariances @ np.swapaxes(A, -1, -2)


def transform_gaussian_arrays(means: np.ndarray, covariances: np.ndarray, T: np.ndarray):
    return transform_points(means, T), transform_covariances(covariances, T)
