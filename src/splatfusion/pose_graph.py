"""SE(3) pose-graph optimization for multi-view camera estimates."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from .geometry import validate_transform


def _skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def so3_exp(w):
    theta = float(np.linalg.norm(w))
    W = _skew(w)
    if theta < 1e-8:
        return np.eye(3) + W + 0.5 * W @ W
    return np.eye(3) + np.sin(theta) / theta * W + (1 - np.cos(theta)) / theta**2 * (W @ W)


def so3_log(R):
    cos_theta = np.clip((np.trace(R) - 1) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-8:
        return 0.5 * vee
    if np.pi - theta < 1e-5:
        eigval, eigvec = np.linalg.eig(R)
        axis = np.real(eigvec[:, np.argmin(np.abs(eigval - 1))])
        axis /= max(np.linalg.norm(axis), 1e-12)
        return theta * axis
    return theta / (2 * np.sin(theta)) * vee


def se3_exp(xi):
    """Exponential map for twist [rotation_vector, translation_vector]."""
    w, v = np.asarray(xi[:3]), np.asarray(xi[3:])
    theta = float(np.linalg.norm(w))
    W = _skew(w)
    R = so3_exp(w)
    if theta < 1e-8:
        V = np.eye(3) + 0.5 * W + (1.0 / 6.0) * (W @ W)
    else:
        V = (np.eye(3) + (1 - np.cos(theta)) / theta**2 * W
             + (theta - np.sin(theta)) / theta**3 * (W @ W))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def se3_log(T):
    T = validate_transform(T)
    w = so3_log(T[:3, :3])
    theta = float(np.linalg.norm(w))
    W = _skew(w)
    if theta < 1e-8:
        V_inv = np.eye(3) - 0.5 * W + (1.0 / 12.0) * (W @ W)
    else:
        a = 1.0 / theta**2 - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))
        V_inv = np.eye(3) - 0.5 * W + a * (W @ W)
    v = V_inv @ T[:3, 3]
    return np.concatenate([w, v])


@dataclass(frozen=True)
class PoseEdge:
    """Measurement T_camera_j_camera_i maps points from camera i to camera j."""
    i: int
    j: int
    transform_j_i: np.ndarray
    weight: float = 1.0
    kind: str = "registration"


def optimize_pose_graph(camera_to_world, edges: list[PoseEdge], prior_weight: float = 0.05,
                        robust_loss: str = "huber", max_nfev: int = 100):
    """Optimize camera-to-world poses while fixing node 0 as the gauge anchor.

    VGGT camera poses provide a connected low-weight prior chain. Registered
    pairwise depth-cloud transforms add stronger local constraints.
    """
    initial = np.asarray([validate_transform(T) for T in camera_to_world], dtype=np.float64)
    n = len(initial)
    if n == 0:
        raise ValueError("At least one camera pose is required")
    if n == 1:
        return initial.copy(), {"success": True, "edges": 0, "cost": 0.0}
    constraints = list(edges)
    for i in range(n - 1):
        j = i + 1
        z_j_i = np.linalg.inv(initial[j]) @ initial[i]
        constraints.append(PoseEdge(i, j, z_j_i, prior_weight, "vggt_prior"))
    for e in constraints:
        if e.i < 0 or e.j < 0 or e.i >= n or e.j >= n or e.i == e.j:
            raise ValueError(f"Invalid pose-graph edge {e}")
        validate_transform(e.transform_j_i)
        if e.weight <= 0:
            raise ValueError("Pose edge weights must be positive")

    def unpack(x):
        poses = initial.copy()
        for k in range(1, n):
            poses[k] = se3_exp(x[(k - 1) * 6:k * 6]) @ initial[k]
        return poses

    def residual(x):
        poses = unpack(x)
        values = []
        for e in constraints:
            error = np.linalg.inv(poses[e.i]) @ poses[e.j] @ e.transform_j_i
            values.extend(np.sqrt(e.weight) * se3_log(error))
        return np.asarray(values)

    result = least_squares(residual, np.zeros((n - 1) * 6), loss=robust_loss,
                           f_scale=0.02, max_nfev=max_nfev)
    refined = unpack(result.x)
    return refined, {
        "success": bool(result.success),
        "message": result.message,
        "edges": len(constraints),
        "registration_edges": sum(e.kind != "vggt_prior" for e in constraints),
        "initial_cost": float(np.dot(residual(np.zeros((n - 1) * 6)), residual(np.zeros((n - 1) * 6)))),
        "final_cost": float(np.dot(result.fun, result.fun)),
    }
