import numpy as np
from splatfusion.ply import transform_vertex, _quat_to_matrix


def test_transform_vertex_transforms_anisotropic_gaussian_covariance():
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    data = np.zeros(1, dtype=dtype)
    data["x"], data["y"], data["z"] = 1, 2, 3
    data["scale_0"], data["scale_1"], data["scale_2"] = np.log(1), np.log(2), np.log(3)
    data["rot_0"] = 1
    T = np.eye(4)
    T[:3, :3] = 2 * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transformed = transform_vertex(data, T)
    assert np.allclose([transformed["x"][0], transformed["y"][0], transformed["z"][0]], [-4, 2, 6])
    scales = np.exp(np.column_stack([transformed[f"scale_{i}"] for i in range(3)]))
    quats = np.column_stack([transformed[f"rot_{i}"] for i in range(4)])
    rotations = _quat_to_matrix(quats)
    actual_cov = rotations @ np.array([np.diag(s * s) for s in scales]) @ np.swapaxes(rotations, 1, 2)
    original_cov = np.diag([1.0, 4.0, 9.0])
    expected_cov = T[:3, :3] @ original_cov @ T[:3, :3].T
    assert np.allclose(actual_cov[0], expected_cov, atol=1e-5)
