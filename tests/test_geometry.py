import numpy as np
from splatfusion.geometry import transform_points, umeyama_similarity, transform_covariances


def test_umeyama_recovers_similarity_transform():
    rng = np.random.default_rng(7)
    src = rng.normal(size=(100, 3))
    angle = 0.4
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])
    expected = np.eye(4)
    expected[:3, :3] = 1.7 * R
    expected[:3, 3] = [0.2, -0.5, 1.1]
    dst = transform_points(src, expected)
    estimated = umeyama_similarity(src, dst)
    assert np.allclose(estimated, expected, atol=1e-6)


def test_covariance_transform_preserves_symmetry():
    cov = np.array([[[2., 0, 0], [0, 1., 0], [0, 0, .5]]])
    T = np.eye(4)
    T[:3, :3] *= 2
    result = transform_covariances(cov, T)
    assert np.allclose(result, result.transpose(0, 2, 1))
    assert np.allclose(np.linalg.eigvalsh(result[0]), [2, 4, 8])
