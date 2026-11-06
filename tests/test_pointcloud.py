import numpy as np
from splatfusion.pointcloud import unproject_depth


def test_unproject_depth_respects_mask():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    K = np.array([[2.0, 0, 1.5], [0, 2.0, 1.5], [0, 0, 1.0]])
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    points = unproject_depth(depth, K, np.eye(4), mask=mask, stride=1)
    assert points.shape == (1, 3)
    assert np.allclose(points[0], [-0.5, -0.5, 2.0])
