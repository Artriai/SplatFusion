import numpy as np
from splatfusion.pose_graph import PoseEdge, optimize_pose_graph, se3_exp, se3_log


def test_se3_exp_log_round_trip():
    xi = np.array([0.1, -0.2, 0.05, 0.4, -0.1, 0.2])
    assert np.allclose(se3_log(se3_exp(xi)), xi, atol=1e-6)


def test_pose_graph_recovers_camera_pose_from_relative_edge():
    true_pose_0 = np.eye(4)
    true_pose_1 = se3_exp(np.array([0.0, 0.0, 0.3, 0.5, 0.0, 0.1]))
    measurement_1_0 = np.linalg.inv(true_pose_1) @ true_pose_0
    initial_pose_1 = se3_exp(np.array([0.0, 0.0, 0.34, 0.58, 0.03, 0.1])) @ true_pose_1
    refined, stats = optimize_pose_graph(
        [true_pose_0, initial_pose_1],
        [PoseEdge(0, 1, measurement_1_0, weight=100.0)],
        prior_weight=0.001,
    )
    assert stats["success"]
    assert np.allclose(refined[1], true_pose_1, atol=2e-3)
