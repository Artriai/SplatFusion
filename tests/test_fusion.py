import numpy as np
from splatfusion.fusion import estimate_sharp_to_vggt_scale


def test_depth_scale_calibration_recovers_constant_scale(tmp_path):
    width = height = 20
    focal = 10.0
    z = 2.0
    rows, cols = np.mgrid[0:10, 0:10]
    u = cols.ravel() + 5
    v = rows.ravel() + 5
    x = (u - width / 2) * z / focal
    y = (v - height / 2) * z / focal
    data = np.empty(100, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    data["x"], data["y"], data["z"] = x, y, z
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, np.full((height, width), 4.0, dtype=np.float32))
    preprocess = {
        "crop_left": 0, "crop_top": 0, "crop_width": width, "crop_height": height,
        "resized_width": width, "resized_height": height,
        "pad_left": 0, "pad_top": 0,
    }
    prediction = {"depth_path": str(depth_path), "preprocess": preprocess}
    metadata = {"intrinsics": np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])}
    scale, stats = estimate_sharp_to_vggt_scale(data, metadata, prediction)
    assert abs(scale - 2.0) < 1e-6
    assert stats["scale_inliers"] == 100
