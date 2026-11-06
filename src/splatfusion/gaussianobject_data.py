"""Prepare GaussianObject's DUSt3R-compatible dataset from VGGT-Ω outputs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from .preprocess import map_original_pixels_to_vggt
from .ply import load_vertex


def _letterbox_image(image: Image.Image, size: int, resample):
    width, height = image.size
    scale = min(size / width, size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    resized = image.resize((new_width, new_height), resample)
    canvas = Image.new(image.mode, (size, size), 0 if image.mode == "L" else (255, 255, 255))
    canvas.paste(resized, (left, top))
    return canvas, {
        "scale_x": new_width / width,
        "scale_y": new_height / height,
        "left": left,
        "top": top,
        "new_width": new_width,
        "new_height": new_height,
        "original_width": width,
        "original_height": height,
    }


def _depth_to_letterbox(depth, confidence, preprocess, letterbox, size):
    output_depth = np.zeros((size, size), dtype=np.float32)
    output_conf = np.zeros((size, size), dtype=np.float32)
    left, top = letterbox["left"], letterbox["top"]
    new_w, new_h = letterbox["new_width"], letterbox["new_height"]
    yy, xx = np.mgrid[0:new_h, 0:new_w]
    u_original = (xx + 0.5) / letterbox["scale_x"] - 0.5
    v_original = (yy + 0.5) / letterbox["scale_y"] - 0.5
    u_model, v_model = map_original_pixels_to_vggt(u_original, v_original, preprocess)
    h_model, w_model = depth.shape
    valid = (
        (u_original >= preprocess["crop_left"])
        & (u_original < preprocess["crop_left"] + preprocess["crop_width"])
        & (v_original >= preprocess["crop_top"])
        & (v_original < preprocess["crop_top"] + preprocess["crop_height"])
        & (u_model >= 0) & (u_model < w_model - 1)
        & (v_model >= 0) & (v_model < h_model - 1)
    )
    sampled_depth = map_coordinates(
        depth, [v_model.ravel(), u_model.ravel()], order=1,
        mode="constant", cval=0.0,
    ).reshape(new_h, new_w)
    if confidence is None:
        sampled_conf = np.ones_like(sampled_depth)
    else:
        sampled_conf = map_coordinates(
            confidence, [v_model.ravel(), u_model.ravel()], order=1,
            mode="constant", cval=0.0,
        ).reshape(new_h, new_w)
    valid &= np.isfinite(sampled_depth) & (sampled_depth > 1e-6)
    sampled_depth[~valid] = 0
    sampled_conf[~valid] = 0
    sampled_conf[~np.isfinite(sampled_conf)] = 0
    output_depth[top:top + new_h, left:left + new_w] = sampled_depth
    output_conf[top:top + new_h, left:left + new_w] = sampled_conf
    return output_depth, output_conf


def _write_colored_pointcloud(path: Path, points: np.ndarray, colors: np.ndarray):
    from plyfile import PlyData, PlyElement
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    vertices = np.empty(len(points), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["nx"] = 0
    vertices["ny"] = 0
    vertices["nz"] = 0
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))


def _sharp_prior_to_pointcloud(source_ply: Path, output_ply: Path, center, scale,
                               max_points=300000, voxel_size=0.002):
    _, data = load_vertex(source_ply)
    names = data.dtype.names or ()
    xyz = np.column_stack([data[k] for k in ("x", "y", "z")]).astype(np.float64)
    if "opacity" in names:
        opacity = 1.0 / (1.0 + np.exp(-np.clip(data["opacity"].astype(np.float64), -30, 30)))
        keep = opacity > 0.05
        xyz, opacity = xyz[keep], opacity[keep]
        data = data[keep]
    else:
        opacity = np.ones(len(xyz))
    xyz = (xyz - center) / scale
    if all(k in names for k in ("red", "green", "blue")):
        colors = np.column_stack([data[k] for k in ("red", "green", "blue")]).astype(np.uint8)
    elif all(k in names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        coeff = np.column_stack([data[k] for k in ("f_dc_0", "f_dc_1", "f_dc_2")]).astype(np.float64)
        colors = (np.clip(0.5 + 0.28209479177387814 * coeff, 0, 1) * 255).astype(np.uint8)
    else:
        colors = np.full((len(xyz), 3), 180, dtype=np.uint8)
    if len(xyz) and voxel_size > 0:
        voxels = np.floor(xyz / voxel_size).astype(np.int64)
        _, inverse = np.unique(voxels, axis=0, return_inverse=True)
        order = np.lexsort((np.arange(len(opacity)), -opacity, inverse))
        sorted_voxels = inverse[order]
        first = np.concatenate(([True], sorted_voxels[1:] != sorted_voxels[:-1]))
        selected = order[first]
        xyz, colors = xyz[selected], colors[selected]
    if len(xyz) < 100:
        raise ValueError("Fused SHARP prior contains fewer than 100 usable points")
    if len(xyz) > max_points:
        rng = np.random.default_rng(0)
        selected = rng.choice(len(xyz), max_points, replace=False)
        xyz, colors = xyz[selected], colors[selected]
    _write_colored_pointcloud(output_ply, xyz, colors)
    return int(len(xyz))


def prepare_gaussianobject_dataset(manifest_path, predictions_path, masks,
                                   output_dir, fused_sharp_ply=None,
                                   image_size=512, stride=4, max_pointcloud_points=500000):
    """Create the exact image/mask/depth/camera/point-cloud layout expected by
    GaussianObject's COLMAP-free DUSt3R data reader, using VGGT-Ω predictions.
    """
    if image_size < 64 or stride < 1 or max_pointcloud_points < 100:
        raise ValueError("image_size must be >=64, stride >=1 and max_pointcloud_points >=100")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    predictions = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    pred_by_image = {str(Path(v["image"]).resolve()): v for v in predictions["views"]}
    views = manifest["views"]
    if len(views) < 2:
        raise ValueError("GaussianObject requires at least two views")
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    cameras, depth_stack, confidence_stack = [], [], []
    world_points, world_colors = [], []
    for index, view in enumerate(views):
        image_path = Path(view.get("processed_image", view["image"]))
        key = str(image_path.resolve())
        if key not in pred_by_image:
            raise ValueError(f"Missing VGGT prediction for {image_path}")
        pred = pred_by_image[key]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(masks[index]) as source_mask:
            mask = source_mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
        image_out, letterbox = _letterbox_image(image, image_size, Image.Resampling.LANCZOS)
        mask_out, _ = _letterbox_image(mask, image_size, Image.Resampling.NEAREST)
        image_name = f"view_{index:04d}.png"
        image_out.save(image_dir / image_name)
        mask_out.save(mask_dir / image_name)

        depth = np.asarray(np.load(pred["depth_path"]), dtype=np.float32).squeeze()
        confidence = (
            np.asarray(np.load(pred["confidence_path"]), dtype=np.float32).squeeze()
            if pred.get("confidence_path") else None
        )
        depth_out, conf_out = _depth_to_letterbox(
            depth, confidence, pred["preprocess"], letterbox, image_size
        )
        mask_array = np.asarray(mask_out, dtype=np.uint8) >= 128
        depth_out[~mask_array] = 0
        conf_out[~mask_array] = 0
        depth_stack.append(depth_out)
        if not np.isfinite(conf_out).any() or float(np.nanmax(conf_out)) <= 0:
            raise RuntimeError(f"VGGT confidence is empty for view {index}")
        confidence_stack.append(conf_out)

        K_model = np.asarray(pred["intrinsics"], dtype=np.float64)
        prep = pred["preprocess"]
        fx_original = K_model[0, 0] * prep["crop_width"] / prep["resized_width"]
        fy_original = K_model[1, 1] * prep["crop_height"] / prep["resized_height"]
        fx = fx_original * letterbox["scale_x"]
        fy = fy_original * letterbox["scale_y"]
        cx = cy = image_size / 2.0
        c2w = np.asarray(pred["camera_to_world"], dtype=np.float64)
        rgb_array = np.asarray(image_out, dtype=np.uint8)

        yy, xx = np.mgrid[0:image_size:stride, 0:image_size:stride]
        z = depth_out[::stride, ::stride]
        m = mask_array[::stride, ::stride]
        conf = conf_out[::stride, ::stride]
        valid_conf = conf[m & np.isfinite(conf) & (z > 0)]
        if valid_conf.size:
            threshold = np.quantile(valid_conf, 0.2)
        else:
            threshold = np.inf
        valid = m & np.isfinite(z) & (z > 1e-6) & np.isfinite(conf) & (conf >= threshold)
        if valid.sum() < 50:
            raise RuntimeError(f"View {index} has too few valid foreground depth pixels for GaussianObject")
        x_cam = (xx - cx) * z / fx
        y_cam = (yy - cy) * z / fy
        camera_xyz = np.stack([x_cam[valid], y_cam[valid], z[valid]], axis=1)
        world_xyz = (c2w[:3, :3] @ camera_xyz.T).T + c2w[:3, 3]
        colors = rgb_array[::stride, ::stride][valid]
        if len(world_xyz):
            world_points.append(world_xyz)
            world_colors.append(colors)

        cameras.append({
            "id": index,
            "img_name": image_name,
            "width": image_size,
            "height": image_size,
            "position": c2w[:3, 3].tolist(),
            "rotation": c2w[:3, :3].tolist(),
            "fx": float(fx),
            "fy": float(fy),
        })

    if not world_points:
        raise RuntimeError("No valid masked VGGT depth points; cannot initialize GaussianObject")
    points = np.concatenate(world_points)
    colors = np.concatenate(world_colors)
    if len(points) > max_pointcloud_points:
        rng = np.random.default_rng(0)
        selected = rng.choice(len(points), max_pointcloud_points, replace=False)
        points, colors = points[selected], colors[selected]
    center = points.mean(axis=0)
    scale = float(np.max(np.abs(points - center)))
    if not np.isfinite(scale) or scale < 1e-6:
        raise RuntimeError("Degenerate VGGT object point cloud")
    points = (points - center) / scale
    cameras_normalized = []
    for camera in cameras:
        item = dict(camera)
        item["position"] = ((np.asarray(camera["position"]) - center) / scale).tolist()
        cameras_normalized.append(item)

    n_views = len(views)
    for i in range(n_views):
        depth_stack[i] /= scale
    np.save(output_dir / f"dust3r_depth_{n_views}.npy", np.stack(depth_stack))
    np.save(output_dir / f"dust3r_confidence_{n_views}.npy", np.stack(confidence_stack))
    (output_dir / f"dust3r_{n_views}.json").write_text(
        json.dumps(cameras_normalized, indent=2), encoding="utf-8"
    )
    (output_dir / f"sparse_{n_views}.txt").write_text(
        "".join(f"{i}\n" for i in range(n_views)), encoding="utf-8"
    )
    (output_dir / "sparse_test.txt").write_text("", encoding="utf-8")
    _write_colored_pointcloud(output_dir / f"dust3r_{n_views}.ply", points, colors)

    sharp_prior_count = None
    sharp_prior_error = None
    if fused_sharp_ply:
        try:
            sharp_prior_count = _sharp_prior_to_pointcloud(
                Path(fused_sharp_ply), output_dir / "sharp_prior.ply", center, scale
            )
        except ValueError as exc:
            sharp_prior_error = str(exc)
            prior_path = output_dir / "sharp_prior.ply"
            if prior_path.exists():
                prior_path.unlink()

    metadata = {
        "dataset_dir": str(output_dir.resolve()),
        "views": n_views,
        "image_size": image_size,
        "normalization_center": center.tolist(),
        "normalization_scale": scale,
        "pointcloud_points": int(len(points)),
        "sharp_prior_points": sharp_prior_count,
        "sharp_prior_error": sharp_prior_error,
        "files": {
            "cameras": str((output_dir / f"dust3r_{n_views}.json").resolve()),
            "depth": str((output_dir / f"dust3r_depth_{n_views}.npy").resolve()),
            "confidence": str((output_dir / f"dust3r_confidence_{n_views}.npy").resolve()),
            "pointcloud": str((output_dir / f"dust3r_{n_views}.ply").resolve()),
        },
    }
    (output_dir / "splatfusion_dataset.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata
