"""Evidence-first fusion of SHARP camera-frame Gaussian predictions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .geometry import validate_transform
from .ply import load_vertex, transform_vertex, write_vertex, read_sharp_metadata
from .preprocess import map_original_pixels_to_vggt


def estimate_sharp_to_vggt_scale(vertex_data, sharp_meta, prediction, min_samples=50):
    """Estimate per-view metric scale by comparing SHARP z with VGGT depth.

    SHARP PLY xyz are already Euclidean camera coordinates. This estimates a
    scalar correction between SHARP's camera-local metric space and VGGT's
    camera-local depth scale by projecting Gaussian centers back to the same
    source pixels and sampling VGGT depth.
    """
    names = vertex_data.dtype.names or ()
    xyz = np.column_stack([vertex_data[k] for k in ("x", "y", "z")]).astype(np.float64)
    z = xyz[:, 2]
    K = sharp_meta["intrinsics"]
    valid = np.isfinite(xyz).all(axis=1) & (z > 1e-5)
    if "opacity" in names:
        opacity = 1.0 / (1.0 + np.exp(-np.clip(vertex_data["opacity"].astype(np.float64), -30, 30)))
        valid &= opacity > 0.05
    xyz = xyz[valid]
    z = z[valid]
    if len(z) > 150000:
        step = int(np.ceil(len(z) / 150000))
        xyz, z = xyz[::step], z[::step]
    u = K[0, 0] * xyz[:, 0] / z + K[0, 2]
    v = K[1, 1] * xyz[:, 1] / z + K[1, 2]
    prep = prediction.get("preprocess")
    if prep is None:
        raise ValueError("VGGT output lacks preprocessing transform needed for SHARP/VGGT pixel correspondence")
    um, vm = map_original_pixels_to_vggt(u, v, prep)
    ui, vi = np.rint(um).astype(int), np.rint(vm).astype(int)
    depth = np.asarray(np.load(prediction["depth_path"]), dtype=np.float64).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D VGGT depth map, got {depth.shape}")
    h, w = depth.shape
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    observed = np.full(len(z), np.nan)
    observed[inside] = depth[vi[inside], ui[inside]]
    keep = inside & np.isfinite(observed) & (observed > 1e-5)
    if prediction.get("confidence_path"):
        conf = np.asarray(np.load(prediction["confidence_path"]), dtype=np.float64).squeeze()
        if conf.shape != depth.shape:
            raise ValueError("VGGT confidence shape does not match depth")
        threshold = np.nanquantile(conf[np.isfinite(conf)], 0.3)
        keep &= np.isfinite(conf[vi.clip(0, h-1), ui.clip(0, w-1)]) & (conf[vi.clip(0, h-1), ui.clip(0, w-1)] >= threshold)
    ratios = observed[keep] / z[keep]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if len(ratios) < min_samples:
        raise ValueError(f"Only {len(ratios)} valid SHARP/VGGT depth correspondences; need {min_samples}")
    logs = np.log(ratios)
    center = np.median(logs)
    mad = np.median(np.abs(logs - center))
    tolerance = max(0.15, 3.0 * 1.4826 * mad)
    inliers = ratios[np.abs(logs - center) <= tolerance]
    scale = float(np.median(inliers))
    if not np.isfinite(scale) or scale < 0.1 or scale > 10.0:
        raise ValueError(f"Implausible SHARP-to-VGGT scale correction: {scale}")
    return scale, {"valid_correspondences": int(len(ratios)), "scale_inliers": int(len(inliers)),
                   "scale": scale, "log_ratio_mad": float(mad)}


def fuse_sharp_with_vggt(manifest_path, output, voxel_size=0.005,
                         max_gaussians=2000000, allow_unscaled=False):
    """Transform SHARP camera-frame PLYs into VGGT world coordinates and dedupe.

    This is an initialization fusion only. It does not perform GaussianObject
    training or visibility-aware surface reconstruction.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pred_path = Path(manifest["vggt_predictions"])
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    by_image = {str(Path(v["image"]).resolve()): v for v in predictions["views"]}
    chunks, template, per_view = [], None, []
    for view in manifest["views"]:
        if not view.get("sharp_ply"):
            continue
        image_key = str(Path(view.get("processed_image", view["image"])).resolve())
        if image_key not in by_image:
            raise ValueError(f"Missing VGGT prediction for {image_key}")
        prediction = by_image[image_key]
        camera_to_world = validate_transform(np.asarray(prediction["camera_to_world"]))
        ply, data = load_vertex(view["sharp_ply"])
        metadata = read_sharp_metadata(ply)
        if template is None:
            template = ply
        try:
            scale, scale_stats = estimate_sharp_to_vggt_scale(data, metadata, prediction)
        except ValueError:
            if not allow_unscaled:
                raise
            scale, scale_stats = 1.0, {"scale": 1.0, "calibrated": False}
        # SHARP CLI has already unprojected NDC Gaussians into Euclidean
        # OpenCV camera coordinates before writing the PLY.
        T_world_sharp_camera = camera_to_world.copy()
        T_world_sharp_camera[:3, :3] *= scale
        chunks.append(transform_vertex(data, T_world_sharp_camera))
        per_view.append({"view_id": view["view_id"], **scale_stats})
    if not chunks:
        raise ValueError("No SHARP predictions found")
    if any(x.dtype != chunks[0].dtype for x in chunks):
        raise ValueError("PLY schemas differ; convert to a common Gaussian schema")
    merged = np.concatenate(chunks)
    names = merged.dtype.names or ()
    before = len(merged)
    if voxel_size > 0 and len(merged):
        xyz = np.column_stack([merged[k] for k in ("x", "y", "z")]).astype(np.float64)
        voxels = np.floor(xyz / voxel_size).astype(np.int64)
        _, inverse = np.unique(voxels, axis=0, return_inverse=True)
        score = merged["opacity"] if "opacity" in names else np.zeros(len(merged))
        best = np.full(inverse.max() + 1, -np.inf)
        np.maximum.at(best, inverse, score)
        merged = merged[score == best[inverse]]
    if len(merged) > max_gaussians:
        score = merged["opacity"] if "opacity" in names else np.ones(len(merged))
        ids = np.argpartition(score, -max_gaussians)[-max_gaussians:]
        merged = merged[ids]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_vertex(template, merged, output)
    return {"output": str(output.resolve()), "gaussians": int(len(merged)),
            "gaussians_before_deduplication": int(before), "views": per_view}
