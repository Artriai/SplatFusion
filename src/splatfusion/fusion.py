"""Evidence-first SHARP Gaussian fusion helpers."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .geometry import validate_transform
from .ply import load_vertex, transform_vertex, write_vertex


def fuse_sharp_with_vggt(manifest_path, output, voxel_size=0.005, max_gaussians=2000000):
    """Transform per-view SHARP PLYs to VGGT world coordinates and voxel-dedupe.

    Requires a VGGT prediction per image. The SHARP camera-frame assumption
    must be checked against the installed SHARP release before production use.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pred_path = Path(manifest["vggt_predictions"])
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    by_image = {str(Path(v["image"]).resolve()): v for v in predictions["views"]}
    chunks, template = [], None
    for view in manifest["views"]:
        if not view.get("sharp_ply"):
            continue
        key = str(Path(view["image"]).resolve())
        if key not in by_image:
            raise ValueError(f"Missing VGGT prediction for {key}")
        camera_to_world = validate_transform(np.asarray(by_image[key]["camera_to_world"]))
        ply, data = load_vertex(view["sharp_ply"])
        if template is None:
            template = ply
        chunks.append(transform_vertex(data, camera_to_world))
    if not chunks:
        raise ValueError("No SHARP predictions found")
    if any(x.dtype != chunks[0].dtype for x in chunks):
        raise ValueError("PLY schemas differ; convert to a common schema")
    merged = np.concatenate(chunks)
    names = merged.dtype.names or ()
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
    return {"output": str(output.resolve()), "gaussians": int(len(merged))}
