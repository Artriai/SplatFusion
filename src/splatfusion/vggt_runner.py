#!/usr/bin/env python3
"""Native VGGT-Ω adapter for SplatFusion.

Run from an environment where facebookresearch/vggt-omega is installed and
its checkpoint is available. Extrinsics returned by encoding_to_camera are
world-to-camera (OpenCV); this runner explicitly inverts them to camera-to-world.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera\nfrom splatfusion.preprocess import vggt_preprocess_metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="JSON array of image paths")
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    paths = [str(Path(x).resolve()) for x in json.loads(args.images)]
    device = torch.device(args.device)
    model = VGGTOmega().to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)

    preprocess = vggt_preprocess_metadata([Path(x) for x in paths], resolution=args.resolution)\n    images = load_and_preprocess_images(paths, image_resolution=args.resolution).to(device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=dtype):
                pred = model(images[None])
        else:
            pred = model(images[None])

    extrinsics, intrinsics = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
    # VGGT-Ω pose encoding stores camera-from-world extrinsics.
    c2w = torch.linalg.inv(torch.cat([
        extrinsics,
        torch.tensor([0, 0, 0, 1], device=device, dtype=extrinsics.dtype).view(1, 1, 1, 4).expand(1, len(paths), 1, 4)
    ], dim=-2))

    depth = pred["depth"]
    depth_conf = pred.get("depth_conf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    depth_dir = output.parent / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    views = []
    for i, image in enumerate(paths):
        d = depth[0, i].detach().float().cpu().numpy()
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]
        conf_path = None
        if depth_conf is not None:
            conf = depth_conf[0, i].detach().float().cpu().numpy()
            if conf.ndim == 3 and conf.shape[-1] == 1:
                conf = conf[..., 0]
            cp = depth_dir / f"view_{i:04d}_confidence.npy"
            np.save(cp, conf)
            conf_path = str(cp.resolve())
        dp = depth_dir / f"view_{i:04d}_depth.npy"
        np.save(dp, d)
        views.append({
            "image": image,
            "camera_to_world": c2w[0, i].detach().float().cpu().numpy().tolist(),
            "intrinsics": intrinsics[0, i].detach().float().cpu().numpy().tolist(),
            "depth_path": str(dp.resolve()),
            "confidence_path": conf_path,
            "depth_shape": list(d.shape),\n            "preprocess": preprocess[i],
        })

    payload = {
        "coordinate_convention": "opencv",
        "world_to_camera": False,
        "depth_units": "model_relative",
        "views": views,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
