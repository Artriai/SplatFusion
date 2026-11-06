"""Native VGGT-Ω inference runner for SplatFusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

from splatfusion.preprocess import vggt_preprocess_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="JSON array of image paths")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    image_paths = [str(Path(x).resolve()) for x in json.loads(args.images)]
    device = torch.device(args.device)
    model = VGGTOmega().to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)

    preprocess = vggt_preprocess_metadata(
        [Path(x) for x in image_paths],
        resolution=args.resolution,
    )
    images = load_and_preprocess_images(
        image_paths,
        image_resolution=args.resolution,
    ).to(device)

    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            predictions = model(images[None])
    else:
        with torch.inference_mode():
            predictions = model(images[None])

    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    # VGGT-Ω pose encoding stores camera-from-world (OpenCV) extrinsics.
    bottom = torch.tensor(
        [0, 0, 0, 1],
        device=device,
        dtype=extrinsics.dtype,
    ).view(1, 1, 1, 4).expand(1, len(image_paths), 1, 4)
    camera_to_world = torch.linalg.inv(torch.cat([extrinsics, bottom], dim=-2))

    depth = predictions["depth"]
    depth_confidence = predictions.get("depth_conf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth_dir = output_path.parent / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    views = []
    for index, image_path in enumerate(image_paths):
        depth_map = depth[0, index].detach().float().cpu().numpy()
        if depth_map.ndim == 3 and depth_map.shape[-1] == 1:
            depth_map = depth_map[..., 0]
        depth_path = depth_dir / f"view_{index:04d}_depth.npy"
        np.save(depth_path, depth_map)

        confidence_path = None
        if depth_confidence is not None:
            confidence = depth_confidence[0, index].detach().float().cpu().numpy()
            if confidence.ndim == 3 and confidence.shape[-1] == 1:
                confidence = confidence[..., 0]
            confidence_path = depth_dir / f"view_{index:04d}_confidence.npy"
            np.save(confidence_path, confidence)

        views.append({
            "image": image_path,
            "camera_to_world": camera_to_world[0, index].detach().float().cpu().numpy().tolist(),
            "intrinsics": intrinsics[0, index].detach().float().cpu().numpy().tolist(),
            "depth_path": str(depth_path.resolve()),
            "confidence_path": str(confidence_path.resolve()) if confidence_path else None,
            "depth_shape": list(depth_map.shape),
            "preprocess": preprocess[index],
        })

    payload = {
        "coordinate_convention": "opencv",
        "world_to_camera": False,
        "depth_units": "model_relative",
        "views": views,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
