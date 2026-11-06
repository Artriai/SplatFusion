from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .adapters import run_sharp, run_vggt_omega

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"Input directory does not exist: {directory}")
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def build_manifest(images: list[Path], output: Path) -> dict:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "splatfusion",
        "stage": "input_indexed",
        "output_dir": str(output.resolve()),
        "views": [{
            "view_id": f"view_{i:04d}",
            "image": str(image.resolve()),
            "sharp_ply": None,
            "camera_to_world": None,
            "intrinsics": None,
            "status": "pending",
        } for i, image in enumerate(images)],
        "notes": [
            "Per-view SHARP predictions are camera-local priors, not registered world-space geometry.",
            "Global fusion is not performed until camera poses and scale are validated.",
        ],
    }


def run(args: argparse.Namespace) -> int:
    images = collect_images(args.input)
    if len(images) < 2:
        raise SystemExit(f"Need at least two supported images; found {len(images)}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(images, args.output)

    if args.sharp and not args.dry_run:
        for view in manifest["views"]:
            image = Path(view["image"])
            view_dir = args.output / "sharp" / view["view_id"]
            ply = run_sharp(image, view_dir)
            view["sharp_ply"] = str(ply.resolve())
            view["status"] = "sharp_complete"
        manifest["stage"] = "sharp_complete"

    if args.vggt and not args.dry_run:
        checkpoint = args.vggt_checkpoint
        result_path = args.output / "vggt" / "predictions.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result = run_vggt_omega(images, result_path, checkpoint)
        manifest["vggt_predictions"] = str(result_path.resolve())
        by_image = {str(Path(v["image"]).resolve()): v for v in manifest["views"]}
        for prediction in result.get("views", []):
            key = str(Path(prediction["image"]).resolve())
            if key in by_image:
                by_image[key]["camera_to_world"] = prediction.get("camera_to_world")
                by_image[key]["intrinsics"] = prediction.get("intrinsics")
                by_image[key]["depth_path"] = prediction.get("depth_path")
                by_image[key]["confidence_path"] = prediction.get("confidence_path")
                by_image[key]["status"] = "vggt_complete"
        manifest["stage"] = "vggt_complete" if not args.sharp else "sharp_and_vggt_complete"

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Indexed {len(images)} images. Manifest: {manifest_path}")
    if args.dry_run:
        print("Dry run: no model inference executed.")
    else:
        print(f"Pipeline stage: {manifest['stage']}")
        print("Registration, cross-view Gaussian fusion and GaussianObject optimization remain separate next-stage tasks.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="splatfusion")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="Index photos and run selected model adapters")
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, default=Path("runs/default"))
    p.add_argument("--sharp", action="store_true", help="Run Apple's SHARP once per image")
    p.add_argument("--vggt", action="store_true", help="Run configured VGGT-Ω adapter")
    p.add_argument("--vggt-checkpoint", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=run)
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
