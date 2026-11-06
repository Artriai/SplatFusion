from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from .adapters import run_sharp, run_vggt_omega
from .fusion import fuse_sharp_with_vggt
from .pose_refinement import refine_prediction_file

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"Input directory does not exist: {directory}")
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def build_manifest(images: list[Path], output: Path) -> dict:
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "splatfusion", "stage": "input_indexed", "output_dir": str(output.resolve()),
        "views": [{"view_id": f"view_{i:04d}", "image": str(image.resolve()),
                   "sharp_ply": None, "camera_to_world": None, "intrinsics": None, "status": "pending"}
                  for i, image in enumerate(images)],
        "notes": ["SHARP CLI exports Euclidean camera-frame Gaussians; transform them with validated VGGT camera-to-world poses."]
    }


def run(args: argparse.Namespace) -> int:
    images = collect_images(args.input)
    if len(images) < 2:
        raise SystemExit(f"Need at least two supported images; found {len(images)}")
    if args.fuse and not (args.sharp and args.vggt):
        raise SystemExit("--fuse requires both --sharp and --vggt")
    if args.register and not args.vggt:
        raise SystemExit("--register requires --vggt")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(images, args.output)

    if args.sharp and not args.dry_run:
        for view in manifest["views"]:
            image = Path(view["image"])
            ply = run_sharp(image, args.output / "sharp" / view["view_id"])
            view["sharp_ply"] = str(ply.resolve())
            view["status"] = "sharp_complete"
        manifest["stage"] = "sharp_complete"

    if args.vggt and not args.dry_run:
        pred_path = args.output / "vggt" / "predictions.json"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        run_vggt_omega(images, pred_path, args.vggt_checkpoint)
        if args.register:
            registered_path = args.output / "vggt" / "predictions_registered.json"
            stats = refine_prediction_file(
                pred_path, registered_path, stride=args.registration_stride,
                voxel_size=args.registration_voxel_size, neighbors=args.registration_neighbors)
            manifest["pose_graph"] = stats
            pred_path = registered_path
            manifest["stage"] = "pose_refined"
        manifest["vggt_predictions"] = str(pred_path.resolve())
        result = json.loads(pred_path.read_text(encoding="utf-8"))
        by_image = {str(Path(v["image"]).resolve()): v for v in manifest["views"]}
        for prediction in result.get("views", []):
            key = str(Path(prediction["image"]).resolve())
            if key in by_image:
                by_image[key].update({k: prediction.get(k) for k in
                    ("camera_to_world", "intrinsics", "depth_path", "confidence_path")})
                by_image[key]["status"] = "vggt_complete"
        if not args.register:
            manifest["stage"] = "sharp_and_vggt_complete" if args.sharp else "vggt_complete"

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Indexed {len(images)} images. Manifest: {manifest_path}")
    if args.dry_run:
        print("Dry run: no model inference executed.")
        return 0
    if args.fuse:
        result = fuse_sharp_with_vggt(manifest_path, args.output / "fused" / "fused.ply",
                                      voxel_size=args.voxel_size, max_gaussians=args.max_gaussians)
        manifest["fusion"] = result
        manifest["stage"] = "initial_fusion_complete"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Initial fused PLY: {result['output']} ({result['gaussians']} Gaussians)")
        print("This is a Gaussian initialization fusion, not GaussianObject training or a validated final reconstruction.")
    else:
        print(f"Completed stage: {manifest['stage']}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="splatfusion")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="Run multi-view Gaussian reconstruction stages")
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, default=Path("runs/default"))
    p.add_argument("--sharp", action="store_true")
    p.add_argument("--vggt", action="store_true")
    p.add_argument("--vggt-checkpoint", type=Path)
    p.add_argument("--register", action="store_true", help="Refine VGGT poses with FPFH RANSAC and a robust pose graph (requires Open3D)")
    p.add_argument("--registration-stride", type=int, default=4)
    p.add_argument("--registration-voxel-size", type=float, default=0.01)
    p.add_argument("--registration-neighbors", type=int, default=4)
    p.add_argument("--fuse", action="store_true", help="Transform and fuse SHARP camera-frame Gaussian priors")
    p.add_argument("--voxel-size", type=float, default=0.005)
    p.add_argument("--max-gaussians", type=int, default=2000000)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=run)
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
