from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .adapters import run_sharp, run_vggt_omega
from .fusion import fuse_sharp_with_vggt
from .pose_refinement import refine_prediction_file
from .preprocess import normalize_images
from .masking import prepare_masks
from .gaussianobject_data import prepare_gaussianobject_dataset
from .gaussianobject_adapter import run_gaussianobject

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"Input directory does not exist: {directory}")
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_manifest(images: list[Path], output: Path) -> dict:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "splatfusion",
        "stage": "input_indexed",
        "output_dir": str(output.resolve()),
        "views": [
            {
                "view_id": f"view_{i:04d}",
                "image": str(image.resolve()),
                "processed_image": str(image.resolve()),
                "mask": None,
                "sharp_ply": None,
                "camera_to_world": None,
                "intrinsics": None,
                "status": "pending",
            }
            for i, image in enumerate(images)
        ],
        "notes": [
            "SHARP CLI exports Euclidean camera-frame Gaussians.",
            "VGGT camera poses place all views into a shared world frame.",
            "GaussianObject receives VGGT-derived depth/cameras and SHARP-derived point initialization.",
        ],
    }



def preflight(args: argparse.Namespace) -> None:
    """Fail early on missing binaries, checkpoints or optional dependencies."""
    if args.dry_run:
        return
    if args.vggt or args.gaussianobject:
        try:
            import torch
        except ImportError as exc:
            raise SystemExit("VGGT-Ω and GaussianObject require PyTorch in this environment") from exc
        if args.gaussianobject and not torch.cuda.is_available():
            raise SystemExit("GaussianObject training requires a CUDA-capable PyTorch environment")
    if args.sharp and shutil.which("sharp") is None:
        raise SystemExit("SHARP CLI is missing; install apple-aiml-research/ml-sharp and ensure sharp is on PATH")
    if args.vggt:
        checkpoint = args.vggt_checkpoint or (
            Path(os.environ["VGGT_OMEGA_CHECKPOINT"])
            if os.environ.get("VGGT_OMEGA_CHECKPOINT") else None
        )
        if checkpoint is None or not checkpoint.is_file():
            raise SystemExit("VGGT-Ω checkpoint missing; pass --vggt-checkpoint or set VGGT_OMEGA_CHECKPOINT")
    if args.register:
        try:
            import open3d  # noqa: F401
        except ImportError as exc:
            raise SystemExit('Pose registration requires Open3D: pip install -e ".[registration]"') from exc
    if args.mask_dir is not None and not args.mask_dir.is_dir():
        raise SystemExit(f"Mask directory does not exist: {args.mask_dir}")
    if args.sam_checkpoint is not None and args.mask_dir is None:
        if not args.sam_checkpoint.is_file():
            raise SystemExit(f"SAM checkpoint does not exist: {args.sam_checkpoint}")
        try:
            import segment_anything  # noqa: F401
        except ImportError as exc:
            raise SystemExit('SAM inference requires the SAM extra: pip install -e ".[sam]"') from exc
    if args.gaussianobject:
        root = args.gaussianobject_root
        if root is None or not (root / "train_gs.py").is_file():
            raise SystemExit("--gaussianobject requires a valid --gaussianobject-root checkout")
        if args.gaussianobject_mode == "full":
            required = [
                root / "leave_one_out_stage1.py",
                root / "leave_one_out_stage2.py",
                root / "train_lora.py",
                root / "train_repair.py",
                root / "models" / "control_v11f1e_sd15_tile.yaml",
                root / "models" / "v1-5-pruned.ckpt",
                root / "models" / "control_v11f1e_sd15_tile.pth",
            ]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise SystemExit(f"GaussianObject checkout or model assets are incomplete: {missing}")


def run(args: argparse.Namespace) -> int:
    if args.gaussianobject:
        args.sharp = True
        args.vggt = True
        args.register = True
        args.fuse = True
        if args.gaussianobject_root is None:
            raise SystemExit("--gaussianobject requires --gaussianobject-root")
        if args.mask_dir is None and args.sam_checkpoint is None:
            raise SystemExit("--gaussianobject requires --mask-dir or --sam-checkpoint")
        if args.dry_run:
            raise SystemExit("--dry-run cannot be combined with --gaussianobject")

    images = collect_images(args.input)
    if len(images) < 2:
        raise SystemExit(f"Need at least two supported images; found {len(images)}")
    if args.gaussianobject and args.gaussianobject_mode == "full" and len(images) < 4:
        raise SystemExit("Full GaussianObject repair expects at least four input views")
    if args.fuse and not (args.sharp and args.vggt):
        raise SystemExit("--fuse requires both --sharp and --vggt")
    if args.register and not args.vggt:
        raise SystemExit("--register requires --vggt")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(images, args.output)
    if args.dry_run:
        processed_images = images
    else:
        processed_images = normalize_images(images, args.output / "preprocessed")
    for view, processed in zip(manifest["views"], processed_images):
        view["processed_image"] = str(processed.resolve())

    mask_paths = None
    if not args.dry_run and (args.mask_dir is not None or args.sam_checkpoint is not None):
        mask_paths = prepare_masks(
            processed_images,
            args.output / "masks",
            mask_dir=args.mask_dir,
            sam_checkpoint=args.sam_checkpoint,
            model_type=args.sam_model_type,
            device=args.sam_device,
            source_paths=images,
        )
        for view, mask in zip(manifest["views"], mask_paths):
            view["mask"] = str(mask.resolve())

    if args.sharp and not args.dry_run:
        for index, view in enumerate(manifest["views"]):
            ply = run_sharp(
                processed_images[index],
                args.output / "sharp" / view["view_id"],
            )
            view["sharp_ply"] = str(ply.resolve())
            view["status"] = "sharp_complete"
        manifest["stage"] = "sharp_complete"

    if args.vggt and not args.dry_run:
        pred_path = args.output / "vggt" / "predictions.json"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        run_vggt_omega(processed_images, pred_path, args.vggt_checkpoint)

        if args.register:
            registered_path = args.output / "vggt" / "predictions_registered.json"
            graph_stats = refine_prediction_file(
                pred_path,
                registered_path,
                stride=args.registration_stride,
                voxel_size=args.registration_voxel_size,
                neighbors=args.registration_neighbors,
                masks=mask_paths,
            )
            manifest["pose_graph"] = graph_stats
            pred_path = registered_path
            manifest["stage"] = "pose_refined"

        manifest["vggt_predictions"] = str(pred_path.resolve())
        prediction_payload = json.loads(pred_path.read_text(encoding="utf-8"))
        by_image = {
            str(Path(view["processed_image"]).resolve()): view
            for view in manifest["views"]
        }
        for prediction in prediction_payload.get("views", []):
            key = str(Path(prediction["image"]).resolve())
            if key in by_image:
                by_image[key].update({
                    key_name: prediction.get(key_name)
                    for key_name in (
                        "camera_to_world",
                        "intrinsics",
                        "depth_path",
                        "confidence_path",
                    )
                })
                by_image[key]["status"] = "vggt_complete"
        if not args.register:
            manifest["stage"] = (
                "sharp_and_vggt_complete" if args.sharp else "vggt_complete"
            )

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Indexed {len(images)} images. Manifest: {manifest_path}")

    if args.dry_run:
        print("Dry run: no model inference executed.")
        return 0

    fused_path = args.output / "fused" / "fused.ply"
    if args.fuse:
        fusion_result = fuse_sharp_with_vggt(
            manifest_path,
            fused_path,
            voxel_size=args.voxel_size,
            max_gaussians=args.max_gaussians,
            allow_unscaled=args.allow_unscaled,
        )
        manifest["fusion"] = fusion_result
        manifest["stage"] = "initial_fusion_complete"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Initial fused PLY: {fusion_result['output']} ({fusion_result['gaussians']} Gaussians)")

    if args.gaussianobject:
        if not mask_paths:
            raise RuntimeError("GaussianObject requires foreground masks")
        dataset_dir = args.output / "gaussianobject_data"
        dataset_metadata = prepare_gaussianobject_dataset(
            manifest_path,
            manifest["vggt_predictions"],
            mask_paths,
            dataset_dir,
            fused_sharp_ply=fused_path,
            image_size=args.gaussianobject_image_size,
            stride=args.gaussianobject_stride,
            max_pointcloud_points=args.gaussianobject_max_points,
        )
        manifest["gaussianobject_dataset"] = dataset_metadata
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        gaussianobject_result = run_gaussianobject(
            args.gaussianobject_root,
            dataset_dir,
            args.output / "gaussianobject",
            mode=args.gaussianobject_mode,
            gpu=args.gaussianobject_gpu,
            coarse_iterations=args.gaussianobject_coarse_iterations,
            loo_iterations=args.gaussianobject_loo_iterations,
            lora_steps=args.gaussianobject_lora_steps,
            repair_steps=args.gaussianobject_repair_steps,
            prompt_token=args.gaussianobject_prompt_token,
            sh_degree=args.gaussianobject_sh_degree,
            use_sharp_prior=not args.no_sharp_prior,
        )
        manifest["gaussianobject"] = gaussianobject_result
        manifest["stage"] = (
            "complete" if args.gaussianobject_mode == "full"
            else "coarse_gaussianobject_complete"
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"GaussianObject output: {gaussianobject_result['output_ply']}")
    elif not args.fuse:
        print(f"Completed stage: {manifest['stage']}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="splatfusion")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser(
        "run", help="Run casual multi-view Gaussian reconstruction"
    )
    run_parser.add_argument("input", type=Path)
    run_parser.add_argument("--output", type=Path, default=Path("runs/default"))
    run_parser.add_argument("--sharp", action="store_true")
    run_parser.add_argument("--vggt", action="store_true")
    run_parser.add_argument("--vggt-checkpoint", type=Path)
    run_parser.add_argument(
        "--register",
        action="store_true",
        help="Refine VGGT poses with FPFH RANSAC and a robust pose graph (requires Open3D)",
    )
    run_parser.add_argument("--registration-stride", type=int, default=4)
    run_parser.add_argument("--registration-voxel-size", type=float, default=0.01)
    run_parser.add_argument("--registration-neighbors", type=int, default=4)
    run_parser.add_argument("--fuse", action="store_true")
    run_parser.add_argument("--voxel-size", type=float, default=0.005)
    run_parser.add_argument("--max-gaussians", type=int, default=2000000)
    run_parser.add_argument("--allow-unscaled", action="store_true")
    run_parser.add_argument("--mask-dir", type=Path)
    run_parser.add_argument("--sam-checkpoint", type=Path)
    run_parser.add_argument("--sam-model-type", choices=("vit_h", "vit_l", "vit_b"), default="vit_h")
    run_parser.add_argument("--sam-device", default=None)
    run_parser.add_argument("--gaussianobject", action="store_true")
    run_parser.add_argument("--gaussianobject-root", type=Path)
    run_parser.add_argument("--gaussianobject-mode", choices=("coarse", "full"), default="full")
    run_parser.add_argument("--gaussianobject-gpu", default="0")
    run_parser.add_argument("--gaussianobject-coarse-iterations", type=int, default=15000)
    run_parser.add_argument("--gaussianobject-loo-iterations", type=int, default=15000)
    run_parser.add_argument("--gaussianobject-lora-steps", type=int, default=1800)
    run_parser.add_argument("--gaussianobject-repair-steps", type=int, default=4000)
    run_parser.add_argument("--gaussianobject-prompt-token", default="xxy5syt00")
    run_parser.add_argument("--gaussianobject-sh-degree", type=int, default=2)
    run_parser.add_argument("--gaussianobject-image-size", type=int, default=512)
    run_parser.add_argument("--gaussianobject-stride", type=int, default=4)
    run_parser.add_argument("--gaussianobject-max-points", type=int, default=500000)
    run_parser.add_argument("--no-sharp-prior", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.set_defaults(func=run)
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
