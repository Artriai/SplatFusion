from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"Input directory does not exist: {directory}")
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_manifest(images: list[Path], output: Path) -> dict:
    views = []
    for index, image in enumerate(images):
        views.append({
            "view_id": f"view_{index:04d}",
            "image": str(image.resolve()),
            "sharp_ply": None,
            "camera_to_world": None,
            "intrinsics": None,
            "status": "pending",
        })
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "splatfusion",
        "stage": "input_indexed",
        "output_dir": str(output.resolve()),
        "views": views,
        "notes": [
            "Per-view SHARP predictions are priors, not registered world-space geometry.",
            "No global fusion is performed by the initial scaffold.",
        ],
    }


def run(args: argparse.Namespace) -> int:
    images = collect_images(args.input)
    if not images:
        raise SystemExit(f"No supported images found in: {args.input}")
    if len(images) < 2:
        raise SystemExit("At least two images are required for multi-view reconstruction.")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    ply_dir = output / "sharp"
    ply_dir.mkdir(exist_ok=True)
    manifest = build_manifest(images, output)

    if args.sharp and not args.dry_run:
        executable = shutil.which("sharp")
        if executable is None:
            raise SystemExit("Could not find 'sharp' on PATH. Install apple/ml-sharp or use --dry-run.")
        for view in manifest["views"]:
            image = Path(view["image"])
            target = ply_dir / f"{view['view_id']}.ply"
            command = [executable, "predict", "-i", str(image), "-o", str(target)]
            subprocess.run(command, check=True)
            view["sharp_ply"] = str(target.resolve())
            view["status"] = "sharp_complete"
        manifest["stage"] = "sharp_complete"

    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Indexed {len(images)} images.")
    print(f"Manifest: {manifest_path}")
    if args.dry_run:
        print("Dry run: no models were executed.")
    elif args.sharp:
        print("SHARP predictions completed per image. Global registration/fusion is not implemented yet.")
    else:
        print("Input indexing completed. Use --sharp to run per-image SHARP inference.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="splatfusion", description="Casual multi-view Gaussian reconstruction scaffold")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Index photos and optionally run SHARP")
    run_parser.add_argument("input", type=Path, help="Directory containing photos of one rigid object")
    run_parser.add_argument("--output", type=Path, default=Path("runs/default"))
    run_parser.add_argument("--sharp", action="store_true", help="Run the installed SHARP CLI on each image")
    run_parser.add_argument("--dry-run", action="store_true", help="Only validate and write the input manifest")
    run_parser.set_defaults(func=run)
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
