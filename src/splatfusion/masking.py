"""Automatic foreground masks using SAM or user-provided masks."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image


def _select_automatic_mask(annotations, height: int, width: int) -> np.ndarray:
    if not annotations:
        raise RuntimeError("SAM returned no masks")
    yy, xx = np.mgrid[0:height, 0:width]
    center = (np.abs(xx - (width - 1) / 2) < width * 0.12) & (
        np.abs(yy - (height - 1) / 2) < height * 0.12
    )
    candidates = []
    for item in annotations:
        mask = np.asarray(item["segmentation"], dtype=bool)
        area = float(mask.mean())
        if area < 0.01 or area > 0.92:
            continue
        center_coverage = float(mask[center].mean()) if center.any() else 0.0
        touches = (
            mask[0].mean() + mask[-1].mean() + mask[:, 0].mean() + mask[:, -1].mean()
        ) / 4.0
        predicted_iou = float(item.get("predicted_iou", 0.5))
        stability = float(item.get("stability_score", 0.5))
        score = predicted_iou * stability * (0.35 + 0.65 * center_coverage)
        score *= np.sqrt(area) * (1.0 - 0.65 * touches)
        candidates.append((score, mask))
    if not candidates:
        raise RuntimeError("SAM produced no plausible foreground mask (area outside 1%-92%)")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def prepare_masks(image_paths: list[Path], output_dir: Path,
                  mask_dir: Path | None = None, sam_checkpoint: Path | None = None,
                  model_type: str = "vit_h", device: str | None = None) -> list[Path]:
    """Return one binary PNG mask per input image.

    External masks take precedence. Otherwise SAM's automatic mask generator is
    used and the most plausible central foreground segment is selected.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = None
    if mask_dir is None and sam_checkpoint is None:
        raise ValueError("Provide --mask-dir or --sam-checkpoint to build object masks")
    if mask_dir is None:
        try:
            import torch
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc:
            raise RuntimeError(
                "SAM support requires the GaussianObject segment-anything dependency "
                "or install segment-anything separately"
            ) from exc
        if not Path(sam_checkpoint).is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
        sam.to(device=device)
        generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.90,
            crop_n_layers=1,
            min_mask_region_area=100,
        )

    outputs = []
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
        h, w = image.shape[:2]
        if mask_dir is not None:
            candidates = [
                mask_dir / f"{image_path.stem}.png",
                mask_dir / f"{image_path.stem}.jpg",
                mask_dir / f"{image_path.stem}.jpeg",
                mask_dir / f"view_{index:04d}.png",
            ]
            mask_path = next((p for p in candidates if p.is_file()), None)
            if mask_path is None:
                raise FileNotFoundError(f"No foreground mask found for {image_path.name} in {mask_dir}")
            with Image.open(mask_path) as source_mask:
                mask_image = source_mask.convert("L").resize((w, h), Image.Resampling.NEAREST)
            mask = np.asarray(mask_image) >= 128
        else:
            annotations = generator.generate(image)
            mask = _select_automatic_mask(annotations, h, w)
        area = float(mask.mean())
        if area < 0.005 or area > 0.98:
            raise RuntimeError(f"Foreground mask for {image_path.name} has implausible area {area:.3f}")
        output = output_dir / f"view_{index:04d}.png"
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(output)
        outputs.append(output.resolve())
    return outputs
