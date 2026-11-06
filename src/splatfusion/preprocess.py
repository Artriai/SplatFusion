"""Shared image normalization and VGGT resize/crop coordinate bookkeeping."""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageOps


def normalize_images(images: list[Path], output_dir: Path) -> list[Path]:
    """Normalize EXIF orientation and alpha handling once for every model.

    Original files are reused when already RGB and orientation-neutral. When a
    copy is needed, EXIF metadata (including focal length) is retained and the
    orientation tag is reset to 1 after pixels are physically rotated.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = []
    for index, path in enumerate(images):
        with Image.open(path) as source:
            exif = source.getexif()
            orientation = int(exif.get(274, 1))
            needs_alpha = source.mode in ("RGBA", "LA") or "transparency" in source.info
            needs_copy = orientation != 1 or needs_alpha or source.mode != "RGB"
            if not needs_copy:
                normalized.append(path.resolve())
                continue
            image = ImageOps.exif_transpose(source)
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, rgba)
            image = image.convert("RGB")
            exif[274] = 1
            target = output_dir / f"view_{index:04d}{path.suffix.lower()}"
            save_kwargs = {"exif": exif.tobytes()}
            if target.suffix in (".jpg", ".jpeg"):
                save_kwargs.update({"quality": 98, "subsampling": 0})
            image.save(target, **save_kwargs)
            normalized.append(target.resolve())
    return normalized


def _target_shape(height: int, width: int, resolution: int, patch_size: int, mode: str):
    aspect = height / max(width, 1)
    if mode == "balanced":
        tokens = (resolution // patch_size) ** 2
        wp = max(1, int(round((tokens / aspect) ** 0.5)))
        hp = max(1, int(round(tokens / wp)))
        return hp * patch_size, wp * patch_size
    if mode == "max_size":
        if aspect >= 1:
            h = resolution
            w = max(patch_size, int(round((resolution / aspect) / patch_size)) * patch_size)
        else:
            w = resolution
            h = max(patch_size, int(round((resolution * aspect) / patch_size)) * patch_size)
        return h, w
    raise ValueError("mode must be balanced or max_size")


def vggt_preprocess_metadata(image_paths: list[Path], resolution: int = 512,
                             patch_size: int = 16, mode: str = "balanced"):
    """Return exact crop/resize/pad metadata matching VGGT-Ω load_fn."""
    records = []
    shapes = []
    for path in image_paths:
        with Image.open(path) as im:
            width, height = im.size
        left = top = 0
        right, bottom = width, height
        aspect = height / max(width, 1)
        if aspect < 0.5:
            crop_width = min(width, max(1, int(round(height / 0.5))))
            left = max((width - crop_width) // 2, 0)
            right = left + crop_width
        elif aspect > 2.0:
            crop_height = min(height, max(1, int(round(width * 2.0))))
            top = max((height - crop_height) // 2, 0)
            bottom = top + crop_height
        crop_w, crop_h = right - left, bottom - top
        resized_h, resized_w = _target_shape(crop_h, crop_w, resolution, patch_size, mode)
        records.append({
            "original_width": width, "original_height": height,
            "crop_left": left, "crop_top": top,
            "crop_width": crop_w, "crop_height": crop_h,
            "resized_width": resized_w, "resized_height": resized_h,
        })
        shapes.append((resized_h, resized_w))
    max_h = max(x[0] for x in shapes)
    max_w = max(x[1] for x in shapes)
    for record in records:
        record["pad_left"] = (max_w - record["resized_width"]) // 2
        record["pad_top"] = (max_h - record["resized_height"]) // 2
        record["model_width"] = max_w
        record["model_height"] = max_h
    return records


def map_original_pixels_to_vggt(u, v, metadata):
    """Map original-image pixel coordinates into the padded VGGT input image."""
    import numpy as np
    m = metadata
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u_crop = u - m["crop_left"]
    v_crop = v - m["crop_top"]
    u_resized = (u_crop + 0.5) * m["resized_width"] / m["crop_width"] - 0.5
    v_resized = (v_crop + 0.5) * m["resized_height"] / m["crop_height"] - 0.5
    return u_resized + m["pad_left"], v_resized + m["pad_top"]
