# SplatFusion

Research prototype for unstructured multi-view object reconstruction with 3D Gaussian Splatting.

The target workflow accepts casual photos of one rigid object (different viewpoints, backgrounds and camera distances) and produces one globally consistent Gaussian initialization. The longer-term target is a fully optimized object model.

## Model roles

- **SHARP** ([apple/ml-sharp](https://github.com/apple/ml-sharp)): per-image Gaussian predictions.
- **VGGT-Ω** ([facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega)): multi-view camera, depth and confidence estimation.
- **GaussianObject** ([GaussianObject](https://github.com/chensjtu/GaussianObject)): intended final object-level optimization and repair stage.

## Implemented

- EXIF orientation normalization shared by SHARP and VGGT-Ω, retaining focal-length metadata.
- Native VGGT-Ω inference runner with camera-to-world, intrinsics, depth and confidence export.
- Per-image SHARP CLI inference.
- Exact bookkeeping for VGGT crop/resize/pad transforms.
- Open3D FPFH + RANSAC global point-cloud registration, point-to-plane ICP refinement, and robust SE(3) pose-graph optimization.
- SHARP PLY metadata reader and Gaussian transforms for xyz, log-scales and wxyz quaternions.
- Per-view SHARP-to-VGGT depth scale calibration before world-frame fusion.
- Voxel-based duplicate reduction and Gaussian count cap.
- Unit tests and GitHub Actions CI.

## Important SHARP coordinate detail

Apple's official `sharp predict` implementation predicts Gaussians in NDC internally, then calls `unproject_gaussians` with identity extrinsics and resized intrinsics before writing the PLY. Therefore the exported PLY vertex positions are already Euclidean coordinates in that image's OpenCV camera frame (+x right, +y down, +z forward), not raw NDC. SplatFusion reads the embedded intrinsics/image-size metadata, calibrates scale against VGGT depth, and then applies VGGT camera-to-world.

See Apple's implementation:
- [sharp/cli/predict.py](https://github.com/apple-aiml-research/ml-sharp/blob/main/src/sharp/cli/predict.py)
- [sharp/utils/gaussians.py](https://github.com/apple-aiml-research/ml-sharp/blob/main/src/sharp/utils/gaussians.py)

## Install

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install Apple's SHARP separately following its official instructions. For pose refinement, install the optional Open3D extra:

```bash
pip install -e ".[registration]"
```

VGGT-Ω has a separate upstream installation and checkpoint access requirement. Install it in the same Python environment and obtain a checkpoint from [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega).

## Run

Place at least two photos of the same rigid object in a folder:

```bash
splatfusion run ./photos --output ./runs/object --sharp --vggt \
  --vggt-checkpoint /path/to/vggt_omega_checkpoint.pt
```

Run global registration and pose-graph refinement:

```bash
splatfusion run ./photos --output ./runs/object --sharp --vggt --register \
  --vggt-checkpoint /path/to/vggt_omega_checkpoint.pt
```

Generate an initial fused Gaussian PLY:

```bash
splatfusion run ./photos --output ./runs/object --sharp --vggt --register --fuse \
  --vggt-checkpoint /path/to/vggt_omega_checkpoint.pt
```

Scale calibration is strict by default. If too few pixel/depth correspondences exist, fusion stops rather than silently mixing incompatible scales. `--allow-unscaled` explicitly opts into a scale-1 fallback for debugging only.

A dry run validates and indexes input images without model inference:

```bash
splatfusion run ./photos --output ./runs/dry --dry-run
```

## Current output semantics

- `sharp/*.ply`: per-view camera-frame SHARP predictions.
- `vggt/predictions.json`: VGGT-Ω camera/depth outputs.
- `vggt/predictions_registered.json`: refined camera poses when `--register` is used.
- `fused/fused.ply`: initial Gaussian fusion after coordinate conversion, scale calibration and voxel deduplication.

**The fused PLY is not yet the final GaussianObject reconstruction.** Visibility-aware conflict removal, masks, GaussianObject training/repair orchestration, final rendering and end-to-end validation remain in the roadmap.

## Python geometry API

```python
import numpy as np
from splatfusion.geometry import umeyama_similarity, transform_points
from splatfusion.ply import transform_ply, merge_plys

T_dst_src = umeyama_similarity(source_xyz, target_xyz, with_scale=True)
aligned = transform_points(source_xyz, T_dst_src)
transform_ply("camera_frame.ply", "world_frame.ply", T_dst_src)
# Only concatenate after every input has been registered into the same frame.
merge_plys(["view0_world.ply", "view1_world.ply"], "merged.ply")
```

## Known limitations

- FPFH/RANSAC can fail on textureless, reflective, symmetric or very low-overlap views. Failed edges are reported; VGGT poses remain as low-weight graph priors.
- The current depth-based scale calibration assumes the projected SHARP Gaussian centers and VGGT depth refer to corresponding visible surfaces. Strong occlusion or incorrect depth can make calibration fail; strict mode stops.
- Current fusion uses voxel deduplication, not full visibility-aware surface reasoning.
- No automatic SAM 2 masks or GaussianObject joint optimization is wired into the final command yet.
- Real GPU integration and reconstruction quality have not been validated in this repository environment.

## License

SplatFusion code is licensed under MIT. Upstream models, checkpoints and dependencies retain their own licenses; check each upstream repository and checkpoint terms before use.
