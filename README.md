# SplatFusion

Unstructured multi-view object reconstruction with 3D Gaussian Splatting.

The target workflow accepts casual photos of the same rigid object (different viewpoints, backgrounds and camera distances) and produces a globally consistent 3DGS model. The implementation combines single-view Gaussian priors, multi-view geometry, global pose refinement and object-level Gaussian optimization.

## Model roles

- **SHARP** ([apple/ml-sharp](https://github.com/apple/ml-sharp)): per-image Gaussian predictions used as geometric/color initialization.
- **VGGT-Ω** ([facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega)): multi-view camera, depth and confidence estimation.
- **SAM** ([facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)): foreground masks for object-only reconstruction.
- **GaussianObject** ([chensjtu/GaussianObject](https://github.com/chensjtu/GaussianObject)): coarse 3DGS training, leave-one-out data generation, LoRA training and Gaussian repair.

## Implemented pipeline

1. Normalize EXIF orientation once so every model sees identical pixels.
2. Generate foreground masks with SAM or load supplied masks.
3. Run SHARP independently on each photo.
4. Run VGGT-Ω jointly on all photos and export camera-to-world, intrinsics, depth, confidence and exact preprocessing transforms.
5. Build masked depth point clouds; use Open3D FPFH/RANSAC + ICP to refine pairwise camera relations, then optimize an SE(3) pose graph.
6. Read SHARP PLY camera metadata, calibrate per-view scale against VGGT depth, transform Gaussian means/covariances into world coordinates, mask background splats and deduplicate.
7. Prepare GaussianObject's DUSt3R-compatible dataset using VGGT camera/depth outputs and a SHARP-derived point-cloud initialization.
8. Optionally run GaussianObject coarse training, leave-one-out stages, LoRA training and Gaussian repair to export the final PLY.

## SHARP coordinate detail

Apple's official `sharp predict` implementation predicts Gaussians in NDC internally, then calls `unproject_gaussians` with identity extrinsics and resized intrinsics before writing the PLY. Exported PLY vertex positions are therefore already Euclidean coordinates in that image's OpenCV camera frame (+x right, +y down, +z forward), not raw NDC. SplatFusion reads the embedded intrinsics/image-size metadata, calibrates scale against VGGT depth, and applies VGGT camera-to-world.

See Apple's implementation:
- [sharp/cli/predict.py](https://github.com/apple-aiml-research/ml-sharp/blob/main/src/sharp/cli/predict.py)
- [sharp/utils/gaussians.py](https://github.com/apple-aiml-research/ml-sharp/blob/main/src/sharp/utils/gaussians.py)

## Install

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,registration,sam]"
```

Install Apple's SHARP separately following its official instructions and ensure `sharp` is on PATH.

Install VGGT-Ω in the same Python environment and obtain a checkpoint from [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega). The checkpoint may require access approval.

GaussianObject is invoked as an external checkout; its code, submodules and pretrained model assets retain their own licenses and installation requirements:

```bash
git clone --recursive https://github.com/chensjtu/GaussianObject.git third_party/GaussianObject
cd third_party/GaussianObject
# Follow the upstream README to install CUDA extensions and download model assets.
# The full mode expects models/v1-5-pruned.ckpt and ControlNet Tile weights.
```

For SAM, download a compatible Segment Anything checkpoint (for example ViT-H) and pass its path. Alternatively, prepare one binary mask per image and supply `--mask-dir`.

## One-command reconstruction

Full pipeline (VGGT-Ω + SHARP + SAM + pose graph + GaussianObject coarse/repair):

```bash
splatfusion run ./photos \
  --output ./runs/my_object \
  --gaussianobject \
  --gaussianobject-root ./third_party/GaussianObject \
  --sam-checkpoint ./checkpoints/sam_vit_h_4b8939.pth \
  --vggt-checkpoint ./checkpoints/vggt_omega_1b_512.pt
```

This automatically enables SHARP, VGGT-Ω, pose registration and Gaussian fusion. By default it also runs GaussianObject coarse training, both leave-one-out stages, LoRA training and repair. This is computationally expensive and requires a supported CUDA environment plus all upstream dependencies and weights.

For only the GaussianObject coarse stage (skip leave-one-out, LoRA and repair):

```bash
splatfusion run ./photos \
  --output ./runs/my_object \
  --gaussianobject \
  --gaussianobject-mode coarse \
  --gaussianobject-root ./third_party/GaussianObject \
  --sam-checkpoint ./checkpoints/sam_vit_h_4b8939.pth \
  --vggt-checkpoint ./checkpoints/vggt_omega_1b_512.pt
```

If masks are already available:

```bash
splatfusion run ./photos \
  --output ./runs/my_object \
  --gaussianobject \
  --gaussianobject-root ./third_party/GaussianObject \
  --mask-dir ./my_masks \
  --vggt-checkpoint ./checkpoints/vggt_omega_1b_512.pt
```

The external mask directory should contain one binary image per input photo, named after the original image stem or normalized `view_XXXX` stem.

Scale calibration is strict by default. If too few SHARP/VGGT depth correspondences exist, fusion stops rather than silently mixing incompatible scales. `--allow-unscaled` explicitly opts into a scale-1 fallback for debugging only; it is not recommended for quality evaluation.

A dry run validates and indexes photos without model inference:

```bash
splatfusion run ./photos --output ./runs/dry --dry-run
```

## Outputs

- `sharp/`: per-view SHARP PLYs.
- `vggt/`: cameras, depth maps, confidence maps and pose-graph diagnostics.
- `masks/`: normalized foreground masks.
- `fused/fused.ply`: world-space SHARP Gaussian initialization after scale calibration, masking and voxel deduplication.
- `gaussianobject_data/`: prepared images, masks, depth, confidence, cameras and initialization point clouds.
- `gaussianobject/final.ply`: final GaussianObject repair output when full mode succeeds.
- `manifest.json`: stage status, model paths, pose-graph and fusion diagnostics.
- `gaussianobject/logs/`: complete logs for each upstream training stage.

## Important limitations

- FPFH/RANSAC can fail on textureless, reflective, symmetric or very low-overlap views. Rejected edges are recorded; VGGT poses remain as low-weight graph priors.
- Depth-based scale calibration assumes SHARP Gaussian centers and VGGT depth correspond to the same visible surfaces. Occlusion, incorrect focal metadata or poor depth can cause strict fusion to stop.
- SAM's automatic mask selection is a central-object heuristic. For scenes with several similarly prominent objects, supply masks explicitly.
- Voxel deduplication is an initialization strategy, not complete visibility-aware surface reasoning.
- GaussianObject's full training chain is expensive and depends on its external CUDA extensions, Stable Diffusion/ControlNet assets and compatible environment.
- GPU execution and final reconstruction quality have not been validated in this repository environment. The pipeline must be run on a machine with the upstream dependencies and checkpoints installed.

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

## License

SplatFusion orchestration code is licensed under MIT. Upstream model code, checkpoints, CUDA extensions and datasets retain their own licenses. Check the terms of SHARP, VGGT-Ω, SAM, GaussianObject and all submodules before use.
