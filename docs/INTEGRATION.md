# Upstream integration contract

## SHARP

Install Apple's official repository and checkpoint using its instructions:
https://github.com/apple/ml-sharp

The adapter calls `sharp predict -i IMAGE -o OUTPUT_DIR` once per input image.

Important coordinate detail: Apple's official `sharp predict` code predicts an internal NDC Gaussian set, then calls `unproject_gaussians(gaussians_ndc, identity_extrinsics, resized_intrinsics, internal_shape)` before calling `save_ply`. Thus the exported PLY vertex positions are already Euclidean coordinates in the input camera frame, not raw NDC. The camera convention is OpenCV (+x right, +y down, +z forward). The PLY contains the image intrinsics, image size and identity export extrinsic.

Relevant upstream implementation:
- https://github.com/apple-aiml-research/ml-sharp/blob/main/src/sharp/cli/predict.py
- https://github.com/apple-aiml-research/ml-sharp/blob/main/src/sharp/utils/gaussians.py

SplatFusion reads this metadata, projects SHARP Gaussian centers back into the source image, samples VGGT depth at corresponding preprocessed pixels, estimates a robust per-view scale correction, and then applies the VGGT camera-to-world transform. Do not apply an additional NDC unprojection to official SHARP CLI PLY files.

The PLY transform helper supports standard 3DGS vertex fields: `x,y,z`, `scale_0..2` (log scales), and `rot_0..3` (wxyz). SHARP's quaternion implementation confirms wxyz ordering. Unknown PLY schemas are rejected rather than silently reinterpreted.

## VGGT-Ω

Upstream: https://github.com/facebookresearch/vggt-omega

SplatFusion includes `scripts/vggt_omega_runner.py`, which imports the upstream Python API and expects the checkpoint path supplied by `--vggt-checkpoint` or `VGGT_OMEGA_CHECKPOINT`.

The runner:
- normalizes the camera pose encoding to camera-to-world by inverting the upstream camera-from-world extrinsics;
- exports OpenCV intrinsics, depth and confidence arrays;
- records the exact crop/resize/pad mapping used by the VGGT input loader.

VGGT-Ω checkpoint access and dependency installation remain governed by the upstream repository.

## Pose refinement

`--register` builds camera-local point clouds by unprojecting VGGT depth maps. It tries FPFH feature matching + RANSAC global registration and point-to-plane ICP for nearby camera pairs, then robustly optimizes an SE(3) pose graph. VGGT poses are retained as low-weight adjacent-edge priors, so a failed pairwise registration does not invent a replacement pose.

Open3D is optional and required for this stage:
`pip install -e ".[registration]"`.

This refinement is not guaranteed to recover disconnected views or pairs with insufficient geometric overlap. The diagnostics record accepted and rejected edges.

## GaussianObject

Upstream: https://github.com/chensjtu/GaussianObject

GaussianObject's released scripts are research training workflows rather than a stable callable library API. Its training data contract, masks, camera convention and optimization entry points still need a dedicated adapter. The current fused PLY is only an initialization artifact; it is not a trained GaussianObject output.

## Scale and visibility

SHARP's metric camera-local coordinates and VGGT's predicted depth may have different scale. SplatFusion estimates a robust scale ratio from projected SHARP centers and VGGT depth samples. By default, fusion stops if there are too few valid correspondences. `--allow-unscaled` is a debugging escape hatch and should not be used for quality evaluation.

Current voxel deduplication is not full visibility-aware fusion. It does not yet reason about occlusion ordering, surface normals, object masks, or hidden-region hallucination. Those are required before treating the fused PLY as a final reconstruction.
