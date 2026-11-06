# Upstream integration contract

## SHARP

Install Apple's official repository and checkpoint using its instructions:
https://github.com/apple/ml-sharp

SplatFusion calls `sharp predict -i IMAGE -o OUTPUT_DIR` once per normalized input image.

Apple's official CLI predicts an internal NDC Gaussian set, then calls `unproject_gaussians` with identity extrinsics and resized intrinsics before `save_ply`. Exported PLY xyz are already Euclidean coordinates in the input OpenCV camera frame (+x right, +y down, +z forward), not raw NDC. The PLY contains the input intrinsics, image size and identity export extrinsic.

Relevant upstream implementation:
- https://github.com/apple-aiml-research/ml-sharp/blob/main/src/sharp/cli/predict.py
- https://github.com/apple-aiml-research/ml-sharp/blob/main/src/sharp/utils/gaussians.py

SplatFusion projects SHARP Gaussian centers back to the source pixels, samples VGGT depth, estimates a robust per-view scale correction, optionally filters centers with foreground masks, and then applies VGGT camera-to-world. It does not apply a second NDC unprojection.

The PLY transform helper supports standard 3DGS vertex fields: xyz, log-scales `scale_0..2`, and wxyz quaternions `rot_0..3`.

## VGGT-Ω

Upstream: https://github.com/facebookresearch/vggt-omega

SplatFusion includes a native Python runner at `splatfusion.vggt_runner`. Install the upstream package and supply a checkpoint with `--vggt-checkpoint` or `VGGT_OMEGA_CHECKPOINT`.

The runner:
- decodes VGGT-Ω camera-from-world extrinsics and inverts them to camera-to-world;
- exports OpenCV intrinsics, depth and confidence arrays;
- records the exact crop/resize/pad mapping used by the upstream input loader;
- normalizes EXIF orientation once before SHARP and VGGT inference so their pixels correspond.

## SAM masks

SplatFusion supports Segment Anything's automatic mask generator. Install the optional SAM extra and pass `--sam-checkpoint`, or supply a directory of per-image binary masks with `--mask-dir`.

Automatic mask selection is a central-object heuristic based on mask stability, predicted IoU, area and center coverage. It is not semantic object detection. When a scene contains several equally prominent objects, provide masks explicitly.

Masks are used for:
- filtering the per-view depth point clouds used by pose refinement;
- filtering SHARP Gaussian centers before fusion;
- preparing the masks expected by GaussianObject.

## Pose refinement

`--register` unprojects VGGT depth into camera-local point clouds, optionally filters by object masks, then attempts FPFH feature matching + RANSAC and point-to-plane ICP for nearby camera pairs. Pairwise camera-to-camera constraints are combined with low-weight VGGT pose priors in a robust SE(3) pose graph.

Open3D is optional:
`pip install -e ".[registration]"`.

This is not guaranteed to connect disconnected views or recover pairs with insufficient overlap. Rejected pairwise edges and optimization diagnostics are stored in the registered prediction JSON.

## GaussianObject data contract and training

Upstream: https://github.com/chensjtu/GaussianObject

SplatFusion prepares the upstream COLMAP-free DUSt3R-style directory contract from VGGT-Ω outputs:
- `images/` and `masks/`;
- `dust3r_N.json` camera records (id, img_name, dimensions, camera-to-world rotation/position, fx/fy);
- `dust3r_depth_N.npy` and `dust3r_confidence_N.npy`;
- `sparse_N.txt`;
- `dust3r_N.ply` depth-derived point cloud;
- `sharp_prior.ply` SHARP-derived point-cloud initialization.

The GaussianObject adapter can run:
1. `train_gs.py` coarse 3DGS training using SHARP prior points and VGGT depth/cameras;
2. `leave_one_out_stage1.py` and `leave_one_out_stage2.py`;
3. `train_lora.py`;
4. `train_repair.py` with the upstream COLMAP-free config.

The full mode requires at least four views, a recursive GaussianObject checkout, its CUDA extensions, Stable Diffusion/ControlNet assets and the relevant upstream checkpoints. It is computationally expensive. SplatFusion invokes upstream scripts but does not vendor or modify their code.

## Scale and visibility

SHARP camera-local coordinates and VGGT depth may differ in scale. SplatFusion estimates a robust scale ratio from projected SHARP centers and VGGT depth samples. Strict mode stops if there are too few correspondences; `--allow-unscaled` is a debugging escape hatch only.

The fused SHARP PLY is an initialization, not the final reconstruction. Voxel deduplication does not fully reason about occlusion ordering, surface normals, view visibility or monocular hallucination. GaussianObject's subsequent multi-view optimization/repair is the intended final stage.

## Licensing

SplatFusion orchestration code is MIT-licensed. SHARP, VGGT-Ω, SAM, GaussianObject, Stable Diffusion, ControlNet, CUDA extensions and submodules have independent license terms. Check each upstream license and checkpoint terms before commercial or redistributed use.
