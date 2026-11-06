# Implementation roadmap

## Implemented in the repository

- Shared EXIF normalization and photo indexing.
- Per-image SHARP CLI inference.
- Native VGGT-Ω inference with camera-to-world, intrinsics, depth, confidence and preprocessing metadata.
- SAM automatic foreground mask generation and external mask support.
- Masked FPFH/RANSAC global point-cloud registration, point-to-plane ICP and robust SE(3) pose-graph refinement.
- SHARP PLY metadata parsing, camera-frame Gaussian transforms and scale calibration against VGGT depth.
- Mask-aware SHARP Gaussian filtering and voxel duplicate reduction.
- GaussianObject-compatible data preparation from VGGT cameras/depth and SHARP point initialization.
- GaussianObject training orchestration for coarse 3DGS, leave-one-out stages, LoRA training and repair.
- Unit tests and GitHub Actions CI.

## Remaining engineering and validation work

1. **Robustness**: add learned correspondence/track matching, view-overlap scoring, loop closure, disconnected-component detection and optional Sim(3) pose graph.
2. **Mask quality**: improve automatic subject selection for multiple prominent objects and offer optional text-guided segmentation.
3. **Visibility-aware fusion**: use occlusion ordering, normals, view support and confidence to suppress contradictory monocular priors rather than relying only on voxel deduplication.
4. **Training resilience**: add resume/checkpoint detection, subprocess timeouts, disk-space checks, GPU memory budgeting and recovery from interrupted GaussianObject stages.
5. **Quality report**: export reprojection metrics, surface coverage, confidence/uncertainty maps and novel-view render videos.
6. **Validation**: run synthetic geometry tests, public benchmark objects and user-captured casual photo sets on supported CUDA environments; tune defaults based on measured results.

## Completion criteria

A full run should accept a folder of arbitrary casual views and, with one command, generate a jointly optimized final 3DGS PLY plus diagnostics. It should not require manual camera poses or PLY alignment. It must fail clearly when masks, scale calibration, view overlap, checkpoints or GPU resources are insufficient.

The repository now contains the full orchestration path, but GPU execution and reconstruction quality have not been validated in this environment. The first real run should be treated as an integration test, not as a guaranteed quality result.
