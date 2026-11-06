# Implementation roadmap

## Implemented in the current codebase

- Photo indexing, manifest generation and shared EXIF orientation normalization.
- Per-image SHARP CLI inference.
- Native VGGT-Ω Python runner with camera-to-world, intrinsics, depth, confidence and preprocessing metadata.
- Open3D FPFH/RANSAC global point-cloud registration and point-to-plane ICP refinement.
- Robust SE(3) pose-graph optimization with VGGT pose priors.
- SHARP PLY metadata parsing and camera-frame Gaussian transforms.
- Per-view SHARP/VGGT depth-based scale calibration.
- Voxel duplicate reduction and output Gaussian count cap.
- Unit tests and GitHub Actions CI.

## Remaining work before a complete end-to-end object reconstruction system

1. **Object segmentation and visibility**: integrate SAM 2 masks; use original unmasked images for geometry estimation and masks for object-only point clouds/Gaussians.
2. **Registration robustness**: add learned dense correspondence/track matching, overlap scoring, loop closure, disconnected-component handling and optional Sim(3) scale graph.
3. **Scale calibration validation**: add confidence-weighted robust scale estimation and independent consistency checks across views; account for EXIF focal uncertainty and model depth bias.
4. **Visibility-aware Gaussian fusion**: remove duplicate splats using view visibility, depth ordering, normals and appearance; reject monocular priors that contradict observed multi-view geometry.
5. **GaussianObject adapter**: prepare the exact upstream dataset layout, masks and camera files; launch coarse training, floater elimination and repair; collect the optimized final PLY.
6. **Rendering and quality report**: export novel-view videos, reprojection errors, coverage maps, confidence/uncertainty and per-view diagnostics.
7. **Production orchestration**: batching, GPU memory budgeting, resumable stages, checkpoint caching, robust error recovery and CLI presets.
8. **Validation**: run synthetic transform tests, public benchmark scenes and user-captured casual photo sets on supported GPU environments.

## Completion criteria

A complete release must accept a folder of arbitrary casual views and, with one command, generate a final jointly optimized 3DGS PLY plus a quality report. It must not require manual camera poses or PLY alignment. It must also fail clearly when views are disconnected or the geometry is insufficient, rather than silently returning a misleading result.

The current `fused.ply` is an initialization artifact, not yet the final GaussianObject reconstruction. GPU execution and reconstruction quality remain unverified in the repository environment.
