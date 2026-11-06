# Implementation roadmap

## Implemented in the scaffold

- Photo folder indexing and manifest generation.
- Optional per-image invocation of Apple's SHARP CLI.
- Optional VGGT-Ω runner hook with an explicit JSON interchange contract.
- Similarity registration (Umeyama) and homogeneous point transforms.
- Standard 3DGS PLY vertex transformation for means, log-scales and wxyz quaternions.
- Concatenation of already co-registered PLYs (not a registration algorithm).

## Still required for an end-to-end reconstruction

1. **VGGT-Ω native adapter**: load the upstream checkpoint and emit camera-to-world matrices, intrinsics, depth and confidence arrays. Validate coordinate convention against upstream code.
2. **Foreground masks**: integrate SAM 2 or another segmenter, with original unmasked frames retained for camera estimation.
3. **SHARP frame calibration**: verify whether SHARP PLY coordinates coincide with its input camera frame; estimate per-view scale/rigid corrections from multi-view correspondences.
4. **Robust registration**: match point maps/depth across overlapping views; use confidence-weighted RANSAC/Umeyama and a global pose graph. Reject disconnected or low-overlap views.
5. **Visibility-aware fusion**: remove duplicate splats and reject monocular hallucinations that contradict observed multi-view depth.
6. **GaussianObject adapter**: convert the globally aligned initialization and camera/mask data into the exact data layout expected by upstream scripts; launch training and collect checkpoint/PLY.
7. **Evaluation**: novel-view rendering, reprojection error, point-cloud consistency, surface coverage, and runtime.

## Why the pipeline is not yet called end-to-end

The three upstream projects are independent research codebases with different data contracts. A correct integration must explicitly validate pose direction, coordinate handedness, scale, camera intrinsics, SHARP PLY property conventions and GaussianObject training inputs. The current repository provides the orchestration and geometry foundation, but does not fabricate missing camera poses or silently merge unregistered predictions.
