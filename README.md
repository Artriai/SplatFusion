# SplatFusion

Research prototype for unstructured multi-view object reconstruction with 3D Gaussian Splatting.

The goal: accept casual photos of the same rigid object (different viewpoints, backgrounds and camera distances) and produce one globally consistent 3D representation.

## Model roles

- **SHARP** ([apple/ml-sharp](https://github.com/apple/ml-sharp)): fast per-image Gaussian predictions, retained as view-local priors.
- **VGGT-Ω** ([facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega)): multi-view camera and geometry estimation.
- **GaussianObject** ([GaussianObject](https://github.com/GaussianObject/GaussianObject)): object-level Gaussian optimization and repair.

These upstream projects are independent research codebases. The repository explicitly avoids pretending that their outputs are plug-and-play compatible: camera direction, coordinate frames, scale, PLY field conventions, masks and visibility need validation.

## What's implemented

- Photo indexing and a reproducible JSON manifest.
- Optional per-image SHARP CLI invocation.
- Optional VGGT-Ω external runner contract.
- Similarity alignment via Umeyama, point and covariance transforms.
- Standard 3DGS PLY transformation (xyz, log-scales, wxyz quaternion) and concatenation of already registered PLYs.
- Unit tests and GitHub Actions CI.

**Not yet implemented:** native VGGT-Ω inference, automatic robust multi-view registration/pose graph, visibility-aware Gaussian fusion, and GaussianObject training orchestration. See [ROADMAP](docs/ROADMAP.md) and [integration contract](docs/INTEGRATION.md).

## Install

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install Apple's SHARP separately following its official instructions and ensure `sharp` is on PATH.

## Run

Place at least two photos of one rigid object in a folder:

```bash
splatfusion run ./photos --output ./runs/demo --dry-run
```

Run SHARP once per image:

```bash
splatfusion run ./photos --output ./runs/demo --sharp
```

Run both SHARP and a configured VGGT-Ω runner:

```bash
export SPLATFUSION_VGGT_OMEGA_RUNNER=/path/to/your/vggt_omega_runner
splatfusion run ./photos --output ./runs/demo --sharp --vggt
```

The VGGT-Ω runner contract is documented in [docs/INTEGRATION.md](docs/INTEGRATION.md). It must emit camera-to-world transforms, intrinsics and optional depth/confidence file paths. This hook is intentional: the upstream repository currently documents a Python API, not a stable CLI/export contract for SplatFusion.

## Python geometry API

```python
import numpy as np
from splatfusion.geometry import umeyama_similarity, transform_points
from splatfusion.ply import transform_ply, merge_plys

# Estimate transform mapping source 3D correspondences to reference points.
T_ref_src = umeyama_similarity(source_xyz, reference_xyz, with_scale=True)
aligned = transform_points(source_xyz, T_ref_src)

# Transform a standard 3DGS PLY after confirming its frame and field convention.
transform_ply("view_local.ply", "view_world.ply", T_ref_src)

# Only concatenate PLYs after all of them are in the same validated world frame.
merge_plys(["view0_world.ply", "view1_world.ply"], "merged.ply")
```

The PLY merge helper does not estimate alignment, visibility or duplicate removal. Do not use it as a substitute for multi-view registration.

## Design principles

- Never concatenate independent SHARP PLYs and call that multi-view reconstruction.
- Keep observed geometry separate from monocular prior/hallucinated geometry.
- Treat pose, scale and coordinate conventions explicitly.
- Reject disconnected or low-confidence views instead of silently producing a misleading model.
- Preserve per-stage artifacts for reproducible experiments.

## License

SplatFusion code is licensed under MIT. Upstream models, checkpoints and dependencies retain their own licenses; check each upstream repository and checkpoint terms before use.
