# SplatFusion

Research prototype for unstructured multi-view object reconstruction with 3D Gaussian Splatting.

The target workflow accepts casual photos of the same rigid object (different viewpoints, backgrounds and camera distances) and produces one globally consistent 3D representation.

## Research direction

- **SHARP** ([apple/ml-sharp](https://github.com/apple/ml-sharp)): fast single-image Gaussian predictions. Used as per-view priors/initialization, not naively concatenated.
- **VGGT-Ω** ([facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega)): multi-view camera and geometry estimation.
- **GaussianObject** ([GaussianObject](https://github.com/GaussianObject/GaussianObject)): object-level Gaussian optimization and repair.

Important: these repositories do not currently expose a plug-and-play compatible pipeline. Camera conventions, scale, masks, Gaussian parameterization and visibility must be validated before transforming SHARP PLYs into a shared frame. SHARP's hallucinated unseen regions must not be treated as observed geometry.

## Current status

This repository starts with an integration scaffold. The first runnable milestone is per-image SHARP inference plus a reproducible manifest. VGGT-Ω inference, cross-view registration, Gaussian fusion and GaussianObject optimization are deliberately separate adapters and are not claimed to be implemented yet.

## Quick start

Python 3.10+ is recommended for the orchestration package. Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install Apple's SHARP separately using its official instructions and make sure the `sharp` executable is on PATH.

Put photos of one rigid object in a directory (JPG/PNG/WebP):

```bash
printf 'your photos here\n'
splatfusion run ./photos --output ./runs/demo --sharp
```

The command runs SHARP on each image and writes per-view PLYs plus `manifest.json`. It does **not** yet claim to produce a globally fused object.

For a dry run without running inference:

```bash
splatfusion run ./photos --output ./runs/demo --dry-run
```

## Planned pipeline

1. Validate and index input images; retain originals and record hashes.
2. Segment the object while retaining unmasked originals for geometry estimation.
3. Estimate all camera poses/depth with VGGT-Ω; check overlap and confidence.
4. Run SHARP per image to obtain Gaussian priors.
5. Calibrate coordinate frames and scale; align observations using multi-view geometry.
6. Fuse only supported/visible Gaussian evidence; suppress duplicate and conflicting splats.
7. Initialize and jointly optimize one object representation with GaussianObject-style losses/repair.
8. Export PLY and diagnostics, including coverage and uncertain regions.

## Design principles

- Never concatenate independent SHARP PLY files and call that multi-view reconstruction.
- Keep observed geometry separate from monocular prior/hallucinated geometry.
- Treat pose, scale and coordinate conventions explicitly.
- Fail with diagnostics when views cannot be reliably registered.
- Preserve per-stage artifacts to make experiments reproducible.

## License

This repository's code is licensed under the MIT License. The upstream models and dependencies retain their own licenses; check each upstream repository and checkpoint terms before use.
