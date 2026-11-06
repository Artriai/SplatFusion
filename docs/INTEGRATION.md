# Upstream integration contract

## SHARP

Install Apple's official repository and checkpoint using its instructions:
https://github.com/apple/ml-sharp

The adapter calls `sharp predict -i IMAGE -o OUTPUT_DIR` once per input image.
The resulting PLY is retained as a *view-local prior*. It is not in world
coordinates and must not be merged until the corresponding camera transform
and scale have been estimated and validated.

The PLY transform helper currently supports standard 3DGS vertex fields:
`x,y,z`, optional `scale_0..2` (log scales), and `rot_0..3` (wxyz).
If an upstream PLY uses different field names or quaternion conventions, add
an explicit converter; do not silently reinterpret fields.

## VGGT-Ω

Upstream: https://github.com/facebookresearch/vggt-omega

VGGT-Ω is a Python research package with checkpoint access requirements. Its
upstream README does not define a stable CLI or a ready-made JSON export format
for this project. SplatFusion therefore delegates to a runner script specified
by `SPLATFUSION_VGGT_OMEGA_RUNNER`.

The runner receives:
- `--images`: JSON-encoded absolute image path array
- `--output`: destination JSON path
- optional `--checkpoint`: checkpoint path

It must write JSON with this schema:

```json
{
  "coordinate_convention": "opencv",
  "world_to_camera": false,
  "views": [
    {
      "image": "/absolute/path/to/image.jpg",
      "camera_to_world": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "intrinsics": [[1000,0,500],[0,1000,500],[0,0,1]],
      "depth_path": "/absolute/path/to/depth.npy",
      "confidence_path": "/absolute/path/to/confidence.npy"
    }
  ]
}
```

The runner author must confirm whether the upstream extrinsics are camera-to-world
or world-to-camera and convert exactly once. The schema deliberately requires
that declaration to avoid the most common pose inversion bug.

## GaussianObject

Upstream: https://github.com/GaussianObject/GaussianObject

GaussianObject's released scripts are research training workflows, not a
stable callable library API. Integrate it through a separate adapter after
the camera/depth/mask and Gaussian initialization artifacts have been
validated. Do not claim that PLY concatenation is GaussianObject optimization.

## Coordinate alignment

SHARP PLYs are local to their input camera. Given camera-to-world transform
`T_cw`, a SHARP Gaussian center can be transformed to world coordinates only
if the SHARP prediction coordinate frame is confirmed to match that camera
frame. Confirm this against SHARP's own metadata and rendering convention.
Scale correction must be explicit.

For a similarity transform with linear component `A=sR`:
- mean: `mu' = A mu + t`
- covariance: `Sigma' = A Sigma A^T`

The helper in `splatfusion.ply` implements this for standard 3DGS log-scale
and quaternion fields. It does not estimate T; registration is a separate
step. Never fuse unregistered views.
