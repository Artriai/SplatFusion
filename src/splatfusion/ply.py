"""PLY utilities for standard 3DGS Gaussian vertex properties.

Supports ASCII and binary PLY through plyfile. The transform path preserves
unknown vertex properties. It transforms xyz and, when standard 3DGS scale /
rotation fields are present, transforms anisotropic Gaussian covariance and
re-encodes scale/quaternion. Verify field conventions for each upstream export.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

try:
    from plyfile import PlyData, PlyElement
except ImportError as exc:  # pragma: no cover
    PlyData = PlyElement = None
    _PLY_IMPORT_ERROR = exc
else:
    _PLY_IMPORT_ERROR = None


def _require_plyfile():
    if PlyData is None:
        raise RuntimeError("PLY support requires 'plyfile'. Install with: pip install plyfile") from _PLY_IMPORT_ERROR


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    # Standard 3DGS PLY stores rot_0..rot_3 as w,x,y,z.
    q = np.asarray(q, dtype=np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2*(y*y + z*z); R[..., 0, 1] = 2*(x*y-z*w); R[..., 0, 2] = 2*(x*z+y*w)
    R[..., 1, 0] = 2*(x*y+z*w); R[..., 1, 1] = 1-2*(x*x+z*z); R[..., 1, 2] = 2*(y*z-x*w)
    R[..., 2, 0] = 2*(x*z-y*w); R[..., 2, 1] = 2*(y*z+x*w); R[..., 2, 2] = 1-2*(x*x+y*y)
    return R


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    out = np.empty((len(R), 4), dtype=np.float64)
    for i, m in enumerate(R):
        tr = np.trace(m)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            out[i] = [0.25*s, (m[2,1]-m[1,2])/s, (m[0,2]-m[2,0])/s, (m[1,0]-m[0,1])/s]
        else:
            k = int(np.argmax(np.diag(m)))
            if k == 0:
                s = np.sqrt(1+m[0,0]-m[1,1]-m[2,2])*2
                out[i] = [(m[2,1]-m[1,2])/s, 0.25*s, (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s]
            elif k == 1:
                s = np.sqrt(1+m[1,1]-m[0,0]-m[2,2])*2
                out[i] = [(m[0,2]-m[2,0])/s, (m[0,1]+m[1,0])/s, 0.25*s, (m[1,2]+m[2,1])/s]
            else:
                s = np.sqrt(1+m[2,2]-m[0,0]-m[1,1])*2
                out[i] = [(m[1,0]-m[0,1])/s, (m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s, 0.25*s]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def load_vertex(path: str | Path):
    _require_plyfile()
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"No vertex element in {path}")
    return ply, ply["vertex"].data.copy()


def transform_vertex(data, T: np.ndarray):
    T = np.asarray(T, dtype=np.float64)
    names = data.dtype.names or ()
    if not all(k in names for k in ("x", "y", "z")):
        raise ValueError("PLY vertex needs x, y, z fields")
    out = data.copy()
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    A, t = T[:3, :3], T[:3, 3]
    out["x"], out["y"], out["z"] = (xyz @ A.T + t).T
    scale_names = [f"scale_{i}" for i in range(3)]
    rot_names = [f"rot_{i}" for i in range(4)]
    if all(k in names for k in scale_names) and all(k in names for k in rot_names):
        scales = np.exp(np.column_stack([data[k] for k in scale_names]).astype(np.float64))
        quats = np.column_stack([data[k] for k in rot_names]).astype(np.float64)
        Rg = _quat_to_matrix(quats)
        cov = Rg @ np.array([np.diag(s*s) for s in scales]) @ np.swapaxes(Rg, 1, 2)
        cov2 = A @ cov @ A.T
        eigval, eigvec = np.linalg.eigh(cov2)
        order = np.argsort(eigval, axis=1)[:, ::-1]
        eigval = np.take_along_axis(eigval, order, axis=1)
        eigvec = np.take_along_axis(eigvec, order[:, None, :], axis=2)
        # Keep right-handed rotation matrices.
        neg = np.linalg.det(eigvec) < 0
        eigvec[neg, :, -1] *= -1
        new_scales = np.sqrt(np.maximum(eigval, 1e-16))
        new_quats = _matrix_to_quat(eigvec)
        for i, k in enumerate(scale_names):
            out[k] = np.log(new_scales[:, i])
        for i, k in enumerate(rot_names):
            out[k] = new_quats[:, i]
    return out


def write_vertex(template_ply, vertex_data, output: str | Path):
    _require_plyfile()
    elements = []
    for element in template_ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(vertex_data, "vertex"))
        else:
            elements.append(element)
    PlyData(elements, text=template_ply.text, byte_order=template_ply.byte_order,
            comments=template_ply.comments, obj_info=template_ply.obj_info).write(str(output))


def transform_ply(source: str | Path, output: str | Path, T: np.ndarray):
    ply, data = load_vertex(source)
    transformed = transform_vertex(data, T)
    write_vertex(ply, transformed, output)


def merge_plys(paths: list[str | Path], output: str | Path):
    """Concatenate already co-registered PLY vertices; does not perform registration."""
    _require_plyfile()
    if not paths:
        raise ValueError("No PLY files supplied")
    template, first = load_vertex(paths[0])
    chunks = [first]
    for path in paths[1:]:
        _, data = load_vertex(path)
        if data.dtype != first.dtype:
            raise ValueError(f"Incompatible PLY vertex schemas: {path}")
        chunks.append(data)
    merged = np.concatenate(chunks)
    write_vertex(template, merged, output)
    return len(merged)


def read_sharp_metadata(ply):
    """Read camera metadata embedded by Apple's official SHARP PLY exporter.

    Official sharp predict first calls unproject_gaussians with identity
    extrinsics before save_ply. Vertex xyz are therefore already Euclidean
    coordinates in the input camera frame (OpenCV: +x right, +y down,
    +z forward), not NDC. The PLY stores focal intrinsics, image size and a
    dummy identity extrinsic for that camera-local frame.
    """
    names = {element.name for element in ply.elements}
    if "intrinsic" not in names or "image_size" not in names:
        raise ValueError("PLY lacks SHARP intrinsic/image_size metadata; cannot verify camera frame")
    intrinsic_values = np.asarray(ply["intrinsic"].data["intrinsic"], dtype=np.float64).reshape(-1)
    if intrinsic_values.size != 9:
        raise ValueError(f"Expected 9 intrinsic values, got {intrinsic_values.size}")
    K = intrinsic_values.reshape(3, 3)
    size_values = np.asarray(ply["image_size"].data["image_size"], dtype=np.int64).reshape(-1)
    if size_values.size != 2:
        raise ValueError("Expected image_size metadata [width,height]")
    width, height = int(size_values[0]), int(size_values[1])
    if "extrinsic" in names:
        ext_values = np.asarray(ply["extrinsic"].data["extrinsic"], dtype=np.float64).reshape(-1)
        if ext_values.size == 16:
            extrinsic = ext_values.reshape(4, 4)
        elif ext_values.size == 12:
            extrinsic = np.eye(4)
            extrinsic[:3] = ext_values.reshape(3, 4)
            extrinsic[:3, :3] = extrinsic[:3, :3].T
        else:
            raise ValueError(f"Unsupported SHARP extrinsic metadata length: {ext_values.size}")
    else:
        extrinsic = np.eye(4)
    if not np.allclose(extrinsic, np.eye(4), atol=1e-5):
        raise ValueError("Expected SHARP CLI camera-local PLY with identity export extrinsic")
    if width <= 0 or height <= 0 or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid SHARP camera metadata")
    return {"intrinsics": K, "width": width, "height": height, "extrinsic": extrinsic}


def write_vertex_only(vertex_data, output: str | Path):
    """Write a standard 3DGS PLY without stale per-camera SHARP metadata."""
    _require_plyfile()
    PlyData([PlyElement.describe(vertex_data, "vertex")], text=False).write(str(output))
