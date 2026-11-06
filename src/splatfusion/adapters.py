"""Thin adapters around upstream research repositories.

SHARP exposes a CLI and is invoked directly. VGGT-Ω currently requires its
Python package/checkpoint; this adapter writes a clear integration contract
rather than guessing undocumented camera/point-map semantics.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def run_sharp(image: Path, output_dir: Path) -> Path:
    executable = shutil.which("sharp")
    if executable is None:
        raise RuntimeError("SHARP CLI not found. Install https://github.com/apple/ml-sharp")
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([executable, "predict", "-i", str(image), "-o", str(output_dir)], check=True)
    candidates = sorted(output_dir.glob("*.ply"))
    if len(candidates) == 1:
        return candidates[0]
    # SHARP may derive the output filename from input; require unambiguous result.
    expected = output_dir / f"{image.stem}.ply"
    if expected.exists():
        return expected
    if not candidates:
        raise RuntimeError(f"SHARP produced no PLY for {image}")
    raise RuntimeError(f"Expected one SHARP PLY for {image}, found {len(candidates)}")


def run_vggt_omega(images: list[Path], output_json: Path, checkpoint: Path | None = None):
    """Run VGGT-Ω via an installed local checkout.

    The upstream project currently provides a Python API rather than a stable
    CLI contract. Set SPLATFUSION_VGGT_OMEGA_RUNNER to a user-written runner
    implementing the JSON contract documented in docs/INTEGRATION.md.
    """
    import os
    runner = os.environ.get("SPLATFUSION_VGGT_OMEGA_RUNNER")
    if not runner:
        raise RuntimeError(
            "Set SPLATFUSION_VGGT_OMEGA_RUNNER to a VGGT-Ω runner script. "
            "See docs/INTEGRATION.md; no guessed upstream CLI is invoked."
        )
    command = [runner, "--images", json.dumps([str(p.resolve()) for p in images]),
               "--output", str(output_json)]
    if checkpoint:
        command += ["--checkpoint", str(checkpoint)]
    subprocess.run(command, check=True)
    if not output_json.exists():
        raise RuntimeError("VGGT-Ω runner exited without writing its output JSON")
    return json.loads(output_json.read_text(encoding="utf-8"))
