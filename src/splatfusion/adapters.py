"""Adapters for upstream inference tools."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_sharp(image: Path, output_dir: Path) -> Path:
    executable = shutil.which("sharp")
    if not executable:
        raise RuntimeError("SHARP CLI not found; install https://github.com/apple/ml-sharp")
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([executable, "predict", "-i", str(image), "-o", str(output_dir)], check=True)
    candidates = sorted(output_dir.glob("*.ply"))
    if len(candidates) == 1:
        return candidates[0]
    expected = output_dir / f"{image.stem}.ply"
    if expected.exists():
        return expected
    raise RuntimeError(f"Cannot uniquely identify SHARP output for {image}: {candidates}")


def run_vggt_omega(images: list[Path], output_json: Path, checkpoint: Path | None = None):
    runner = os.environ.get("SPLATFUSION_VGGT_OMEGA_RUNNER")
    if runner:
        command = [runner]
    else:
        command = [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts" / "vggt_omega_runner.py")]
    command += ["--images", json.dumps([str(p.resolve()) for p in images]),
                "--output", str(output_json)]
    checkpoint = checkpoint or (Path(os.environ["VGGT_OMEGA_CHECKPOINT"]) if os.environ.get("VGGT_OMEGA_CHECKPOINT") else None)
    if checkpoint:
        command += ["--checkpoint", str(checkpoint)]
    else:
        raise RuntimeError("Provide --vggt-checkpoint or VGGT_OMEGA_CHECKPOINT")
    subprocess.run(command, check=True)
    if not output_json.exists():
        raise RuntimeError("VGGT-Ω runner did not create its JSON output")
    return json.loads(output_json.read_text(encoding="utf-8"))
