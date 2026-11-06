"""Orchestrate the upstream GaussianObject coarse/repair workflow."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _run(command, cwd: Path, env: dict, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(map(str, command)) + "\n\n")
        log.flush()
        subprocess.run(
            list(map(str, command)),
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _latest_iteration_ply(model_dir: Path) -> Path:
    candidates = []
    point_cloud = model_dir / "point_cloud"
    if point_cloud.is_dir():
        for folder in point_cloud.glob("iteration_*"):
            try:
                iteration = int(folder.name.split("_")[-1])
            except ValueError:
                continue
            ply = folder / "point_cloud.ply"
            if ply.is_file():
                candidates.append((iteration, ply))
    if not candidates:
        raise FileNotFoundError(f"No trained Gaussian point cloud found under {point_cloud}")
    return max(candidates, key=lambda x: x[0])[1]


def run_gaussianobject(
    gaussianobject_root,
    dataset_dir,
    output_dir,
    mode="full",
    gpu="0",
    coarse_iterations=15000,
    loo_iterations=15000,
    lora_steps=1800,
    repair_steps=4000,
    prompt_token="xxy5syt00",
    sh_degree=2,
    use_sharp_prior=True,
):
    """Run GaussianObject training using SplatFusion's VGGT-compatible dataset.

    The upstream repository and its research-only dependencies/checkpoints must
    be installed separately. This function invokes upstream scripts without
    copying or modifying their source tree.
    """
    root = Path(gaussianobject_root).resolve()
    data = Path(dataset_dir).resolve()
    output = Path(output_dir).resolve()
    if mode not in ("coarse", "full"):
        raise ValueError("mode must be 'coarse' or 'full'")
    if not (root / "train_gs.py").is_file():
        raise FileNotFoundError(f"GaussianObject root does not contain train_gs.py: {root}")
    required = ["leave_one_out_stage1.py", "leave_one_out_stage2.py", "train_lora.py", "train_repair.py"]
    if mode == "full":
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"GaussianObject checkout is incomplete: {missing}")
    if not (data / "splatfusion_dataset.json").is_file():
        raise FileNotFoundError(f"Missing prepared GaussianObject dataset metadata in {data}")
    if coarse_iterations < 1000:
        raise ValueError("coarse_iterations must be >=1000")
    if mode == "full" and loo_iterations < 6001:
        raise ValueError("loo_iterations must be >6000 because stage 2 resumes from checkpoint 6000")
    if mode == "full" and (lora_steps < 1 or repair_steps < 1):
        raise ValueError("LoRA and repair step counts must be positive")

    metadata = json.loads((data / "splatfusion_dataset.json").read_text(encoding="utf-8"))
    n_views = int(metadata["views"])
    if mode == "full" and n_views < 4:
        raise ValueError("Full GaussianObject repair expects at least four views")
    if mode == "full":
        model_assets = [
            root / "models" / "control_v11f1e_sd15_tile.yaml",
            root / "models" / "v1-5-pruned.ckpt",
            root / "models" / "control_v11f1e_sd15_tile.pth",
        ]
        missing_assets = [str(p) for p in model_assets if not p.is_file()]
        if missing_assets:
            raise FileNotFoundError(f"Missing GaussianObject pretrained model assets: {missing_assets}")
    init_name = "sharp_prior" if use_sharp_prior and (data / "sharp_prior.ply").is_file() else f"dust3r_{n_views}"
    run_name = output.parent.name.replace(" ", "_")
    gs_dir = output / "gs_init"
    loo_dir = output / "gs_loo"
    exp_name = f"controlnet_finetune/{run_name}"
    gs_dir.parent.mkdir(parents=True, exist_ok=True)
    loo_dir.parent.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    common = [
        "-s", data,
        "-r", "1",
        "--sparse_view_num", str(n_views),
        "--sh_degree", str(sh_degree),
        "--init_pcd_name", init_name,
        "--white_background",
        "--random_background",
        "--use_dust3r",
    ]

    coarse_command = [
        sys.executable, "train_gs.py", *common,
        "-m", gs_dir,
        "--iterations", str(coarse_iterations),
        "--densify_until_iter", str(int(coarse_iterations * 0.6)),
        "--save_iterations", str(coarse_iterations),
        "--test_iterations", str(coarse_iterations),
        "--quiet",
    ]
    _run(coarse_command, root, env, logs / "01_coarse.log")
    coarse_ply = _latest_iteration_ply(gs_dir)
    result = {
        "mode": mode,
        "dataset": str(data),
        "coarse_model_dir": str(gs_dir),
        "coarse_ply": str(coarse_ply),
        "initialization": init_name,
        "views": n_views,
    }
    if mode == "coarse":
        output.mkdir(parents=True, exist_ok=True)
        final_coarse = output / "coarse.ply"
        shutil.copy2(coarse_ply, final_coarse)
        result["output_ply"] = str(final_coarse)
        (output / "gaussianobject_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    if not (gs_dir / "refined_cams.json").is_file():
        raise FileNotFoundError("GaussianObject coarse stage did not produce refined_cams.json")

    loo_common = [
        "-s", data,
        "-r", "1",
        "--sparse_view_num", str(n_views),
        "--sh_degree", str(sh_degree),
        "--init_pcd_name", init_name,
        "--white_background",
        "--random_background",
        "--use_dust3r",
        "--iterations", str(loo_iterations),
        "--densify_until_iter", str(int(loo_iterations * 0.6)),
        "--save_iterations", str(loo_iterations),
        "--test_iterations", str(loo_iterations),
    ]
    stage1 = [
        sys.executable, "leave_one_out_stage1.py", *loo_common,
        "-m", loo_dir,
        "--checkpoint_iterations", "6000",
        "--quiet",
    ]
    _run(stage1, root, env, logs / "02_leave_one_out_stage1.log")

    stage2 = [
        sys.executable, "leave_one_out_stage2.py", *loo_common,
        "-m", loo_dir,
        "--quiet",
    ]
    _run(stage2, root, env, logs / "03_leave_one_out_stage2.log")

    model_yaml = root / "models" / "control_v11f1e_sd15_tile.yaml"
    sd_checkpoint = root / "models" / "v1-5-pruned.ckpt"
    control_checkpoint = root / "models" / "control_v11f1e_sd15_tile.pth"
    for required_path in (model_yaml, sd_checkpoint, control_checkpoint):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Missing GaussianObject pretrained model asset: {required_path}. "
                "Follow the upstream model download instructions."
            )

    lora_command = [
        sys.executable, "train_lora.py",
        "--sh_degree", str(sh_degree),
        "--image_size", "512",
        "--resolution", "1",
        "--sparse_num", str(n_views),
        "--gs_dir", gs_dir,
        "--data_dir", data,
        "--loo_dir", loo_dir,
        "--prompt", prompt_token,
        "--exp_name", exp_name,
        "--bg_white",
        "--sd_locked",
        "--train_lora",
        "--use_prompt_list",
        "--add_diffusion_lora",
        "--add_control_lora",
        "--add_clip_lora",
        "--use_dust3r",
        "--max_steps", str(lora_steps),
    ]
    _run(lora_command, root, env, logs / "04_train_lora.log")
    lora_dir = root / "output" / exp_name
    lora_candidates = sorted(lora_dir.glob("lora-step=*.ckpt"))
    if not lora_candidates:
        raise FileNotFoundError(f"LoRA training produced no checkpoint in {lora_dir}")
    lora_checkpoint = max(
        lora_candidates,
        key=lambda p: int(p.stem.split("=")[-1]) if p.stem.split("=")[-1].isdigit() else -1,
    )

    repair_tag = run_name
    repair_command = [
        sys.executable, "train_repair.py",
        "--config", "configs/gaussian-object-colmap-free.yaml",
        "--train",
        "--gpu", "0",
        f"tag={repair_tag}",
        f"system.init_dreamer={gs_dir}",
        f"system.exp_name={root / 'output' / exp_name}",
        f"system.lora_name={lora_checkpoint.name}",
        "system.refresh_size=8",
        f"data.data_dir={data}",
        "data.resolution=1",
        f"data.sparse_num={n_views}",
        f"data.prompt=a photo of a {prompt_token}",
        f"data.json_path={gs_dir / 'refined_cams.json'}",
        "data.refresh_size=8",
        f"system.sh_degree={sh_degree}",
        f"system.gaussian_opt_params.iterations={repair_steps}",
        f"trainer.max_steps={repair_steps}",
    ]
    _run(repair_command, root, env, logs / "05_train_repair.log")

    repaired = root / "output" / "gaussian_object" / repair_tag / "save" / "last.ply"
    if not repaired.is_file():
        candidates = list((root / "output" / "gaussian_object" / repair_tag).rglob("last.ply"))
        if not candidates:
            raise FileNotFoundError("GaussianObject repair stage did not produce save/last.ply")
        repaired = candidates[0]
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "final.ply"
    shutil.copy2(repaired, final_path)
    result.update({
        "loo_dir": str(loo_dir),
        "lora_checkpoint": str(lora_checkpoint),
        "gaussianobject_output": str(repaired),
        "output_ply": str(final_path),
        "repair_steps": repair_steps,
    })
    (output / "gaussianobject_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
