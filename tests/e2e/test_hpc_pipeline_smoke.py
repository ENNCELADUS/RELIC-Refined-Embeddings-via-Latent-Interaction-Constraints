"""Optional HPC-style DDP smoke tests."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import torch

from src.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "tests" / "e2e" / "artifacts" / "v3_hpc.yaml"


@pytest.mark.e2e
def test_hpc_config_artifact_is_valid() -> None:
    """Validate the HPC test config can be parsed and has required DDP flags."""
    config = load_config(CONFIG_PATH)
    run_cfg = config["run_config"]
    device_cfg = config["device_config"]
    assert isinstance(run_cfg, dict)
    assert isinstance(device_cfg, dict)
    assert run_cfg["mode"] == "train_only"
    assert device_cfg["ddp_enabled"] is True


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.slow
def test_hpc_ddp_train_only_smoke() -> None:
    """Run optional torchrun smoke test for cluster environments."""
    if os.environ.get("RELIC_RUN_HPC_E2E", "0") != "1":
        pytest.skip("Set RELIC_RUN_HPC_E2E=1 to run HPC smoke test.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("HPC smoke test requires at least 2 CUDA devices.")

    world_size = 2
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={world_size}",
        "-m",
        "src.run",
        "--config",
        str(CONFIG_PATH),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)

    expected_checkpoint = (
        REPO_ROOT / "models" / "v3" / "pretrain" / "hpc_e2e_pretrain" / "best_model.pth"
    )
    assert expected_checkpoint.exists()
