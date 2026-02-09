"""Integration smoke test for end-to-end pipeline runner."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from src.run import execute_pipeline


def _small_full_pipeline_config() -> dict[str, object]:
    run_token = uuid.uuid4().hex[:8]
    data_root = Path("data/PRING").resolve()
    split_dir = data_root / "species_processed_data" / "human" / "BFS"
    return {
        "run_config": {
            "mode": "full_pipeline",
            "seed": 7,
            "pretrain_run_id": f"itest_pre_{run_token}",
            "finetune_run_id": f"itest_ft_{run_token}",
            "eval_run_id": f"itest_eval_{run_token}",
            "load_checkpoint_path": None,
            "save_best_only": True,
        },
        "device_config": {
            "device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
        },
        "data_config": {
            "benchmark": {
                "name": "PRING",
                "root_dir": str(data_root),
                "species": "human",
                "split_strategy": "BFS",
                "processed_dir": str(split_dir),
            },
            "embeddings": {"source": "esm3", "cache_dir": "data/embeddings/esm3"},
            "max_sequence_length": 8,
            "dataloader": {
                "train_dataset": str(split_dir / "human_train_ppi.txt"),
                "valid_dataset": str(split_dir / "human_val_ppi.txt"),
                "test_dataset": str(split_dir / "human_test_ppi.txt"),
                "max_samples_per_split": 8,
                "num_workers": 0,
                "pin_memory": False,
                "drop_last": False,
                "sampling": {
                    "strategy": "ohem",
                    "warmup_epochs": 1,
                    "keep_ratio": 0.5,
                    "min_keep": 2,
                },
            },
        },
        "model_config": {
            "model": "v3",
            "input_dim": 8,
            "d_model": 8,
            "encoder_layers": 1,
            "cross_attn_layers": 1,
            "n_heads": 2,
            "mlp_head": {
                "hidden_dims": [8, 4],
                "dropout": 0.1,
                "activation": "gelu",
                "norm": "layernorm",
            },
            "regularization": {
                "dropout": 0.1,
                "token_dropout": 0.0,
                "cross_attention_dropout": 0.1,
                "stochastic_depth": 0.0,
            },
        },
        "training_config": {
            "epochs": 1,
            "batch_size": 4,
            "early_stopping_patience": 1,
            "monitor_metric": "auprc",
            "optimizer": {
                "type": "adamw",
                "lr": 1e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
            "scheduler": {"type": "none"},
            "strategy": {"type": "none"},
        },
        "evaluate": {"metrics": ["accuracy", "f1", "auprc", "auroc"]},
    }


def test_execute_pipeline_full_smoke() -> None:
    config = _small_full_pipeline_config()
    run_cfg = config["run_config"]
    assert isinstance(run_cfg, dict)
    pretrain_id = str(run_cfg["pretrain_run_id"])
    finetune_id = str(run_cfg["finetune_run_id"])
    eval_id = str(run_cfg["eval_run_id"])

    execute_pipeline(config)

    pretrain_model = Path("models/v3/pretrain") / pretrain_id / "best_model.pth"
    finetune_model = Path("models/v3/finetune") / finetune_id / "best_model.pth"
    eval_csv = Path("logs/v3/evaluate") / eval_id / "evaluate.csv"
    assert pretrain_model.exists()
    assert finetune_model.exists()
    assert eval_csv.exists()

    shutil.rmtree(Path("models/v3/pretrain") / pretrain_id, ignore_errors=True)
    shutil.rmtree(Path("models/v3/finetune") / finetune_id, ignore_errors=True)
    shutil.rmtree(Path("logs/v3/pretrain") / pretrain_id, ignore_errors=True)
    shutil.rmtree(Path("logs/v3/finetune") / finetune_id, ignore_errors=True)
    shutil.rmtree(Path("logs/v3/evaluate") / eval_id, ignore_errors=True)
