"""Unit tests for model construction entrypoints."""

import pytest
import torch
from src.model.v6 import V6
from src.run import build_model
from torch import nn


def _base_config(model_name: str) -> dict[str, object]:
    return {
        "model_config": {
            "model": model_name,
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
        }
    }


def _v5_config() -> dict[str, object]:
    config = _base_config("v5")
    model_config = config["model_config"]
    assert isinstance(model_config, dict)
    model_config["pair_dim"] = 8
    model_config["cnn_dim"] = 8
    model_config["cnn_blocks"] = 1
    model_config["interaction_map"] = {
        "include_pair_features": True,
        "similarity": "cosine",
        "eps": 1.0e-8,
    }
    model_config["pooling"] = {"mode": "max_mean"}
    model_config["regularization"] = {
        "dropout": 0.1,
        "token_dropout": 0.0,
        "cross_attention_dropout": 0.1,
        "stochastic_depth": 0.0,
        "cnn_dropout": 0.0,
    }
    return config


def _v6_config() -> dict[str, object]:
    return {
        "model_config": {
            "model": "v6",
            "input_dim": 8,
            "d_model": 8,
            "cross_attn_layers": 1,
            "n_heads": 2,
            "mlp_head": {"dropout": 0.1},
            "regularization": {
                "dropout": 0.1,
                "cross_attention_dropout": 0.1,
                "projection_dropout": 0.1,
            },
            "esm3": {
                "strip_cls_eos": True,
                "embed_batch_size": 0,
                "combine_pairs": True,
            },
            "lora": {
                "last_n_layers": 1,
                "target_modules": ["ffn"],
                "r": 2,
                "alpha": 4,
                "dropout": 0.0,
            },
        }
    }


def test_build_model_v3() -> None:
    model = build_model(_base_config("v3"))
    assert model.__class__.__name__ == "V3"


def test_build_model_v4() -> None:
    model = build_model(_base_config("v4"))
    assert model.__class__.__name__ == "V4"


def test_build_model_v5() -> None:
    model = build_model(_v5_config())
    assert model.__class__.__name__ == "V5"


def test_build_model_v6(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeProtein:
        def __init__(self, sequence: str) -> None:
            self.sequence = sequence

    class _FakeBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.ffn = nn.Linear(dim, dim)

    class _FakeESM3(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dim = dim
            self.transformer = nn.Module()
            self.transformer.blocks = nn.ModuleList([_FakeBlock(dim), _FakeBlock(dim)])

        def encode(self, proteins: _FakeProtein | list[_FakeProtein]) -> torch.Tensor:
            if isinstance(proteins, list):
                max_len = max(len(protein.sequence) + 2 for protein in proteins)
                encoded = torch.zeros((len(proteins), max_len, self.dim), dtype=torch.float32)
                for index, protein in enumerate(proteins):
                    seq_len = len(protein.sequence) + 2
                    encoded[index, :seq_len] = 1.0
                return encoded

            seq_len = len(proteins.sequence) + 2
            return torch.ones((1, seq_len, self.dim), dtype=torch.float32)

        def logits(self, protein_tensor: torch.Tensor, logits_config: object) -> object:
            del logits_config
            return type("FakeOutput", (), {"embeddings": protein_tensor})()

    def _fake_load_esm3(self: V6) -> nn.Module:
        return _FakeESM3(self.input_dim)

    def _fake_load_esm_sdk(self: V6) -> tuple[type[_FakeProtein], object]:
        return _FakeProtein, object()

    monkeypatch.setattr(V6, "_load_esm3", _fake_load_esm3)
    monkeypatch.setattr(V6, "_load_esm_sdk", _fake_load_esm_sdk)

    model = build_model(_v6_config())
    assert model.__class__.__name__ == "V6"
    output = model(
        seq_a=["AAAA", "BBB"],
        seq_b=["CCCC", "DD"],
        label=torch.tensor([1.0, 0.0], dtype=torch.float32),
    )
    assert output["logits"].shape == (2, 1)
    assert output["loss"].ndim == 0


def test_build_model_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(_base_config("unknown"))
