from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    width: int = 32
    depth: int = 32
    activation: str = "relu"
    input_dim: int = 784
    output_dim: int = 10


@dataclass(frozen=True)
class MethodConfig:
    name: str = "pcalm"
    budget: int = 64
    alpha: float = 1.0
    rho: float = 1.0
    state_lr: float = 0.25
    inner_steps: int = 1
    weight_credit_timing: str = "pre_dual_energy"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float | None = None
    eta0: float = 1e-3
    gamma0: float = 1.0
    train_subset: int = 1024
    test_subset: int = 512
    drop_last: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "synthetic"
    output_dir: str = "results/run"
    model: ModelConfig = field(default_factory=ModelConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def config_from_dict(values: dict[str, Any]) -> ExperimentConfig:
    values = dict(values)
    model = ModelConfig(**(values.pop("model", {}) or {}))
    method = MethodConfig(**(values.pop("method", {}) or {}))
    training = TrainingConfig(**(values.pop("training", {}) or {}))
    return ExperimentConfig(**values, model=model, method=method, training=training)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    if not isinstance(values, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return config_from_dict(values)


def with_updates(config: ExperimentConfig, **kwargs: Any) -> ExperimentConfig:
    model_updates = kwargs.pop("model", {})
    method_updates = kwargs.pop("method", {})
    training_updates = kwargs.pop("training", {})
    model = replace(config.model, **{k: v for k, v in model_updates.items() if v is not None})
    method = replace(config.method, **{k: v for k, v in method_updates.items() if v is not None})
    training = replace(config.training, **{k: v for k, v in training_updates.items() if v is not None})
    top = {k: v for k, v in kwargs.items() if v is not None}
    return replace(config, **top, model=model, method=method, training=training)
