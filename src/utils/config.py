"""Dataclass-based YAML config loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class DataConfig:
    json_path: str = "dataset/dataset.json"
    dataset_root: str = "dataset"
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4


@dataclass
class ModelConfig:
    families: List[str] = field(default_factory=lambda: ["object"])
    include_changeflag: bool = True
    pretrained_backbone: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 30
    lr_backbone: float = 1.0e-5
    lr_head: float = 1.0e-4
    weight_decay: float = 1.0e-4
    changeflag_weight: float = 0.5
    grad_clip: float = 1.0
    early_stop_patience: int = 10
    eta_min: float = 1.0e-6
    pos_weight_clamp_max: float = 50.0
    pos_weight_clamp_min: float = 1.0
    use_amp: bool = True
    subset_train: int | None = None
    subset_val: int | None = None


@dataclass
class EvaluationConfig:
    threshold: float = 0.5


@dataclass
class ExperimentConfig:
    run_name: str = "track1_object"
    output_dir: str = "results/track1/object"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _build_dataclass(cls, raw: dict):
    """Recursively populate a dataclass from a dict, falling back to defaults."""
    if raw is None:
        return cls()
    kwargs = {}
    for f in cls.__dataclass_fields__.values():
        if f.name not in raw:
            continue
        value = raw[f.name]
        if hasattr(f.type, "__dataclass_fields__") or (
            isinstance(f.type, type) and hasattr(f.type, "__dataclass_fields__")
        ):
            kwargs[f.name] = _build_dataclass(f.type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = ExperimentConfig(
        run_name=raw.get("run_name", "track1_object"),
        output_dir=raw.get("output_dir", "results/track1/object"),
        seed=raw.get("seed", 42),
        data=_build_dataclass(DataConfig, raw.get("data")),
        model=_build_dataclass(ModelConfig, raw.get("model")),
        training=_build_dataclass(TrainingConfig, raw.get("training")),
        evaluation=_build_dataclass(EvaluationConfig, raw.get("evaluation")),
    )

    valid_families = {"object", "event", "attribute"}
    bad = set(cfg.model.families) - valid_families
    if bad:
        raise ValueError(f"Unknown families in config: {bad}")
    if not cfg.model.families:
        raise ValueError("model.families must list at least one family")
    return cfg
