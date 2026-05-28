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
    # Phase 2 (v2): dropout applied to family heads only (not changeflag).
    # 0.0 reproduces Phase 1 (v1.0.0) exactly.
    head_dropout: float = 0.0
    # Architecture family. "siamese_convnext" (default) reproduces v1-v4;
    # "dinov2_crossattn_mldecoder" selects the Track 3 v5 stack.
    arch_kind: str = "siamese_convnext"
    # Track 3 v5 hyperparameters (unused when arch_kind == "siamese_convnext").
    dinov2_model: str = "dinov2_vitb14"
    freeze_backbone: bool = True
    fusion_n_heads: int = 8
    fusion_n_encoder_layers: int = 2
    fusion_dropout: float = 0.0
    decoder_n_heads: int = 8
    decoder_n_layers: int = 1
    decoder_dropout: float = 0.0


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
    # Track 2 v3 (A3) additions: LDAM-DRW + cRT.
    loss_type: str = "bce"             # "bce" (default) | "ldam"
    ldam_max_m: float = 0.5
    ldam_s: float = 30.0
    ldam_drw_epoch: int = 10           # switch to inverse-freq pos_weight from this epoch on
    crt_epochs: int = 0                # 0 = no cRT phase
    crt_lr_head: float = 1.0e-5        # head-only LR during cRT (typically lower than Phase A)


@dataclass
class EvaluationConfig:
    threshold: float = 0.5
    # Track 2 v3 additions: post-hoc adjustments at inference.
    logit_adjust_tau: float = 0.0      # 0 = no adjustment; 1.0 = full Bayes-optimal
    tta_views: str = "1"               # "1" = none, "4" = dihedral


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
