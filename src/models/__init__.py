"""Model factory for the project.

Two architecture families:

* ``arch_kind="siamese_convnext"`` (default) — ConvNeXt-Tiny Siamese +
  concat-diff fusion + linear heads. Used by v1, v2, v3, v4.

* ``arch_kind="dinov2_crossattn_mldecoder"`` — DINOv2-Base frozen +
  cross-attention bi-temporal fusion + ML-Decoder heads. Used by Track 3 v5.

The factory dispatches based on ``cfg.model.arch_kind``, falling back to
the existing ``ChangeClassifier`` when not specified for backward
compatibility with v1/v2/v3/v4 configs.
"""

from __future__ import annotations

from .classifier import ChangeClassifier
from .classifier_v5 import ChangeClassifierV5


def build_model(cfg):
    """Instantiate the model selected by ``cfg.model.arch_kind``.

    Args:
        cfg: an :class:`ExperimentConfig` (see ``src.utils.config``).
    """
    arch = getattr(cfg.model, "arch_kind", "siamese_convnext")

    if arch == "siamese_convnext":
        return ChangeClassifier(
            families=cfg.model.families,
            include_changeflag=cfg.model.include_changeflag,
            pretrained_backbone=cfg.model.pretrained_backbone,
            head_dropout=cfg.model.head_dropout,
        )

    if arch == "dinov2_crossattn_mldecoder":
        return ChangeClassifierV5(
            families=cfg.model.families,
            include_changeflag=cfg.model.include_changeflag,
            pretrained_backbone=cfg.model.pretrained_backbone,
            dinov2_model=getattr(cfg.model, "dinov2_model", "dinov2_vitb14"),
            freeze_backbone=getattr(cfg.model, "freeze_backbone", True),
            fusion_n_heads=getattr(cfg.model, "fusion_n_heads", 8),
            fusion_n_encoder_layers=getattr(cfg.model, "fusion_n_encoder_layers", 2),
            fusion_dropout=getattr(cfg.model, "fusion_dropout", 0.0),
            decoder_n_heads=getattr(cfg.model, "decoder_n_heads", 8),
            decoder_n_layers=getattr(cfg.model, "decoder_n_layers", 1),
            decoder_dropout=getattr(cfg.model, "decoder_dropout", 0.0),
        )

    raise ValueError(
        f"Unknown model.arch_kind={arch!r}. Expected one of "
        f"['siamese_convnext', 'dinov2_crossattn_mldecoder']."
    )


__all__ = ["ChangeClassifier", "ChangeClassifierV5", "build_model"]
