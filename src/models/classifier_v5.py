"""Track 3 v5 model — DINOv2 + cross-attention fusion + ML-Decoder heads.

Architecture summary (see ``docs/track3.md`` for the rationale):

    RGB_A ─┐
           ├─► DINOv2-Base (frozen)
    RGB_B ─┘    -> patch tokens A (B, N, 768)
                -> patch tokens B (B, N, 768)
                   |
                   v
    CROSS-ATTENTION FUSION
        block 1: Q from A, K/V from B   -> A_cond
        block 2: Q from B, K/V from A   -> B_cond
        diff token: pool(A_cond - B_cond)
        2 self-attn encoder layers over [A_cond; B_cond; diff]
                   |
                   v
    Joint token sequence (B, 2N+1, 768)
        |---> ML-Decoder per family (Object/Event/Attribute) -> per-class logits
        |---> mean-pool + Linear(768, 1) -> changeflag logit
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from .cross_attn_fusion import CrossAttentionFusion
from .dinov2_backbone import DINOv2Backbone
from .ml_decoder import MLDecoder

NUM_CLASSES: Dict[str, int] = {"object": 12, "event": 12, "attribute": 24}


class ChangeClassifierV5(nn.Module):
    """v5 model: foundation backbone + cross-attention fusion + ML-Decoder heads.

    Compatible with the trainer's ``head_parameters`` / ``backbone_parameters``
    convention. The DINOv2 backbone is frozen by default, so its parameters
    are still returned from ``backbone_parameters()`` but they have
    ``requires_grad=False`` — the AdamW optimizer will skip them.
    """

    def __init__(
        self,
        families: Iterable[str],
        include_changeflag: bool = True,
        pretrained_backbone: bool = True,
        dinov2_model: str = "dinov2_vitb14",
        freeze_backbone: bool = True,
        fusion_n_heads: int = 8,
        fusion_n_encoder_layers: int = 2,
        fusion_dropout: float = 0.0,
        decoder_n_heads: int = 8,
        decoder_n_layers: int = 1,
        decoder_dropout: float = 0.0,
    ):
        super().__init__()
        self.families: List[str] = list(families)
        for fam in self.families:
            if fam not in NUM_CLASSES:
                raise ValueError(f"Unknown family: {fam}")
        self.include_changeflag = include_changeflag

        self.backbone = DINOv2Backbone(
            model_name=dinov2_model,
            pretrained=pretrained_backbone,
            freeze=freeze_backbone,
        )
        d = self.backbone.embed_dim

        self.fusion = CrossAttentionFusion(
            d_model=d,
            n_heads=fusion_n_heads,
            n_encoder_layers=fusion_n_encoder_layers,
            dropout=fusion_dropout,
        )

        self.heads = nn.ModuleDict(
            {
                fam: MLDecoder(
                    num_classes=NUM_CLASSES[fam],
                    d_model=d,
                    n_heads=decoder_n_heads,
                    n_layers=decoder_n_layers,
                    dropout=decoder_dropout,
                )
                for fam in self.families
            }
        )

        self.changeflag_head: nn.Linear | None = nn.Linear(d, 1) if include_changeflag else None

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens_a = self.backbone(image_a)
        tokens_b = self.backbone(image_b)
        fused_seq = self.fusion(tokens_a, tokens_b)  # (B, 2N+1, d)

        out: Dict[str, torch.Tensor] = {
            fam: self.heads[fam](fused_seq) for fam in self.families
        }
        if self.changeflag_head is not None:
            pooled = fused_seq.mean(dim=1)  # (B, d)
            out["changeflag"] = self.changeflag_head(pooled).squeeze(-1)
        return out

    def head_parameters(self):
        """Trainable params (fusion + ML-Decoder heads + changeflag).

        Backbone is excluded here. With ``freeze_backbone=True`` (default),
        the backbone has ``requires_grad=False`` so AdamW would skip it
        anyway, but we keep the partition explicit to match the trainer
        convention used by v1/v4.
        """
        params = list(self.fusion.parameters())
        for head in self.heads.values():
            params.extend(head.parameters())
        if self.changeflag_head is not None:
            params.extend(self.changeflag_head.parameters())
        return params

    def backbone_parameters(self):
        """Returns *trainable* backbone params (empty when frozen)."""
        return [p for p in self.backbone.parameters() if p.requires_grad]
