"""Track 4 v6 model — ConvNeXt-Tiny Siamese + Mamba bi-temporal fusion + linear heads.

Architecture summary (see ``docs/track4.md`` for full rationale):

    RGB_A ─┐
           ├─► ConvNeXt-Tiny (ImageNet-1k, unfrozen, shared weights)
    RGB_B ─┘    output: (B, 768, 7, 7) per side
                |
                v
    Tokenize + interleave [A_0, B_0, A_1, B_1, ..., A_48, B_48]  (98 tokens)
                |
                v
    6 Mamba SSM blocks (linear-time selective recurrence)
                |
                v
    Mean-pool over 98 tokens -> (B, 768)
                |
                |---> Linear(768 -> 12) Object  + RAL
                |---> Linear(768 -> 12) Event   + RAL
                |---> Linear(768 -> 24) Attribute + RAL
                |---> Linear(768 -> 1) changeflag + BCE

Inductive-bias contrast with v1/v5:
* v1 = local CNN + concat-diff fusion + linear heads + BCE
* v5 = attention everywhere (DINOv2) + cross-attn fusion + ML-Decoder + BCE
* v6 = local CNN + Mamba SSM fusion + linear heads + RAL  (this file)

The full ConvNeXt backbone is *unfrozen* (unlike v5 where DINOv2 was
frozen) because v5 showed that adaptation to the aerial-bi-temporal
distribution matters more than raw representation quality. v6 keeps the
backbone trainable; only the fusion is the new piece.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from .backbone import SiameseBackbone
from .mamba_fusion import InterleavedMambaFusion

NUM_CLASSES: Dict[str, int] = {"object": 12, "event": 12, "attribute": 24}


class ChangeClassifierV6(nn.Module):
    """v6 model: ConvNeXt-Tiny + Mamba bi-temporal fusion + linear heads."""

    def __init__(
        self,
        families: Iterable[str],
        include_changeflag: bool = True,
        pretrained_backbone: bool = True,
        backbone_name: str = "convnext_tiny.fb_in1k",
        n_mamba_blocks: int = 6,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        fusion_dropout: float = 0.0,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.families: List[str] = list(families)
        for fam in self.families:
            if fam not in NUM_CLASSES:
                raise ValueError(f"Unknown family: {fam}")
        self.include_changeflag = include_changeflag
        self.head_dropout = head_dropout

        self.backbone = SiameseBackbone(backbone_name, pretrained=pretrained_backbone)
        ch = self.backbone.feature_channels  # 768 for convnext_tiny

        self.fusion = InterleavedMambaFusion(
            in_channels=ch,
            d_model=ch,
            n_blocks=n_mamba_blocks,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=fusion_dropout,
        )

        # Linear heads (optionally with dropout). RAL is applied at the loss
        # level in the trainer.
        def _make_head(in_ch: int, out_ch: int) -> nn.Module:
            if head_dropout > 0:
                return nn.Sequential(nn.Dropout(head_dropout), nn.Linear(in_ch, out_ch))
            return nn.Linear(in_ch, out_ch)

        self.heads = nn.ModuleDict(
            {fam: _make_head(ch, NUM_CLASSES[fam]) for fam in self.families}
        )
        self.changeflag_head: nn.Linear | None = (
            nn.Linear(ch, 1) if include_changeflag else None
        )

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat_a = self.backbone(image_a)
        feat_b = self.backbone(image_b)
        fused = self.fusion(feat_a, feat_b)

        out: Dict[str, torch.Tensor] = {fam: self.heads[fam](fused) for fam in self.families}
        if self.changeflag_head is not None:
            out["changeflag"] = self.changeflag_head(fused).squeeze(-1)
        return out

    def head_parameters(self):
        params = list(self.fusion.parameters())
        for head in self.heads.values():
            params.extend(head.parameters())
        if self.changeflag_head is not None:
            params.extend(self.changeflag_head.parameters())
        return params

    def backbone_parameters(self):
        return list(self.backbone.parameters())
