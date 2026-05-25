"""ChangeClassifier — full model assembling backbone + fusion + heads."""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from .backbone import SiameseBackbone
from .fusion import ConcatDiffFusion

NUM_CLASSES: Dict[str, int] = {"object": 12, "event": 12, "attribute": 24}


class ChangeClassifier(nn.Module):
    """Siamese-encoder change classifier with per-family heads + changeflag head.

    Phase 1 = instantiate one family head; Phase 2 = instantiate all three.
    ``changeflag`` head is always present (cheap auxiliary signal).
    Sigmoid is applied at loss/inference time, never inside the model.
    """

    def __init__(
        self,
        families: Iterable[str],
        include_changeflag: bool = True,
        pretrained_backbone: bool = True,
        backbone_name: str = "convnext_tiny.fb_in1k",
    ):
        super().__init__()
        self.families: List[str] = list(families)
        for fam in self.families:
            if fam not in NUM_CLASSES:
                raise ValueError(f"Unknown family: {fam}")
        self.include_changeflag = include_changeflag

        self.backbone = SiameseBackbone(backbone_name, pretrained=pretrained_backbone)
        ch = self.backbone.feature_channels
        self.fusion = ConcatDiffFusion(channels=ch)

        self.heads = nn.ModuleDict(
            {fam: nn.Linear(ch, NUM_CLASSES[fam]) for fam in self.families}
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
