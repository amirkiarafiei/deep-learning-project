"""DINOv2 Siamese backbone (frozen) for Track 3 v5.

Loads `dinov2_vitb14` via `torch.hub` from the official facebookresearch/dinov2
repo. Backbone weights are frozen (`requires_grad_(False)` + `.eval()` mode).
Outputs patch token sequence: shape ``(B, num_patches, embed_dim)`` where for
224x224 input + patch size 14, ``num_patches = (224/14)**2 = 256`` and
``embed_dim = 768`` for the Base variant.

The ``[CLS]`` token is dropped; downstream cross-attention fusion composes its
own global representation from patch tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DINOv2Backbone(nn.Module):
    """Frozen DINOv2 Siamese backbone returning patch token sequence.

    Args:
        model_name: torch.hub model id. Default ``dinov2_vitb14`` = Base / 86M params.
        pretrained: load LVD-142M SSL pretrained weights.
        freeze: if True (default), backbone weights are frozen and in eval mode.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        pretrained: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        self.encoder = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=pretrained,
            trust_repo=True,
        )
        # DINOv2's vit-base has ``embed_dim = 768``; expose as attribute so
        # callers don't have to peek inside.
        self.embed_dim: int = self.encoder.embed_dim
        self.patch_size: int = self.encoder.patch_size
        self.freeze = freeze
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        # When frozen we want eval mode regardless of the outer .train()/.eval()
        # call so BatchNorm-style stats (DINOv2 uses only LayerNorm so it's
        # mostly cosmetic) and dropout remain in inference mode.
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch tokens, shape ``(B, num_patches, embed_dim)``."""
        # ``forward_features`` returns a dict with keys including
        # ``x_norm_patchtokens`` and ``x_norm_clstoken``.
        with torch.set_grad_enabled(not self.freeze):
            out = self.encoder.forward_features(x)
        return out["x_norm_patchtokens"]
