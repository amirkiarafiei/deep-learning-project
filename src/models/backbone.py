"""Shared Siamese backbone: ConvNeXt-Tiny via timm."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class SiameseBackbone(nn.Module):
    """Wrap a timm model in ``features_only`` mode so we get spatial features.

    For ``convnext_tiny.fb_in1k`` at 224 input, ``out_indices=(3,)`` yields a
    ``(B, 768, 7, 7)`` feature map from the last stage. We return that
    directly (no pooling here — the fusion module handles pooling).
    """

    def __init__(self, model_name: str = "convnext_tiny.fb_in1k", pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,),
        )
        info = self.encoder.feature_info.channels()
        self.feature_channels: int = info[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return feats[-1]
