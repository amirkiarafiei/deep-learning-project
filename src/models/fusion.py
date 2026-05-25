"""Concat + diff fusion module."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConcatDiffFusion(nn.Module):
    """Fuse (feat_A, feat_B) into a single pooled feature vector.

    Channel-wise concat of [A, B, A-B, |A-B|] → 1×1 conv to ``channels`` →
    BatchNorm → GELU → global average pool → flatten.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.project = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        diff = feat_a - feat_b
        x = torch.cat([feat_a, feat_b, diff, diff.abs()], dim=1)
        x = self.act(self.bn(self.project(x)))
        x = self.pool(x).flatten(1)
        return x
