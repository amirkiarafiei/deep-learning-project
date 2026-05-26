"""LDAM (Label-Distribution-Aware Margin) loss adapted for multi-label sigmoid heads.

The single-label LDAM of Cao et al. (NeurIPS 2019) computes per-class margins
``m_c = C / n_c^{1/4}`` where ``n_c`` is positive-sample count, and the
margins are subtracted from the *correct* class's logit before softmax. We
adapt the same margin concept to **sigmoid multi-label** by subtracting the
margin from the logits of *positive* labels in each sample before BCE:

    z̃_{ic} = z_{ic} − m_c · y_{ic}            (margin only on positives)
    L = BCEWithLogitsLoss(z̃, y; pos_weight)

Combined with DRW (Deferred Re-Weighting): the per-class ``pos_weight`` is
identity (all ones) during Phase A and switches to ``neg/pos`` weighting at
``drw_epoch`` to avoid early-training instability with hard margins on
ultra-rare classes.

References
----------
- Cao et al. 2019, "Learning Imbalanced Datasets with Label-Distribution-Aware
  Margin Loss", NeurIPS.
- Sulake 2026, "Loss Design and Architecture Selection for Long-Tailed
  Multi-Label CXR Classification" (LDAM-DRW SOTA recipe for multi-label).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMMultiLabelLoss(nn.Module):
    """LDAM loss for multi-label sigmoid heads, with optional DRW reweighting.

    Args:
        class_pos_counts: 1-D tensor of length C, positive-sample count per class.
        max_m: cap on the largest margin (typical: 0.3–0.5; larger = stronger
            pull on rare classes but more training instability).
        s: logit scale, applied to the (margin-shifted) logits before BCE
            (typical: 30 — boosts the gradient signal).
        pos_weight: per-class pos_weight tensor or None. When None, defaults
            to all-ones (DRW Phase A). Call ``set_pos_weight(...)`` to switch
            to inverse-frequency weighting at the DRW transition epoch.
    """

    def __init__(
        self,
        class_pos_counts: torch.Tensor,
        max_m: float = 0.5,
        s: float = 30.0,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if class_pos_counts.ndim != 1:
            raise ValueError(f"class_pos_counts must be 1-D, got {tuple(class_pos_counts.shape)}")
        # m_c = C / n_c^{1/4}, normalized so that max(m_c) = max_m
        m_list = 1.0 / torch.sqrt(torch.sqrt(class_pos_counts.clamp(min=1.0).float()))
        m_list = m_list * (max_m / (m_list.max().item() + 1e-12))
        self.register_buffer("margins", m_list)  # shape (C,)
        self.s = s

        pw = pos_weight if pos_weight is not None else torch.ones_like(m_list)
        self.register_buffer("pos_weight", pw.float())

    def set_pos_weight(self, pos_weight: torch.Tensor) -> None:
        """Switch to inverse-frequency reweighting (DRW Phase B)."""
        self.pos_weight = pos_weight.to(self.pos_weight.device).float()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Subtract margin from logits of positive labels only.
        # Equivalent to: z̃_c = z_c − m_c · y_c
        targets_f = targets.float()
        shifted = logits - self.margins.view(1, -1) * targets_f
        shifted = self.s * shifted
        # Adjust pos_weight for scale: BCE with logit scale s effectively
        # scales the loss; pos_weight stays as-is per-class.
        return F.binary_cross_entropy_with_logits(
            shifted, targets_f, pos_weight=self.pos_weight
        )


def make_ldam_pos_weight(class_pos_counts: torch.Tensor, total_samples: int,
                        clamp_min: float = 1.0, clamp_max: float = 10.0) -> torch.Tensor:
    """Inverse-frequency pos_weight: neg/pos clamped to [clamp_min, clamp_max]."""
    pos = class_pos_counts.clamp(min=1.0)
    neg = (float(total_samples) - class_pos_counts).clamp(min=0.0)
    pw = (neg / pos).clamp(min=clamp_min, max=clamp_max)
    return pw.float()
