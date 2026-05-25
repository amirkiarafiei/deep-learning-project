"""Per-family BCE with pos_weight, plus Phase-2 naive-sum aggregator."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class FamilyBCELoss(nn.Module):
    """BCEWithLogitsLoss with a fixed per-class pos_weight."""

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        if pos_weight.ndim != 1:
            raise ValueError(f"pos_weight must be 1-D, got shape {tuple(pos_weight.shape)}")
        self.register_buffer("pos_weight", pos_weight)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets)


class ChangeflagBCELoss(nn.Module):
    """BCEWithLogitsLoss for the binary changeflag with a scalar pos_weight."""

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        # BCEWithLogitsLoss expects pos_weight broadcastable; scalar is fine.
        self.register_buffer("pos_weight", pos_weight.view(()))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets)


class MultiHeadLoss(nn.Module):
    """Naive-sum aggregator for Phase-2 multi-task training.

    ``total = sum(family_losses) + changeflag_weight * changeflag_loss``.
    Logs per-head losses so the trainer can monitor whether one head dominates.
    """

    def __init__(
        self,
        family_losses: Dict[str, FamilyBCELoss],
        changeflag_loss: ChangeflagBCELoss | None = None,
        changeflag_weight: float = 0.5,
    ):
        super().__init__()
        self.family_losses = nn.ModuleDict(family_losses)
        self.changeflag_loss = changeflag_loss
        self.changeflag_weight = changeflag_weight

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        per_head: Dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=next(iter(outputs.values())).device)
        for fam, loss_fn in self.family_losses.items():
            li = loss_fn(outputs[fam], targets[f"{fam}_labels"])
            per_head[fam] = li
            total = total + li
        if self.changeflag_loss is not None and "changeflag" in outputs:
            cl = self.changeflag_loss(outputs["changeflag"], targets["changeflag"])
            per_head["changeflag"] = cl
            total = total + self.changeflag_weight * cl
        per_head["total"] = total
        return per_head
