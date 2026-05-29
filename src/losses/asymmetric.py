"""Asymmetric Loss family for multi-label long-tail classification.

Two implementations:

* :class:`AsymmetricLoss` — original ASL from Ridnik et al. 2021 (ICCV).
  Asymmetric focal-style loss with different ``gamma`` for positive and
  negative samples, and a probability margin ``m`` clamping negative
  probabilities ``p_neg = max(p - m, 0)``.

* :class:`RobustAsymmetricLoss` (RAL) — Park et al. 2023 (ICCV CVAMD).
  Polynomial-asymmetric loss with Hill-style regularization. More
  robust to noisy positive labels — directly applicable to our long-tail
  multi-label change classification task where rare-class labels are
  empirically noisy (see Track 2 v3/v4 analysis).

Reference impls:
- ASL:  https://github.com/Alibaba-MIIL/ASL
- RAL:  https://github.com/kalelpark/RALoss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (Ridnik et al., ICCV 2021).

    Args:
        gamma_neg: focal exponent for negative samples (default 4).
        gamma_pos: focal exponent for positive samples (default 0).
        clip: probability margin for negatives. ``p_neg = max(p - clip, 0)``.
        eps: log clamp.
        disable_torch_grad_focal_loss: avoid backward through focal exponent.
        pos_weight: optional per-class positive weight tensor (B,).
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Probabilities.
        p = torch.sigmoid(logits)
        p_neg = p
        if self.clip > 0:
            p_neg = (p - self.clip).clamp(min=0)

        # BCE-like terms.
        los_pos = targets * torch.log(p.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log((1 - p_neg).clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric focal.
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            # Use a context manager rather than torch.set_grad_enabled(...) so
            # we don't leak the grad state when the caller is already inside
            # @torch.no_grad() (e.g. Trainer.validate). The previous global
            # set_grad_enabled(True) at the end of this block clobbered the
            # outer no_grad and caused autograd-graph accumulation across
            # validation batches → OOM on long val loops.
            import contextlib
            ctx = torch.no_grad() if self.disable_torch_grad_focal_loss else contextlib.nullcontext()
            with ctx:
                pt0 = p * targets
                pt1 = p_neg * (1 - targets)
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        # Per-class positive weight (gives tail classes extra gradient).
        if self.pos_weight is not None:
            w = targets * self.pos_weight.to(logits.dtype) + (1 - targets)
            loss = loss * w

        return -loss.sum(dim=-1).mean()


class RobustAsymmetricLoss(nn.Module):
    """Robust Asymmetric Loss (Park et al., ICCV CVAMD 2023).

    Differences from plain ASL:
      * Replaces the focal-style ``(1-p)^gamma`` factor with a polynomial.
      * Adds a Hill-style regularization term that reweighs noisy positives
        (positive samples where the model is highly confident in negative).

    Args:
        gamma_neg: polynomial exponent for negative samples (default 4).
        gamma_pos: polynomial exponent for positive samples (default 0).
        clip: probability margin for negatives.
        lambda_hill: weight of the Hill regularization term.
        eps: log clamp.
        pos_weight: optional per-class positive weight tensor.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        lambda_hill: float = 1.5,
        eps: float = 1e-8,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.lambda_hill = lambda_hill
        self.eps = eps
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)

        # Negative-sample probability margin.
        p_neg = p
        if self.clip > 0:
            p_neg = (p - self.clip).clamp(min=0)

        # Log terms (always <= 0).
        log_p = torch.log(p.clamp(min=self.eps))
        log_1mp = torch.log((1 - p_neg).clamp(min=self.eps))

        # Polynomial-asymmetric multiplicative factors (always >= 0).
        pos_w = (1 - p).pow(self.gamma_pos)
        neg_w = p.pow(self.gamma_neg)

        # Hill regularization (multiplicative). For positives, add the Hill
        # weight ``(lambda - p)^2`` to the polynomial weight; this UP-weights
        # uncertain positives (p << lambda) and DOWN-weights confident ones.
        # Hill is a multiplier on log_p (never a separate sign-flipping
        # subtraction — that was the v6-first-run bug).
        if self.lambda_hill > 0:
            hill_w = (self.lambda_hill - p).pow(2)
            pos_w_eff = pos_w + hill_w
        else:
            pos_w_eff = pos_w

        loss_pos = targets * pos_w_eff * log_p          # <= 0
        loss_neg = (1 - targets) * neg_w * log_1mp      # <= 0
        loss = loss_pos + loss_neg                      # <= 0

        # Per-class positive weight.
        if self.pos_weight is not None:
            w = targets * self.pos_weight.to(logits.dtype) + (1 - targets)
            loss = loss * w

        return -loss.sum(dim=-1).mean()
