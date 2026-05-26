"""Multi-label Logit Adjustment for long-tailed classification (closed-form).

Subtracts ``τ · log(π_c)`` from each class logit at inference time. ``π_c``
is the marginal probability of class c being positive on the training set.
This is mathematically equivalent to a Bayes-optimal correction for the
class-frequency prior under the assumption that train and test marginals are
matched (Menon et al. 2021, "Long-tail learning via logit adjustment").

The adjustment moves the decision boundary toward producing more positive
predictions on rare classes, mirroring what `pos_weight` does at training
time — but without retraining.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable

import torch


def compute_log_priors(class_pos_counts: torch.Tensor, total_samples: int,
                       eps: float = 1e-6) -> torch.Tensor:
    """log(π_c) per class, where π_c = pos_count_c / total_samples."""
    priors = (class_pos_counts.float() / max(float(total_samples), 1.0)).clamp(min=eps)
    return torch.log(priors)


def save_log_priors(per_family_log_priors: Dict[str, torch.Tensor], path: Path) -> None:
    """Save {family: [log_pi_c]} to JSON."""
    payload = {fam: t.tolist() for fam, t in per_family_log_priors.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_log_priors(path: Path) -> Dict[str, torch.Tensor]:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {fam: torch.tensor(v, dtype=torch.float32) for fam, v in raw.items()}


def apply_logit_adjustment(logits: torch.Tensor, log_priors: torch.Tensor,
                           tau: float = 1.0) -> torch.Tensor:
    """logit_c ← logit_c − τ · log(π_c). Shapes: logits (N,C), log_priors (C,)."""
    if log_priors.shape[0] != logits.shape[1]:
        raise ValueError(
            f"log_priors length {log_priors.shape[0]} != logits classes {logits.shape[1]}"
        )
    return logits - tau * log_priors.view(1, -1).to(logits.device)
