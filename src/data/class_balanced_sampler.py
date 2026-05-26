"""Class-aware (re-)sampling for multi-label cRT.

For each sample, weight = sum_{c in positive_labels} 1 / pos_count_c. This
oversamples rare-class examples in proportion to their inverse frequency,
exactly the protocol used in the cRT stage of long-tail learning.

If a sample has no positive labels (all-zero target — the no-change case), it
gets a small constant weight so the loss still sees some no-change examples.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch.utils.data import WeightedRandomSampler


def class_aware_sample_weights(
    targets_per_family: dict,
    families: Sequence[str],
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Compute per-sample sampling weights for class-aware resampling.

    Args:
        targets_per_family: dict ``fam -> Tensor(N, C_fam)`` of multi-hot
            targets across all training samples.
        families: which families to include (e.g. multi-task = all three).
        epsilon: baseline weight given to samples with no positive labels.

    Returns 1-D tensor of length N (sum of weights ≈ ?, doesn't matter — the
    sampler normalizes internally).
    """
    n = next(iter(targets_per_family.values())).shape[0]
    weights = torch.full((n,), epsilon, dtype=torch.float64)

    for fam in families:
        tgt = targets_per_family[fam].float()  # (N, C_fam)
        pos_count = tgt.sum(dim=0).clamp(min=1.0)  # (C_fam,)
        inv = 1.0 / pos_count  # (C_fam,)
        # per-sample contribution = sum_c inv[c] * y[c]
        sample_contrib = (tgt * inv.view(1, -1)).sum(dim=1)  # (N,)
        weights = weights + sample_contrib

    return weights


def build_class_aware_sampler(
    targets_per_family: dict,
    families: Sequence[str],
    num_samples: int = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Convenience wrapper that returns a torch ``WeightedRandomSampler``."""
    weights = class_aware_sample_weights(targets_per_family, families)
    n = num_samples if num_samples is not None else len(weights)
    return WeightedRandomSampler(weights, num_samples=n, replacement=replacement)
