"""Metric helpers built on sklearn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

ThresholdSpec = Union[float, Sequence[float], torch.Tensor, np.ndarray]


@dataclass
class FamilyMetrics:
    family: str
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: List[Dict[str, float]]  # one dict per class with p/r/f/support

    def as_flat_dict(self, prefix: str = "") -> Dict[str, float]:
        return {
            f"{prefix}{self.family}_micro_p": self.micro_precision,
            f"{prefix}{self.family}_micro_r": self.micro_recall,
            f"{prefix}{self.family}_micro_f1": self.micro_f1,
            f"{prefix}{self.family}_macro_p": self.macro_precision,
            f"{prefix}{self.family}_macro_r": self.macro_recall,
            f"{prefix}{self.family}_macro_f1": self.macro_f1,
        }


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def family_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str],
    threshold: ThresholdSpec = 0.5,
) -> FamilyMetrics:
    """Compute micro/macro F1 + per-class breakdown for one multi-label family.

    ``logits`` and ``targets`` are shape ``(N, C)``. Sigmoid is applied here.

    ``threshold`` is either a scalar (applied uniformly across classes) or a
    per-class iterable / 1-D tensor of length ``C`` (Phase 2 val-tuned mode).
    """
    probs = torch.sigmoid(logits)
    if isinstance(threshold, (int, float)):
        thr_t = torch.full((probs.shape[1],), float(threshold), dtype=probs.dtype)
    else:
        thr_t = torch.as_tensor(threshold, dtype=probs.dtype).view(-1)
        if thr_t.shape[0] != probs.shape[1]:
            raise ValueError(
                f"per-class threshold must have length {probs.shape[1]}, "
                f"got {thr_t.shape[0]}"
            )
    preds = (probs >= thr_t.view(1, -1)).to(torch.int32)
    y_true = _to_np(targets.to(torch.int32))
    y_pred = _to_np(preds)

    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    Mp, Mr, Mf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_per, r_per, f_per, supp_per = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_class = []
    for i, name in enumerate(class_names):
        per_class.append(
            {
                "class": name,
                "precision": float(p_per[i]),
                "recall": float(r_per[i]),
                "f1": float(f_per[i]),
                "support": int(supp_per[i]),
            }
        )

    return FamilyMetrics(
        family="",
        micro_precision=float(mp),
        micro_recall=float(mr),
        micro_f1=float(mf),
        macro_precision=float(Mp),
        macro_recall=float(Mr),
        macro_f1=float(Mf),
        per_class=per_class,
    )


def changeflag_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Binary metrics on the changeflag head."""
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).to(torch.int32)
    y_true = _to_np(targets.to(torch.int32))
    y_pred = _to_np(preds)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"changeflag_p": float(p), "changeflag_r": float(r), "changeflag_f1": float(f)}
