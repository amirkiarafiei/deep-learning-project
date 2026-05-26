"""Test-Time Augmentation for the Siamese A/B classifier.

Averages sigmoid scores across the 4 dihedral views (identity, hflip, vflip,
hflip+vflip). Both A and B images receive the SAME transformation in every
view so the bi-temporal correspondence is preserved.

Usage:

    tta = TTAEnsemble(views="4")
    logits_dict = tta(model, image_a, image_b)   # averaged sigmoid then logit?
                                                  # actually averages PROBS, not logits

Note: we average probabilities (sigmoid space), not raw logits, because the
mapping is non-linear and averaging logits gives a different answer.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torchvision.transforms.functional as TF


def _apply_view(img: torch.Tensor, view: str) -> torch.Tensor:
    """Apply a dihedral view to an image tensor."""
    if view == "identity":
        return img
    if view == "hflip":
        return TF.hflip(img)
    if view == "vflip":
        return TF.vflip(img)
    if view == "hflip_vflip":
        return TF.vflip(TF.hflip(img))
    raise ValueError(f"Unknown view: {view}")


def _supported_views(views: str) -> List[str]:
    if views == "1":
        return ["identity"]
    if views == "4":
        return ["identity", "hflip", "vflip", "hflip_vflip"]
    raise ValueError(f"views must be '1' or '4', got {views}")


@torch.no_grad()
def tta_predict(model, image_a: torch.Tensor, image_b: torch.Tensor,
                views: str = "4") -> Dict[str, torch.Tensor]:
    """Run the model under multiple dihedral views, average sigmoid output.

    Returns a dict family → probabilities (NOT logits) of shape (N, C).
    For binary changeflag the returned tensor has shape (N,).
    """
    model.eval()
    views_list = _supported_views(views)
    accum: Dict[str, torch.Tensor] = {}
    for v in views_list:
        a = _apply_view(image_a, v)
        b = _apply_view(image_b, v)
        out = model(a, b)
        for k, logits in out.items():
            probs = torch.sigmoid(logits)
            if k not in accum:
                accum[k] = probs
            else:
                accum[k] = accum[k] + probs
    return {k: v / len(views_list) for k, v in accum.items()}
