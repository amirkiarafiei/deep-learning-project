"""Torch Dataset over ``dataset.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .label_encoder import LabelEncoder, build_encoders
from .transforms import PairedTransform, build_transforms

LABEL_KEYS = {
    "object": "object_labels",
    "event": "event_labels",
    "attribute": "attribute_labels",
}


@dataclass
class DatasetSample:
    image_A: torch.Tensor
    image_B: torch.Tensor
    object_labels: torch.Tensor
    event_labels: torch.Tensor
    attribute_labels: torch.Tensor
    changeflag: torch.Tensor  # 0-d float
    sample_id: str


class ChangeDataset(Dataset):
    """Multi-label bi-temporal change dataset.

    Returns dict per sample with keys: image_A, image_B, object_labels,
    event_labels, attribute_labels, changeflag, sample_id.

    The returned label tensors *always* cover all three families regardless
    of which families a particular run trains on. The trainer/loss decides
    which to consume.
    """

    def __init__(
        self,
        json_path: str | Path,
        dataset_root: str | Path,
        split: str,
        transform: PairedTransform,
        encoders: Optional[Dict[str, LabelEncoder]] = None,
        subset: Optional[int] = None,
    ):
        json_path = Path(json_path)
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        self.split = split
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.encoders = encoders or build_encoders(("object", "event", "attribute"))

        self.records: List[dict] = [r for r in data["images"] if r["split"] == split]
        if subset is not None:
            self.records = self.records[:subset]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        path_a = self.dataset_root / rec["rgb_A"]
        path_b = self.dataset_root / rec["rgb_B"]

        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        img_a, img_b = self.transform(img_a, img_b)

        out: Dict[str, object] = {
            "image_A": img_a,
            "image_B": img_b,
            "sample_id": rec["sample_id"],
            "changeflag": torch.tensor(float(rec["changeflag"]), dtype=torch.float32),
        }
        for fam, key in LABEL_KEYS.items():
            out[f"{fam}_labels"] = self.encoders[fam].encode(rec.get(key, []))
        return out

    def label_records(self) -> List[dict]:
        """Return the raw record list (used for computing pos_weight on train)."""
        return self.records


def compute_pos_weight(
    records: Sequence[dict],
    encoder: LabelEncoder,
    label_key: str,
    clamp_min: float = 1.0,
    clamp_max: float = 50.0,
) -> torch.Tensor:
    """pos_weight[c] = num_negatives[c] / num_positives[c], clamped.

    Ultra-rare classes (e.g. ``plant`` with 22 train samples) would otherwise
    produce huge weights that destabilize training — hence the clamp.
    """
    counts = torch.zeros(encoder.num_classes, dtype=torch.float64)
    total = len(records)
    for rec in records:
        for name in rec.get(label_key, []):
            if name == "none":
                continue
            counts[encoder.label_to_idx[name] - 1] += 1.0

    pos = counts.clamp(min=1.0)  # avoid division by zero
    neg = total - counts
    pw = (neg / pos).clamp(min=clamp_min, max=clamp_max).to(torch.float32)
    return pw


def compute_changeflag_pos_weight(
    records: Sequence[dict],
    clamp_min: float = 1.0,
    clamp_max: float = 50.0,
) -> torch.Tensor:
    pos = sum(1 for r in records if r["changeflag"] == 1)
    neg = len(records) - pos
    if pos == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    pw = float(neg) / float(pos)
    return torch.tensor(max(clamp_min, min(clamp_max, pw)), dtype=torch.float32)
