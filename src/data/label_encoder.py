"""Multi-hot label encoder driven by ``results/eda/label_vocab.json``.

``none`` is index 0 in the vocab file and is *excluded* from the output
multi-hot vector. A sample with ``["none"]`` encodes to an all-zero vector.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

VOCAB_PATH = Path("results/eda/label_vocab.json")
FAMILIES: tuple[str, ...] = ("object", "event", "attribute")


class LabelEncoder:
    """Encode/decode multi-hot label vectors for a single label family."""

    def __init__(self, family: str, vocab_path: str | Path = VOCAB_PATH):
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family}")
        vp = Path(vocab_path)
        if not vp.exists():
            raise FileNotFoundError(
                f"Label vocabulary not found at {vp.resolve()}. "
                f"This file is canonical (per AGENTS.md) and must exist at "
                f"results/eda/label_vocab.json relative to the current working "
                f"directory. If running from a different cwd, pass an absolute "
                f"vocab_path or `cd` to the repo root first."
            )
        with vp.open("r", encoding="utf-8") as fh:
            vocab = json.load(fh)[family]

        self.family: str = family
        self.label_to_idx: Dict[str, int] = vocab["label_to_idx"]
        self.idx_to_label: Dict[int, str] = {int(k): v for k, v in vocab["idx_to_label"].items()}

        # Output indexing drops `none` (vocab idx 0): output_idx = vocab_idx - 1
        self.num_classes: int = len(self.label_to_idx) - 1
        self.classes: List[str] = [self.idx_to_label[i + 1] for i in range(self.num_classes)]

    def encode(self, labels: Sequence[str]) -> torch.Tensor:
        """Convert a list of label names into a fixed-order multi-hot tensor."""
        vec = torch.zeros(self.num_classes, dtype=torch.float32)
        for name in labels:
            if name == "none":
                continue
            if name not in self.label_to_idx:
                raise KeyError(f"{self.family!r}: unknown label {name!r}")
            vocab_idx = self.label_to_idx[name]
            vec[vocab_idx - 1] = 1.0
        return vec

    def decode(self, vec: torch.Tensor, threshold: float = 0.5) -> List[str]:
        """Convert a multi-hot or probability tensor back to label names."""
        if vec.ndim != 1 or vec.shape[0] != self.num_classes:
            raise ValueError(f"expected shape ({self.num_classes},), got {tuple(vec.shape)}")
        above = (vec >= threshold).nonzero(as_tuple=False).flatten().tolist()
        return [self.classes[i] for i in above]

    def decode_with_scores(
        self, scores: torch.Tensor, threshold: float = 0.5, top_k: int = 5
    ) -> List[tuple[str, float]]:
        """Return (label, score) pairs above threshold; if none, return top-k."""
        if scores.ndim != 1 or scores.shape[0] != self.num_classes:
            raise ValueError(f"expected shape ({self.num_classes},), got {tuple(scores.shape)}")
        idx = (scores >= threshold).nonzero(as_tuple=False).flatten().tolist()
        if not idx:
            idx = torch.topk(scores, k=min(top_k, self.num_classes)).indices.tolist()
        return [(self.classes[i], float(scores[i])) for i in idx]


def build_encoders(
    families: Iterable[str], vocab_path: str | Path = VOCAB_PATH
) -> Dict[str, LabelEncoder]:
    return {fam: LabelEncoder(fam, vocab_path) for fam in families}


if __name__ == "__main__":
    enc = LabelEncoder("object")
    print(f"object num_classes = {enc.num_classes}")
    print(f"classes = {enc.classes}")

    cases = [["building", "tree"], ["none"], [], ["plant", "asphalt", "green"]]
    for c in cases:
        v = enc.encode(c)
        back = enc.decode(v)
        print(f"  encode({c}) -> nonzero at {v.nonzero().flatten().tolist()}, decode -> {back}")
        assert sorted(back) == sorted(x for x in c if x != "none"), "round-trip failed"
    print("round-trip OK")
