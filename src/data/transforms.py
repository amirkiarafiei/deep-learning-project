"""Paired transforms for the (A, B) image pair.

Both images must receive the *same* geometric transform decisions
(same flip outcome, same resize) so the bi-temporal correspondence
is preserved. The dataset already ships with offline augmentation
(see ``results/eda/eda_summary.md``); on-the-fly aug is kept minimal.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PairedTransform:
    """Apply resize → optional hflip → ToTensor → ImageNet normalize to both A and B."""

    image_size: int = 224
    hflip_prob: float = 0.5
    train: bool = True

    def __post_init__(self):
        self._to_tensor = T.ToTensor()
        self._normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, img_a: Image.Image, img_b: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        img_a = TF.resize(img_a, [self.image_size, self.image_size])
        img_b = TF.resize(img_b, [self.image_size, self.image_size])

        if self.train and self.hflip_prob > 0 and random.random() < self.hflip_prob:
            img_a = TF.hflip(img_a)
            img_b = TF.hflip(img_b)

        a = self._normalize(self._to_tensor(img_a))
        b = self._normalize(self._to_tensor(img_b))
        return a, b


def build_transforms(image_size: int, train: bool) -> PairedTransform:
    return PairedTransform(image_size=image_size, train=train, hflip_prob=0.5 if train else 0.0)
