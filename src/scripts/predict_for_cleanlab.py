"""Dump multi-task v1 model predictions on the TRAIN split for Cleanlab analysis.

Writes ``predictions_train.pt`` containing logits (per family) + targets +
sample_ids, in the same format eval.py produces for val/test. This file is
consumed by ``find_noisy_labels.py`` to rank training samples by suspected
label noise.

Usage:
    python -m src.scripts.predict_for_cleanlab \\
        --config configs/track1_multitask.yaml \\
        --ckpt results/track1_v1/multitask/checkpoints/best.pt \\
        --output-dir results/track2_v3/cleanup
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ChangeDataset
from src.data.label_encoder import build_encoders
from src.data.transforms import build_transforms
from src.models.classifier import ChangeClassifier
from src.utils.config import load_config
from src.utils.logging import build_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict on train split for Cleanlab")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--json-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset_root: cfg.data.dataset_root = args.dataset_root
    if args.json_path:    cfg.data.json_path = args.json_path

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger("predict_train", out_dir / "predict_train.log")
    logger.info(f"Predicting on TRAIN split with {args.ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = build_encoders(("object", "event", "attribute"))
    t = build_transforms(cfg.data.image_size, train=False)  # no augmentation
    ds = ChangeDataset(cfg.data.json_path, cfg.data.dataset_root, "train", t, encoders=encoders)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    # Auto-detect head architecture from the checkpoint's state_dict keys.
    # v1 saved plain Linear heads (`heads.object.weight`); v2 wraps them in
    # Sequential(Dropout, Linear) (`heads.object.1.weight`). Without this
    # detection, loading a v1 ckpt with a v2 config raises a state_dict mismatch.
    sd_keys = ckpt["model_state"].keys()
    ckpt_head_dropout = cfg.model.head_dropout if any(
        k.startswith("heads.") and k.endswith(".1.weight") for k in sd_keys
    ) else 0.0
    if ckpt_head_dropout != cfg.model.head_dropout:
        logger.info(
            f"head_dropout: using {ckpt_head_dropout} (auto-detected from checkpoint) "
            f"instead of {cfg.model.head_dropout} from config"
        )
    model = ChangeClassifier(
        families=cfg.model.families,
        include_changeflag=cfg.model.include_changeflag,
        pretrained_backbone=False,
        head_dropout=ckpt_head_dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}; {len(ds)} train samples")

    all_logits = {fam: [] for fam in cfg.model.families}
    all_targets = {fam: [] for fam in cfg.model.families}
    if cfg.model.include_changeflag:
        all_logits["changeflag"] = []; all_targets["changeflag"] = []
    sample_ids = []

    with torch.no_grad():
        for batch in loader:
            ia = batch["image_A"].to(device, non_blocking=True)
            ib = batch["image_B"].to(device, non_blocking=True)
            out = model(ia, ib)
            for fam in cfg.model.families:
                all_logits[fam].append(out[fam].float().cpu())
                all_targets[fam].append(batch[f"{fam}_labels"])
            if cfg.model.include_changeflag:
                all_logits["changeflag"].append(out["changeflag"].float().cpu())
                all_targets["changeflag"].append(batch["changeflag"])
            sample_ids.extend(batch["sample_id"])

    cat_logits = {k: torch.cat(v, dim=0) for k, v in all_logits.items()}
    cat_targets = {k: torch.cat(v, dim=0) for k, v in all_targets.items()}
    pred_path = out_dir / "predictions_train.pt"
    torch.save({"logits": cat_logits, "targets": cat_targets, "sample_ids": sample_ids},
              pred_path)
    logger.info(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
