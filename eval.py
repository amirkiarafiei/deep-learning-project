"""Thin CLI: evaluate a checkpoint on val or test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ChangeDataset
from src.data.label_encoder import build_encoders
from src.data.transforms import build_transforms
from src.models.classifier import ChangeClassifier
from src.training.metrics import changeflag_metrics, family_metrics
from src.utils.config import load_config
from src.utils.logging import build_logger
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Track 1 evaluator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt to load")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)

    output_dir = Path(cfg.output_dir)
    logger = build_logger(f"{cfg.run_name}_eval", output_dir / "logs" / f"eval_{args.split}.txt")
    logger.info(f"Loading checkpoint: {args.ckpt}")
    logger.info(f"Eval split: {args.split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = build_encoders(("object", "event", "attribute"))

    transform = build_transforms(cfg.data.image_size, train=False)
    ds = ChangeDataset(
        cfg.data.json_path, cfg.data.dataset_root, args.split, transform, encoders=encoders
    )
    loader = DataLoader(
        ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = ChangeClassifier(
        families=cfg.model.families,
        include_changeflag=cfg.model.include_changeflag,
        pretrained_backbone=False,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    all_logits = {fam: [] for fam in cfg.model.families}
    all_targets = {fam: [] for fam in cfg.model.families}
    if cfg.model.include_changeflag:
        all_logits["changeflag"] = []
        all_targets["changeflag"] = []
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

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary = {"split": args.split, "ckpt": str(args.ckpt), "checkpoint_epoch": ckpt.get("epoch")}
    for fam in cfg.model.families:
        fm = family_metrics(
            cat_logits[fam], cat_targets[fam],
            class_names=encoders[fam].classes,
            threshold=cfg.evaluation.threshold,
        )
        fm.family = fam
        summary[fam] = {
            "micro_p": fm.micro_precision, "micro_r": fm.micro_recall, "micro_f1": fm.micro_f1,
            "macro_p": fm.macro_precision, "macro_r": fm.macro_recall, "macro_f1": fm.macro_f1,
        }
        logger.info(
            f"{fam}: micro_f1={fm.micro_f1:.4f}  macro_f1={fm.macro_f1:.4f}  "
            f"micro_p={fm.micro_precision:.4f}  micro_r={fm.micro_recall:.4f}"
        )
        per_class_path = metrics_dir / f"per_class_{fam}_{args.split}.csv"
        with per_class_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["class", "precision", "recall", "f1", "support"])
            for row in fm.per_class:
                w.writerow([row["class"], row["precision"], row["recall"], row["f1"], row["support"]])
        logger.info(f"  per-class metrics → {per_class_path}")

    if cfg.model.include_changeflag:
        cm = changeflag_metrics(cat_logits["changeflag"], cat_targets["changeflag"])
        summary["changeflag"] = cm
        logger.info(f"changeflag: f1={cm['changeflag_f1']:.4f}  p={cm['changeflag_p']:.4f}  r={cm['changeflag_r']:.4f}")

    summary_path = metrics_dir / f"eval_{args.split}.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Summary → {summary_path}")

    pred_path = metrics_dir / f"predictions_{args.split}.pt"
    torch.save(
        {"logits": cat_logits, "targets": cat_targets, "sample_ids": sample_ids},
        pred_path,
    )
    logger.info(f"Full prediction matrix → {pred_path}")


if __name__ == "__main__":
    main()
