"""Thin CLI: parse args → load config → run Trainer."""

from __future__ import annotations

import argparse

from src.training.trainer import Trainer
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Track 1 trainer")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--subset-train", type=int, default=None, help="Subset train set (smoke-test)")
    parser.add_argument("--subset-val", type=int, default=None, help="Subset val set (smoke-test)")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output_dir")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override data.dataset_root")
    parser.add_argument("--json-path", type=str, default=None, help="Override data.json_path")
    parser.add_argument("--run-name", type=str, default=None, help="Override run_name")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint (typically last.pt) to resume from")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override data.num_workers (drop to 0 on Colab if Drive I/O stalls)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.subset_train is not None:
        cfg.training.subset_train = args.subset_train
    if args.subset_val is not None:
        cfg.training.subset_val = args.subset_val
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.dataset_root is not None:
        cfg.data.dataset_root = args.dataset_root
    if args.json_path is not None:
        cfg.data.json_path = args.json_path
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    trainer = Trainer(cfg, resume_from=args.resume)
    trainer.fit()

    # Track 2 v3 hooks: save log-priors for inference-time logit adjustment,
    # and run the cRT phase if configured.
    if hasattr(trainer, "save_log_priors"):
        trainer.save_log_priors()
    if cfg.training.crt_epochs and cfg.training.crt_epochs > 0:
        trainer.fit_crt()


if __name__ == "__main__":
    main()
