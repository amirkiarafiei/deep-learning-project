"""Plain-PyTorch training loop for Track 1.

One Trainer covers both Phase 1 (single-family) and Phase 2 (multi-task);
the difference is the set of family heads instantiated on the model.
"""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


def _seed_worker(worker_id: int) -> None:
    """Re-seed Python/NumPy RNG per worker so PairedTransform's hflip is
    actually diverse across workers (and reproducible from cfg.seed)."""
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)

from ..data.dataset import (
    ChangeDataset,
    compute_changeflag_pos_weight,
    compute_pos_weight,
)
from ..data.label_encoder import LabelEncoder, build_encoders
from ..data.transforms import build_transforms
from ..losses.bce import ChangeflagBCELoss, FamilyBCELoss, MultiHeadLoss
from ..models.classifier import ChangeClassifier
from ..utils.config import ExperimentConfig
from ..utils.logging import build_logger
from ..utils.seed import seed_everything
from .metrics import changeflag_metrics, family_metrics

LABEL_KEYS = {"object": "object_labels", "event": "event_labels", "attribute": "attribute_labels"}


class Trainer:
    """Owns the full training/eval lifecycle for one experiment."""

    def __init__(self, cfg: ExperimentConfig, resume_from: Optional[Path] = None):
        self.cfg = cfg
        seed_everything(cfg.seed)
        self._resume_from: Optional[Path] = Path(resume_from) if resume_from else None
        self.start_epoch: int = 1

        self.output_dir = Path(cfg.output_dir)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(parents=True, exist_ok=True)

        self.logger = build_logger(
            cfg.run_name, self.output_dir / "logs" / "train.txt"
        )
        self.logger.info(f"Run: {cfg.run_name}")
        self.logger.info(f"Output dir: {self.output_dir}")
        self.logger.info(f"Config: {json.dumps(_dataclass_to_dict(cfg), indent=2)}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device: {self.device}")
        if self.device.type == "cuda":
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        # Encoders for all three families so the Dataset always returns the full
        # label tensor dict; loss/model only consume the active families.
        self.encoders: Dict[str, LabelEncoder] = build_encoders(("object", "event", "attribute"))

        self.train_loader, self.val_loader, train_records = self._build_loaders()

        self.model = ChangeClassifier(
            families=cfg.model.families,
            include_changeflag=cfg.model.include_changeflag,
            pretrained_backbone=cfg.model.pretrained_backbone,
            head_dropout=cfg.model.head_dropout,
        ).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model parameters: {n_params/1e6:.2f}M")

        self.loss_fn = self._build_loss(train_records)

        self.optimizer = AdamW(
            [
                {"params": self.model.backbone_parameters(), "lr": cfg.training.lr_backbone},
                {"params": self.model.head_parameters(),     "lr": cfg.training.lr_head},
            ],
            weight_decay=cfg.training.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.eta_min
        )

        self.use_amp = cfg.training.use_amp and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp else torch.float32

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            **{f"train_{fam}_loss": [] for fam in cfg.model.families},
            **{f"val_{fam}_loss":   [] for fam in cfg.model.families},
            **{f"val_{fam}_macro_f1": [] for fam in cfg.model.families},
        }
        if cfg.model.include_changeflag:
            self.history["train_changeflag_loss"] = []
            self.history["val_changeflag_loss"] = []
            self.history["val_changeflag_f1"] = []

        self.best_val_macro_f1: float = -1.0
        self.best_epoch: int = -1
        self.epochs_since_improvement: int = 0

        if self._resume_from is not None:
            self._load_resume_checkpoint(self._resume_from)

    # -- resume ------------------------------------------------------------

    def _load_resume_checkpoint(self, ckpt_path: Path) -> None:
        """Restore model, optimizer, scheduler, history, and early-stop state.

        History is merged defensively: keys present in the current run config
        but missing from the checkpoint (e.g., new per-head loss columns added
        between runs) are zero-padded to the checkpoint's epoch count; keys in
        the checkpoint but not in the current schema are dropped.
        """
        self.logger.info(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Pre-flight: refuse to resume across head-architecture mismatches.
        # Catches the v1→v2 case (Phase 1 plain Linear heads vs Phase 2
        # Sequential(Dropout, Linear) heads) with a clear error instead of
        # cryptic state_dict missing/unexpected keys.
        ckpt_model_cfg = ckpt.get("config", {}).get("model", {}) or {}
        ckpt_drop = ckpt_model_cfg.get("head_dropout", 0.0)
        ckpt_families = ckpt_model_cfg.get("families", [])
        if ckpt_drop != self.cfg.model.head_dropout:
            raise ValueError(
                f"Resume incompatibility: checkpoint trained with "
                f"head_dropout={ckpt_drop}, current config has "
                f"head_dropout={self.cfg.model.head_dropout}. Head shapes will "
                f"not match (plain Linear vs Sequential(Dropout, Linear)). "
                f"Either start a fresh run or use a checkpoint built with the "
                f"same head_dropout. (This usually means you're trying to "
                f"resume v2 training from a v1 checkpoint.)"
            )
        if ckpt_families and list(ckpt_families) != list(self.cfg.model.families):
            raise ValueError(
                f"Resume incompatibility: checkpoint families={ckpt_families} "
                f"vs current config families={self.cfg.model.families}."
            )

        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])

        ckpt_history: Dict[str, List[float]] = ckpt.get("history", {})
        prior_epochs = max((len(v) for v in ckpt_history.values()), default=0)
        merged: Dict[str, List[float]] = {}
        for key in self.history:  # current-run schema is authoritative
            if key in ckpt_history:
                merged[key] = list(ckpt_history[key])
            else:
                merged[key] = [0.0] * prior_epochs
                self.logger.info(f"  history key {key!r} missing from checkpoint — zero-padded.")
        self.history = merged

        self.best_val_macro_f1 = ckpt.get("best_val_macro_f1", -1.0)
        self.best_epoch = ckpt.get("best_epoch", -1)
        self.epochs_since_improvement = ckpt.get("epochs_since_improvement", 0)
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.logger.info(
            f"  resumed at epoch={self.start_epoch}  best_macro_f1={self.best_val_macro_f1:.4f}  "
            f"best_epoch={self.best_epoch}  patience_used={self.epochs_since_improvement}"
        )

    # -- loaders & loss ----------------------------------------------------

    def _build_loaders(self) -> Tuple[DataLoader, DataLoader, List[dict]]:
        d = self.cfg.data
        t_train = build_transforms(d.image_size, train=True)
        t_val   = build_transforms(d.image_size, train=False)

        train_ds = ChangeDataset(
            d.json_path, d.dataset_root, "train", t_train,
            encoders=self.encoders, subset=self.cfg.training.subset_train,
        )
        val_ds = ChangeDataset(
            d.json_path, d.dataset_root, "val", t_val,
            encoders=self.encoders, subset=self.cfg.training.subset_val,
        )
        self.logger.info(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

        train_loader = DataLoader(
            train_ds, batch_size=d.batch_size, shuffle=True,
            num_workers=d.num_workers, pin_memory=(self.device.type == "cuda"),
            drop_last=True, worker_init_fn=_seed_worker if d.num_workers > 0 else None,
            persistent_workers=d.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=d.batch_size, shuffle=False,
            num_workers=d.num_workers, pin_memory=(self.device.type == "cuda"),
            worker_init_fn=_seed_worker if d.num_workers > 0 else None,
            persistent_workers=d.num_workers > 0,
        )
        return train_loader, val_loader, train_ds.label_records()

    def _build_loss(self, train_records: List[dict]) -> MultiHeadLoss:
        family_losses: Dict[str, FamilyBCELoss] = {}
        clamp_min = self.cfg.training.pos_weight_clamp_min
        clamp_max = self.cfg.training.pos_weight_clamp_max
        for fam in self.cfg.model.families:
            pw = compute_pos_weight(
                train_records, self.encoders[fam], LABEL_KEYS[fam],
                clamp_min=clamp_min, clamp_max=clamp_max,
            )
            self.logger.info(f"pos_weight[{fam}] min={pw.min():.2f} max={pw.max():.2f} mean={pw.mean():.2f}")
            family_losses[fam] = FamilyBCELoss(pw.to(self.device))

        cf_loss = None
        if self.cfg.model.include_changeflag:
            cf_pw = compute_changeflag_pos_weight(
                train_records, clamp_min=clamp_min, clamp_max=clamp_max,
            )
            self.logger.info(f"pos_weight[changeflag] = {cf_pw.item():.3f}")
            cf_loss = ChangeflagBCELoss(cf_pw.to(self.device))

        return MultiHeadLoss(
            family_losses=family_losses,
            changeflag_loss=cf_loss,
            changeflag_weight=self.cfg.training.changeflag_weight,
        ).to(self.device)

    # -- core loops --------------------------------------------------------

    def _move(self, batch: Dict) -> Dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _forward_loss(self, batch: Dict) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        outputs = self.model(batch["image_A"], batch["image_B"])
        losses = self.loss_fn(outputs, batch)
        return outputs, losses

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running: Dict[str, float] = {"total": 0.0}
        for fam in self.cfg.model.families:
            running[fam] = 0.0
        if self.cfg.model.include_changeflag:
            running["changeflag"] = 0.0
        total_iters = len(self.train_loader)
        log_every = max(1, total_iters // 5)  # ~5 progress lines per epoch
        epochs = self.cfg.training.epochs
        t_epoch = time.time()
        n_batches = 0
        for batch in self.train_loader:
            batch = self._move(batch)
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    _, losses = self._forward_loss(batch)
                total = losses["total"]
                total.backward()
            else:
                _, losses = self._forward_loss(batch)
                total = losses["total"]
                total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
            self.optimizer.step()
            for k in running:
                if k in losses:
                    running[k] += float(losses[k].detach())
            n_batches += 1
            if n_batches % log_every == 0 or n_batches == total_iters:
                elapsed = time.time() - t_epoch
                eta_sec = int(elapsed / n_batches * (total_iters - n_batches))
                avg = running["total"] / n_batches
                if self.device.type == "cuda":
                    mem = torch.cuda.memory_allocated() / 1e9
                    tot = torch.cuda.get_device_properties(0).total_memory / 1e9
                    mem_str = f"mem={mem:.1f}/{tot:.0f}GB  "
                else:
                    mem_str = ""
                self.logger.info(
                    f"  [ep {epoch:02d}/{epochs} iter {n_batches:>3d}/{total_iters}] "
                    f"loss={avg:.4f}  {mem_str}eta_epoch={eta_sec}s"
                )
        self.scheduler.step()
        n = max(n_batches, 1)
        return {k: v / n for k, v in running.items()}

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        self.model.eval()
        running: Dict[str, float] = {"total": 0.0}
        for fam in self.cfg.model.families:
            running[fam] = 0.0
        if self.cfg.model.include_changeflag:
            running["changeflag"] = 0.0
        n_batches = 0
        all_logits: Dict[str, List[torch.Tensor]] = {fam: [] for fam in self.cfg.model.families}
        all_targets: Dict[str, List[torch.Tensor]] = {fam: [] for fam in self.cfg.model.families}
        if self.cfg.model.include_changeflag:
            all_logits["changeflag"] = []
            all_targets["changeflag"] = []

        for batch in self.val_loader:
            batch = self._move(batch)
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    outputs, losses = self._forward_loss(batch)
            else:
                outputs, losses = self._forward_loss(batch)
            for k in running:
                if k in losses:
                    running[k] += float(losses[k])
            n_batches += 1
            for fam in self.cfg.model.families:
                all_logits[fam].append(outputs[fam].float().cpu())
                all_targets[fam].append(batch[f"{fam}_labels"].cpu())
            if self.cfg.model.include_changeflag:
                all_logits["changeflag"].append(outputs["changeflag"].float().cpu())
                all_targets["changeflag"].append(batch["changeflag"].cpu())

        cat_logits = {k: torch.cat(v, dim=0) for k, v in all_logits.items()}
        cat_targets = {k: torch.cat(v, dim=0) for k, v in all_targets.items()}

        metrics: Dict[str, float] = {}
        macro_f1s = []
        for fam in self.cfg.model.families:
            fm = family_metrics(
                cat_logits[fam], cat_targets[fam],
                class_names=self.encoders[fam].classes,
                threshold=self.cfg.evaluation.threshold,
            )
            fm.family = fam
            metrics.update(fm.as_flat_dict("val_"))
            macro_f1s.append(fm.macro_f1)
        if self.cfg.model.include_changeflag:
            metrics.update({f"val_{k}": v for k, v in changeflag_metrics(
                cat_logits["changeflag"], cat_targets["changeflag"]
            ).items()})

        avg_macro = sum(macro_f1s) / max(len(macro_f1s), 1)
        n = max(n_batches, 1)
        for k, v in running.items():
            metrics[f"val_loss_{k}" if k != "total" else "val_loss"] = v / n
        metrics["val_avg_macro_f1"] = avg_macro
        return avg_macro, metrics, cat_logits, cat_targets

    # -- orchestration -----------------------------------------------------

    def fit(self) -> None:
        epochs = self.cfg.training.epochs
        patience = self.cfg.training.early_stop_patience
        start = time.time()
        if self.start_epoch > epochs:
            self.logger.info(
                f"start_epoch ({self.start_epoch}) > epochs ({epochs}); nothing to do."
            )
            return
        for epoch in range(self.start_epoch, epochs + 1):
            t0 = time.time()
            train_losses = self.train_one_epoch(epoch)
            val_macro, val_metrics, _, _ = self.validate()

            self.history["train_loss"].append(train_losses["total"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            for fam in self.cfg.model.families:
                self.history[f"train_{fam}_loss"].append(train_losses.get(fam, 0.0))
                self.history[f"val_{fam}_loss"].append(val_metrics.get(f"val_loss_{fam}", 0.0))
                self.history[f"val_{fam}_macro_f1"].append(val_metrics[f"val_{fam}_macro_f1"])
            if self.cfg.model.include_changeflag:
                self.history["train_changeflag_loss"].append(train_losses.get("changeflag", 0.0))
                self.history["val_changeflag_loss"].append(val_metrics.get("val_loss_changeflag", 0.0))
                self.history["val_changeflag_f1"].append(val_metrics.get("val_changeflag_f1", 0.0))

            elapsed = time.time() - t0
            lr0 = self.optimizer.param_groups[0]["lr"]
            lr1 = self.optimizer.param_groups[1]["lr"]
            self.logger.info(
                f"Epoch {epoch}/{epochs}  "
                f"train_loss={train_losses['total']:.4f}  val_loss={val_metrics['val_loss']:.4f}  "
                f"val_avg_macro_f1={val_macro:.4f}  "
                f"lr=({lr0:.2e}, {lr1:.2e})  time={elapsed:.1f}s"
            )
            per_head_train = "  ".join(
                f"{k}={train_losses[k]:.4f}" for k in train_losses if k != "total"
            )
            self.logger.info(f"  train per-head: {per_head_train}")
            for fam in self.cfg.model.families:
                self.logger.info(
                    f"  {fam}: val_loss={val_metrics.get(f'val_loss_{fam}', 0.0):.4f}  "
                    f"micro_f1={val_metrics[f'val_{fam}_micro_f1']:.4f}  "
                    f"macro_f1={val_metrics[f'val_{fam}_macro_f1']:.4f}"
                )
            if self.cfg.model.include_changeflag:
                self.logger.info(
                    f"  changeflag: val_loss={val_metrics.get('val_loss_changeflag', 0.0):.4f}  "
                    f"f1={val_metrics.get('val_changeflag_f1', 0):.4f}"
                )

            improved = val_macro > self.best_val_macro_f1 + 1e-6
            if improved:
                self.best_val_macro_f1 = val_macro
                self.best_epoch = epoch
                self.epochs_since_improvement = 0
                self._save_checkpoint("best.pt", epoch, val_metrics)
                self.logger.info(f"  ✓ new best val_avg_macro_f1={val_macro:.4f}")
            else:
                self.epochs_since_improvement += 1

            # Resilience: rewrite last.pt + history.csv + curves every epoch so a
            # crash, OOM, or credit exhaustion bounds the loss to one epoch.
            self._save_checkpoint("last.pt", epoch, val_metrics)
            self._save_history_csv()
            self._save_history_plot()

            if not improved and self.epochs_since_improvement >= patience:
                self.logger.info(
                    f"Early stopping at epoch {epoch} (no improvement in {patience} epochs). "
                    f"Best epoch: {self.best_epoch} (macro_f1={self.best_val_macro_f1:.4f})."
                )
                break

        self.logger.info(f"Training complete. Total time: {(time.time() - start)/60:.1f} min")
        self.logger.info(f"Best epoch: {self.best_epoch}  best val_avg_macro_f1: {self.best_val_macro_f1:.4f}")

    def _save_checkpoint(self, name: str, epoch: int, metrics: Dict[str, float]) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": _dataclass_to_dict(self.cfg),
            "metrics": metrics,
            "history": self.history,
            "best_val_macro_f1": self.best_val_macro_f1,
            "best_epoch": self.best_epoch,
            "epochs_since_improvement": self.epochs_since_improvement,
        }
        # Atomic-ish write so a crash mid-save can't leave a half-written .pt
        # file. Google Drive's FUSE mount has historically been finicky about
        # `replace()` overwriting an existing file across the FUSE boundary —
        # unlink the old file first if it exists, then rename.
        path = self.output_dir / "checkpoints" / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(ckpt, tmp)
        if path.exists():
            path.unlink()
        tmp.rename(path)

    def _save_history_csv(self) -> None:
        path = self.output_dir / "metrics" / "history.csv"
        keys = list(self.history.keys())
        rows = list(zip(*[self.history[k] for k in keys]))
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["epoch"] + keys)
            for i, row in enumerate(rows, start=1):
                w.writerow([i, *row])

    def _save_history_plot(self) -> None:
        path = self.output_dir / "plots" / "train_curves.png"
        fams = self.cfg.model.families
        rows = 2
        cols = max(1, len(fams))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 8), squeeze=False)

        ax = axes[0, 0]
        ax.plot(self.history["train_loss"], label="train")
        ax.plot(self.history["val_loss"], label="val")
        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        ax.legend()

        for i in range(1, cols):
            axes[0, i].axis("off")

        for i, fam in enumerate(fams):
            ax = axes[1, i]
            ax.plot(self.history[f"val_{fam}_macro_f1"], label=f"{fam} macro F1")
            if self.cfg.model.include_changeflag and i == cols - 1 and "val_changeflag_f1" in self.history:
                ax.plot(self.history["val_changeflag_f1"], label="changeflag F1", linestyle="--")
            ax.set_title(f"val macro F1 — {fam}")
            ax.set_xlabel("Epoch")
            ax.set_ylim(0, 1)
            ax.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close(fig)


def _dataclass_to_dict(cfg: ExperimentConfig) -> Dict:
    return {
        "run_name": cfg.run_name,
        "output_dir": cfg.output_dir,
        "seed": cfg.seed,
        "data": asdict(cfg.data),
        "model": asdict(cfg.model),
        "training": asdict(cfg.training),
        "evaluation": asdict(cfg.evaluation),
    }
