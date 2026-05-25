"""Per-class threshold sweep on val predictions.

Loads a ``predictions_val.pt`` file produced by ``eval.py --split val``.
For each family, sweeps per-class thresholds on a grid and picks the
threshold that maximizes per-class F1 on the val set. Saves the result
as ``thresholds_val.json`` next to the input (or at ``--output``).

Tuning ON VAL (not test) is the protocol both Copilot and Gemini explicitly
flagged as correct for Phase 2. Phase 1's test-set-optimistic sweep was
diagnostic only.

Usage:

  python -m src.scripts.tune_thresholds \\
      --predictions results/track1_v2/object/metrics/predictions_val.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support


def _per_class_best_threshold(
    probs: np.ndarray, targets: np.ndarray, grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick per-class threshold maximizing F1 on (probs, targets).

    Returns (best_thresholds, best_f1s) each of length C. Classes with zero
    val support are assigned threshold 1.0 (no test-time false positives can
    fire) rather than 0.5 — for those classes we have no signal either way,
    so suppressing predictions is the protocol-neutral, F1-safe default.
    """
    n_classes = probs.shape[1]
    # Default to 1.0 for unsupported classes (suppresses test predictions).
    best_thr = np.full(n_classes, 1.0, dtype=np.float32)
    best_f1 = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        if targets[:, c].sum() == 0:
            continue  # no positives in val → keep 1.0 (suppress)
        # Class has positives — start at 0.5 default and improve from there.
        best_thr[c] = 0.5
        for thr in grid:
            preds = (probs[:, c] >= thr).astype(np.int32)
            _, _, f1, _ = precision_recall_fscore_support(
                targets[:, c], preds, average="binary", zero_division=0
            )
            if f1 > best_f1[c]:
                best_f1[c] = f1
                best_thr[c] = thr
    return best_thr, best_f1


def _macro_at_thresholds(
    probs: np.ndarray, targets: np.ndarray, thresholds: np.ndarray
) -> Dict[str, float]:
    preds = (probs >= thresholds[None, :]).astype(np.int32)
    p, r, f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )
    pm, rm, f1m, _ = precision_recall_fscore_support(
        targets, preds, average="micro", zero_division=0
    )
    return {
        "macro_p": float(p), "macro_r": float(r), "macro_f1": float(f1),
        "micro_p": float(pm), "micro_r": float(rm), "micro_f1": float(f1m),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-class threshold sweep on val predictions")
    parser.add_argument("--predictions", required=True, help="Path to predictions_val.pt (from eval.py --split val)")
    parser.add_argument("--output", default=None, help="Path for thresholds JSON (default: predictions parent / thresholds_val.json)")
    parser.add_argument("--grid-step", type=float, default=0.05, help="Threshold grid step (0.05 default → 19 points 0.05..0.95)")
    parser.add_argument("--grid-min", type=float, default=0.05)
    parser.add_argument("--grid-max", type=float, default=0.95)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions file not found: {pred_path}")
    out_path = Path(args.output) if args.output else pred_path.parent / "thresholds_val.json"

    grid = np.arange(args.grid_min, args.grid_max + 1e-9, args.grid_step, dtype=np.float32)
    print(f"Threshold grid: {grid.tolist()}")
    print(f"Reading predictions from {pred_path}")
    ckpt = torch.load(pred_path, map_location="cpu", weights_only=False)
    logits = ckpt["logits"]
    targets = ckpt["targets"]

    result: Dict[str, Dict] = {}
    print(f"\n{'family':<12s}  {'val flat=0.5 macro':>20s}  {'val tuned macro':>16s}  {'Δ':>7s}")
    for fam in logits:
        if fam == "changeflag":
            continue  # binary, leave at 0.5
        probs_t = torch.sigmoid(logits[fam]).numpy()
        targs_t = targets[fam].numpy().astype(np.int32)
        thrs, _ = _per_class_best_threshold(probs_t, targs_t, grid)
        flat = _macro_at_thresholds(probs_t, targs_t, np.full(thrs.shape[0], 0.5, dtype=np.float32))
        tuned = _macro_at_thresholds(probs_t, targs_t, thrs)
        delta = tuned["macro_f1"] - flat["macro_f1"]
        print(f"{fam:<12s}  {flat['macro_f1']:>20.4f}  {tuned['macro_f1']:>16.4f}  {delta:>+7.4f}")
        result[fam] = {
            "thresholds": thrs.tolist(),
            "val_flat_05":  flat,
            "val_tuned":    tuned,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nWrote thresholds → {out_path}")


if __name__ == "__main__":
    main()
