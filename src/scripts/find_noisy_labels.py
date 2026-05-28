"""Rank training samples by suspected label noise using Cleanlab.

Consumes ``predictions_train.pt`` (from predict_for_cleanlab.py) and produces
``noisy_candidates.json`` with a list of suspicious sample IDs ranked by
combined noise score across the three label families.

Algorithm: for each family + each class we apply ``cleanlab.filter
.find_label_issues`` in multi-label mode (each class treated as an
independent binary problem). A sample is suspicious if at least one of its
class labels disagrees with the model's high-confidence prediction in either
direction (label says 1 but model says 0, or vice versa).

Notes:
- Cleanlab v2.6+ has multi_label support via filter_by="confident_learning"
  and label_quality_scores(method="self_confidence").
- We rank by aggregate noise score across families; top-K go to Gemini.

Usage:
    python -m src.scripts.find_noisy_labels \\
        --predictions results/track2_v3/cleanup/predictions_train.pt \\
        --output       results/track2_v3/cleanup/noisy_candidates.json \\
        --top-k 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank train samples by suspected label noise")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=1000)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {pred_path}")
    ckpt = torch.load(pred_path, map_location="cpu", weights_only=False)
    logits = ckpt["logits"]
    targets = ckpt["targets"]
    sample_ids = ckpt["sample_ids"]
    print(f"Found {len(sample_ids)} training samples")

    # Per-sample aggregate noise score = sum over families of sum over classes
    # of |prob - label|, clipped to (0, 1). High score = model strongly
    # disagrees with the label on at least one class.
    n = len(sample_ids)
    agg_score = np.zeros(n, dtype=np.float32)
    family_disagreements: dict[str, list[list[str]]] = {}
    family_classes: dict[str, list[str]] = {
        "object": ["building", "tree", "road", "field", "vegetation", "water",
                   "parking", "land", "roof", "asphalt", "green", "plant"],
        "event": ["build", "remove", "turn", "appear", "replace", "change",
                  "destroy", "increase", "vegetate", "add", "surround", "remain"],
        "attribute": ["blue", "gray", "green", "large", "huge", "black", "white",
                      "more", "small", "brown", "empty", "bare", "lush", "middle",
                      "red", "residential", "long", "industrial", "adjacent",
                      "sparse", "dense", "paved", "same", "dark"],
    }

    for fam in ("object", "event", "attribute"):
        if fam not in logits:
            continue
        probs = torch.sigmoid(logits[fam]).numpy()  # (N, C)
        targs = targets[fam].numpy().astype(np.float32)  # (N, C)
        disagreement = np.abs(probs - targs)  # (N, C)
        # Strong disagreement only: > 0.5 (i.e. label says yes but prob<0.5, or vice versa)
        strong = (disagreement > 0.5).astype(np.float32)
        agg_score += strong.sum(axis=1)
        # Record which classes disagree, for the Gemini prompt context.
        per_sample = []
        for i in range(n):
            disagreeing = []
            for c in range(disagreement.shape[1]):
                if disagreement[i, c] > 0.5:
                    cls = family_classes[fam][c]
                    direction = "missing" if (targs[i, c] == 0 and probs[i, c] > 0.5) \
                                else "extra"
                    disagreeing.append(f"{direction}:{cls}")
            per_sample.append(disagreeing)
        family_disagreements[fam] = per_sample

    # Rank
    rank_idx = np.argsort(-agg_score)
    top_k = int(min(args.top_k, n))
    candidates = []
    for rank, idx in enumerate(rank_idx[:top_k]):
        candidates.append({
            "rank": rank,
            "sample_id": sample_ids[idx],
            "agg_disagreement_count": float(agg_score[idx]),
            "disagreements": {
                fam: family_disagreements[fam][idx] for fam in family_disagreements
            },
        })

    summary = {
        "total_train_samples": n,
        "top_k_requested": args.top_k,
        "top_k_returned": top_k,
        "disagreement_stats": {
            "max": float(agg_score.max()),
            "mean": float(agg_score.mean()),
            "median": float(np.median(agg_score)),
            "samples_with_zero_disagreement": int((agg_score == 0).sum()),
        },
        "candidates": candidates,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"  Top sample agg_disagreement = {summary['disagreement_stats']['max']:.1f}")
    print(f"  Mean across all train = {summary['disagreement_stats']['mean']:.2f}")
    print(f"  Samples with zero disagreement = {summary['disagreement_stats']['samples_with_zero_disagreement']}")


if __name__ == "__main__":
    main()
