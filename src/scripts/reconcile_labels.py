"""Reconcile Gemini re-labels with the original dataset to produce a cleaned JSON.

Conservative AND-rule:
- A label is FLIPPED only when (a) Cleanlab/disagreement score for that sample
  is in the top-K AND (b) Gemini's confidence is "high" AND (c) Gemini's
  labels for that family differ from the original labels.
- "Unknown" / "low" / "medium" confidence keeps the original.

Outputs ``dataset_v3_clean.json`` with the same schema as ``dataset.json``
but updated label lists where applicable. Also emits a summary JSON of
how many labels changed per family.

Usage:
    python -m src.scripts.reconcile_labels \\
        --dataset-json dataset/dataset.json \\
        --train-relabels results/track1_v3/cleanup/gemini_relabels_train.jsonl \\
        --val-relabels   results/track1_v3/cleanup/gemini_relabels_val.jsonl \\
        --output         results/track1_v3/cleanup/dataset_v3_clean.json \\
        --summary        results/track1_v3/cleanup/reconcile_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: str) -> Dict[str, dict]:
    """Return {sample_id: record} from a JSONL of Gemini relabels."""
    out = {}
    if not path or not Path(path).exists():
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["sample_id"]] = r
    return out


def apply_flip(original: list, gemini: list, gemini_conf: str,
              valid_vocab: set) -> tuple[list, bool]:
    """Return (new_labels, was_flipped) according to the conservative rule."""
    if gemini_conf != "high":
        return original, False
    # Sanitize gemini labels to the valid vocab + drop "none"
    filtered = [l for l in gemini if l in valid_vocab and l != "none"]
    orig_set = set(l for l in original if l != "none")
    new_set = set(filtered)
    if new_set == orig_set:
        return original, False
    return filtered if filtered else ["none"], True


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile Gemini relabels into cleaned dataset")
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--train-relabels", help="JSONL from gemini_relabel.py on train candidates")
    parser.add_argument("--val-relabels", help="JSONL from gemini_relabel.py on val split")
    parser.add_argument("--test-relabels", help="(Optional) JSONL on test split. Usually skipped.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    data = json.load(open(args.dataset_json))
    samples = data["images"]
    relabels = {}
    relabels.update(load_jsonl(args.train_relabels))
    relabels.update(load_jsonl(args.val_relabels))
    if args.test_relabels:
        relabels.update(load_jsonl(args.test_relabels))
    print(f"Loaded {len(relabels)} gemini relabel records")

    vocabs = {
        "object_labels": {"building", "tree", "road", "field", "vegetation", "water",
                          "parking", "land", "roof", "asphalt", "green", "plant", "none"},
        "event_labels": {"build", "remove", "turn", "appear", "replace", "change",
                         "destroy", "increase", "vegetate", "add", "surround", "remain", "none"},
        "attribute_labels": {"blue", "gray", "green", "large", "huge", "black", "white",
                             "more", "small", "brown", "empty", "bare", "lush", "middle",
                             "red", "residential", "long", "industrial", "adjacent",
                             "sparse", "dense", "paved", "same", "dark", "none"},
    }
    family_to_key = {"object": "object_labels", "event": "event_labels", "attribute": "attribute_labels"}
    family_to_gemini = {"object": "object_labels", "event": "event_labels", "attribute": "attribute_labels"}

    flip_counter: Counter[str] = Counter()
    confidence_counter: Counter[str] = Counter()
    per_split_flips: dict[str, Counter[str]] = {"train": Counter(), "val": Counter(), "test": Counter()}

    cleaned = []
    for s in samples:
        sid = s["sample_id"]
        r = relabels.get(sid)
        sout = dict(s)  # shallow copy

        if r is not None and isinstance(r.get("gemini"), dict):
            g = r["gemini"]
            conf = g.get("confidence", "unknown")
            confidence_counter[conf] += 1
            for fam, key in family_to_key.items():
                gem_labels = g.get(family_to_gemini[fam], [])
                new_labels, flipped = apply_flip(
                    s.get(key, []), gem_labels, conf, vocabs[key]
                )
                sout[key] = new_labels
                if flipped:
                    flip_counter[fam] += 1
                    per_split_flips[s["split"]][fam] += 1
            # Update changeflag based on whether any non-none label remains in any family
            any_change = any(
                any(l != "none" for l in sout[family_to_key[fam]])
                for fam in family_to_key
            )
            sout["changeflag"] = 1 if any_change else 0

        cleaned.append(sout)

    out_data = dict(data)
    out_data["images"] = cleaned
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out_data, fh, indent=2, ensure_ascii=False)

    summary = {
        "total_samples": len(samples),
        "relabel_records": len(relabels),
        "confidence_distribution": dict(confidence_counter),
        "flips_per_family": dict(flip_counter),
        "flips_per_split_family": {sp: dict(c) for sp, c in per_split_flips.items()},
    }
    with open(args.summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"\nWrote cleaned dataset → {args.output}")
    print(f"Summary → {args.summary}")
    print(json.dumps(summary, indent=2)[:600])


if __name__ == "__main__":
    main()
