"""Quantify scene-level cross-split leakage in ``dataset.json``.

Writes a JSON + CSV summary to ``results/eda/`` so the report can cite
exact numbers. Per the course PDF the splits must be used as-is, so this
is documentation, not a re-split.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

DATASET_JSON = Path("dataset/dataset.json")
OUT_DIR = Path("results/eda")


def canonical_scene_id(filename: str) -> str:
    name = filename.removesuffix(".png")
    name = name.removesuffix("_random_augment")
    return name.replace("_ters_", "_")


def main() -> None:
    with DATASET_JSON.open("r", encoding="utf-8") as fh:
        samples = json.load(fh)["images"]

    scene_to_splits: dict[str, set[str]] = defaultdict(set)
    scene_to_files: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for s in samples:
        cid = canonical_scene_id(s["filename"])
        scene_to_splits[cid].add(s["split"])
        scene_to_files[cid].append((s["split"], s["filename"]))

    leaks = {cid: splits for cid, splits in scene_to_splits.items() if len(splits) > 1}

    distribution: dict[tuple[str, ...], int] = defaultdict(int)
    for splits in scene_to_splits.values():
        distribution[tuple(sorted(splits))] += 1

    summary = {
        "total_samples": len(samples),
        "unique_base_scene_ids": len(scene_to_splits),
        "leaking_scene_ids": len(leaks),
        "leak_rate_pct": round(100 * len(leaks) / len(scene_to_splits), 3),
        "distribution_scenes_per_split_set": {
            "|".join(k): v for k, v in distribution.items()
        },
        "notes": [
            "Leakage = same base scene appearing in more than one split via the "
            "_ters_ (A/B-swapped) twin file.",
            "Labels differ between an original and its _ters_ variant (events flip, "
            "annotators re-labeled), so this is soft leakage not direct memorization.",
            "Per course PDF instruction (\"Verilerin ayrımları arasında "
            "karıştırılmaması, olduğu gibi kullanılması gerekmektedir.\"), splits "
            "are used as-is.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "split_leakage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = OUT_DIR / "split_leakage_scenes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["base_scene_id", "splits", "files"])
        for cid, splits in sorted(leaks.items()):
            files = "; ".join(f"[{sp}]{fn}" for sp, fn in scene_to_files[cid])
            w.writerow([cid, "|".join(sorted(splits)), files])

    print(f"Summary → {summary_path}")
    print(f"Leaking scenes → {csv_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
