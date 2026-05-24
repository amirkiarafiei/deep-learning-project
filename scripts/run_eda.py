#!/usr/bin/env python3
"""Exploratory Data Analysis for the BLM5135 bi-temporal multi-label dataset.

Outputs land in `results/eda/`:
- figures/*.png and *.pdf  (300 DPI, IEEE-report-ready)
- label_vocab.json
- per_split_stats.json
- eda_summary.md
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# --------------------------------------------------------------------------- #
# Style                                                                       #
# --------------------------------------------------------------------------- #
mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FAMILY_COLORS = {
    "object":    "steelblue",
    "event":     "indianred",
    "attribute": "seagreen",
}
SPLIT_COLORS = {
    "train": "#4477AA",
    "val":   "#EE6677",
    "test":  "#228833",
}
FAMILIES = ["object", "event", "attribute"]
SPLITS = ["train", "val", "test"]
EXPECTED_COUNTS = {"object": 13, "event": 13, "attribute": 25}

# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "dataset"
DATA_JSON = DATA_ROOT / "dataset.json"
OUT = ROOT / "results" / "eda"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def savefig(fig, name: str) -> None:
    """Save a matplotlib figure as both PNG (300 DPI) and PDF."""
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Load                                                                        #
# --------------------------------------------------------------------------- #
print("Loading dataset.json ...")
with DATA_JSON.open() as f:
    data = json.load(f)

samples = data["images"]
N = len(samples)
print(f"  loaded {N} samples")


def family_labels(sample: dict, family: str) -> list[str]:
    return sample[f"{family}_labels"]


def non_none(labels: list[str]) -> list[str]:
    return [l for l in labels if l != "none"]


# --------------------------------------------------------------------------- #
# ANALYSIS 1 — Label vocab + per-split stats                                  #
# --------------------------------------------------------------------------- #
print("\n[1/8] Building vocab + per-split stats ...")

global_counts: dict[str, Counter] = {fam: Counter() for fam in FAMILIES}
split_counts: dict[str, dict[str, Counter]] = {
    s: {fam: Counter() for fam in FAMILIES} for s in SPLITS
}
split_sample_n: Counter = Counter()
changeflag_by_split: dict[str, Counter] = {s: Counter() for s in SPLITS}

for s in samples:
    split = s["split"]
    split_sample_n[split] += 1
    changeflag_by_split[split][s["changeflag"]] += 1
    for fam in FAMILIES:
        # Use set to dedupe accidental repeats within a single sample.
        for label in set(family_labels(s, fam)):
            global_counts[fam][label] += 1
            split_counts[split][fam][label] += 1

vocab: dict[str, dict] = {}
for fam in FAMILIES:
    counts = global_counts[fam]
    # 'none' goes to idx 0, the rest sorted by frequency desc, then alpha.
    others = sorted(
        [(lbl, c) for lbl, c in counts.items() if lbl != "none"],
        key=lambda kv: (-kv[1], kv[0]),
    )
    ordered = [("none", counts.get("none", 0))] + others
    label_to_idx = {lbl: i for i, (lbl, _) in enumerate(ordered)}
    idx_to_label = {str(i): lbl for i, (lbl, _) in enumerate(ordered)}
    frequencies = {lbl: int(c) for lbl, c in ordered}
    vocab[fam] = {
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "frequencies": frequencies,
    }
    unique = len(ordered)
    flag = "OK" if unique == EXPECTED_COUNTS[fam] else f"!! expected {EXPECTED_COUNTS[fam]}"
    print(f"  {fam:>9s}: {unique} labels [{flag}]")

with (OUT / "label_vocab.json").open("w") as f:
    json.dump(vocab, f, indent=2, ensure_ascii=False)
print(f"  -> wrote {OUT / 'label_vocab.json'}")

# per-split stats JSON
per_split_stats = {
    "totals": {s: int(split_sample_n[s]) for s in SPLITS},
    "changeflag": {
        s: {str(k): int(v) for k, v in changeflag_by_split[s].items()} for s in SPLITS
    },
    "by_split": {
        s: {fam: dict(sorted(split_counts[s][fam].items())) for fam in FAMILIES}
        for s in SPLITS
    },
    "global": {
        fam: dict(sorted(global_counts[fam].items())) for fam in FAMILIES
    },
}
with (OUT / "per_split_stats.json").open("w") as f:
    json.dump(per_split_stats, f, indent=2, ensure_ascii=False)
print(f"  -> wrote {OUT / 'per_split_stats.json'}")

# --------------------------------------------------------------------------- #
# ANALYSIS 2 — Label frequency histograms                                     #
# --------------------------------------------------------------------------- #
print("\n[2/8] Plotting label frequency histograms ...")

for fam in FAMILIES:
    items = [(lbl, c) for lbl, c in vocab[fam]["frequencies"].items()]
    # For frequency plot, sort all labels (including 'none') by count desc.
    items_sorted = sorted(items, key=lambda kv: -kv[1])
    labels = [k for k, _ in items_sorted]
    counts = np.array([v for _, v in items_sorted], dtype=float)
    pct = 100 * counts / N

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.32 * len(labels) + 1.5)))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, counts, color=FAMILY_COLORS[fam], edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of samples containing label")
    ax.set_title(f"{fam.capitalize()} Label Frequency (n={N})")
    xmax = counts.max() * 1.18
    ax.set_xlim(0, xmax)
    for i, (c, p) in enumerate(zip(counts, pct)):
        ax.text(c + counts.max() * 0.01, i, f"{int(c):,} ({p:.1f}%)",
                va="center", ha="left", fontsize=8.5)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    savefig(fig, f"freq_{fam}")
    print(f"  -> figures/freq_{fam}.png/.pdf")

# --------------------------------------------------------------------------- #
# ANALYSIS 3 — Per-split distribution drift                                   #
# --------------------------------------------------------------------------- #
print("\n[3/8] Per-split drift check ...")

drift_findings: dict[str, list[tuple[str, float]]] = {}

for fam in FAMILIES:
    # Labels sorted by global frequency desc, including 'none' for visual context.
    labels = list(vocab[fam]["frequencies"].keys())
    # Per-split percentage (% of samples in that split that contain the label).
    per_split_pct = {s: np.zeros(len(labels)) for s in SPLITS}
    for s in SPLITS:
        denom = max(split_sample_n[s], 1)
        for i, lbl in enumerate(labels):
            per_split_pct[s][i] = 100 * split_counts[s][fam].get(lbl, 0) / denom

    # Relative-difference detection vs. mean across splits.
    drifts = []
    for i, lbl in enumerate(labels):
        if lbl == "none":
            continue
        vals = np.array([per_split_pct[s][i] for s in SPLITS])
        mu = vals.mean()
        if mu < 0.05:
            continue
        rel_spread = (vals.max() - vals.min()) / mu * 100
        drifts.append((lbl, float(rel_spread), {s: float(per_split_pct[s][i]) for s in SPLITS}))
    drifts.sort(key=lambda t: -t[1])
    drift_findings[fam] = drifts[:3]
    print(f"  {fam}: top-3 drifted labels (>50% rel diff is notable)")
    for lbl, rel, vals in drifts[:3]:
        print(f"    {lbl:<14s}  rel_spread={rel:6.1f}%  "
              f"train={vals['train']:.2f}% val={vals['val']:.2f}% test={vals['test']:.2f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(labels) + 2), 4.8))
    x = np.arange(len(labels))
    w = 0.27
    for k, s in enumerate(SPLITS):
        ax.bar(x + (k - 1) * w, per_split_pct[s], width=w,
               label=s, color=SPLIT_COLORS[s], edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("% of samples in split containing label")
    ax.set_title(f"{fam.capitalize()} — per-split label distribution")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    savefig(fig, f"split_drift_{fam}")
    print(f"  -> figures/split_drift_{fam}.png/.pdf")

# --------------------------------------------------------------------------- #
# ANALYSIS 4 — Labels-per-sample distributions                                #
# --------------------------------------------------------------------------- #
print("\n[4/8] Labels-per-sample distributions ...")

per_sample_counts: dict[str, list[int]] = {fam: [] for fam in FAMILIES}
per_sample_counts_changed: dict[str, list[int]] = {fam: [] for fam in FAMILIES}
for s in samples:
    is_changed = bool(s["changeflag"])
    for fam in FAMILIES:
        n = len(set(non_none(family_labels(s, fam))))
        per_sample_counts[fam].append(n)
        if is_changed:
            per_sample_counts_changed[fam].append(n)

intensity_stats = {}
for fam in FAMILIES:
    arr_all = np.array(per_sample_counts[fam])
    arr_ch = np.array(per_sample_counts_changed[fam])
    intensity_stats[fam] = {
        "all":     {"mean": float(arr_all.mean()), "median": float(np.median(arr_all)),
                    "max": int(arr_all.max()), "std": float(arr_all.std())},
        "changed": {"mean": float(arr_ch.mean()), "median": float(np.median(arr_ch)),
                    "max": int(arr_ch.max()), "std": float(arr_ch.std())},
    }
    print(f"  {fam:>9s}  all  mean={arr_all.mean():.3f} med={np.median(arr_all):.1f} "
          f"max={arr_all.max()} std={arr_all.std():.3f}")
    print(f"  {fam:>9s}  ch   mean={arr_ch.mean():.3f} med={np.median(arr_ch):.1f} "
          f"max={arr_ch.max()} std={arr_ch.std():.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
for ax, fam in zip(axes, FAMILIES):
    arr_all = np.array(per_sample_counts[fam])
    arr_ch = np.array(per_sample_counts_changed[fam])
    max_v = max(arr_all.max(), arr_ch.max())
    bins = np.arange(-0.5, max_v + 1.5, 1)
    ax.hist(arr_all, bins=bins, color=FAMILY_COLORS[fam], alpha=0.55,
            label="all samples", edgecolor="black", linewidth=0.4)
    ax.hist(arr_ch, bins=bins, color=FAMILY_COLORS[fam], alpha=1.0,
            histtype="step", linewidth=1.8, label="changeflag=1")
    ax.set_title(f"{fam.capitalize()}")
    ax.set_xlabel("# non-'none' labels per sample")
    ax.set_ylabel("# samples")
    ax.set_xticks(np.arange(0, max_v + 1))
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
fig.suptitle(f"Multi-label intensity per family (n={N})", y=1.02)
fig.tight_layout()
savefig(fig, "labels_per_sample")
print("  -> figures/labels_per_sample.png/.pdf")

# --------------------------------------------------------------------------- #
# ANALYSIS 5 — Co-occurrence heatmaps                                         #
# --------------------------------------------------------------------------- #
print("\n[5/8] Co-occurrence heatmaps ...")

cooc_notes: dict[str, list[tuple[str, str, int]]] = {}

for fam in FAMILIES:
    items_sorted = sorted(
        [(lbl, c) for lbl, c in vocab[fam]["frequencies"].items() if lbl != "none"],
        key=lambda kv: -kv[1],
    )
    labels = [k for k, _ in items_sorted]
    idx = {lbl: i for i, lbl in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for s in samples:
        present = set(l for l in family_labels(s, fam) if l in idx)
        if len(present) < 2:
            # Still record diagonal counts.
            for l in present:
                i = idx[l]
                M[i, i] += 1
            continue
        for l1 in present:
            for l2 in present:
                M[idx[l1], idx[l2]] += 1
    # Mask diagonal in displayed matrix
    display = M.astype(float).copy()
    np.fill_diagonal(display, np.nan)

    # Top off-diagonal pairs
    pairs = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] > 0:
                pairs.append((labels[i], labels[j], int(M[i, j])))
    pairs.sort(key=lambda t: -t[2])
    cooc_notes[fam] = pairs[:5]

    fig_w = max(6, 0.5 * n + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.9))
    cmap = plt.cm.get_cmap({"object": "Blues", "event": "Reds", "attribute": "Greens"}[fam]).copy()
    cmap.set_bad(color="lightgray")
    im = ax.imshow(display, cmap=cmap, aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    # Annotate non-diagonal cells with counts (skip zeros to reduce clutter)
    max_val = np.nanmax(display) if n > 1 else 1
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = M[i, j]
            if v == 0:
                continue
            color = "white" if v > max_val * 0.55 else "black"
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=7.5, color=color)
    ax.set_title(f"{fam.capitalize()} co-occurrence (off-diagonal counts)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("# samples with both labels")
    fig.tight_layout()
    savefig(fig, f"cooccurrence_{fam}")
    print(f"  -> figures/cooccurrence_{fam}.png/.pdf")
    print(f"     top pairs: {pairs[:3]}")

# --------------------------------------------------------------------------- #
# ANALYSIS 6 — Image inspection                                               #
# --------------------------------------------------------------------------- #
print("\n[6/8] Image inspection ...")

rng = random.Random(42)


def pick(predicate, k, exclude):
    pool = [s for s in samples if predicate(s) and s["sample_id"] not in exclude]
    rng.shuffle(pool)
    return pool[:k]


taken_ids: set[str] = set()
group_nochange = pick(lambda s: s["changeflag"] == 0, 5, taken_ids)
taken_ids.update(s["sample_id"] for s in group_nochange)

group_single_obj = pick(
    lambda s: s["changeflag"] == 1 and len(set(non_none(s["object_labels"]))) == 1
              and len(set(non_none(s["event_labels"]))) <= 1
              and len(set(non_none(s["attribute_labels"]))) <= 1,
    5, taken_ids,
)
taken_ids.update(s["sample_id"] for s in group_single_obj)

group_dense = pick(
    lambda s: s["changeflag"] == 1 and (
        len(set(non_none(s["object_labels"]))) >= 3 or
        len(set(non_none(s["event_labels"]))) >= 3 or
        len(set(non_none(s["attribute_labels"]))) >= 3
    ),
    5, taken_ids,
)
taken_ids.update(s["sample_id"] for s in group_dense)

group_ters = pick(lambda s: "_ters_" in s["filename"], 5, taken_ids)

inspection = []
for g, tag in [
    (group_nochange, "no-change"),
    (group_single_obj, "single-object"),
    (group_dense, "dense (≥3 labels)"),
    (group_ters, "_ters_ pair"),
]:
    for s in g:
        inspection.append((tag, s))

print(f"  picked groups: nochange={len(group_nochange)} single={len(group_single_obj)} "
      f"dense={len(group_dense)} ters={len(group_ters)}")

# Image stats (resolution + filesize) — sample broadly, not just the 20.
res_counter: Counter = Counter()
filesizes: list[int] = []
# Sample up to ~500 images for stats to stay quick on big datasets.
stat_pool = samples if N <= 600 else rng.sample(samples, 500)
for s in stat_pool:
    p = DATA_ROOT / s["rgb_A"]
    if not p.exists():
        continue
    try:
        with Image.open(p) as im:
            res_counter[im.size] += 1
        filesizes.append(p.stat().st_size)
    except Exception as e:
        print(f"  warn: could not read {p}: {e}")
filesize_arr = np.array(filesizes) if filesizes else np.array([0])
img_stats = {
    "resolution_distribution": {f"{w}x{h}": int(c) for (w, h), c in res_counter.most_common()},
    "filesize_bytes": {
        "mean": float(filesize_arr.mean()),
        "median": float(np.median(filesize_arr)),
        "min": int(filesize_arr.min()),
        "max": int(filesize_arr.max()),
        "n_sampled": len(filesizes),
    },
}
print(f"  resolutions seen (sample {len(filesizes)}): {dict(res_counter.most_common(5))}")
print(f"  filesize bytes: mean={filesize_arr.mean():.0f} med={np.median(filesize_arr):.0f} "
      f"min={filesize_arr.min()} max={filesize_arr.max()}")


def load_rgb(rel_path: str):
    p = DATA_ROOT / rel_path
    return np.array(Image.open(p).convert("RGB"))


def render_pair_row(ax_a, ax_b, sample, tag):
    img_a = load_rgb(sample["rgb_A"])
    img_b = load_rgb(sample["rgb_B"])
    ax_a.imshow(img_a); ax_a.set_xticks([]); ax_a.set_yticks([])
    ax_b.imshow(img_b); ax_b.set_xticks([]); ax_b.set_yticks([])
    header = (f"[{tag}]  {sample['sample_id']}\n"
              f"split={sample['split']}  changeflag={sample['changeflag']}")
    ax_a.set_title(header, fontsize=8.5, loc="left", fontweight="bold")
    ax_b.set_title("B (t2)", fontsize=8.5)
    o = ", ".join(sample["object_labels"]) or "-"
    e = ", ".join(sample["event_labels"]) or "-"
    a = ", ".join(sample["attribute_labels"]) or "-"
    caption = f"object: {o}\nevent: {e}\nattribute: {a}"
    ax_a.text(0.0, -0.04, caption, transform=ax_a.transAxes,
              fontsize=8, ha="left", va="top",
              family="monospace")


# Multi-page PDF: 4 samples per page, with generous vertical spacing
pdf_path = FIG / "sample_grid.pdf"
print(f"  writing {pdf_path} ...")
per_page = 4
with PdfPages(pdf_path) as pdf:
    for page_start in range(0, len(inspection), per_page):
        page = inspection[page_start:page_start + per_page]
        fig = plt.figure(figsize=(10, 16))
        # Each "row" of the page is one sample = (header strip via title + two image axes).
        # Generous hspace so the caption below row r doesn't run into the title of row r+1.
        gs = fig.add_gridspec(per_page, 2, hspace=0.95, wspace=0.06,
                              left=0.06, right=0.97, top=0.96, bottom=0.04)
        for r, (tag, s) in enumerate(page):
            ax_a = fig.add_subplot(gs[r, 0])
            ax_b = fig.add_subplot(gs[r, 1])
            render_pair_row(ax_a, ax_b, s, tag)
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
        if page_start == 0:
            fig.savefig(FIG / "sample_grid_page1.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
print("  -> figures/sample_grid.pdf  + figures/sample_grid_page1.png")

# --------------------------------------------------------------------------- #
# ANALYSIS 7 — A/B registration diff for no-change samples                    #
# --------------------------------------------------------------------------- #
print("\n[7/8] Diff maps for no-change samples ...")

diff_samples = group_nochange
diff_stats = []
fig, axes = plt.subplots(len(diff_samples), 3, figsize=(11, 3.4 * len(diff_samples)))
if len(diff_samples) == 1:
    axes = np.array([axes])
for r, s in enumerate(diff_samples):
    img_a = np.array(Image.open(DATA_ROOT / s["rgb_A"]).convert("L"), dtype=np.float32) / 255.0
    img_b = np.array(Image.open(DATA_ROOT / s["rgb_B"]).convert("L"), dtype=np.float32) / 255.0
    if img_a.shape != img_b.shape:
        # Resize B to match A
        b_im = Image.open(DATA_ROOT / s["rgb_B"]).convert("L").resize(img_a.shape[::-1])
        img_b = np.array(b_im, dtype=np.float32) / 255.0
    diff = np.abs(img_a - img_b)
    diff_stats.append({
        "sample_id": s["sample_id"],
        "mean_abs_diff": float(diff.mean()),
        "p95_abs_diff":  float(np.percentile(diff, 95)),
        "max_abs_diff":  float(diff.max()),
    })
    axes[r, 0].imshow(np.array(Image.open(DATA_ROOT / s["rgb_A"]).convert("RGB")))
    axes[r, 0].set_title(f"A (t1) — {s['sample_id']}", fontsize=9)
    axes[r, 1].imshow(np.array(Image.open(DATA_ROOT / s["rgb_B"]).convert("RGB")))
    axes[r, 1].set_title("B (t2)", fontsize=9)
    im = axes[r, 2].imshow(diff, cmap="magma", vmin=0, vmax=max(0.1, diff.max()))
    axes[r, 2].set_title(f"|A-B| gray  (mean={diff.mean():.3f}, p95={np.percentile(diff,95):.3f})",
                         fontsize=9)
    for c in range(3):
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.colorbar(im, ax=axes[r, 2], fraction=0.04)

fig.suptitle("Registration sanity: no-change pairs (changeflag=0)", y=0.995)
fig.tight_layout()
savefig(fig, "nochange_diffmaps")
print("  -> figures/nochange_diffmaps.png/.pdf")
for d in diff_stats:
    print(f"    {d['sample_id']}: mean={d['mean_abs_diff']:.4f} "
          f"p95={d['p95_abs_diff']:.4f} max={d['max_abs_diff']:.4f}")

# --------------------------------------------------------------------------- #
# ANALYSIS 8 — Augmentation breakdown                                         #
# --------------------------------------------------------------------------- #
print("\n[8/8] Augmentation breakdown ...")


def categorize(fn: str) -> str:
    has_aug = fn.endswith("_random_augment.png")
    has_ters = "_ters_" in fn
    if has_aug and has_ters:
        return "ters+aug"
    if has_aug:
        return "random_augment"
    if has_ters:
        return "ters"
    return "original"


CATS = ["original", "random_augment", "ters", "ters+aug"]
CAT_COLORS = {
    "original":        "#888888",
    "random_augment":  "#4477AA",
    "ters":            "#EE6677",
    "ters+aug":        "#CCBB44",
}

cat_split: dict[str, Counter] = {s: Counter() for s in SPLITS}
cat_total: Counter = Counter()
for s in samples:
    cat = categorize(s["filename"])
    cat_split[s["split"]][cat] += 1
    cat_total[cat] += 1

print(f"  totals: {dict(cat_total)}")
for s in SPLITS:
    print(f"  {s}: {dict(cat_split[s])}")

fig, ax = plt.subplots(figsize=(7, 4.5))
bottom = np.zeros(len(SPLITS))
x = np.arange(len(SPLITS))
for cat in CATS:
    vals = np.array([cat_split[s].get(cat, 0) for s in SPLITS], dtype=float)
    ax.bar(x, vals, bottom=bottom, label=cat, color=CAT_COLORS[cat],
           edgecolor="black", linewidth=0.4)
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels(SPLITS)
ax.set_ylabel("# samples")
ax.set_title("Augmentation breakdown per split")
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.grid(axis="x", visible=False)
# Annotate totals on top of each stack
for i, s in enumerate(SPLITS):
    total = int(sum(cat_split[s].values()))
    ax.text(i, total + max(bottom) * 0.01, f"n={total:,}", ha="center",
            va="bottom", fontsize=9, fontweight="bold")
fig.tight_layout()
savefig(fig, "augmentation_breakdown")
print("  -> figures/augmentation_breakdown.png/.pdf")

# --------------------------------------------------------------------------- #
# SUMMARY — eda_summary.md                                                    #
# --------------------------------------------------------------------------- #
print("\nWriting eda_summary.md ...")

# Imbalance for each family: max/min ratio (ignoring 'none')
imbalance_info: dict[str, dict] = {}
for fam in FAMILIES:
    freqs = [(lbl, c) for lbl, c in vocab[fam]["frequencies"].items() if lbl != "none"]
    freqs_sorted = sorted(freqs, key=lambda kv: -kv[1])
    most = freqs_sorted[0]
    least = freqs_sorted[-1]
    ratio = most[1] / max(least[1], 1)
    imbalance_info[fam] = {
        "most": most,
        "least": least,
        "ratio": ratio,
    }


def md_freq_table(fam: str) -> str:
    rows = ["| idx | label | count | % |", "|---:|---|---:|---:|"]
    for i, (lbl, c) in enumerate(vocab[fam]["frequencies"].items()):
        rows.append(f"| {i} | `{lbl}` | {c:,} | {100*c/N:.2f}% |")
    return "\n".join(rows)


lines: list[str] = []
lines.append("# Dataset EDA Summary")
lines.append("")
lines.append(f"_Source: `dataset/dataset.json` (n = {N:,} samples)._")
lines.append("")

lines.append("## 1. Overall scale")
lines.append("")
lines.append(f"- **Total samples:** {N:,}")
lines.append("- **Per-split counts:**")
for s in SPLITS:
    n = split_sample_n[s]
    lines.append(f"  - `{s}`: {n:,} ({100*n/N:.2f}%)")
lines.append("- **Changeflag distribution:**")
total_changed = sum(changeflag_by_split[s].get(1, 0) for s in SPLITS)
total_no = sum(changeflag_by_split[s].get(0, 0) for s in SPLITS)
lines.append(f"  - `changeflag=1` (changed): {total_changed:,} ({100*total_changed/N:.2f}%)")
lines.append(f"  - `changeflag=0` (no-change): {total_no:,} ({100*total_no/N:.2f}%)")
lines.append("- **Per-split changeflag:**")
for s in SPLITS:
    c1 = changeflag_by_split[s].get(1, 0)
    c0 = changeflag_by_split[s].get(0, 0)
    lines.append(f"  - `{s}`: changed={c1:,} ({100*c1/max(split_sample_n[s],1):.1f}%), "
                 f"no-change={c0:,} ({100*c0/max(split_sample_n[s],1):.1f}%)")
lines.append("")

lines.append("## 2. Label vocabularies")
lines.append("")
for fam in FAMILIES:
    n_unique = len(vocab[fam]["frequencies"])
    ok = "✅" if n_unique == EXPECTED_COUNTS[fam] else "⚠️"
    lines.append(f"### `{fam}` — {n_unique} unique labels {ok} (expected {EXPECTED_COUNTS[fam]})")
    lines.append("")
    lines.append(md_freq_table(fam))
    lines.append("")

lines.append("## 3. Class imbalance")
lines.append("")
for fam in FAMILIES:
    info = imbalance_info[fam]
    lines.append(f"### `{fam}`")
    lines.append(f"- Most frequent (non-none): `{info['most'][0]}` — {info['most'][1]:,} samples "
                 f"({100*info['most'][1]/N:.2f}%)")
    lines.append(f"- Least frequent (non-none): `{info['least'][0]}` — {info['least'][1]:,} samples "
                 f"({100*info['least'][1]/N:.2f}%)")
    lines.append(f"- Max/min frequency ratio: **{info['ratio']:.1f}×**")
    if info["ratio"] > 50:
        suggested = "Asymmetric Loss (ASL) + class-aware reweighting; consider Distribution-Balanced loss"
    elif info["ratio"] > 10:
        suggested = "ASL or focal loss on positives; per-class thresholds at inference"
    else:
        suggested = "BCE may be adequate; still recommend per-class threshold tuning"
    lines.append(f"- Suggested treatment: {suggested}")
    lines.append("")
lines.append("Also note the dominance of `none` (no-change) labels in every family — the model "
             "must learn an implicit \"no change\" prior, and changeflag prediction is essentially "
             "an auxiliary task already.")
lines.append("")

lines.append("## 4. Per-split distribution drift")
lines.append("")
lines.append("Drift score = (max-min)/mean across the three splits, where each value is the "
             "percentage of samples in a split that contain the label. Labels with mean < 0.05% "
             "are excluded (too rare to be reliable).")
lines.append("")
any_serious = False
for fam in FAMILIES:
    lines.append(f"### `{fam}` — top 3 drifted labels")
    lines.append("")
    lines.append("| label | rel_spread | train% | val% | test% | flag |")
    lines.append("|---|---:|---:|---:|---:|:---:|")
    for lbl, rel, vals in drift_findings[fam]:
        flag = "⚠️" if rel > 50 else "OK"
        if rel > 50:
            any_serious = True
        lines.append(f"| `{lbl}` | {rel:.1f}% | {vals['train']:.2f} | {vals['val']:.2f} | "
                     f"{vals['test']:.2f} | {flag} |")
    lines.append("")
if any_serious:
    lines.append("**Verdict:** at least one label has >50% relative spread across splits — "
                 "use stratified resampling for hyperparameter tuning, and report per-class "
                 "metrics rather than relying on global mAP.")
else:
    lines.append("**Verdict:** no label exhibits >50% relative spread. The splits are well "
                 "balanced for label-level evaluation.")
lines.append("")

lines.append("## 5. Multi-label intensity")
lines.append("")
lines.append("Number of non-`none` labels per sample, per family. `all` includes no-change "
             "samples (forced to zero), `changed` restricts to `changeflag=1`.")
lines.append("")
lines.append("| family | scope | mean | median | max | std |")
lines.append("|---|---|---:|---:|---:|---:|")
for fam in FAMILIES:
    for scope in ("all", "changed"):
        s = intensity_stats[fam][scope]
        lines.append(f"| `{fam}` | {scope} | {s['mean']:.3f} | {s['median']:.0f} | {s['max']} | {s['std']:.3f} |")
lines.append("")

lines.append("## 6. Co-occurrences")
lines.append("")
for fam in FAMILIES:
    lines.append(f"### `{fam}` — top 5 co-occurring pairs")
    lines.append("")
    if not cooc_notes[fam]:
        lines.append("_No multi-label samples in this family — only singletons._")
        lines.append("")
        continue
    lines.append("| label A | label B | # samples |")
    lines.append("|---|---|---:|")
    for a, b, c in cooc_notes[fam]:
        lines.append(f"| `{a}` | `{b}` | {c:,} |")
    lines.append("")
lines.append("Strong off-diagonal mass indicates exploitable label correlations — candidate "
             "for ML-GCN / C-Tran / Query2Label-style heads that model dependencies.")
lines.append("")

lines.append("## 7. Image properties")
lines.append("")
lines.append(f"Sampled {img_stats['filesize_bytes']['n_sampled']} images for resolution and "
             "file-size measurement.")
lines.append("")
lines.append("**Resolution distribution (top entries):**")
lines.append("")
lines.append("| resolution (WxH) | # images (in sample) |")
lines.append("|---|---:|")
for res, cnt in list(img_stats["resolution_distribution"].items())[:10]:
    lines.append(f"| {res} | {cnt:,} |")
lines.append("")
fs = img_stats["filesize_bytes"]
lines.append(f"**File-size stats (bytes):** mean = {fs['mean']:.0f}, median = {fs['median']:.0f}, "
             f"min = {fs['min']:,}, max = {fs['max']:,}.")
lines.append("")
if len(img_stats["resolution_distribution"]) == 1:
    lines.append("All inspected images share a single resolution — safe to assume a fixed input "
                 "size in the data loader.")
else:
    lines.append("Multiple resolutions observed — the data loader must resize/crop to a uniform "
                 "size before feeding the encoder.")
lines.append("")

lines.append("## 8. Augmentation breakdown")
lines.append("")
lines.append("| split | original | random_augment | ters | ters+aug | total |")
lines.append("|---|---:|---:|---:|---:|---:|")
for s in SPLITS:
    row = [cat_split[s].get(c, 0) for c in CATS]
    total = sum(row)
    lines.append(f"| `{s}` | {row[0]:,} | {row[1]:,} | {row[2]:,} | {row[3]:,} | {total:,} |")
totals_row = [cat_total.get(c, 0) for c in CATS]
lines.append(f"| **total** | **{totals_row[0]:,}** | **{totals_row[1]:,}** | "
             f"**{totals_row[2]:,}** | **{totals_row[3]:,}** | **{N:,}** |")
lines.append("")
lines.append("`random_augment` and `_ters_` are pre-baked augmentations in the dataset folder. "
             "If your training pipeline also applies on-the-fly augmentation, ensure you don't "
             "double-augment. The `_ters_` reversed pairs are particularly important to keep "
             "in the same split as their originals to avoid information leakage; verify this "
             "with the split assignment script before training.")
lines.append("")

lines.append("## 9. Implications for modeling")
lines.append("")
lines.append("- TODO: discuss with Claude after EDA")
lines.append("")

(OUT / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
print(f"  -> {OUT / 'eda_summary.md'}")

# Persist intensity + drift + image stats for downstream code reuse
extras = {
    "intensity": intensity_stats,
    "drift_top3": {fam: [{"label": lbl, "rel_spread_pct": rel, "values_pct": vals}
                         for lbl, rel, vals in drift_findings[fam]] for fam in FAMILIES},
    "image_stats": img_stats,
    "augmentation": {
        "totals": dict(cat_total),
        "per_split": {s: dict(cat_split[s]) for s in SPLITS},
    },
    "cooccurrence_top5": {fam: [{"a": a, "b": b, "n": c} for a, b, c in cooc_notes[fam]]
                          for fam in FAMILIES},
    "diff_nochange_stats": diff_stats,
}
with (OUT / "extras.json").open("w") as f:
    json.dump(extras, f, indent=2, ensure_ascii=False)
print(f"  -> {OUT / 'extras.json'}")

print("\nEDA COMPLETE — outputs in results/eda/")
