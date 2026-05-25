"""Thin CLI: produce qualitative example PNGs.

Each PNG shows: rgb_A | rgb_B | a right-side text block with sample metadata,
ground-truth labels per family, and predicted labels with sigmoid scores
formatted as ``label (0.xxx)`` — matches Image 2 of the project PDF.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from src.data.dataset import ChangeDataset
from src.data.label_encoder import build_encoders
from src.data.transforms import build_transforms
from src.models.classifier import ChangeClassifier
from src.utils.config import load_config
from src.utils.logging import build_logger
from src.utils.seed import seed_everything


def render_panel(
    img_a: Image.Image,
    img_b: Image.Image,
    record: dict,
    pred_text: str,
    gt_text: str,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [1, 1, 1.4]})
    axes[0].imshow(img_a); axes[0].set_title("RGB A"); axes[0].axis("off")
    axes[1].imshow(img_b); axes[1].set_title("RGB B"); axes[1].axis("off")
    axes[2].axis("off")
    meta = (
        f"sample_id: {record['sample_id']}\n"
        f"filename: {record['filename']}\n"
        f"split: {record['split']}\n"
        f"changeflag: {record['changeflag']}\n\n"
        f"Ground truth:\n{gt_text}\n\n"
        f"Predictions:\n{pred_text}"
    )
    axes[2].text(0.0, 1.0, meta, va="top", ha="left", family="monospace", fontsize=9)
    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Track 1 visualizer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(args.seed)

    output_dir = Path(cfg.output_dir)
    qual_dir = output_dir / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(f"{cfg.run_name}_viz", output_dir / "logs" / "visualize.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = build_encoders(("object", "event", "attribute"))

    transform = build_transforms(cfg.data.image_size, train=False)
    ds = ChangeDataset(
        cfg.data.json_path, cfg.data.dataset_root, args.split, transform, encoders=encoders
    )

    model = ChangeClassifier(
        families=cfg.model.families,
        include_changeflag=cfg.model.include_changeflag,
        pretrained_backbone=False,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    indices = list(range(len(ds)))
    random.shuffle(indices)
    chosen = indices[: args.num_samples]
    logger.info(f"Rendering {len(chosen)} samples from split={args.split}")

    pdf_path = qual_dir / f"qualitative_{args.split}.pdf"
    with PdfPages(pdf_path) as pdf:
        for i, idx in enumerate(chosen):
            record = ds.records[idx]
            raw_a = Image.open(Path(cfg.data.dataset_root) / record["rgb_A"]).convert("RGB")
            raw_b = Image.open(Path(cfg.data.dataset_root) / record["rgb_B"]).convert("RGB")

            sample = ds[idx]
            with torch.no_grad():
                out = model(
                    sample["image_A"].unsqueeze(0).to(device),
                    sample["image_B"].unsqueeze(0).to(device),
                )

            gt_lines, pred_lines = [], []
            for fam in cfg.model.families:
                gt_labels = record.get(f"{fam}_labels", [])
                gt_lines.append(f"  {fam}: {', '.join(gt_labels) if gt_labels else '(empty)'}")
                probs = torch.sigmoid(out[fam])[0].cpu()
                pairs = encoders[fam].decode_with_scores(
                    probs, threshold=cfg.evaluation.threshold, top_k=5
                )
                pred_str = ", ".join(f"{n} ({s:.3f})" for n, s in pairs) or "(none)"
                pred_lines.append(f"  {fam}: {pred_str}")
            if cfg.model.include_changeflag:
                cf_prob = float(torch.sigmoid(out["changeflag"])[0])
                pred_lines.append(f"  changeflag: {cf_prob:.3f}")

            fig = render_panel(
                raw_a, raw_b, record,
                pred_text="\n".join(pred_lines),
                gt_text="\n".join(gt_lines),
            )
            png_path = qual_dir / f"sample_{i:03d}_{record['sample_id']}.png"
            fig.savefig(png_path, dpi=120, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    logger.info(f"Wrote {len(chosen)} PNGs and a combined PDF: {pdf_path}")


if __name__ == "__main__":
    main()
