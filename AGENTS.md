# AGENTS.md

Project context for AI coding agents. Read this first; for full implementation details read `docs/track1.md`.

## What this project is

Multi-label classification of bi-temporal remote sensing image pairs. Given two co-registered RGB aerial images (A = before, B = after), predict what changed across three label families:

- **Object** (12 classes after dropping `none`): building, tree, road, field, vegetation, water, parking, land, roof, asphalt, green, plant
- **Event** (12 classes): build, remove, turn, appear, replace, change, destroy, increase, vegetate, add, surround, remain
- **Attribute** (24 classes): blue, gray, green, large, huge, black, white, more, small, brown, empty, bare, lush, middle, red, residential, long, industrial, adjacent, sparse, dense, paved, same, dark

`none` is handled as a separate binary `changeflag` head, not as a per-family class — see `docs/track1.md` § Architecture for the rationale.

Course: BLM5135 Deep Learning (Yıldız Technical University). Deadline: **2026-05-31**. The dataset is ~10k pairs derived from LEVIR-CC.

## Where things live

- `src/` — library code (data, models, losses, training, utils)
- `train.py` / `eval.py` / `visualize.py` — thin CLI entry points; logic lives in `src/`
- `configs/` — one YAML per experiment (`track1_object.yaml`, `track1_event.yaml`, `track1_attribute.yaml`, `track1_multitask.yaml`)
- `results/track1/<run>/` — checkpoints, logs, metrics, plots, qualitative examples
- `results/eda/` — pre-computed dataset stats; **`label_vocab.json` is canonical** for label ordering
- `docs/track1.md` — full architecture and implementation spec (read this before coding)
- `docs/literature/` — three independent literature reviews; reference material, not requirements

## Where compute runs

Training runs on **Google Colab Pro (A100)**, not locally. Use the `colab` CLI via bash (see the `colab-session-operator` skill for full reference).

Dataset lives at `MyDrive/dataset/` on Google Drive. Standard session prep: `colab drivemount`, then read directly from `/content/drive/MyDrive/dataset/`. Don't copy the dataset every session.

The user's laptop is for code, EDA, and visualization only. Don't train locally.

## Stack

PyTorch + timm. Plain PyTorch training loops, no Lightning. bf16 mixed precision on A100. No `einops`, no `kornia`. See `docs/track1.md` for backbone, fusion, loss, and optimizer specifics.

## Boundaries

**Track 1 only** unless the user explicitly switches to Track 2. The following are reserved for Track 2 and must not be added now (the point of Track 1 is a clean baseline we can measure against later):

- Foundation backbones (DINOv2, RemoteCLIP, SAM)
- Asymmetric Loss, focal loss, distribution-balanced loss — plain BCE with `pos_weight` only
- Per-class threshold tuning — fixed 0.5
- Cross-attention or transformer fusion — concat + diff only
- Advanced multi-task balancing (GradNorm, PCGrad, Kendall uncertainty) — naive sum only
- Test-time augmentation, ensembling, EMA, SWA
- Query2Label / ML-Decoder heads — linear heads only

If tempted to add one of these, ask first.

## House rules

- Seed everything from config (default 42); reproducibility matters.
- All experiments write text logs to `results/track1/<run>/logs/*.txt` — course rubric requires it.
- Outputs go under `results/track1/`, never to repo root or `dataset/`.
- `pathlib.Path` over `os.path.join`. `argparse` only in top-level scripts.
- Always `colab stop` when done; idle VMs burn compute units.

## Required submission files

The course rubric deducts 10 points each if missing:

- `ReadMe.txt` — exact filename, capital R and M. Plain text in Turkish. Describes project, dependencies, and how to run training/eval/visualization. Mirror content from `README.md` but keep concise.
- `requirements.txt` — pinned versions. Generate with `pip freeze` after env is set up; don't hand-write.

## How to communicate

- The user is competent but not a DL expert. Explain *why*, not just *what*.
- Ask when ambiguous; cheap to ask, expensive to redo.
- High course grade is the primary goal. Paper publication (Track 2) is secondary and post-deadline.

## Maintenance

This file is for durable, universally-applicable context. Project-specific changes (new files, new conventions) belong here only if they survive months. Anything fast-moving belongs in `docs/track1.md` or commit messages. If this file exceeds ~100 lines, refactor — long instruction files get ignored.