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

**Primary (full training runs):** Google Colab Pro (A100) via the `colab` CLI (see the `colab-session-operator` skill).

**Local fallback (debug + small runs):** the user's laptop has an **RTX 4060 (8 GB VRAM) + 64 GB RAM**. Usable for smoke tests, dataloader debugging, single-batch overfit checks, and small-subset training. ConvNeXt-Tiny at batch 32–64, 224×224 fits comfortably. Don't run the full 30-epoch training plan locally — that's what Colab A100 is for.

**Dataset paths:**
- On Colab: `/content/drive/MyDrive/dataset/` after `colab drivemount`. Don't copy the dataset every session.
- Locally: `dataset/` in the repo (gitignored).

### Budget hygiene (~200 Colab credits)

Credits are finite. Burn them only on runs that produce usable results.

- **Local first for light work, Colab when you need real GPU throughput.** The 4060 handles debug iterations cheaper than any Colab session.
- **CPU Colab sessions for non-training work on Colab** (smoke tests, vocab inspection, Drive I/O checks). ~50× cheaper than A100.
- **Right-size GPU per task, not smallest-possible.** User prefers faster wall-clock over saved credits when the speedup is significant — don't downgrade an A100 run to T4 just to save units if A100 cuts training from 4 h to 1 h. Rough guide: T4 for subset sanity runs, A100 for full Track-1 training runs.
- **High-end GPUs (H100, A100-80GB) need explicit user approval before provisioning.** They're available on Colab Pro but pricey. Ask first when the task is computation-heavy — full multi-hour training, hyperparameter sweep, large-scale ablation. No need to ask for routine debug runs, smoke tests, or single-config Track-1 training (A100 is fine by default).
- **One Colab session at a time.** `colab stop -s <previous>` before `colab new -s <next>`.
- **`colab stop` immediately when a task is done** — including before context-switching. Idle VMs burn credits with nothing executing.
- **Smoke-test (locally or CPU) before paying for a GPU session.** Run end-to-end on a ~200-sample subset, then provision the GPU.
- **Verify the GPU before training.** Exec `nvidia-smi`; if Colab handed you the wrong device, stop and re-provision.
- **All training outputs go to Drive, never `/content/`.** `/content/` is wiped on session loss. Use `--output-dir /content/drive/MyDrive/dl_project_outputs/results/track1/<run>/` so checkpoints, logs, history.csv, and curves all survive a crash.
- **Cache the timm pretrained weights to Drive** by exporting `HF_HOME=/content/drive/MyDrive/dl_project_outputs/hf_cache` *before* the first `python train.py` invocation. ConvNeXt-Tiny (~110 MB) then survives session loss and re-runs across configs don't re-download.
- **Trainer writes `last.pt` + `history.csv` + curves after every epoch** and flushes logs per record. Resume any dropped run with `python train.py --config ... --resume <output_dir>/checkpoints/last.pt` — reloads model + optimizer + scheduler + best-so-far + early-stop counter + history; loss bounded to one epoch. Keep `--epochs` consistent across original and resumed runs (the cosine schedule's `T_max` is set at construction).

Guardrails — stop and ask before provisioning if either applies:
- The task hasn't been smoke-tested (locally or on CPU) yet.
- The task is computation-heavy enough to want a higher GPU tier (H100 / A100-80GB) or will run for several hours.

## Stack

PyTorch + timm. Plain PyTorch training loops, no Lightning. bf16 mixed precision on A100. No `einops`, no `kornia`. See `docs/track1.md` for backbone, fusion, loss, and optimizer specifics.

## Boundaries

**Track 1 only** unless the user explicitly switches to Track 2. Track 1 has its own internal iterations — Phase 1 (`v1.0.0`, shipped 2026-05-26) and Phase 2 (`v2`, planned) — both within Track 1 scope; see `docs/track1.md` § Phase structure for the map. Phase 2 stays inside Track 1; do not confuse it with Track 2.

The following are reserved for Track 2 and must not be added in either Phase 1 or Phase 2 of Track 1 (the point of Track 1 is a clean baseline we can measure against later):

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