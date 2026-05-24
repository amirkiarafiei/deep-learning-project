# AGENTS.md

Context document for AI coding assistants (Claude Code, etc.) working on this project. Read this first before touching any code.

## What this project is

A course project for **BLM5135 — Deep Learning and Artificial Neural Networks** (Yıldız Technical University). The task is **multi-label classification of bi-temporal remote sensing image pairs**: given two co-registered aerial RGB images of the same scene at different times (a "before" image and an "after" image), predict what changed between them across three label families simultaneously.

**Three label families** (independent multi-label outputs, sigmoid not softmax):
- **Object** (13 labels including `none`): `building`, `road`, `tree`, `field`, `water`, `vegetation`, `parking`, `land`, `roof`, `asphalt`, `green`, `plant`, `none`.
- **Event** (13 labels including `none`): `build`, `remove`, `turn`, `appear`, `replace`, `change`, `destroy`, `increase`, `vegetate`, `add`, `surround`, `remain`, `none`.
- **Attribute** (25 labels including `none`): `blue`, `gray`, `green`, `large`, `huge`, `black`, `white`, `more`, `small`, `brown`, `empty`, `bare`, `lush`, `middle`, `red`, `residential`, `long`, `industrial`, `adjacent`, `sparse`, `dense`, `paved`, `same`, `dark`, `none`.

The `none` label means "no change in this family", and is perfectly correlated across families with `changeflag == 0`. Decision in this project: predict an auxiliary `changeflag` head separately, and have the family heads only predict the real labels (12 + 12 + 24 = 48 outputs total). At inference, if `changeflag` < threshold, output `none` for all families.

**Dataset:** 10,855 image pairs at 256×256 resolution, derived from LEVIR-CC (Liu et al., TGRS 2022). Splits are 77.7% train / 11.0% val / 11.3% test, with identical change-flag distribution (~72% changed) across splits. The professor pre-augmented the data: ~48% are originals, ~44% are `_random_augment` copies, ~14% are `_ters_` (reversed A↔B pairs). Crucially, the test split contains *only* originals and ters pairs — no random augments — so test is the cleanest distribution. See `results/eda/eda_summary.md` for the full picture.

## What we're building (Track 1)

A safe, well-tested baseline that satisfies every item in the course rubric:

- **Phase 1 (worth 75% of project grade):** Three *separate* models, one per label family. Each is a Siamese encoder + fusion + single classification head. We analyze and report which family is hardest to learn.
- **Phase 2 (worth 25%):** One *joint* model with a shared encoder and three parallel heads. We compare against Phase 1 and report whether multi-task learning helps each family.

We have a follow-on Track 2 planned (more advanced methods aimed at publication) but Track 2 is *not in scope for the current implementation*. Stick to Track 1 unless the user explicitly says we're moving to Track 2.

The architecture spec for Track 1 is in `docs/track1.md`. Read it before implementing.

## What the architecture looks like

At the highest level:

```
img_A ──┐
        ├── Siamese backbone (shared weights, ConvNeXt-Tiny) ── feat_A, feat_B
img_B ──┘                                                         │
                                                                  ▼
                              fusion: concat[A, B, A−B, |A−B|] → 1x1 conv → GAP
                                                                  │
                                                                  ▼
                                              ┌─── changeflag head (binary)
                                              ├─── object head     (12 logits, Phase 1 single / Phase 2 all)
                                              ├─── event head      (12 logits)
                                              └─── attribute head  (24 logits)
```

Phase 1 instantiates one family head at a time. Phase 2 instantiates all three. The `changeflag` head is always present.

Loss: `BCEWithLogitsLoss` with per-class `pos_weight` to handle class imbalance. Loss aggregation across families in Phase 2: naive sum (no GradNorm or Kendall uncertainty weighting in Track 1).

## What we explicitly are NOT doing (in Track 1)

These are reserved for Track 2 and should not be added now even if they seem like easy wins:

- Foundation-model backbones (DINOv2, RemoteCLIP, SAM2). ConvNeXt-Tiny only.
- Asymmetric Loss, focal loss, distribution-balanced loss. Plain BCE with `pos_weight` only.
- Per-class threshold tuning. Fixed 0.5 only.
- Cross-attention or transformer fusion modules. Feature concat + difference only.
- Advanced multi-task loss balancing (GradNorm, PCGrad, Kendall uncertainty).
- Test-time augmentation, ensembling, EMA, SWA.
- Optical-flow A↔B alignment.
- Query2Label / ML-Decoder heads. Linear heads only.
- Heavy on-the-fly augmentation. The professor's pre-augmentation is enough; we add only horizontal flip (applied identically to A and B).

If the implementation tempts you to add one of these, **stop and ask the user first**. Track 2 will revisit these one at a time so we can measure their contribution properly.

## Important context from the EDA

These facts shape several architectural choices and should be respected during implementation and reporting:

- **Class imbalance is severe.** Object family: `building` is 64.7% of all samples, `plant` is 0.24%. Macro-F1 will be dragged down by the long tail no matter what we do. Per-class metrics matter more than aggregate F1.
- **The label vocabulary is loaded from `results/eda/label_vocab.json`** — this file is canonical. The order of labels in any tensor or output table must match this file. The `none` label is index 0 in the file; when building the 12/12/24-dim multi-hot vectors, skip index 0 and shift the rest down by 1.
- **"No-change" pairs aren't pixel-identical.** Mean pixel-level abs-diff on no-change pairs ranges 0.09–0.23. This is seasonal / illumination variation, not registration error. Implication: pixel-level differencing is a bad fusion strategy; we fuse at the feature level (which the spec already specifies).
- **Some labels have ≤ 100 training samples** (asphalt 36, green 26, plant 22 in train; dark = 0 in val). For these, F1 ≈ 0 is expected and not a model failure. Report it honestly.
- **All images are 256×256.** We resize to 224×224 to match ImageNet-pretrained backbones. No aspect-ratio gymnastics needed.
- **Val is somewhat duplicated** (contains both originals and random_augment copies of the same samples) so val metrics may be ~1–2 points optimistic relative to test. Test contains no random_augment, so it's the honest distribution to report.

## How to work with this repository

- **Compute:** Training happens on Google Colab Pro (A100 40GB). The user's local machine is for code, EDA, and visualization only. Don't try to train locally.
- **Project tree:** Already mostly set up. Code lives in `src/`, top-level CLI entry points (`train.py`, `eval.py`, `visualize.py`) at the repo root, configs in `configs/`, experiment outputs in `results/track1/`. See `docs/track1.md` for the full layout.
- **Framework:** PyTorch + timm. Plain PyTorch training loops, not Lightning. bf16 mixed precision on A100. No `einops`, no `kornia`, no fancy abstractions.
- **Determinism:** Seed everything from config (default 42). The user expects reproducible numbers between runs.
- **Logging:** All experiments write text logs to `results/track1/<run>/logs/*.txt`. The course rubric specifically requires console + .txt logs of training output.
- **Outputs go under `results/track1/`** — never write outputs to the project root or to `dataset/`.

## How to communicate with the user

- The user is competent but not a deep learning expert. Explain *why* you make choices, not just *what* you're doing.
- When something is ambiguous in the spec, ask before guessing. Cheap to ask, expensive to redo.
- The user will run training on Colab and paste back logs / numbers. Make code that produces *legible, copy-pasteable* console output.
- The user wants a high grade and (separately) a publishable result. The high grade is primary; the publishable result depends on Track 2 work after the course deadline (May 31, 2026). Don't compromise Track 1 for Track 2 aspirations.

## Background context you might want to know

- **The data is a derivative of LEVIR-CC** (a remote sensing change captioning dataset). The professor decomposed the captions into structured (object, event, attribute) multi-label triples to create this classification task. This task formulation appears to be novel — no published paper does exactly this on this data. This is what makes Track 2 a potential publication target later.
- **Three independent literature reviews are in `docs/literature/`** (Codex, Flash, Opus versions). They were generated by different AI assistants; the user verified that the papers cited do exist on arXiv. Use them as a reference when writing the report's related-work section but don't cite uncritically — paraphrase findings and check claims against the original papers when possible.
- **Three deliverables are due May 31:** code (modular, runnable), IEEE-format report (5 pages max), and presentation slides (6 min talk on June 5/12). Code and report quality are equally weighted (each ~50% of project grade).

## Reading list before you start

1. `docs/track1.md` — full implementation spec (architecture, file layout, conventions, defaults).
2. `results/eda/eda_summary.md` — what the data actually looks like.
3. `results/eda/label_vocab.json` — the canonical label index mapping.
4. `docs/clasroom/classroom.md` — the original course assignment (in Turkish; the gist is in this file's "What this project is" section).
5. This file (you're reading it).

Don't read all three literature reviews end-to-end before implementing — they're reference material, not requirements.

## Quick test before assuming things work

After implementing the dataset class but before training anything:

1. Instantiate `ChangeDataset(split='train')` and fetch sample 0. Print all dict keys and tensor shapes.
2. Verify `image_A.shape == image_B.shape == (3, 224, 224)` and labels are float multi-hot tensors with correct dimensions (12 / 12 / 24).
3. Verify that the encoded labels round-trip through `MultiLabelEncoder.decode` back to the original label list.
4. Verify that for a `_ters_` sample, the A/B images really are the reversed-order version (you can spot-check visually).

These are 5 minutes of verification that save hours of mid-training debugging.