# Track 3 — Foundation-Model Representations + Change-Aware Attention Fusion (v5)

**Status:** spec, not yet implemented.
**Last revised:** 2026-05-28.
**Companion docs:** [`track4.md`](track4.md) (v6 = state-space-model path), [`track1.md`](track1.md) (v1+v2 baseline), [`literature_v6.md`](literature_v6.md) (shared literature survey).

> **Read `AGENTS.md` first.** This spec assumes the project context, dataset, and Track 1 baseline are already understood.

---

## Phase summary

| Track / version | Status | Headline |
|---|---|---|
| Track 1 v1.0.0 | ✅ shipped 2026-05-26 | CNN baseline, test avg macro F1 **0.270** (winner so far) |
| Track 1 v2.0.0 | ✅ shipped 2026-05-26 | + regularization, 0.246 (tuned) |
| Track 2 v3 | ✅ shipped 2026-05-27 | A3 Hybrid (LDAM-DRW + cRT + LA + cleanup) — failed, 0.175 |
| Track 2 v4 | ✅ shipped 2026-05-27 | A2 isolation (cleanup + v1 recipe) — tied with v1 (0.268) |
| **Track 3 v5** | 📋 planned (this doc) | Foundation backbone + cross-attention fusion + transformer multi-label head |
| Track 4 v6 | 📋 planned (`track4.md`) | Vision State-Space Model + interleaved bi-temporal SSM + RAL |

---

## Goal

Test the hypothesis that *representation quality is the bottleneck*. v1–v4 all use ConvNeXt-Tiny ImageNet-1k pretraining (28M params, 1.4M training images). Foundation models pretrained at SSL scale on 100M+ images have demonstrably stronger features. If representations are the bottleneck on this dataset, a frozen foundation backbone + a tiny trainable head should lift macro F1 above v1's 0.270.

A secondary, complementary bet: **cross-attention fusion** explicitly models "what changed" as `Q_from_A · K_from_B^T`, instead of v1's concat-difference (which throws features into a linear projection and hopes). This is the inductive bias the field has converged on for bi-temporal change tasks.

A tertiary bet: **transformer multi-label head (ML-Decoder)** treats classes as learnable queries that attend to feature tokens — natively handles multi-label co-occurrence, instead of v1's sigmoid-on-pooled-features.

---

## Architecture (v5)

```
RGB_A ─┐
       ├─► DINOv2-Base (frozen) ─►  patch tokens A (256 × 768) — at 224 input, patch size 14
RGB_B ─┘                            patch tokens B (256 × 768)
                                              │
                                              ▼
         ┌───────────────────────────────────────────────────────┐
         │  CROSS-ATTENTION FUSION                               │
         │  ─ block 1: Q from A, K/V from B  →  A_cond (B-aware) │
         │  ─ block 2: Q from B, K/V from A  →  B_cond (A-aware) │
         │  ─ diff token = (A_cond − B_cond) projected to 768    │
         │  ─ 2 transformer encoder layers on [A_cond; B_cond; D]│
         │  d_model = 768, n_heads = 8, ffn = 4×                 │
         │  Trainable params here: ~3M                           │
         └───────────────────────────────────────────────────────┘
                                              │
                                              ▼
                          ┌─────────────────────────────────────┐
                          │  ML-DECODER per family              │
                          │  (transformer decoder; class queries│
                          │   attend to fusion-output tokens →  │
                          │   per-class sigmoid)                │
                          │  ─ Object:    12 queries → 12 logits│
                          │  ─ Event:     12 queries → 12 logits│
                          │  ─ Attribute: 24 queries → 24 logits│
                          │  + Changeflag: 1 learnable query    │
                          │  Trainable params here: ~1M total    │
                          └─────────────────────────────────────┘
```

Total trainable parameters: ~5M (out of ~91M total — DINOv2-Base is 86M frozen).

### Component rationale

#### Backbone — DINOv2-Base frozen

- **Model.** `dinov2_vitb14` from [`torch.hub`](https://github.com/facebookresearch/dinov2) (or `facebook/dinov2-base` via HuggingFace `transformers`).
- **Pretraining.** Self-supervised on LVD-142M images via DINOv2's distillation objective [Oquab et al. 2024]. 142M images vs ConvNeXt-Tiny's 1.4M ImageNet-1k.
- **Why frozen.** With only 8,438 training samples and a 91M-param backbone, fine-tuning would overfit. Freezing the backbone means *only ~5M params train*, dramatically reducing overfitting risk while keeping the foundation-quality features. This recipe (frozen foundation + small adapter) won MIDOG 2025 Task 2 [Balezo et al. 2025].
- **Output.** 256 patch tokens at 768-d (for 224×224 input with patch 14). We drop the `[CLS]` token; downstream attention can compose its own global representation.
- **Inductive bias.** Attention-everywhere on patches; learned without supervision so the representations are more general than supervised classification pretraining.

#### Fusion — Cross-Attention

- **Mechanism.** Two cross-attention blocks: one queries A with B as key/value (produces "A seen through B's lens"), the other queries B with A. Difference of the two is the change signal. Then two self-attention encoder layers integrate.
- **Why not concat-diff (v1)?** Concat-difference is an ablation-baseline — it works but is generic. Cross-attention has *explicit alignment*: for each query position in A, it looks for the most semantically similar position in B and forms the residual. This is closer to how a human compares two scenes.
- **Citations.** Cross-attention bi-temporal feature fusion is the dominant paradigm in recent change-detection literature: CSTSUNet [Wu et al. 2023], MIFNet [Xie et al. 2025], SCADNet [Xu et al. 2022].

#### Head — ML-Decoder

- **Mechanism.** Transformer decoder where each *class* is represented by a learnable query embedding. The query attends to the fusion-output tokens via cross-attention, then a single linear projection produces the per-class logit. One ML-Decoder per family.
- **Why not sigmoid on pooled features (v1)?** Pooling discards spatial information; a learned class query can find its evidence wherever it lies. ML-Decoder also implicitly models label co-occurrence via decoder self-attention between query embeddings.
- **Citations.** ML-Decoder [Ridnik et al. 2023, WACV] reports gains on every standard multi-label benchmark (COCO-ML, VOC-ML, Open Images, NUS-WIDE) over sigmoid-on-features. Related family: Query2Label [Liu et al. 2021], C-Tran [Lanchantin et al. 2021].

#### Loss

Plain `BCEWithLogitsLoss` with per-class `pos_weight` clamped `[1, 50]`, identical to v1. We're not changing the loss because the failure mode we want to test (representation quality) is orthogonal to loss-side intervention, and v3 already showed that stacking LDAM-DRW + cRT + LA on weakened pretraining only makes things worse.

If v5 plateaus, a follow-on ablation with Robust Asymmetric Loss [Park et al. 2023] is cheap to add — but the first run is the cleanest test of the representation hypothesis.

---

## Course-rubric compliance

The course mandates Aşama 1 (three single-task models, weight 75%) and Aşama 2 (one multi-task model with shared encoder, weight 25%).

- **Aşama 1** — three single-task variants, each with one family's ML-Decoder head only. The frozen DINOv2 backbone + cross-attention fusion are shared design but trained separately. Each ~30 minutes on A100 (only ~3.5M params train per variant).
- **Aşama 2** — one multi-task variant: the same shared fusion plus three ML-Decoders for the three families simultaneously. ~1 hour A100.
- **Architecture diagram** to draw in the IEEE report: the ASCII figure above. No fundamental difference from the v1 diagram except the boxes are replaced by their v5 equivalents.

---

## Implementation plan

| Task | Effort | Files touched |
|---|---|---|
| Add DINOv2-Base loader (frozen) — wraps `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')` | ~1h | `src/models/dinov2_backbone.py` (new) |
| Implement cross-attention bi-temporal fusion (two `nn.MultiheadAttention` calls + 2 encoder layers) | ~2h | `src/models/cross_attn_fusion.py` (new) |
| Implement ML-Decoder (~150 LOC; reference: official Ridnik repo) | ~2h | `src/models/ml_decoder.py` (new) |
| Wire into `ChangeClassifier` via a new `backbone_kind="dinov2"` / `fusion_kind="cross_attn"` / `head_kind="ml_decoder"` config branch | ~1h | `src/models/change_classifier.py` (extend) |
| Write 4 configs: `track3_v5_{object,event,attribute,multitask}.yaml` | ~30 min | `configs/` |
| Add 2 Colab cells: v5 train + v5 eval, mirroring v3.7 / v3.8 structure | ~30 min | `colab_runbook.ipynb` |
| **Total** | **~7h dev** | |
| Training (3 single-task @ ~30 min + 1 multi-task @ ~1h) | **~2.5h A100** | |

---

## Risks and failure modes

1. **DINOv2 input resolution.** Default DINOv2 expects 224×224 with patch 14 → 16×16 = 256 tokens. Our images are originally 256×256 — we resize to 224 to match. *Should* be fine.
2. **Frozen backbone with BatchNorm.** None — DINOv2 uses LayerNorm. Safe to freeze.
3. **Cross-attention with long sequences.** 256 tokens × 2 = 512-token sequence to the 2 encoder layers. Cheap for an 8-head attention block.
4. **Cold-start ML-Decoder queries.** Class queries are randomly initialized. Convergence is usually fast (~10 epochs) per the original paper. Use lr=1e-3 for queries, 1e-4 for fusion.
5. **Course-rubric "Aşama 1 is single-task" concern.** Aşama 1 demands *independent* models per family. Our three v5 single-task variants share the *frozen* backbone (no information leakage during training, since DINOv2 doesn't update) but train independent fusions and heads. This is defensible — the frozen backbone is just feature extraction, no different from using ImageNet features. If the grader objects, v1's three truly-independent single-task models are also on file.

---

## Expected outcome

| Scenario | Test avg macro F1 | Probability |
|---|---|---|
| Best | 0.32 – 0.40 (representation quality breaks the ceiling) | ~25% |
| Likely | 0.28 – 0.31 (small lift over v1) | ~55% |
| Worst | 0.24 – 0.27 (foundation features can't compensate for label noise) | ~20% |

Most likely outcome: a small but real lift (likely range), which is enough to make v5 the headline of Track 3 and the basis for the Track 3 IEEE report.

---

## Publishable angle

- **Positive:** *"Foundation vision-model representations and cross-attention bi-temporal fusion improve multi-label change classification under extreme long-tail imbalance on LEVIR-CC-derived data."*
- **Mixed:** *"Foundation backbones lift mid-frequency classes but cannot compensate for tail-class label noise on LEVIR-CC; the ~0.30 macro F1 ceiling appears to be partly representation-driven and partly annotation-driven."*
- **Negative:** *"DINOv2 with cross-attention fusion matches but does not exceed strong CNN baselines on LEVIR-CC-derived multi-label change classification, indicating the data-side (annotation, vocabulary) is the dominant ceiling."*

Track 3 + Track 4 (v6, the SSM path in `track4.md`) together form the cross-architectural triangulation that lets us speak with authority about the ceiling.

---

## References (this track)

- **Oquab et al. 2024.** *DINOv2: Learning Robust Visual Features without Supervision.* Transactions on Machine Learning Research.
- **Ridnik et al. 2023.** *ML-Decoder: Scalable and Versatile Classification Head.* WACV.
- **Liu et al. 2021.** *Query2Label: A Simple Transformer Way to Multi-Label Classification.* arXiv:2107.10834.
- **Lanchantin et al. 2021.** *General Multi-Label Image Classification with Transformers (C-Tran).* CVPR.
- **Wu et al. 2023.** *CSTSUNet: A Cross Swin Transformer-Based Siamese U-Shape Network for Change Detection.* IEEE TGRS.
- **Xu et al. 2022.** *SCAD: A Siamese Cross-Attention Discrimination Network for Bitemporal Building Change Detection.* Remote Sensing 14(24).
- **Xie et al. 2025.** *MIFNet: Multi-Scale Interaction Fusion Network for Remote Sensing Image Change Detection.* IEEE TCSVT.
- **Balezo et al. 2025.** *Efficient Fine-Tuning of DINOv3 Pretrained on Natural Images for Atypical Mitotic Figure Classification (MIDOG 2025 Task 2 Winner).* arXiv:2508.21041.

See also the consolidated literature survey in [`literature_v6.md`](literature_v6.md) for the methodology (queries, papers reviewed) shared between Tracks 3 and 4.
