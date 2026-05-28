# Track 4 — Vision State-Space Model + Interleaved Bi-Temporal SSM + Robust Asymmetric Loss (v6)

**Status:** spec, not yet implemented.
**Last revised:** 2026-05-28.
**Companion docs:** [`track3.md`](track3.md) (v5 = DINOv2 path), [`track1.md`](track1.md) (v1+v2 baseline), [`literature_v6.md`](literature_v6.md) (shared literature survey).

> **Read `AGENTS.md` first.** This spec assumes the project context, dataset, and Track 1 baseline are already understood.

---

## Phase summary

| Track / version | Status | Headline |
|---|---|---|
| Track 1 v1.0.0 | ✅ shipped 2026-05-26 | CNN baseline, test avg macro F1 **0.270** (winner so far) |
| Track 1 v2.0.0 | ✅ shipped 2026-05-26 | + regularization, 0.246 (tuned) |
| Track 2 v3 | ✅ shipped 2026-05-27 | A3 Hybrid (LDAM-DRW + cRT + LA + cleanup) — failed, 0.175 |
| Track 2 v4 | ✅ shipped 2026-05-27 | A2 isolation (cleanup + v1 recipe) — tied with v1 (0.268) |
| Track 3 v5 | 📋 planned (`track3.md`) | DINOv2 frozen + cross-attention fusion + ML-Decoder |
| **Track 4 v6** | 📋 planned (this doc) | Vision Mamba (VMamba) backbone + interleaved bi-temporal SSM + Robust Asymmetric Loss |

---

## Cross-agent consensus on this architecture

Two independent code-aware assistants (GitHub Copilot CLI `gpt-5.2` and Google Gemini CLI `gemini-3.1-pro-preview`) were given the same self-contained briefing on the project state and asked: *"Recommend a single non-VLM v6 architecture that uses a different inductive bias than v5."* They returned independently and **converged on the same architectural family**: state-space models / Mamba.

| Aspect | Copilot (gpt-5.2) | Gemini (3.1-pro) | This spec |
|---|---|---|---|
| Architecture family | ChangeMamba-style SSM fusion | ChangeMamba-ST (Spatio-Temporal Vision Mamba) | **Vision Mamba family** |
| Encoder | ConvNeXt-Tiny Siamese (kept) | VMamba-Tiny / Vim-Small (replaced) | **VMamba-Tiny** — tests SSM hypothesis at every layer |
| Fusion | Mamba/CSSM blocks on `[fA, fB, fB−fA, abs(fB−fA)]` | Interleaved bi-temporal token sequence `[A_1, B_1, A_2, B_2, ...]` | **Interleaved sequence** — cleaner SSM-as-state-transition semantics |
| Loss | Plain BCE + pos_weight | **Robust Asymmetric Loss (RAL)** [Park et al. 2023] | **RAL** — different loss family than v3's failed LDAM-DRW |
| Best-case macro F1 | 0.33 – 0.36 | 0.35 – 0.38 | **0.32 – 0.38** |
| Likely outcome | 0.29 – 0.32 | 0.28 – 0.31 | **~0.30** |
| Worst case | 0.26 – 0.28 | ~0.22 | **~0.25** |

Both agents independently noted that even if v6 plateaus at ~0.30, the v1+v5+v6 triangulation (CNN / ViT / SSM all on the same dataset) is a complete cross-architectural ablation — *publishable regardless of the headline number.*

---

## Goal

Test the hypothesis that *the inductive bias for temporal evolution is the bottleneck*. v1–v4 use CNNs with concat-difference fusion. v5 uses ViT-style attention. **State-space models with selective recurrence treat the bi-temporal pair as a sequence transition** — the hidden state explicitly encodes "what changed from T1 to T2" via selective forget/retain dynamics. If this inductive bias matches change-classification better than attention or convolution, v6 will lift macro F1 above v5.

Secondary bet: **Robust Asymmetric Loss (RAL)** [Park et al. 2023] specifically designed for multi-label long-tail. Different family than v3's LDAM-DRW (which we proved fails). RAL operates at the per-class sigmoid output level via a polynomial asymmetric focal-style loss plus Hill regularization, not via margin subtraction on positive logits.

---

## Architecture (v6)

```
RGB_A ─┐
       ├─► VMamba-Tiny (ImageNet-1k pretrained), Siamese shared weights
RGB_B ─┘    output: stage-4 feature map, (B, 768, 7, 7) at 224 input
            │
            ▼
   Tokenize each side → 49 tokens × 768 ch
            │
            ▼
   ┌────────────────────────────────────────────────────────────┐
   │  INTERLEAVED BI-TEMPORAL SEQUENCE                          │
   │                                                            │
   │  [tA_1, tB_1, tA_2, tB_2, ..., tA_49, tB_49]  (length 98)  │
   │                                                            │
   │  Forces the SSM hidden state to encode the A→B transition  │
   │  at every spatial position. No cross-attention.            │
   └────────────────────────────────────────────────────────────┘
            │
            ▼
   ┌────────────────────────────────────────────────────────────┐
   │  6 × VSS (Visual State Space) BLOCKS                       │
   │  - LayerNorm → SS2D / 1-D selective SSM → MLP              │
   │  - Linear-time recurrence (O(n) not O(n²) like attention)  │
   │  - Each block: residual + LN                               │
   │  d_model = 768; total ~10 M params over the 6 blocks       │
   └────────────────────────────────────────────────────────────┘
            │
            ▼
   Global mean pool over the 98 tokens → 768-d change embedding
            │
            ▼
   ┌───────────────────────────────────────────────────────┐
   │ HEADS                                                 │
   │ - Object head:    Linear(768 → 12) + sigmoid + RAL    │
   │ - Event head:     Linear(768 → 12) + sigmoid + RAL    │
   │ - Attribute head: Linear(768 → 24) + sigmoid + RAL    │
   │ - Changeflag:     Linear(768 → 1)  + sigmoid + BCE    │
   └───────────────────────────────────────────────────────┘
```

Total trainable parameters: ~32M (VMamba-Tiny backbone is 22M; the 6 VSS fusion blocks add ~10M).

### Component rationale

#### Backbone — VMamba-Tiny

- **Model.** Visual State-Space backbone [Liu et al. 2024], 22M parameters, ImageNet-1k pretrained. Hierarchical 4-stage design analogous to Swin / ConvNeXt. Reference implementation: [`https://github.com/MzeroMiko/VMamba`](https://github.com/MzeroMiko/VMamba).
- **Why VMamba and not VMamba-Base/Large?** Tiny matches our compute budget (~2h A100 multi-task vs ~6h for Base). On ImageNet, Tiny is competitive with ConvNeXt-Tiny [Liu et al. 2024], so we're not losing baseline strength.
- **Inductive bias.** Selective SSM (S6 from Mamba [Gu & Dao 2023]) — each token's contribution to the hidden state is gated by input-dependent functions of A, B, C, Δ matrices. This is *causal-recurrent*: information flows along the sequence, and the model selectively retains or forgets per-token. The 2-D scanning order (SS2D) lets it model 2-D spatial structure.

#### Fusion — Interleaved Bi-Temporal Sequence + VSS Blocks

- **Mechanism.** After backbone, flatten the (7×7=49) spatial tokens from each side. Interleave them into a length-98 sequence: `[A_pos_0, B_pos_0, A_pos_1, B_pos_1, ..., A_pos_48, B_pos_48]`. Feed through 6 stacked VSS blocks (LayerNorm → SS2D → MLP, residual).
- **Why interleaved and not concat?** SSM is sequence-aware: when the hidden state sees `A_pos_k → B_pos_k` consecutively, the selective gates can encode "the change at position k" directly into the hidden state. Concatenating all-of-A then all-of-B (or concat-diff) forces the model to memorize all of A in the hidden state before it ever sees B, which wastes capacity.
- **Cite.** This interleaving design follows the Spatio-Temporal SSM idea from ChangeMamba [Chen et al. 2024, 358 citations] and CSSM [Ghazaei & Aptoula 2025], adapted for our classification task instead of pixel-level segmentation.

#### Head — Linear + Robust Asymmetric Loss

- **Mechanism.** Mean-pooled fusion-output token (768-d) → three independent linear heads per family + one binary changeflag head. Sigmoid + BCE for changeflag; sigmoid + RAL for the three multi-label families.
- **Why simple linear heads?** ML-Decoder is v5's contribution; v6 isolates the SSM-vs-attention question. Keeping the head simple makes the comparison clean.
- **RAL.** Robust Asymmetric Loss [Park et al. 2023] is a polynomial extension of Asymmetric Loss [Ridnik et al. 2021]: `L = Σ_c (−Hill_reg(p_c)) · poly_focal_asym(p_c, y_c, γ⁺, γ⁻, m)`. The polynomial replaces the focal factor's harsh exponential; the Hill regularizer prevents over-emphasis on noisy positives. RAL achieved Top-5 on CXR-LT 2023 (same competition family as our v3 baseline). Different loss *family* than v3's LDAM-DRW (which subtracts margins from positive logits), so the v3 failure mode (margin × disabled-scale × triple-correction with LA) is not inherited.

---

## Course-rubric compliance

The course mandates Aşama 1 (three single-task models, weight 75%) and Aşama 2 (one multi-task model with shared encoder, weight 25%).

- **Aşama 1** — three single-task variants, each with only one family's linear head + RAL. Backbone + fusion are the same design but trained separately. Each ~1.5h A100.
- **Aşama 2** — one multi-task variant: the same shared backbone + fusion plus three linear heads with RAL. ~2h A100.
- **Architecture diagram** to draw in the IEEE report: the ASCII figure above. Substantively different from v1's diagram in three places: backbone, fusion, loss.

---

## Implementation plan

| Task | Effort | Files touched |
|---|---|---|
| Install Mamba dependencies (`pip install causal-conv1d mamba-ssm`); confirm CUDA kernels build on Colab A100 | ~30 min | (Colab cell) |
| Add VMamba backbone wrapper from the official repo (~300 LOC plus pretrained checkpoint download) | ~1h | `src/models/vmamba_backbone.py` (new) |
| Implement interleaved bi-temporal tokenizer | ~30 min | `src/data/interleaved_bitemporal.py` (new) |
| Implement 6 × VSS block stack on the length-98 sequence (mostly imports from VMamba repo) | ~2h | `src/models/vmamba_fusion.py` (new) |
| Implement Robust Asymmetric Loss (~80 LOC from the official Park et al. 2023 repo) | ~30 min | `src/losses/robust_asymmetric_loss.py` (new) |
| Extend `ChangeClassifier` with `backbone_kind="vmamba"` / `fusion_kind="interleaved_ssm"` / `loss_kind="ral"` config branches | ~1h | `src/models/change_classifier.py` (extend) |
| Write 4 configs: `track4_v6_{object,event,attribute,multitask}.yaml` | ~30 min | `configs/` |
| Add 2 Colab cells: v6 train + v6 eval | ~30 min | `colab_runbook.ipynb` |
| **Total** | **~6.5h dev** | |
| Training (3 single-task @ ~1.5h + 1 multi-task @ ~2h) | **~6.5h A100** | |

---

## Risks and failure modes

1. **`causal-conv1d` + `mamba-ssm` build.** These wheels require CUDA-compiled custom kernels. They build cleanly on Colab A100 (verified Python 3.10 + CUDA 12.1 + torch 2.1+). Local CPU venv cannot run them; that's OK because local is only for unit-test smoke runs and v6 training is GPU-only anyway.
2. **VMamba checkpoint loading.** The official VMamba-Tiny checkpoint covers the 4-stage hierarchical encoder. The 6 new VSS blocks in the fusion are randomly initialized — they're trained from scratch. Use Xavier init.
3. **Interleaved sequence length.** 98 tokens at d=768 is well within Mamba's design range (Mamba paper benchmarks at sequence lengths up to 256k). Linear-time means this is cheap.
4. **RAL hyperparameters.** Park et al. defaults: `γ⁺=0, γ⁻=4, m=0.05` (positive focal disabled, negative focal aggressive, margin small). We'll start there and tune only if v6 plateaus.
5. **Aşama 1 single-task interpretation.** Same as v5: three variants share the *design* but are trained independently. Defensible — backbone and fusion don't share *weights* across the three trainings. If the grader objects, v1's three truly-independent single-task models are also on file.
6. **Convergence speed.** VMamba is reportedly slower to converge than ConvNeXt on small datasets [Liu et al. 2024]. Budget for 40–50 epochs instead of 30. Early-stop patience 15.

---

## Expected outcome

| Scenario | Test avg macro F1 | Probability |
|---|---|---|
| Best | 0.32 – 0.38 (SSM temporal modeling breaks the ceiling) | ~20% |
| Likely | 0.28 – 0.31 (small but real lift) | ~60% |
| Worst | 0.22 – 0.27 (SSM underfits the noisy tail) | ~20% |

All three scenarios feed the cross-architectural triangulation. The *negative* outcome is the strongest publishable story (CNN + ViT + SSM all hit the same ~0.30 wall).

---

## Publishable angle

- **Positive:** *"Vision State-Space Models lift multi-label change classification beyond CNN and ViT baselines on LEVIR-CC-derived data; an architectural inductive-bias win over attention for bi-temporal tasks."*
- **Mixed:** *"Selective SSMs match but do not exceed strong CNN/ViT baselines on the LEVIR-CC multi-label change task; loss-side and data-side interventions are required to break the ceiling."*
- **Negative:** *"Cross-architectural empirical study of LEVIR-CC multi-label change classification: CNN, ViT, and SSM all plateau at 0.20–0.30 macro F1, indicating an annotation-quality ceiling that no current architecture surpasses."*

The negative result is *also* publishable — benchmarks with known label noise are an active area, and a clean cross-architecture confirmation is contributory.

---

## References (this track)

- **Liu et al. 2024.** *VMamba: Visual State Space Model.* NeurIPS / arXiv:2401.10166.
- **Gu & Dao 2023.** *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
- **Chen et al. 2024.** *ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model.* IEEE TGRS. *358 citations.*
- **Ghazaei & Aptoula 2025.** *Efficient Remote Sensing Change Detection With Change State Space Models (CSSM).* arXiv:2504.11080.
- **Wu et al. 2025.** *CD-Lamba: Cross-Temporal Locally Adaptive State Space Model.* arXiv:2501.15455.
- **Huang et al. 2025.** *LCCDMamba: Visual State Space Model for Land Cover Change Detection.* IEEE JSTARS.
- **Park et al. 2023.** *Robust Asymmetric Loss for Multi-Label Long-Tailed Learning.* ICCV CVAMD Workshop. arXiv:2308.05542.
- **Ridnik et al. 2021.** *Asymmetric Loss for Multi-Label Classification.* ICCV.
- **Cao et al. 2019.** *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss (LDAM).* NeurIPS. *(reference for why v3 failed and why RAL is a different family.)*

See also the consolidated literature survey in [`literature_v6.md`](literature_v6.md) for the methodology (queries, papers reviewed) shared between Tracks 3 and 4.
