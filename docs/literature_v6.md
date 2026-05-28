# Literature Review for v6 Architecture Selection

**Date:** 2026-05-28
**Purpose:** survey of recent (2022–2026) literature relevant to bi-temporal multi-label change classification, to inform the choice of a v6 architecture that is (a) genuinely different from v5's DINOv2 + cross-attention + ML-Decoder approach, (b) publishable in remote-sensing / multi-label venues, (c) implementable in 1–2 days.

The search was conducted via Semantic Scholar (`paper_relevance_search`, 6 diverse queries), arXiv (`search_papers`, 1 query), and cross-agent recommendations from Copilot (gpt-5.2) and Gemini (3.1-pro).

---

## 1. Most-relevant body of work — LEVIR-CC change captioning (same dataset!)

The LEVIR-CC dataset is the source of our adapted multi-label dataset. The captioning literature is therefore very directly relevant — most papers extract bi-temporal features from the *same* image pairs we use, just feed them to a caption decoder instead of multi-label classification heads.

| Paper | Year | Citations | Architecture | Relevance |
|---|---|---|---|---|
| **RSCaMa** (Liu et al.) | 2024 | 104 | Mamba SSM bi-temporal encoder + caption decoder | High — direct ref for Mamba on LEVIR-CC |
| **Diffusion-RSCC** (Yu et al.) | 2024 | 13 | Diffusion probabilistic model | High — diffusion features path |
| **Semantic-CC** (Zhu et al.) | 2024 | 28 | SAM-based bi-temporal encoder + LLM | High — SAM as foundation encoder |
| **TISDNet** (Li et al.) | 2024 | 15 | Symmetric difference transformer | High — bi-temporal feature design |
| **IntelliChange-RSCC** | 2024 | 0 | ResNet + difference-aware transformer | Medium |
| **MTI-CC / MTH-Net** | 2025 | 0–2 | Mamba–Transformer hybrid | Medium |
| **DGAT** | 2025 | 1 | Dynamic Gaussian attenuated transformer | Medium |
| **Text-Augmented (TACC)** | 2025 | 5 | CLIP-prompted SAM features | Medium |

## 2. Mamba / State-Space Models for change detection (hottest direction)

Mamba is the most-cited recent architectural family for change tasks on LEVIR-style data.

| Paper | Year | Citations | Architecture | Notes |
|---|---|---|---|---|
| **ChangeMamba** (Chen et al.) | 2024 | **358** | Visual Mamba backbone + 3 spatiotemporal relationship modules | The foundational paper; reference implementation https://github.com/ChenHongruixuan/MambaCD |
| **CD-Lamba** (Wu et al.) | 2025 | 15 | Locally Adaptive State-Space Scan + Cross-Temporal Scan | Improves locality awareness |
| **LCCDMamba** (Huang et al.) | 2025 | 15 | Siam-VMamba + multi-scale fusion + dual token modeling | F1 94.18 on WHU-CD, 91.68 on LEVIR-CD |
| **CSSM** (Ghazaei & Aptoula) | 2025 | 2 | Task-specific Change State-Space Model | Designed exclusively for CD, lower params |
| **NeXt2Former-CD** | 2026 | 0 | DINOv3 + Siamese ConvNeXt + deformable temporal fusion + Mask2Former | Recent hybrid; suggests DINOv3 weights work well |
| **FAPMNet** | 2026 | 1 | Mamba VSSM + flow alignment + prototype memory | Semantic-CD focused |

## 3. Foundation models for remote sensing

| Paper | Year | Citations | Approach | Suitability for our problem |
|---|---|---|---|---|
| **SpectralGPT** (Hong et al.) | 2023 | **774** | 3D GPT pre-trained on 1M spectral RS images, 600M params | Strong, but spectral — we have RGB |
| **MTP** (Wang et al.) | 2024 | 143 | Multi-task supervised pre-training on RS SAM-annotated segmentation, 300M+ params | Strong; pretrained ViT-B available; downstream CD shown |
| **SatMAE / ScaleMAE** | 2023+ | — | MAE pretraining on satellite imagery | Tailored to satellite, but our images are aerial-RGB |
| **Changen2** (Zheng et al.) | 2024 | 68 | Generative change foundation model + diffusion synthesis | Zero-shot CD capabilities |

## 4. Long-tail multi-label loss functions

| Paper | Year | Citations | Method | Take-away |
|---|---|---|---|---|
| **Asymmetric Loss (ASL)** (Ridnik et al.) | 2021 | many | γ⁺ ≠ γ⁻ focal-style loss for multi-label | Original strong baseline; in our `references.bib` already |
| **Robust Asymmetric Loss (RAL)** (Park et al.) | 2023 | 24 | Polynomial ASL + Hill regularization | Top-5 CXR-LT 2023, generic for multi-label long-tail |
| **LM-CLIP / BAL** (Timmermann et al.) | 2025 | 3 | Balanced Asymmetric Loss + contrastive embedding | Designed for VOC-MLT, COCO-MLT long-tail |
| **Sulake CXR-LT 2026** (Sulake) | 2026 | 0 | LDAM-DRW + ConvNeXt-Large + cRT + TTA | Same recipe family we tried in v3; placed 5th |

## 5. Other promising directions

- **DINOv3 + LoRA + Focal** — MIDOG 2025 Task 2 *winner* (Balezo et al., 2025-08): fine-tuned DINOv3-H+ with LoRA on ~1.3M params + domain-weighted Focal Loss + heavy augmentation → first place on histopathology classification. Hard evidence that *foundation backbone + tiny adapter + class-imbalance-aware loss* recipe works at SOTA.
- **Mask2Former / SAM-style mask classification** (MaskChanger 2024, IEEE MVIP): converted CD from per-pixel classification to mask classification using Mask2Former + Siamese encoder → 91.96% F1 on LEVIR-CD.
- **Cross Swin transformer** (CSTSUNet, Wu et al. 2023, 20 citations): cross-attention between Siamese branches at each scale.
- **Multi-scale interaction fusion** (MIFNet, Xie et al. 2025, 25 citations): early-fusion + dual complementary attention + collection-allocation fusion.

---

## 6. Synthesis — what v6 should look like

The v5 architecture (DINOv2 frozen + cross-attention fusion + ML-Decoder) targets **representation quality + change-aware fusion** via an attention-based, transformer-only stack with foundation-model pretraining.

For v6 to test a *different* hypothesis, the inductive bias and/or paradigm should differ on at least one of these axes:
- **Sequential vs attention-based processing**: Mamba / SSM
- **Discriminative vs generative**: diffusion / Changen2-style
- **General vs change-specific architecture**: ChangeMamba family vs general ViT
- **Loss-side**: Asymmetric / RAL instead of plain BCE
- **Mask-classification reframing**: Mask2Former-style per-class queries

The strongest single recommendation from the literature, in my reading, is **a Mamba/SSM-based bi-temporal encoder adapted for multi-label classification**, optionally with Asymmetric Loss (RAL). This is captured by [`docs/track4_v6.md`](track4_v6.md) (to be written after cross-agent opinions arrive).

## Cross-agent opinions

Copilot (gpt-5.2) and Gemini (3.1-pro) were briefed with the same context and constraints. Their independent picks are summarized in [`docs/track4_v6.md`](track4_v6.md) once they return.

---

## Notes on what was searched

Queries used:
1. `bitemporal change detection siamese transformer multi-label classification` — 55 results, 8 read
2. `long-tail multi-label classification asymmetric loss decoder transformer` — 26 results, 6 read
3. `remote sensing foundation model self-supervised pretraining LEVIR-CC change captioning` — 25 results, 8 read
4. `state-space model mamba vision remote sensing change detection` — 302 results, 6 read
5. `masked autoencoder satellite image SatMAE ScaleMAE change detection` — 1 result, 1 read
6. `change captioning bi-temporal image transformer LEVIR-CC` — 77 results, 8 read
7. `cross-attention bitemporal feature fusion deep learning Siamese` — 1996 results, 5 read
8. (arXiv) `change detection bi-temporal multi-label classification transformer 2024 2025` — 8 results, mostly off-topic; one hit (DINOv3 + LoRA MIDOG winner) added above.

Total: ~50 unique papers read or skim-reviewed.
