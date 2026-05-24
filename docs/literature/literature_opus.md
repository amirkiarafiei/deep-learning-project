# Literature Review — Multi-Label Classification of Bi-Temporal Remote Sensing Image Pairs

**Task framing.** Given two co-registered RGB satellite/aerial images of the same scene at times *t₁* (before) and *t₂* (after), predict three sets of multi-label outputs describing what changed: **Object** (13 classes), **Event** (13 classes), **Attribute** (25 classes). Dataset is decomposed from LEVIR-CC (Liu et al. 2022), ~10K image pairs, train/val/test = 78/11/11.

**Top-line finding.** After exhaustive search across Semantic Scholar, arXiv, Google Scholar, and the open web, **no published paper performs the exact task** of decomposing LEVIR-CC captions into three independent multi-label families (Object × Event × Attribute) for scene-level classification. The closest analogues are: (a) **CDVQA** — single-label categorical change answers; (b) **TransWCD / AdvCP / DISep** — weakly supervised CD using binary scene labels; (c) **ChangeIMTI / ChangeVG** — captioning + binary change classification + counting; (d) **Change-Agent / ChangeMinds / Semantic-CC** — joint pixel-level CD + free-form captioning on LEVIR-MCI/LEVIR-CC. Each is a *single-head* version of what we propose. This is positive for novelty but means no comparable scene-level multi-label classification numbers exist as direct baselines.

---

## RING 1 — Directly Related Work (Highest Priority)

### 1. Change-Agent: Toward Interactive Comprehensive Remote Sensing Change Interpretation and Analysis
1. Title: Change-Agent: Toward Interactive Comprehensive Remote Sensing Change Interpretation and Analysis
2. Authors: Chenyang Liu, Keyan Chen, Haotian Zhang et al.
3. Venue + Year: IEEE TGRS 2024
4. arXiv: 2403.19646 — DOI: 10.1109/TGRS.2024.3425815
5. Task: Multi-task CD + change captioning, orchestrated by an LLM agent; introduces **LEVIR-MCI** (extension of LEVIR-CC with both masks and captions on the same image pairs).
6. Datasets: LEVIR-MCI (direct sibling of our source data).
7. Architecture: Dual-branch ResNet/ViT Siamese encoder + Bi-temporal Iterative Interaction (BI³) fusion layer + two heads (segmentation decoder for CD, Transformer decoder for CC), with an LLM agent on top.
8. Loss: Pixel-wise CE (CD) + token-level CE (CC), jointly optimized.
9. Headline metric: LEVIR-CC: BLEU-4 = 64.39, CIDEr-D = 136.61, avg S\*ₘ = 76.15; LEVIR-MCI building CD: F1 = 91.69, IoU = 84.66.
10. Why relevant: Canonical multi-task paper on our exact image collection. The LEVIR-MCI extension is the direct predecessor to our structured-label decomposition — cite as the SOTA multi-task baseline and compare label-recall against their captioning head argmax.
11. Code: https://github.com/Chen-Yang-Liu/Change-Agent

### 2. ChangeIMTI / ChangeVG: Comprehensive Interactive Change Understanding with a Dual-Granularity VLM
1. Title: Towards Comprehensive Interactive Change Understanding in Remote Sensing: A Large-scale Dataset and Dual-granularity Enhanced VLM
2. Authors: Junxiao Xue, Quan Deng, Xuecheng Wu et al.
3. Venue + Year: arXiv 2025
4. arXiv: 2509.23105
5. Task: Multi-task instruction tuning on bi-temporal RS for change captioning + **binary change classification** + change counting + change localization. The only paper that treats scene-level change classification as a *first-class* task alongside captioning — but it is binary changed/unchanged, not our 51-way multi-label decomposition.
6. Datasets: ChangeIMTI (large instruction-tuning dataset built on LEVIR-CC, LEVIR-MCI, and other bi-temporal sources).
7. Architecture: Dual-branch (fine-grained spatial + high-level semantic) vision-guided module feeding Qwen2.5-VL-7B as auxiliary prompts during instruction tuning.
8. Loss: Autoregressive cross-entropy (instruction-tuning style).
9. Headline metric: +1.39 on S\*ₘ over Semantic-CC for LEVIR-CC change captioning; binary-classification accuracy not reported in abstract.
10. Why relevant: Closest conceptual cousin to our structured-label task — explicitly carves change understanding into discrete sibling tasks. Architectural pattern (dual-granularity vision guidance) is a baseline candidate.
11. Code: Released on GitHub (link in paper).

### 3. ChangeMinds: Multi-task Framework for Detecting and Describing Changes
1. Title: ChangeMinds: Multi-task Framework for Detecting and Describing Changes in Remote Sensing
2. Authors: Yuduo Wang, Weikang Yu, Michael Kopp, Pedram Ghamisi
3. Venue + Year: arXiv 2024
4. arXiv: 2410.10047
5. Task: Joint CD + CC end-to-end.
6. Datasets: LEVIR-MCI (primary), standard CD benchmarks for comparison.
7. Architecture: Siamese ResNet encoder + ChangeLSTM (change-aware spatio-temporal LSTM) + multi-task predictor with cross-attention; one head segments, one head captions.
8. Loss: Pixel-wise CE (CD) + token-level CE (CC), jointly optimized.
9. Headline metric: LEVIR-MCI: BLEU-4 = 64.42, CIDEr-D = 138.10, IoU(building) = 85.42.
10. Why relevant: Direct architectural template — shared Siamese encoder + task-specific heads is viable on this data; we can adapt by replacing the captioning head with three multi-label heads.
11. Code: Promised; track repository for Wang/Ghamisi.

### 4. Semantic-CC: Captioning + Pixel-Level Semantic CD via SAM
1. Title: Semantic-CC: Boosting Remote Sensing Image Change Captioning via Foundational Knowledge and Semantic Guidance
2. Authors: Yongshuo Zhu, Lu Li, Keyan Chen, Chenyang Liu et al.
3. Venue + Year: IEEE TGRS 2024
4. arXiv: 2407.14032 — DOI: 10.1109/TGRS.2024.3497338
5. Task: Joint CD + CC where pixel-level semantic CD guides change captioning.
6. Datasets: LEVIR-CC + LEVIR-CD (same image collection as our source data).
7. Architecture: Bi-temporal SAM-based encoder + multi-task semantic aggregation neck + multi-scale CD decoder + LLM-based caption decoder; three-stage staged-supervision training.
8. Loss: Stage-wise BCE (segmentation) + CE (caption) with curriculum.
9. Headline metric: LEVIR-CC: BLEU-4 = 63.47, CIDEr-D = 134.87; LEVIR-CD: IoU = 84.83.
10. Why relevant: Most direct prior art for "Siamese SAM encoder + multi-task heads on LEVIR-CC pairs." Their staged-supervision schedule is a candidate MTL recipe for our three heads.
11. Code: Implementation linked via Liu Chenyang's GitHub.

### 5. SECOND-CC / MModalCC — Turkish-authored RSICC benchmark (FLAG: regional alignment)
1. Title: Robust Change Captioning in Remote Sensing: SECOND-CC Dataset and MModalCC Framework
2. Authors: Ali Can Karaca, M. Enes Ozelbas, Saadettin Berber, Orkhan Karimli, Turabi Yıldırım, M. F. Amasyalı (Yıldız Technical University, Istanbul)
3. Venue + Year: IEEE JSTARS 2025
4. arXiv: 2501.10075 — DOI: 10.1109/JSTARS.2025.3600613
5. Task: Remote sensing change captioning with multimodal (RGB + semantic seg map) inputs; introduces SECOND-CC dataset.
6. Datasets: SECOND-CC (6,041 pairs, 30,205 captions); LEVIR-MCI for comparison.
7. Architecture: Dual-branch ResNet + Cross-Modal Cross-Attention (CMCA) + Multimodal Gated Cross-Attention (MGCA) + Transformer caption decoder.
8. Loss: Captioning cross-entropy with semantic/visual fusion.
9. Headline metric: SECOND-CC: BLEU-4 = 41.6, CIDEr = 113.7 (+4.6% / +9.6% over RSICCformer); LEVIR-MCI: avg S\*ₘ = 83.51.
10. Why relevant: **PROMINENT TURKISH FLAG** — Karaca/Amasyalı group at YTÜ Istanbul publishes at SIU, UBMK, and JSTARS; their `ChangeCapsInRS` GitHub org hosts BLIP-CC, MOSAIC-SEN2-CC, SECOND-CC, X-Change. Aligning our evaluation conventions and citing this group is strategically important for SIU/INISTA/UBMK/ASYU venue acceptance.
11. Code: https://github.com/ChangeCapsInRS/SecondCC

### 6. CDVQA — Change Detection Meets Visual Question Answering
1. Title: Change Detection Meets Visual Question Answering
2. Authors: Zhenghang Yuan, Lichao Mou, Zhitong Xiong, Xiao Xiang Zhu
3. Venue + Year: IEEE TGRS 2022
4. arXiv: 2112.06343 — DOI: 10.1109/TGRS.2022.3203314
5. Task: Answer categorical questions about changes in bi-temporal aerial images (e.g., "what changed?", "increase/decrease?") — effectively a *single-label classification per question* over discrete change concepts.
6. Datasets: CDVQA (built from SECOND segmentation labels with template-generated QA triplets).
7. Architecture: Multi-temporal CNN feature encoding + multi-temporal fusion + multi-modal (image+text) fusion + answer-prediction MLP head.
8. Loss: Cross-entropy over answer vocabulary.
9. Headline metric: Overall accuracy ≈ 73% on CDVQA test set (best variant; paper Table II).
10. Why relevant: Earliest paper to recast change interpretation as a **categorical classification** problem rather than pixel masks or free-form captions — exactly the conceptual move we are making. Strong precedent and baseline architecture.
11. Code: https://github.com/YZHJessica/CDVQA

### 7. TransWCD — Scene-Level Weakly Supervised Change Detection
1. Title: TransWCD: Scene-Adaptive Joint Constrained Framework for Weakly Supervised Change Detection
2. Authors: Zhenghui Zhao, Lixiang Ru, Chen Wu, Di Wang
3. Venue + Year: IEEE TGRS 2025
4. DOI: 10.1109/TGRS.2025.3545051
5. Task: Weakly supervised CD where the supervision is **scene-level binary change labels**; architecture explicitly contains a scene-level change classifier.
6. Datasets: WHU-CD, LEVIR-CD, DSIFN-CD (scene-level labels derived from masks).
7. Architecture: Hierarchical Transformer classifier + multi-scale CAM + scene-adaptive predictor; gated scene constraint via Dirac penalty.
8. Loss: Image-level CE + scene-gated consistency loss.
9. Headline metric: WHU-CD IoU = 65.49, LEVIR-CD IoU = 73.86, DSIFN-CD IoU = 65.04.
10. Why relevant: **Best architectural template for our task** — it is literally a scene-classification transformer trained from binary scene labels on bi-temporal pairs. We extend its head from 2-way to (13+13+25)-way multi-label with BCE/ASL.
11. Code: https://github.com/zhenghuizhao/TransWCD

### 8. AdvCP — Adversarial Class Prompting for Weakly Supervised CD
1. Title: Advancing Weakly-Supervised Change Detection in Satellite Images via Adversarial Class Prompting
2. Authors: Zhenghui Zhao, Chen Wu, Di Wang et al.
3. Venue + Year: IEEE TIP 2025
4. arXiv: 2508.17186 — DOI: 10.1109/TIP.2025.3623260
5. Task: Weakly supervised CD with image-level labels.
6. Datasets: WHU-CD, LEVIR-CD, DSIFN-CD, SYSU-CD, CDD.
7. Architecture: Plug-in adversarial prompt mining + global prototype rectification; works on ConvNet, Transformer, and SAM-based baselines.
8. Loss: Adversarial prompting + rectification under weak supervision.
9. Headline metric: Up to **+7.46 IoU** improvement on LEVIR-CD over previous WSCD baselines.
10. Why relevant: Strongest evidence that scene-level supervision can be made robust against background co-occurrence noise — directly relevant since our caption-derived labels are noisy.
11. Code: https://github.com/zhenghuizhao/AdvCP

### 9. DISep — Instance Separation for Scene-to-Pixel WSCD
1. Title: Plug-and-Play DISep: Separating Dense Instances for Scene-to-Pixel Weakly-Supervised Change Detection in High-Resolution Remote Sensing Images
2. Authors: Zhenghui Zhao, Chen Wu, Lixiang Ru et al.
3. Venue + Year: ISPRS JPRS 2025
4. arXiv: 2501.04934 — DOI: 10.1016/j.isprsjprs.2025.01.007
5. Task: Weakly supervised CD under scene-level supervision.
6. Datasets: LEVIR-CD, WHU-CD, DSIFN-CD, SYSU-CD, CDD.
7. Architecture: Instance localization + retrieval + separation modules, pluggable into Transformer/ConvNet WSCD methods.
8. Loss: Separation loss on per-instance embeddings.
9. Headline metric: Boosts three Transformer-based and four ConvNet-based WSCD methods by 1–3 IoU points on average.
10. Why relevant: Addresses the "instance lumping" problem that appears when supervision is only scene-level — exactly our regime.
11. Code: https://github.com/zhenghuizhao/Plug-and-Play-DISep-for-Change-Detection

### 10. Pix4Cap — Pixel-Level CD Pseudo-Label Learning for RS Change Captioning
1. Title: Pixel-Level Change Detection Pseudo-Label Learning for Remote Sensing Change Captioning
2. Authors: Chenyang Liu, Keyan Chen, Zipeng Qi et al.
3. Venue + Year: IGARSS 2024
4. arXiv: 2312.15311 — DOI: 10.1109/IGARSS53475.2024.10642750
5. Task: RS change captioning with auxiliary CD branch via pseudo-labels.
6. Datasets: LEVIR-CC.
7. Architecture: CD pseudo-label branch + semantic fusion augment + caption decoder.
8. Loss: Captioning loss + pseudo-label CD supervision.
9. Headline metric: SOTA on RSICC at submission; CIDEr-D ≈ 137 on LEVIR-CC per repo benchmarks.
10. Why relevant: Closest "caption-to-label" bridge in the literature; shows that explicit change labels improve captioning — implies the reverse (captions → labels) is information-preserving.
11. Code: https://github.com/Chen-Yang-Liu/Pix4Cap

### 11. RSICCformer — The LEVIR-CC Origin Paper (MUST CITE)
1. Title: Remote Sensing Image Change Captioning With Dual-Branch Transformers: A New Method and a Large Scale Dataset
2. Authors: Chenyang Liu, Rui Zhao, Hao Chen et al.
3. Venue + Year: IEEE TGRS 2022
4. DOI: 10.1109/TGRS.2022.3218921
5. Task: RS change captioning; introduces LEVIR-CC dataset.
6. Datasets: LEVIR-CC (10,077 image pairs, 50,385 captions).
7. Architecture: CNN feature extractor + dual-branch Transformer encoder + caption decoder.
8. Loss: Caption-generation cross-entropy.
9. Headline metric: +4.98 BLEU-4 and +9.86 CIDEr-D over prior methods on LEVIR-CC.
10. Why relevant: Mandatory citation as the LEVIR-CC dataset source and the baseline against which all subsequent RSICC work compares.
11. Code: Implementation linked via author's GitHub.

### 12. KCFI — Key Change Features + Instruction Tuning
1. Title: Enhancing Perception of Key Changes in Remote Sensing Image Change Captioning
2. Authors: Cong Yang, Zuchao Li, Hongzan Jiao, Zhi Gao, Lefei Zhang
3. Venue + Year: arXiv 2024
4. arXiv: 2409.12612
5. Task: Captioning with joint pixel-level CD; dynamic weight averaging across tasks.
6. Datasets: LEVIR-CC.
7. Architecture: ViT encoder + key-feature perceiver + pixel-level CD decoder + instruction-tuned LLM decoder.
8. Loss: CD BCE + caption CE, balanced via dynamic weight averaging.
9. Headline metric: Best LEVIR-CC results at submission; exact BLEU-4/CIDEr-D in Table 2 of paper (not in abstract).
10. Why relevant: Their dynamic-weight-averaging multi-task balancing is directly applicable when training our three classification heads with mismatched loss magnitudes.
11. Code: https://github.com/yangcong356/KCFI

### 13. VisTA — Show Me What and Where Has Changed (CDQAG)
1. Title: Show Me What and Where has Changed? Question Answering and Grounding for Remote Sensing Change Detection
2. Authors: Ke Li, Fuyu Dong, Di Wang, Shaofeng Li, Quan Wang et al.
3. Venue + Year: arXiv 2024
4. arXiv: 2410.23828
5. Task: Question answering (categorical) + grounding (masks) for changes; 8 question types over 10 land-cover categories.
6. Datasets: QAG-360K (360K question/answer/mask triplets).
7. Architecture: VisTA — unified visual + textual answer transformer.
8. Loss: CE on answers + segmentation loss on grounding masks.
9. Headline metric: SOTA on CDVQA + QAG-360K; exact accuracy in paper Tables (not in abstract).
10. Why relevant: Most recent and largest categorical-change benchmark; useful as a cross-task reference — our label space ≈ a small 51-class multi-label QA.
11. Code: https://github.com/like413/VisTA

### 14. Weakly Supervised Bitemporal Scene Change for Building Damage Assessment
1. Title: A Weakly Supervised Bitemporal Scene Change Detection Approach for Pixel-Level Building Damage Assessment
2. Authors: Wenfan Qiao, Li Shen, Wei Wang, Zhilin Li
3. Venue + Year: IEEE TGRS 2024
4. DOI: 10.1109/TGRS.2024.3494257
5. Task: Scene-level change classification (image-level multi-class labels of damage categories) to drive pixel-level damage maps.
6. Datasets: 2010 Haiti earthquake satellite + 2019 Changning UAV imagery.
7. Architecture: Siamese local-global ViT (SLgViT) + cross-Siamese interaction-fusion (CSIF) module; trained on scene patches with image-level multi-class labels.
8. Loss: Scene-level cross-entropy over damage categories.
9. Headline metric: Exact F1 not in abstract; paper reports superior performance vs. WSCD baselines.
10. Why relevant: Operates at exactly our supervision level — scene patches with image-level multi-class labels of changed object types. Direct blueprint for a Siamese ViT scene classifier on bi-temporal pairs.
11. Code: Not stated in abstract.

### 15. Zero-Shot CLIP for Satellite Change Classification (FLAG: Turkish-authored)
1. Title: Zero Shot Classification for Change Detection in Satellite Imagery
2. Authors: Kürşat Kömürcü, Linas Petkevičius
3. Venue + Year: IEEE AIEEE 2024
4. DOI: 10.1109/AIEEE62837.2024.10586705
5. Task: Zero-shot scene-level change classification.
6. Datasets: LEVIR-CD, DSIFN, S2 Looking.
7. Architecture: Frozen CLIP ViT-B/32 image encoder + zero-shot text-image matching on bi-temporal scenes; supervised decision-tree baseline for comparison.
8. Loss: Pre-trained contrastive (no fine-tuning).
9. Headline metric: Establishes zero-shot baseline accuracies across LEVIR-CD and S2 Looking without remote-sensing training data; exact accuracy in paper Tables.
10. Why relevant: Turkish author and regional IEEE venue that directly models change detection as a discrete, scene-level zero-shot classification task (same operational regime as ours).
11. Code: No public code link from abstract.

---

## RING 2 — Architecture Building Blocks (Medium Priority)

### A. Siamese encoders / Remote-sensing foundation backbones

#### 16. SkySense — Multimodal RS Foundation Model
1. Title: SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery
2. Authors: Xin Guo, Jiangwei Lao, Bo Dang et al.
3. Venue + Year: CVPR 2024
4. arXiv: 2312.10115
5. Task: Generic RS foundation model evaluated on 16 datasets across 7 tasks including change detection.
6. Datasets: 21.5M temporal Sentinel-2 / Sentinel-1 / HR optical sequences for pretraining.
7. Architecture: Billion-parameter factorized multimodal spatio-temporal encoder (Swin-Huge optical + ViT-L SAR + temporal fusion) + Multi-Granularity Contrastive Learning + Geo-Context Prototype Learning.
8. Loss: Multi-granularity contrastive + masked image modeling.
9. Headline metric: Outperforms SatLas / GFM / Scale-MAE by +3.67% / +2.76% / +3.61% average across 7 tasks; surpasses 18 RSFMs.
10. Why relevant: Strongest published multimodal RS foundation backbone — primary candidate for a frozen/LoRA-tuned Siamese encoder when scaling beyond DINOv2.
11. Code: Pretrained weights promised; search GitHub for SkySense.

#### 17. SatMAE — Temporal/Spectral Masked Pretraining
1. Title: SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery
2. Authors: Yezhen Cong, Samar Khanna, Chenlin Meng et al.
3. Venue + Year: NeurIPS 2022
4. arXiv: 2207.08051
5. Task: Self-supervised pretraining for satellite imagery.
6. Datasets: fMoW-Sentinel for pretraining; EuroSAT, BigEarthNet, segmentation for downstream.
7. Architecture: ViT-Large MAE with temporal-position embedding + grouped spectral-band positional encoding.
8. Loss: Masked pixel reconstruction.
9. Headline metric: Up to +7% on supervised classification, +14% on land-cover transfer vs. ImageNet-MAE.
10. Why relevant: Canonical "RS pretraining" baseline; reusable temporal-position embedding is directly applicable to bi-temporal Siamese setups.
11. Code: https://sustainlab-group.github.io/SatMAE/

#### 18. RemoteCLIP — Vision-Language Foundation Model for RS
1. Title: RemoteCLIP: A Vision Language Foundation Model for Remote Sensing
2. Authors: Fan Liu, Delong Chen, Zhangqingyun Guan et al.
3. Venue + Year: IEEE TGRS 2024
4. arXiv: 2306.11029
5. Task: Vision-language pretraining for RS.
6. Datasets: ~12× larger pretraining set than prior RS-CLIP (boxes, masks, UAV imagery).
7. Architecture: ViT-L/14 + transformer text encoder, CLIP-style contrastive.
8. Loss: InfoNCE image-text contrastive.
9. Headline metric: +9.14% mean recall on RSITMD, +8.92% on RSICD vs. SOTA; +6.39% avg zero-shot classification across 12 datasets over baseline CLIP.
10. Why relevant: Useful Siamese-encoder option when our class names (Object/Event/Attribute) can serve as text anchors during head training (Query2Label / ML-Decoder style).
11. Code: https://github.com/ChaoFan996/RemoteCLIP

#### 19. PeftCD — PEFT of Foundation Models for CD
1. Title: PeftCD: Leveraging Vision Foundation Models With Parameter-Efficient Fine-Tuning for Remote Sensing Change Detection
2. Authors: Shun Dong, Yong Hu, Lei Wang et al.
3. Venue + Year: IEEE JSTARS 2025
4. arXiv: 2509.09572
5. Task: Bi-temporal binary CD via PEFT of foundation models.
6. Datasets: LEVIR-CD, WHU-CD, SYSU-CD, S2Looking, MSRSCD, MLCD, CDD.
7. Architecture: Shared-weight Siamese encoder = SAM2 or **DINOv3** with LoRA + Adapter PEFT, minimal decoder.
8. Loss: BCE + Dice on change map.
9. Headline metric: IoU = 85.62% on LEVIR-CD, 92.05% WHU-CD, 73.81% SYSU-CD, 97.01% CDD, 52.25% S2Looking.
10. Why relevant: Most direct benchmark of DINOv3 / SAM2 as Siamese RS-CD backbones with PEFT — the same recipe transfers to our three multi-label heads.
11. Code: GitHub link in paper.

#### 20. MTP — Multitask Pretraining for RS Foundation
1. Title: MTP: Advancing Remote Sensing Foundation Model via Multitask Pretraining
2. Authors: Di Wang, Jing Zhang, Minqiang Xu et al.
3. Venue + Year: IEEE JSTARS 2024
4. arXiv: 2403.13430
5. Task: Multitask supervised pretraining for RS foundation models.
6. Datasets: SAMRS (~300M params backbone); downstream on 14 RS datasets.
7. Architecture: Shared CNN/ViT encoder + task-specific decoders (semantic seg, instance seg, rotated detection).
8. Loss: Per-task supervised losses combined.
9. Headline metric: Competitive with much larger SOTA across 14 RS benchmarks (incl. change-detection downstream).
10. Why relevant: Both a candidate backbone *and* an explicit "shared encoder + three task heads" architecture analogue for the project structure we propose.
11. Code: https://github.com/ViTAE-Transformer/MTP

### B/C. Bi-temporal architectures and fusion strategies

#### 21. ChangeFormer — Transformer Siamese + Abs-Diff Fusion (FOUNDATIONAL)
1. Title: ChangeFormer: A Transformer-Based Siamese Network for Change Detection
2. Authors: Wele Gedara Chaminda Bandara, Vishal M. Patel
3. Venue + Year: IGARSS 2022
4. arXiv: 2201.01293
5. Task: Bi-temporal binary CD.
6. Datasets: LEVIR-CD, DSIFN-CD.
7. Architecture: Hierarchical SegFormer (MiT-B encoder, MLP decoder) in Siamese form; **fusion = absolute element-wise difference of multi-scale features → concatenated MLP decoder**.
8. Loss: Weighted cross-entropy on change mask.
9. Headline metric: F1 ≈ 90.4 on LEVIR-CD; F1 = 95.56 on DSIFN-CD.
10. Why relevant: The canonical "transformer-Siamese + abs-diff fusion" baseline; simplest and most-cited reference architecture.
11. Code: https://github.com/wgcban/ChangeFormer

#### 22. BIT — Bitemporal Image Transformer
1. Title: Remote Sensing Image Change Detection With Transformers
2. Authors: Hao Chen, Zipeng Qi, Zhenwei Shi
3. Venue + Year: IEEE TGRS 2021
4. arXiv: 2103.00208
5. Task: Bi-temporal binary CD.
6. Datasets: LEVIR-CD, WHU-CD, DSIFN-CD.
7. Architecture: ResNet-18 Siamese backbone → semantic tokenizer → **transformer encoder over concatenated bi-temporal tokens** → transformer decoder refines pixel features → feature-difference + classifier.
8. Loss: Cross-entropy on change mask.
9. Headline metric: F1 = 89.31 on LEVIR-CD, F1 = 83.98 on WHU-CD; ~3× fewer FLOPs than ConvNet baseline.
10. Why relevant: Canonical efficient token-space cross-temporal transformer; demonstrates value of joint-token attention over later differencing.
11. Code: https://github.com/justchenhao/BIT_CD

#### 23. SiamixFormer — Asymmetric Cross-Temporal Attention
1. Title: SiamixFormer: A Siamese Transformer Network For Building Detection And Change Detection From Bi-Temporal Remote Sensing Images
2. Authors: Amir Mohammadian, Foad Ghaderi
3. Venue + Year: Int. J. Remote Sensing 2023
4. arXiv: 2208.00657 — DOI: 10.1080/01431161.2023.2225228
5. Task: Building detection + bi-temporal CD.
6. Datasets: xBD, WHU, LEVIR-CD, CDD.
7. Architecture: Two hierarchical SegFormer encoders + at every stage a **temporal transformer where Q=pre-image, K/V=post-image (asymmetric cross-attention)**, then MLP decoder.
8. Loss: Cross-entropy + Dice.
9. Headline metric: F1 = 91.32 on LEVIR-CD (paper Table 4); outperforms ChangeFormer.
10. Why relevant: Cleanest example of **temporal cross-attention as fusion at every encoder stage** — directly applicable as the fusion block in our Siamese model.
11. Code: GitHub link in paper.

#### 24. Changer / ChangerEx — Feature Interaction is What You Need
1. Title: Changer: Feature Interaction is What You Need for Change Detection
2. Authors: Sheng Fang, Kaiyu Li, Zhe Li
3. Venue + Year: IEEE TGRS 2023
4. arXiv: 2209.08290 — DOI: 10.1109/TGRS.2023.3277496
5. Task: Bi-temporal binary CD.
6. Datasets: LEVIR-CD, S2Looking, SYSU-CD.
7. Architecture: MetaChanger meta-architecture with **alternative interaction layers inside the feature extractor**; ChangerEx uses parameter-free channel "feature exchange" between t1 and t2; ChangerAD uses aggregation-distribution. Adds **FDAF (Flow-based Dual-Alignment Fusion)** for misalignment-robust fusion.
8. Loss: Cross-entropy + auxiliary.
9. Headline metric: ChangerEx + ResNet-18 → F1 = **92.97** on LEVIR-CD; F1 = 65.99 on S2Looking.
10. Why relevant: Empirically strongest fusion paradigm — feature-level **interaction inside the encoder** (not just late fusion). FDAF directly handles co-registration imperfections in LEVIR-CC-derived data.
11. Code: https://github.com/likyoo/open-cd

#### 25. ChangeMamba — State-Space Spatiotemporal Fusion
1. Title: ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model
2. Authors: Hongruixuan Chen, Jian Song, Chengxi Han, Junshi Xia, Naoto Yokoya
3. Venue + Year: IEEE TGRS 2024
4. arXiv: 2404.03425
5. Task: Binary CD (MambaBCD), Semantic CD (MambaSCD), Building Damage Assessment (MambaBDA).
6. Datasets: LEVIR-CD+, SYSU-CD, WHU-CD, xBD, SECOND.
7. Architecture: **Visual Mamba (VMamba) Siamese encoder** + change decoder with three spatio-temporal interaction mechanisms (sequential / cross / parallel SS scans).
8. Loss: Cross-entropy + Dice.
9. Headline metric: MambaBCD-Tiny achieves F1 = **92.27** on LEVIR-CD+.
10. Why relevant: First SSM-based bi-temporal fusion baseline — linear-complexity alternative to cross-attention, attractive for high-res LEVIR-CC images.
11. Code: https://github.com/ChenHongruixuan/MambaCD

#### 26. TTP — Time Travelling Pixels (Frozen SAM + Cross-Temporal Injection)
1. Title: Time Travelling Pixels: Bitemporal Features Integration with Foundation Model for Remote Sensing Image Change Detection
2. Authors: Keyan Chen, Chenyang Liu, Wenyuan Li, Zili Liu, Hao Chen et al.
3. Venue + Year: IGARSS 2024
4. arXiv: 2312.16202
5. Task: Bi-temporal binary CD.
6. Datasets: LEVIR-CD, WHU-CD, S2Looking.
7. Architecture: **Frozen SAM ViT-H encoder** with LoRA adapters in Siamese form + Time-Travelling Activation Gates (TTAG) that **inject t2 features into the t1 stream and vice-versa across decoder stages**.
8. Loss: Cross-entropy + Dice.
9. Headline metric: F1 = **92.1** on LEVIR-CD; outperforms BAN/ChangeFormer with frozen backbone.
10. Why relevant: Concrete recipe for "frozen large foundation backbone + lightweight bi-temporal injection" — most parameter-efficient SOTA in 2024 and directly transferable to multi-head MTL.
11. Code: https://github.com/KyanChen/TTP

#### 27. BAN — Bi-Temporal Adapter Network
1. Title: A New Learning Paradigm for Foundation Model-Based Remote-Sensing Change Detection
2. Authors: Kaiyu Li, Xiangyong Cao, Deyu Meng
3. Venue + Year: IEEE TGRS 2024
4. arXiv: 2312.01163
5. Task: Bi-temporal CD via foundation-model adaptation.
6. Datasets: LEVIR-CD, WHU-CD, S2Looking, CLCD.
7. Architecture: **Frozen foundation model (CLIP / SAM / RemoteCLIP) + Bi-Temporal Adapter Branch (Bi-TAB)** connected by bridging modules; Bi-TAB is model-agnostic.
8. Loss: BCE + auxiliary.
9. Headline metric: Up to **+4.08 IoU** improvement over existing CD methods (e.g., ChangeFormer + BAN ≈ 84.39 IoU on LEVIR-CD).
10. Why relevant: Reference framework for plugging any frozen backbone (DINOv2/v3, SAM, RemoteCLIP) into a bi-temporal pipeline with adapters — ideal scaffold when full fine-tuning is too expensive on ~10K pairs.
11. Code: https://github.com/likyoo/BAN

#### 28. ChangeBind — Hybrid Local/Global Change Encoder
1. Title: ChangeBind: A Hybrid Change Encoder for Remote Sensing Change Detection
2. Authors: Mubashir Noman, Mustansar Fiaz, Hisham Cholakkal
3. Venue + Year: IGARSS 2024
4. arXiv: 2404.17565
5. Task: Bi-temporal binary CD.
6. Datasets: LEVIR-CD, DSIFN-CD.
7. Architecture: Siamese hybrid (CNN + transformer) backbone → **multi-scale change encoder that concatenates and fuses local + global representations** to estimate subtle and large changes.
8. Loss: Standard CD cross-entropy.
9. Headline metric: ~91.x F1 on LEVIR-CD (paper Table).
10. Why relevant: Explicitly addresses limits of pure-transformer fusion for subtle changes — useful design choice for fine-grained Attribute/Event heads.
11. Code: https://github.com/techmn/changebind

### D. Multi-task learning with shared encoders

#### 29. Uncertainty Weighting (Kendall et al.) — DEFAULT BASELINE
1. Title: Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
2. Authors: Alex Kendall, Yarin Gal, Roberto Cipolla
3. Venue + Year: CVPR 2018
4. arXiv: 1705.07115
5. Task: MTL with regression + classification heads.
6. Datasets: CityScapes, Make3D.
7. Architecture: Shared encoder + task-specific decoders.
8. Loss: Per-task **learned homoscedastic uncertainty** σᵢ as automatic weights: L = Σ (1/2σᵢ²) Lᵢ + log σᵢ.
9. Headline metric: Outperforms separately-trained single-task models on all three tasks (+1.2% mean IoU on segmentation, ~10% relative on depth).
10. Why relevant: Most-cited and simplest principled loss-weighting scheme; ideal default for three classification heads of different intrinsic difficulty.
11. Code: Numerous reimplementations.

#### 30. GradNorm — Gradient-Norm Adaptive Weighting
1. Title: GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks
2. Authors: Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, Andrew Rabinovich
3. Venue + Year: ICML 2018
4. arXiv: 1711.02257
5. Task: General MTL loss balancing.
6. Datasets: NYUv2, synthetic regression.
7. Architecture: Any shared-encoder + task heads.
8. Loss: Dynamically tunes per-task gradient magnitudes via α hyperparameter.
9. Headline metric: Matches/surpasses exhaustive grid-search of fixed loss weights; +0.8% to +3% per task vs. uniform weighting.
10. Why relevant: Reference adaptive-weighting baseline for our three-head setting; lightweight and works at any scale.
11. Code: Multiple PyTorch reimplementations.

#### 31. PCGrad — Gradient Surgery
1. Title: Gradient Surgery for Multi-Task Learning
2. Authors: Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea Finn
3. Venue + Year: NeurIPS 2020
4. arXiv: 2001.06782
5. Task: Multi-task supervised + RL.
6. Datasets: CityScapes, NYUv2, Meta-World RL.
7. Architecture: Shared trunk + task heads.
8. Loss: Projects each task's gradient onto the normal plane of any conflicting task gradient.
9. Headline metric: +30% success rate on Meta-World MT10; +1–3 points on supervised MTL.
10. Why relevant: Best-known fix for gradient interference among related heads; matters if Object/Event/Attribute heads share strong but not identical features.
11. Code: https://github.com/tianheyu927/PCGrad

#### 32. FAMO — Fast Adaptive Multitask Optimization (RECENT)
1. Title: FAMO: Fast Adaptive Multitask Optimization
2. Authors: Bo Liu, Yihao Feng, Peter Stone, Qiang Liu
3. Venue + Year: NeurIPS 2023
4. arXiv: 2306.03792
5. Task: General MTL.
6. Datasets: NYUv2, CityScapes, Meta-World, QM9.
7. Architecture: Any shared-encoder MTL.
8. Loss: Dynamic weights computed in **O(1) memory and time** (vs. PCGrad/CAGrad O(k)) by tracking loss-decrease rates.
9. Headline metric: Δm = −4.10 on NYUv2 (vs. PCGrad's −1.97) while using O(1) memory.
10. Why relevant: Most efficient and recent (2023) loss-balancer; recommended modern default over PCGrad when GPU memory is tight.
11. Code: https://github.com/Cranial-XIX/FAMO

#### 33. DB-MTL — Dual-Balancing for MTL
1. Title: Dual-Balancing for Multi-Task Learning
2. Authors: Baijiong Lin, Weisen Jiang, Feiyang Ye, Yu Zhang et al.
3. Venue + Year: Neural Networks 2026 (arXiv 2023)
4. arXiv: 2308.12029
5. Task: General MTL.
6. Datasets: NYUv2, CityScapes, QM9, Office-31.
7. Architecture: Shared encoder + heads.
8. Loss: **Log-transform** of per-task losses (loss-scale balancing) + gradient normalization by max-grad-norm (gradient-scale balancing).
9. Headline metric: Δm = **−7.66** on NYUv2 (vs. PCGrad −1.97, CAGrad −2.74).
10. Why relevant: Newest balanced MTL method; particularly relevant since our three heads have very different label cardinalities (13/13/25) → very different loss scales.
11. Code: https://github.com/Baijiong-Lin/LibMTL

---

## RING 3 — Multi-Label Classification Tricks (Medium Priority)

### A. Multi-label loss functions for imbalance

#### 34. Asymmetric Loss (ASL) — DEFAULT FIRST LOSS TO TRY
1. Title: Asymmetric Loss For Multi-Label Classification
2. Authors: Emanuel Ben-Baruch, Tal Ridnik, Nadav Zamir et al.
3. Venue + Year: ICCV 2021
4. arXiv: 2009.14119 — DOI: 10.1109/ICCV48922.2021.00015
5. Task: Generic multi-label image classification.
6. Datasets: MS-COCO, PASCAL-VOC, NUS-WIDE, Open Images.
7. Architecture: TResNet backbone + global pooling + binary classifier head.
8. Loss: ASL — decoupled focusing parameters γ_pos / γ_neg + probability margin shift that hard-thresholds easy negatives.
9. Headline metric: **mAP = 86.6 on MS-COCO** (single-crop, TResNet-L); 87.3 mAP with extra tricks.
10. Why relevant: Default loss for sparse-positive multi-label since 2021 — directly addresses the dominant-negative problem in our 51-class scene-level setting.
11. Code: https://github.com/Alibaba-MIIL/ASL

#### 35. Distribution-Balanced Loss
1. Title: Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets
2. Authors: Tong Wu, Qingqiu Huang, Ziwei Liu et al.
3. Venue + Year: ECCV 2020
4. arXiv: 2007.09654
5. Task: Long-tailed multi-label classification.
6. Datasets: VOC-LT, COCO-LT.
7. Architecture: ResNet-50 + global pooling + BCE classifier.
8. Loss: DB loss — rebalanced weighting accounting for label co-occurrence + negative-tolerant regularization.
9. Headline metric: **mAP = 53.55 on COCO-LT** and **78.94 on VOC-LT** (DB-Focal variant).
10. Why relevant: Companion to ASL — explicitly handles long-tail multi-label and over-suppression of negatives; co-occurrence reweighting fits our cross-family label dependencies.
11. Code: https://github.com/wutong16/DistributionBalancedLoss

#### 36. Two-Way Multi-Label Loss
1. Title: Two-Way Multi-Label Loss
2. Authors: Takumi Kobayashi
3. Venue + Year: CVPR 2023
4. DOI: 10.1109/CVPR52729.2023.00722
5. Task: Multi-label image classification.
6. Datasets: MS-COCO, PASCAL-VOC, NUS-WIDE.
7. Architecture: TResNet-L backbone + global pooling + linear classifier.
8. Loss: Two-way loss bridging softmax CE and BCE — relative comparison across classes AND across samples.
9. Headline metric: **mAP = 89.8 on MS-COCO** (TResNet-L, 448×448).
10. Why relevant: 2023 alternative to ASL that may transfer better and produce more discriminative features; worth A/B testing.
11. Code: https://github.com/tk1980/TwowayMultiLabelLoss

#### 37. R-ASL — Robust Asymmetric Loss
1. Title: Robust Asymmetric Loss for Multi-Label Long-Tailed Learning
2. Authors: Wongi Park, Inhyuk Park, Sungeun Kim et al.
3. Venue + Year: ICCVW 2023 (CXR-LT competition, Top-5)
4. arXiv: 2308.05542
5. Task: Long-tailed multi-label medical classification.
6. Datasets: CXR-LT, MIMIC-CXR-LT.
7. Architecture: ConvNeXt / EfficientNet + multi-label head.
8. Loss: Robust polynomial asymmetric loss with Hill-loss regularization, reducing sensitivity to ASL's many hyperparameters.
9. Headline metric: mAP = 0.372 (Top-5 on CXR-LT 2023 leaderboard).
10. Why relevant: Combines both imbalance issues (long-tail + multi-label) we face; Hill-regularization reduces overfitting when ASL is over-tuned.
11. Code: https://github.com/kalelpark/RALoss

### B. Multi-label classification heads with label-attention / correlations

#### 38. Query2Label — Label-as-Query Transformer Head (RECOMMENDED HEAD)
1. Title: Query2Label: A Simple Transformer Way to Multi-Label Classification
2. Authors: Shilong Liu, Lei Zhang, Xiao Yang, Hang Su, Jun Zhu
3. Venue + Year: arXiv 2021
4. arXiv: 2107.10834
5. Task: Multi-label image classification.
6. Datasets: MS-COCO, PASCAL-VOC, NUS-WIDE, Visual Genome.
7. Architecture: CvT/ResNet backbone + Transformer decoder where K learned label embeddings cross-attend over spatial features to produce K binary scores.
8. Loss: ASL.
9. Headline metric: **mAP = 91.3 on MS-COCO** (CvT-w24-384).
10. Why relevant: Canonical "label-as-query" head; cleanly extracts per-class evidence — ideal pattern when we have 3 disjoint label families that can each have their own query bank.
11. Code: https://github.com/SlongLiu/query2labels

#### 39. ML-Decoder — Scalable Multi-Label Head
1. Title: ML-Decoder: Scalable and Versatile Classification Head
2. Authors: Tal Ridnik, Gilad Sharir, Avi Ben-Cohen, Emanuel Ben-Baruch, Asaf Noy
3. Venue + Year: WACV 2023
4. arXiv: 2111.12933
5. Task: Multi-label, single-label, and zero-shot classification.
6. Datasets: MS-COCO, NUS-WIDE (ZSL), ImageNet, OpenImages.
7. Architecture: Any backbone (TResNet/ViT) + ML-Decoder — redesigned Transformer decoder with linear (not quadratic) complexity via group-decoding.
8. Loss: ASL.
9. Headline metric: **mAP = 91.1 on MS-COCO**; ZSL mAP = 31.1 on NUS-WIDE.
10. Why relevant: Linear-complexity head that scales to thousands of classes — efficient drop-in for our 51-class budget; supports word-query generalization, useful since our labels come from captions.
11. Code: https://github.com/Alibaba-MIIL/ML_Decoder

#### 40. CSRA — Class-Specific Residual Attention (CHEAP BASELINE)
1. Title: Residual Attention: A Simple but Effective Method for Multi-Label Recognition
2. Authors: Ke Zhu, Jianxin Wu
3. Venue + Year: ICCV 2021
4. arXiv: 2108.02456
5. Task: Multi-label image recognition.
6. Datasets: MS-COCO, PASCAL-VOC, WIDER-Attribute.
7. Architecture: ResNet/TResNet backbone + class-specific spatial attention combined with class-agnostic GAP (~4 lines of code).
8. Loss: BCE / ASL.
9. Headline metric: **mAP = 86.5 on MS-COCO** (TResNet-L + CSRA, single resolution).
10. Why relevant: Embarrassingly simple class-specific spatial pooling — strong cheap baseline before going to Transformer-decoder heads; works without retraining on top of any backbone.
11. Code: https://github.com/Kevinz-code/CSRA

#### 41. C-Tran — Classification Transformer with Ternary Label States
1. Title: General Multi-label Image Classification with Transformers
2. Authors: Jack Lanchantin, Tianlu Wang, Vicente Ordonez, Yanjun Qi
3. Venue + Year: CVPR 2021
4. arXiv: 2011.14027
5. Task: General multi-label classification (handles partial/extra labels).
6. Datasets: MS-COCO, Visual Genome, News-500, CUB.
7. Architecture: CNN backbone + Transformer encoder over fused visual tokens + ternary-encoded label state tokens (positive/negative/unknown).
8. Loss: BCE with masked-label training objective.
9. Headline metric: **mAP = 85.1 on MS-COCO** (ResNet-101).
10. Why relevant: Label-mask training is directly useful for our partial-label situation (captions may omit some changes); explicit label-state modeling captures Object↔Event↔Attribute dependencies.
11. Code: https://github.com/QData/C-Tran

#### 42. ML-GCN — Label Graph for Multi-Label Recognition
1. Title: Multi-Label Image Recognition with Graph Convolutional Networks
2. Authors: Zhao-Min Chen, Xiu-Shen Wei, Peng Wang, Yanwen Guo
3. Venue + Year: CVPR 2019
4. arXiv: 1904.03582
5. Task: Multi-label image recognition.
6. Datasets: MS-COCO, PASCAL-VOC 2007.
7. Architecture: ResNet-101 + GCN over directed label graph (nodes = word-embedded labels, edges = co-occurrence).
8. Loss: Multi-label CE.
9. Headline metric: **mAP = 83.0 on MS-COCO**, **mAP = 94.0 on VOC-2007**.
10. Why relevant: Foundational label-correlation method; the conditional co-occurrence graph is a natural fit for modeling cross-family dependencies.
11. Code: https://github.com/megvii-research/ML-GCN

### C. Per-class threshold optimization

#### 43. Neural Thresholding for Multi-Label
1. Title: From Scores to Predictions in Multi-Label Classification: Neural Thresholding Strategies
2. Authors: Karol Draszawka, Julian Szymański
3. Venue + Year: Applied Sciences 2023
4. DOI: 10.3390/app13137591
5. Task: Per-class threshold learning post-scoring for multi-label.
6. Datasets: Reuters-21578, Bibtex, Mediamill, Delicious.
7. Architecture: Neural-network thresholding head taking full per-class score vector → per-sample per-class binary decisions.
8. Loss: Custom F1-targeting loss for threshold network.
9. Headline metric: Up to **+40.6% relative improvement in micro-F1** over global-threshold or per-class constant-threshold baselines.
10. Why relevant: Direct solution to our per-class threshold optimization need — learned thresholds conditioned on the full score vector beat fixed thresholds.
11. Code: Referenced in paper; not centrally hosted.

#### 44. Adaptive Thresholding via Global-Local Fusion
1. Title: Adaptive Thresholding for Multi-Label Classification via Global-Local Signal Fusion
2. Authors: Dmytro Shamatrin
3. Venue + Year: arXiv 2025
4. arXiv: 2505.03118
5. Task: Per-instance, per-label adaptive thresholding for noisy/imbalanced multi-label.
6. Datasets: AmazonCat-13K.
7. Architecture: Lightweight MLP fusing global IDF-based label rarity + local KNN-based contextual signal.
8. Loss: BCE + differentiable threshold penalty.
9. Headline metric: Macro-F1 = 0.1712 on AmazonCat-13K; substantially beats tree-based and transformer baselines.
10. Why relevant: 2025 method producing per-label, per-instance thresholds — relevant since both global rarity and local context matter for our 51-class problem.
11. Code: Released by author.

### D. Hierarchical / grouped multi-label

#### 45. C-HMCNN — Coherent Hierarchical Multi-Label
1. Title: Coherent Hierarchical Multi-Label Classification Networks
2. Authors: Eleonora Giunchiglia, Thomas Lukasiewicz
3. Venue + Year: NeurIPS 2020
4. arXiv: 2010.10151
5. Task: Hierarchical multi-label classification with hard taxonomy constraints.
6. Datasets: 20 standard HMC benchmarks (FunCat, Gene Ontology, etc.).
7. Architecture: Any base network + Max Constraint Module enforcing parent-child coherence via max-pool over hierarchy edges.
8. Loss: BCE adapted to back-propagate gradients only through hierarchy-respecting predictions.
9. Headline metric: **Avg AU(PRC) = 0.799** across 20 datasets — beats Clus-HMC, HMC-LMLP, HMCN-F.
10. Why relevant: The MCM trick generalizes to enforcing structured constraints across our label groups (e.g., "no Event without an Object").
11. Code: https://github.com/EGiunchiglia/C-HMCNN

### E. Multi-label remote sensing classification (RS-specific)

#### 46. L-GCMA — Label-Guided Cross-Modal Attention for Aerial Multi-Label
1. Title: Label-Guided Cross-Modal Attention Network for Multi-Label Aerial Image Classification
2. Authors: Ying Chen, Ding Zhang, Tao Han, Xiaoliang Meng, Mianxin Gao, Teng Wang
3. Venue + Year: IEEE GRSL 2024
4. DOI: 10.1109/LGRS.2024.3388568
5. Task: Multi-label aerial scene classification.
6. Datasets: UCM multi-label, AID multi-label.
7. Architecture: Transformer visual encoder + label-sentence mapping attention (BERT-encoded prompts + multi-head attention over class names) + cross-modal attention with text as query.
8. Loss: BCE / ASL.
9. Headline metric: **mAP = 99.10 on UCM multi-label**, **mAP = 85.96 on AID multi-label**.
10. Why relevant: Closest method to our setup — uses class-name text embeddings as queries (a free signal we have since labels are caption-derived) for RS multi-label classification.
11. Code: Not publicly confirmed in abstract.

#### 47. AdaGC — Adaptive Gradient Calibration for SPML in RS
1. Title: Adaptive Gradient Calibration for Single-Positive Multi-Label Learning in RS Scene Classification
2. Authors: Chenying Liu, Gianmarco Perantoni, Lorenzo Bruzzone, Xiao Xiang Zhu
3. Venue + Year: IEEE TGRS 2025
4. arXiv: 2510.08269
5. Task: Single-positive multi-label classification for RS.
6. Datasets: AID multi-label, DFC15 multi-label.
7. Architecture: ResNet/ViT + dual EMA branches generating pseudo-labels + gradient-calibration mechanism + training-dynamics adaptive trigger.
8. Loss: BCE + GC-rectified gradient.
9. Headline metric: +3–7% mAP over SPML baselines on AID-MLC and DFC15-MLC.
10. Why relevant: RS-specific; our caption-derived labels behave like partial-positive data, exactly AdaGC's regime.
11. Code: https://github.com/rslab-unitrento/AdaGC

#### 48. SoftCon — Multi-Label Guided Soft Contrastive RS Pretraining
1. Title: Multi-Label Guided Soft Contrastive Learning for Efficient Earth Observation Pretraining
2. Authors: Yi Wang, Conrad M. Albrecht, Xiao Xiang Zhu
3. Venue + Year: arXiv 2024
4. arXiv: 2405.20462
5. Task: Self-supervised pretraining for Earth observation.
6. Datasets: GeoPile, BigEarthNet.
7. Architecture: DINOv2 backbone continued-pretrained with Siamese masking, supervised by land-cover-generated multi-label soft similarities.
8. Loss: Soft contrastive loss guided by label similarity.
9. Headline metric: 86.8 linear-probing mAP on BigEarthNet-10%.
10. Why relevant: Demonstrates how to leverage multi-label correlations to adapt DINOv2 specifically for RS scene classification — directly relevant to our encoder pretraining strategy.
11. Code: https://github.com/zhu-xlab/softcon

### F. Partial / noisy labels in multi-label

#### 49. CSL — Class-aware Selective Loss for Partial Annotations
1. Title: Multi-label Classification with Partial Annotations using Class-aware Selective Loss
2. Authors: Emanuel Ben-Baruch, Tal Ridnik, Itamar Friedman et al.
3. Venue + Year: CVPR 2022
4. arXiv: 2110.10955
5. Task: Multi-label classification with partial/missing labels.
6. Datasets: OpenImages V6, LVIS, simulated-COCO.
7. Architecture: TResNet-L + ML-Decoder + temporary model estimating per-class label distribution.
8. Loss: Class-aware Selective Loss — asymmetric loss that selectively treats un-annotated labels based on (a) estimated class prior and (b) per-sample label likelihood.
9. Headline metric: **mAP = 87.3 on OpenImages V6** (single TResNet-L).
10. Why relevant: Direct fit for our exact setting — caption-derived labels are partial positives; CSL is the canonical "treat unobserved labels selectively" method.
11. Code: https://github.com/Alibaba-MIIL/PartialLabelingCSL

#### 50. GPR / AEVLP — 2025 SPML SOTA
1. Title: More Reliable Pseudo-Labels, Better Performance: A Generalized Approach to Single Positive Multi-Label Learning
2. Authors: Luong Tran, T. Vo, Anh Nguyen, Sang Dinh, V. Nguyen
3. Venue + Year: ICCV 2025
4. arXiv: 2508.20381
5. Task: Single Positive Multi-Label Learning.
6. Datasets: MS-COCO, PASCAL-VOC, NUS-WIDE, CUB-200-2011 (single-positive variants).
7. Architecture: CLIP-based vision-language encoder + Dynamic Augmented Multi-focus Pseudo-labeling (DAMP) + Generalized Pseudo-Label Robust Loss (GPR).
8. Loss: GPR Loss — noise-robust BCE variant consuming multiple pseudo-label sources.
9. Headline metric: **mAP = 80.7 on MS-COCO single-positive setting** (SOTA).
10. Why relevant: 2025 SPML method; our caption-decomposed labels are exactly the partial-positive case, and multi-source pseudo-label fusion helps bootstrap missing labels from a VLM.
11. Code: Promised in paper.

#### 51. COMIC — Long-Tail + Partial Multi-Label
1. Title: COMIC: Multi-Label Classification with Long-Tailed Distribution and Partial Labels
2. Authors: Wenqiao Zhang, Changshuo Liu, Lingze Zeng et al.
3. Venue + Year: ICCV 2023
4. arXiv: 2304.10539
5. Task: Multi-label classification under long-tail and partial labels.
6. Datasets: PLT-MLC benchmarks.
7. Architecture: Correction → modification → balance pipeline.
8. Loss: Multi-focal modifier loss + balanced classifier.
9. Headline metric: Significantly outperforms LT-MLC and PL-MLC baselines.
10. Why relevant: Our labels are simultaneously long-tail (sparse positive classes) AND partial (captions omit changes) — COMIC addresses both jointly.
11. Code: https://github.com/wannature/COMIC

---

## Synthesis & Strategic Recommendations

### Must-Cite Papers (10)

| # | Paper | Why must-cite |
|---|-------|---------------|
| 1 | **RSICCformer (Liu et al., TGRS 2022)** | LEVIR-CC dataset origin; mandatory citation |
| 2 | **Change-Agent / LEVIR-MCI (Liu et al., TGRS 2024)** | The multi-task superset on our exact image collection |
| 3 | **Semantic-CC (Zhu et al., TGRS 2024)** | Closest prior art: Siamese SAM + multi-task heads on LEVIR-CC |
| 4 | **ChangeIMTI / ChangeVG (Xue et al., 2025)** | First to treat scene-level change classification as a first-class task |
| 5 | **CDVQA (Yuan et al., TGRS 2022)** | First to recast change interpretation as categorical classification |
| 6 | **TransWCD (Zhao et al., TGRS 2025)** | Best architectural template — scene-level classifier on bi-temporal pairs |
| 7 | **ChangeFormer (Bandara & Patel, IGARSS 2022)** | Canonical Siamese-transformer + abs-diff fusion baseline |
| 8 | **Query2Label (Liu et al., 2021)** | Recommended multi-label head for our 3 disjoint families |
| 9 | **Asymmetric Loss (Ben-Baruch et al., ICCV 2021)** | Default first loss for sparse-positive multi-label |
| 10 | **SECOND-CC / MModalCC (Karaca et al., JSTARS 2025)** | Turkish-authored regional alignment; sibling dataset to LEVIR-CC |

### Numerical Baselines to Beat (3)

1. **TransWCD** — adapt the head from 2-way to (13+13+25)-way multi-label BCE/ASL. Code at https://github.com/zhenghuizhao/TransWCD. Most defensible numerical comparison on a scene-level supervision regime.
2. **Query2Label + ASL on a Siamese ConvNeXt/Swin encoder** — a clean, mature, well-instrumented SOTA recipe. Will likely be our reported supervised baseline.
3. **Change-Agent / Semantic-CC captioning head argmax → derived labels** — extract token-level argmax predictions from a captioning model and compute per-family F1. This is an *honest negative baseline* — captioning loss does not optimize label coverage, so we expect poor per-class F1, which is the negative result we should report to motivate our discrete-classification formulation.

### State of the Field (drop-in related-work paragraph)

> Remote sensing change understanding has bifurcated into three partially overlapping lines: pixel-level binary/semantic change detection (ChangeFormer, BIT, ChangeMamba, ChangeBind), free-form change captioning (RSICCformer, SECOND-CC, KCFI, Semantic-CC), and interactive vision-language change interpretation (Change-Agent, ChangeIMTI, VisTA). The strongest recent gains come not from deeper decoders but from better bi-temporal interaction: token-space transformer fusion (BIT), in-encoder feature exchange with flow-based alignment (ChangerEx + FDAF), state-space scans (ChangeMamba), and parameter-efficient adapter injection on top of frozen foundation backbones such as SAM and DINOv3 (TTP, BAN, PeftCD). In parallel, the weakly supervised CD literature (TransWCD, AdvCP, DISep) has demonstrated that image-level supervision is usable but noisy, motivating class-prompt mining, pseudo-label cleanup, and instance separation. On the classification side, multi-label image recognition has converged on Transformer-decoder heads where learnable label embeddings cross-attend to spatial features (Query2Label, ML-Decoder, C-Tran, CSRA), combined with asymmetric or distribution-balanced losses (ASL, DB-Loss, Two-Way Loss) to combat positive-negative imbalance. Despite this progress, the integration of structured multi-label classification directly with bi-temporal remote sensing pairs remains essentially untouched: there is no public benchmark for structured (Object, Event, Attribute) scene-level multi-label change triples, and no method that jointly handles their imbalance, cross-family co-occurrence, and partial-label noise inherited from caption-decomposition.

### Gaps & Open Problems (Our Positioning)

- **Gap 1 — No scene-level structured change classifier exists.** All public datasets are either pixel-level (LEVIR-CD family, SECOND, xBD) or unstructured free-form text (LEVIR-CC, SECOND-CC, ChangeIMTI). Our decomposed LEVIR-CC is the first benchmark in the structured multi-label regime.
- **Gap 2 — Multi-task label constraints are unexploited.** Captioning decoders treat descriptions as flat token sequences, ignoring the hierarchical relationship between *what* changed (Object), *how* it changed (Event), and its *properties* (Attribute). Modeling these as three distinct label families with shared Siamese encoders introduces a lightweight, interpretable, structured formulation.
- **Gap 3 — Suboptimal temporal fusion under asymmetric label noise.** Vanilla concat/difference fusion is noise-sensitive to misregistration and illumination differences. Cross-temporal attention (SiamixFormer) regularized by multi-label co-occurrence loss (DB-loss, ML-GCN) is a promising under-tested combination on small datasets.
- **Gap 4 — Per-class threshold calibration is rarely treated as a first-class problem in RS multi-label.** None of the RS-specific multi-label papers (L-GCMA, SoftCon, AdaGC) include neural thresholding heads. Adopting Draszawka & Szymański 2023 or Shamatrin 2025 inside an RS pipeline is an open win.
- **Gap 5 — Caption-to-label decomposition has no published methodology for RS.** Pix4Cap goes pixel-mask → caption; nobody publishes the reverse direction (caption → structured discrete labels) for satellite imagery. Our preprocessing pipeline (parsing LEVIR-CC captions into Object/Event/Attribute triples) is itself contributable.

### Empirically Defensible Modeling Recipe (Synthesis)

Based on cross-paper evidence:
1. **Encoder.** Siamese ConvNeXt or DINOv3-PEFT (PeftCD, NeXt2Former-CD evidence); 10K pairs is enough to fine-tune with LoRA but probably not enough for full fine-tuning of a >300M parameter ViT.
2. **Fusion.** ChangerEx-style channel feature exchange + FDAF alignment (highest empirical wins in Changer 2023), or alternatively asymmetric temporal cross-attention at every stage (SiamixFormer). Skip plain concat or abs-diff.
3. **Heads.** Three parallel Query2Label / ML-Decoder heads — one per label family — with learned label embeddings initialized from the class names in the LEVIR-CC vocabulary.
4. **Loss.** ASL (γ_neg = 4, γ_pos = 0) per head, summed under Kendall uncertainty weighting (default) or DB-MTL (modern). FAMO if memory-constrained.
5. **Threshold calibration.** Per-class learned thresholds (Draszawka & Szymański 2023) on the val set; report both macro-F1 (per-class threshold) and micro-F1 (single global threshold).
6. **Partial-label correction.** CSL or AdaGC on top, treating captions as partial-positive annotations.
7. **Honest negative baseline.** Run Change-Agent/Semantic-CC, argmax their generated captions into our 51-class space, and report per-family F1. Expect ≤ 0.4 macro-F1, which justifies the discrete-classification reformulation.

### Special Flags Surfaced

- **No LEVIR-CC classification paper.** The exact target task is unpublished. This is genuine novelty.
- **Turkish/regional cluster:** Karaca/Amasyalı group (YTÜ, `ChangeCapsInRS` GitHub org) — SECOND-CC, MModalCC, related work; primary regional anchor. Kömürcü & Petkevičius (IEEE AIEEE 2024) — zero-shot CLIP scene classification. Taskesen et al. (METU, SIU 2017) — foundational regional CD baseline. Akyon & Temizel (METU, 2026) — Query2Label + ASL multi-label recipe. Demir (TU Berlin, Turkish-authored) — Seabed-Net uncertainty MTL.
- **Expected honest negative result:** Captioning-derived label argmax should *underperform* our discrete classifier substantially. This is the result that justifies our task formulation.
- **Empirical fusion ranking** (from the cited literature, not all on the same dataset, treat with care): ChangerEx feature exchange (~F1 92.97 LEVIR-CD) ≥ TTP frozen-SAM + bi-temporal injection (~F1 92.1) ≥ MambaBCD scans (~F1 92.27) ≥ SiamixFormer cross-attention (F1 91.32) ≥ ChangeFormer abs-diff (F1 ~90.4) ≥ BIT token-transformer (F1 89.31). All beat plain concat at comparable scale.
