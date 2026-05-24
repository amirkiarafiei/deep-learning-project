# Remote sensing bi-temporal multi-label classification: literature map

I did not find a paper that exactly matches the target task: **scene-level multi-label classification of bitemporal change into object / event / attribute triples derived from LEVIR-CC**.

The closest published lines are:
- **weakly supervised change detection** with scene-level labels,
- **change captioning / change interpretation** with an auxiliary CD branch,
- **semantic change detection** and **interactive RS change analysis**,
- and generic **multi-label / multi-task** methods for class imbalance and label correlation.

## Ring 1: directly related

### 1) TransWCD: Scene-Adaptive Joint Constrained Framework for Weakly Supervised Change Detection
- **Authors:** Zhenghui Zhao, Lixiang Ru, Chen Wu et al.
- **Venue / year:** IEEE TGRS, 2025
- **ID:** DOI 10.1109/TGRS.2025.3545051
- **Task:** weakly supervised change detection; scene-level binary change classification
- **Datasets:** WHU-CD, LEVIR-CD, DSIFN-CD
- **Architecture:** hierarchical transformer classifier + scene-adaptive predictor, trained end-to-end with joint constraints and multiscale CAMs
- **Loss:** label-prediction consistency / scene-gated constraint plus standard WSCD supervision
- **Headline metric:** significant improvement over prior WSCD baselines; exact number not in abstract
- **Why relevant:** this is the closest scene-level-label formulation to your setup; it shows how far image-level supervision can be pushed before pixel masks become necessary
- **Code:** https://github.com/zhenghuizhao/TransWCD

### 2) Advancing Weakly-Supervised Change Detection in Satellite Images via Adversarial Class Prompting
- **Authors:** Zhenghui Zhao, Chen Wu, Di Wang et al.
- **Venue / year:** IEEE TIP, 2025
- **ID:** arXiv 2508.17186 / DOI 10.1109/TIP.2025.3623260
- **Task:** weakly supervised change detection with image-level labels
- **Datasets:** WHU-CD, LEVIR-CD, DSIFN-CD, SYSU-CD, CDD
- **Architecture:** adversarial prompt mining + global prototype rectification; plug-in for ConvNet / Transformer / SAM baselines
- **Loss:** adversarial prompting + rectification under weak supervision
- **Headline metric:** up to **+7.46 IoU** on LEVIR-CD
- **Why relevant:** strongest evidence that scene-level supervision can be made robust against background co-occurrence noise
- **Code:** https://github.com/zhenghuizhao/AdvCP

### 3) Weakly Supervised Change Detection via Knowledge Distillation and Multiscale Sigmoid Inference
- **Authors:** Binghao Lu, Caiwen Ding, Jinbo Bi et al.
- **Venue / year:** arXiv, 2024
- **ID:** arXiv 2403.05796
- **Task:** weakly supervised change detection from image-level labels
- **Datasets:** WHU-CD, DSIFN-CD, LEVIR-CD
- **Architecture:** teacher-student CAM distillation + MSI refinement
- **Loss:** knowledge distillation with image-level supervision
- **Headline metric:** reported as SOTA on the three benchmarks; exact figure not in abstract
- **Why relevant:** useful baseline for label-only training and CAM-derived pseudo change supervision
- **Code:** not stated in abstract

### 4) Plug-and-Play DISep: Separating Dense Instances for Scene-to-Pixel Weakly-Supervised Change Detection in High-Resolution Remote Sensing Images
- **Authors:** Zhenghui Zhao, Chen Wu, Lixiang Ru et al.
- **Venue / year:** ISPRS JPRS, 2025
- **ID:** arXiv 2501.04934 / DOI 10.1016/j.isprsjprs.2025.01.007
- **Task:** weakly supervised change detection under scene-level supervision
- **Datasets:** LEVIR-CD, WHU-CD, DSIFN-CD, SYSU-CD, CDD
- **Architecture:** instance localization + instance retrieval + instance separation
- **Loss:** separation loss on per-instance embeddings
- **Headline metric:** improves three Transformer-based and four ConvNet-based WSCD methods; exact number not in abstract
- **Why relevant:** directly addresses the "instance lumping" problem that will also appear if your labels are only scene-level
- **Code:** https://github.com/zhenghuizhao/Plug-and-Play-DISep-for-Change-Detection

### 5) Pixel-Level Change Detection Pseudo-Label Learning for Remote Sensing Change Captioning
- **Authors:** Chenyang Liu, Keyan Chen, Zipeng Qi et al.
- **Venue / year:** IGARSS, 2024
- **ID:** arXiv 2312.15311 / DOI 10.1109/IGARSS53475.2024.10642750
- **Task:** RS change captioning with auxiliary CD branch
- **Datasets:** LEVIR-CC
- **Architecture:** CD pseudo-label branch + semantic fusion augment + caption decoder
- **Loss:** captioning loss plus pseudo-label CD supervision
- **Headline metric:** SOTA on RSICC; exact number not in abstract
- **Why relevant:** closest "caption-to-label" style bridge; shows captioning can benefit from explicit change labels
- **Code:** https://github.com/Chen-Yang-Liu/Pix4Cap

### 6) Semantic-CC: Boosting Remote Sensing Image Change Captioning via Foundational Knowledge and Semantic Guidance
- **Authors:** Yongshuo Zhu, Lu Li, Keyan Chen et al.
- **Venue / year:** IEEE TGRS, 2024
- **ID:** arXiv 2407.14032 / DOI 10.1109/TGRS.2024.3497338
- **Task:** RS change captioning + pixel-level semantic CD
- **Datasets:** LEVIR-CC, LEVIR-CD
- **Architecture:** bi-temporal SAM encoder + multitask semantic aggregation neck + multiscale CD decoder + LLM caption decoder
- **Loss:** joint CD + CC with three-stage training
- **Headline metric:** optimal performance on both LEVIR-CC and LEVIR-CD; exact number not in abstract
- **Why relevant:** one of the strongest examples of using change semantics as a helper signal for language generation
- **Code:** not stated in abstract

### 7) Change-Agent: Towards Interactive Comprehensive Remote Sensing Change Interpretation and Analysis
- **Authors:** Chenyang Liu, Keyan Chen, Haotian Zhang et al.
- **Venue / year:** arXiv, 2024
- **ID:** arXiv 2403.19646
- **Task:** interactive remote sensing change interpretation (change detection + captioning + counting + cause analysis)
- **Datasets:** LEVIR-MCI
- **Architecture:** multi-level change interpretation model with pixel-level CD branch and semantic-level captioning branch; BI3 interaction layer
- **Loss:** joint CD + captioning supervision
- **Headline metric:** SOTA for simultaneous detection and description; exact number not in abstract
- **Why relevant:** probably the clearest precedent for a structured change-interpretation system
- **Code:** https://github.com/Chen-Yang-Liu/Change-Agent

### 8) Remote Sensing Image Change Captioning With Dual-Branch Transformers: A New Method and a Large Scale Dataset
- **Authors:** Chenyang Liu, Rui Zhao, Hao Chen et al.
- **Venue / year:** IEEE TGRS, 2022
- **ID:** DOI 10.1109/TGRS.2022.3218921
- **Task:** RS change captioning; dataset introduction
- **Datasets:** LEVIR-CC (10,077 image pairs, 50,385 sentences)
- **Architecture:** CNN feature extractor + dual-branch Transformer encoder + caption decoder
- **Loss:** caption generation loss
- **Headline metric:** **+4.98 BLEU-4** and **+9.86 CIDEr-D** over prior methods
- **Why relevant:** the canonical LEVIR-CC baseline and the main dataset citation
- **Code:** not stated in abstract

### 9) Robust Change Captioning in Remote Sensing: SECOND-CC Dataset and MModalCC Framework
- **Authors:** Ali Can Karaca, M. Enes Ozelbas, Saadettin Berber et al.
- **Venue / year:** arXiv, 2025
- **ID:** arXiv 2501.10075
- **Task:** RS change captioning
- **Datasets:** SECOND-CC (6,041 pairs, 30,205 sentences)
- **Architecture:** multimodal attention with Cross-Modal Cross Attention and Multimodal Gated Cross Attention
- **Loss:** captioning loss with semantic/visual fusion
- **Headline metric:** **+4.6 BLEU-4** and **+9.6 CIDEr** over prior methods
- **Why relevant:** useful if you want a second benchmark beyond LEVIR-CC with richer semantic maps
- **Code:** https://github.com/ChangeCapsInRS/SecondCC

## Ring 2: architecture building blocks

### 1) ChangeFormer: A Transformer-Based Siamese Network for Change Detection
- **Authors:** W. G. C. Bandara, Vishal M. Patel
- **Venue / year:** IGARSS, 2022
- **ID:** arXiv 2201.01293 / DOI 10.1109/IGARSS46834.2022.9883686
- **Task:** binary change detection
- **Datasets:** LEVIR-CD, CDD
- **Architecture:** hierarchical Transformer encoder in a Siamese setup + MLP decoder
- **Loss:** standard CD loss
- **Headline metric:** SOTA on two CD datasets; exact number not in abstract
- **Why relevant:** the baseline transformer-CD architecture almost every later paper compares against
- **Code:** github.com/wgcban/ChangeFormer

### 2) ChangeBind: A Hybrid Change Encoder for Remote Sensing Change Detection
- **Authors:** Mubashir Noman, M. Fiaz, Hisham Cholakkal
- **Venue / year:** IGARSS, 2024
- **ID:** arXiv 2404.17565 / DOI 10.1109/IGARSS53475.2024.10640559
- **Task:** binary change detection
- **Datasets:** two CD benchmarks (not named in abstract)
- **Architecture:** Siamese framework with a hybrid change encoder mixing local and global multiscale features
- **Loss:** standard CD loss
- **Headline metric:** SOTA on two datasets; exact number not in abstract
- **Why relevant:** strong "hybrid encoder" reference for combining local detail with global context
- **Code:** https://github.com/techmn/changebind

### 3) SiamixFormer: A Siamese Transformer Network For Building Detection And Change Detection From Bi-Temporal Remote Sensing Images
- **Authors:** Amir Mohammadian, Foad Ghaderi
- **Venue / year:** International Journal of Remote Sensing, 2022
- **ID:** arXiv 2208.00657 / DOI 10.1080/01431161.2023.2225228
- **Task:** building detection and change detection
- **Datasets:** xBD, WHU, LEVIR-CD, CDD
- **Architecture:** two encoders + temporal Transformer fusion where pre-image queries post-image key/value
- **Loss:** detection / segmentation loss
- **Headline metric:** outperforms prior work on the cited benchmarks; exact number not in abstract
- **Why relevant:** clean reference for explicit cross-temporal attention as the fusion step
- **Code:** not stated in abstract

### 4) Changer: Feature Interaction is What You Need for Change Detection
- **Authors:** Sheng Fang, Kaiyu Li, Zhe Li
- **Venue / year:** TGRS, 2022
- **ID:** arXiv 2209.08290 / DOI 10.1109/TGRS.2023.3277496
- **Task:** binary change detection
- **Datasets:** standard CD benchmarks
- **Architecture:** MetaChanger with interaction layers, plus ChangerAD / ChangerEx variants and FDAF fusion
- **Loss:** standard CD loss
- **Headline metric:** competitive on multiple scale CD datasets; exact number not in abstract
- **Why relevant:** one of the clearest papers arguing that interaction/fusion matters more than deeper backbones
- **Code:** https://github.com/likyoo/open-cd

### 5) A VHR Bi-Temporal Remote-Sensing Image Change Detection Network Based on Swin Transformer
- **Authors:** Yunhe Teng, Shuoxun Liu, Weichao Sun et al.
- **Venue / year:** Remote Sensing, 2023
- **ID:** DOI 10.3390/rs15102645
- **Task:** binary change detection
- **Datasets:** LEVIR-CD, CDD
- **Architecture:** Swin backbone + foreground-aware fusion module with attention gates
- **Loss:** standard CD loss
- **Headline metric:** **F1 = 91.78** on LEVIR-CD and **97.87** on CDD
- **Why relevant:** strong Swin-era baseline with explicit foreground-aware fusion
- **Code:** not stated in abstract

### 6) ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model
- **Authors:** Hongruixuan Chen, Jian Song, Chengxi Han et al.
- **Venue / year:** IEEE TGRS, 2024
- **ID:** arXiv 2404.03425 / DOI 10.1109/TGRS.2024.3417253
- **Task:** binary CD, semantic CD, building damage assessment
- **Datasets:** five benchmark datasets
- **Architecture:** Visual Mamba encoder + three spatiotemporal decoders for BCD/SCD/BDA
- **Loss:** task-specific CD / SCD / BDA losses
- **Headline metric:** beats CNN- and Transformer-based methods across five datasets; exact number not in abstract
- **Why relevant:** strongest recent evidence that state-space models are competitive for bitemporal fusion
- **Code:** https://github.com/ChenHongruixuan/MambaCD

### 7) LCCDMamba: Visual State Space Model for Land Cover Change Detection of VHR Remote Sensing Images
- **Authors:** Jun Huang, Xiaochen Yuan, C. Lam et al.
- **Venue / year:** JSTARS, 2025
- **ID:** DOI 10.1109/JSTARS.2025.3531499
- **Task:** land-cover change detection
- **Datasets:** WHU-CD, LEVIR-CD, GVLM
- **Architecture:** Siam-VMamba backbone + multiscale information spatio-temporal fusion + dual-token modeling SSM decoder
- **Loss:** standard CD loss
- **Headline metric:** **F1 = 94.18 / 91.68 / 87.14** on WHU-CD / LEVIR-CD / GVLM
- **Why relevant:** good example of Mamba-style fusion beating older transformer/CNN baselines
- **Code:** not stated in abstract

### 8) Combining SAM With Limited Data for Change Detection in Remote Sensing
- **Authors:** Junyu Gao, Da Zhang, Feiyu Wang et al.
- **Venue / year:** IEEE TGRS, 2025
- **ID:** DOI 10.1109/TGRS.2025.3545040
- **Task:** change detection with limited data
- **Datasets:** limited-data change detection benchmarks
- **Architecture:** FastSAM frozen backbone + CNN adapter + pixel-level binarization module
- **Loss:** adapter/decoder training plus learned thresholding
- **Headline metric:** strong low-data performance and zero-shot ability; exact number not in abstract
- **Why relevant:** the most direct SAM-based CD baseline for low-label settings
- **Code:** Meta-CD repo mentioned in abstract

### 9) FTA-Net: Frequency-Temporal-Aware Network for Remote Sensing Change Detection
- **Authors:** Taojun Zhu, Zikai Zhao, Min Xia et al.
- **Venue / year:** JSTARS, 2025
- **ID:** DOI 10.1109/JSTARS.2025.3525595
- **Task:** binary CD
- **Datasets:** three CD datasets
- **Architecture:** Transformer-INN feature extractor + frequency-temporal fusion + stepwise modification detection
- **Loss:** supervised reweighting / time-difference losses
- **Headline metric:** fewer params and FLOPs while outperforming SOTA; exact number not in abstract
- **Why relevant:** a compact recent example of multi-stage fusion without heavy decoders
- **Code:** not stated in abstract

## Ring 3: multi-label classification tricks

### 1) Asymmetric Loss For Multi-Label Classification
- **Authors:** Emanuel Ben Baruch, T. Ridnik, Nadav Zamir et al.
- **Venue / year:** ICCV, 2021
- **ID:** arXiv 2009.14119 / DOI 10.1109/ICCV48922.2021.00015
- **Task:** generic multi-label classification
- **Datasets:** MS-COCO, Pascal VOC, NUS-WIDE, Open Images
- **Architecture:** any backbone + sigmoid multi-label head
- **Loss:** asymmetric loss that down-weights easy negatives and suppresses mislabeled negatives
- **Headline metric:** SOTA mAP on major multi-label datasets; exact numbers vary by dataset
- **Why relevant:** still the first loss to try for long-tailed label imbalance in your three-family setup
- **Code:** https://github.com/Alibaba-MIIL/ASL

### 2) Robust Asymmetric Loss for Multi-Label Long-Tailed Learning
- **Authors:** Wongi Park, Inhyuk Park, Sungeun Kim et al.
- **Venue / year:** ICCVW, 2023
- **ID:** arXiv 2308.05542 / DOI 10.1109/ICCVW60793.2023.00286
- **Task:** long-tailed multi-label classification
- **Datasets:** medical multi-label benchmarks; CXR-LT competition
- **Architecture:** loss-only paper
- **Loss:** robust polynomial asymmetric loss with Hill-loss regularization
- **Headline metric:** top-5 on CXR-LT
- **Why relevant:** direct follow-up to ASL with better robustness to hyperparameter sensitivity
- **Code:** https://github.com/kalelpark/RALoss

### 3) Asymmetric Polynomial Loss for Multi-Label Classification
- **Authors:** Yusheng Huang, Jiexing Qi, Xinbing Wang et al.
- **Venue / year:** ICASSP, 2023
- **ID:** arXiv 2304.05361 / DOI 10.1109/ICASSP49357.2023.10095437
- **Task:** generic multi-label classification
- **Datasets:** relation extraction, text classification, image classification
- **Architecture:** loss-only paper
- **Loss:** asymmetric polynomial loss that separates positive/negative gradient contributions
- **Headline metric:** consistent gains across tasks; exact number not central in the abstract
- **Why relevant:** another good loss candidate if ASL is too negative-sample dominant
- **Code:** not stated in abstract

### 4) Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets
- **Authors:** Tong Wu, Qingqiu Huang, Ziwei Liu et al.
- **Venue / year:** ECCV, 2020
- **ID:** arXiv 2007.09654 / DOI 10.1007/978-3-030-58548-8_10
- **Task:** generic multi-label classification
- **Datasets:** Pascal VOC, COCO
- **Architecture:** loss-only paper
- **Loss:** DB loss with co-occurrence-aware reweighting + negative-tolerant regularization
- **Headline metric:** significant gains on VOC and COCO
- **Why relevant:** good alternative to ASL when label co-occurrence is strong
- **Code:** code/models available (as stated in abstract)

### 5) Two-Way Multi-Label Loss
- **Authors:** Takumi Kobayashi
- **Venue / year:** CVPR, 2023
- **ID:** DOI 10.1109/CVPR52729.2023.00722
- **Task:** generic multi-label classification
- **Datasets:** standard multi-label image benchmarks
- **Architecture:** loss-only paper
- **Loss:** two-way loss bridging softmax-style relative comparison and BCE
- **Headline metric:** competitive vs other multi-label losses
- **Why relevant:** a good choice when you want stronger margin structure than plain BCE
- **Code:** https://github.com/tk1980/TwowayMultiLabelLoss

### 6) Query2Label: A Simple Transformer Way to Multi-Label Classification
- **Authors:** Shilong Liu, Lei Zhang, Xiao Yang et al.
- **Venue / year:** arXiv, 2021
- **ID:** arXiv 2107.10834
- **Task:** generic multi-label image classification
- **Datasets:** MS-COCO, Pascal VOC, NUS-WIDE, Visual Genome
- **Architecture:** Transformer decoder queries label existence over CNN features
- **Loss:** label-mask training objective
- **Headline metric:** **91.3 mAP** on MS-COCO
- **Why relevant:** probably the cleanest structured-label classifier for your three label families
- **Code:** https://github.com/SlongLiu/query2labels

### 7) General Multi-label Image Classification with Transformers (C-Tran)
- **Authors:** Jack Lanchantin, Tianlu Wang, Vicente Ordonez et al.
- **Venue / year:** CVPR, 2021
- **ID:** arXiv 2011.14027 / DOI 10.1109/CVPR46437.2021.01621
- **Task:** generic multi-label classification
- **Datasets:** COCO, Visual Genome, News-500, CUB
- **Architecture:** Transformer encoder over masked label tokens and CNN visual features
- **Loss:** ternary label-mask objective
- **Headline metric:** SOTA on five datasets; exact number not needed for the core takeaway
- **Why relevant:** strong label-correlation model for grouped outputs
- **Code:** not stated in abstract

### 8) Multi-Label Remote Sensing Image Classification with Deformable Convolutions and Graph Neural Networks
- **Authors:** Ying Diao, Jingzhou Chen, Y. Qian et al.
- **Venue / year:** IGARSS, 2020
- **ID:** DOI 10.1109/IGARSS39084.2020.9324530
- **Task:** remote sensing multi-label classification
- **Datasets:** UC-Merced, DOTA
- **Architecture:** deformable conv backbone + attention + directed label graph
- **Loss:** multi-label classification loss
- **Headline metric:** effective on two RS benchmarks
- **Why relevant:** RS-native label-graph baseline; useful when label co-occurrence matters
- **Code:** not stated in abstract

### 9) Cross-Modal Feature Representation Learning and Label Graph Mining in a Residual Multi-Attentional CNN-LSTM Network for Multi-Label Aerial Scene Classification
- **Authors:** Peng Li, Peng Chen, D. Zhang
- **Venue / year:** Remote Sensing, 2022
- **ID:** DOI 10.3390/rs14102424
- **Task:** multi-label aerial scene classification
- **Datasets:** aerial multi-label benchmarks
- **Architecture:** residual multi-attentional CNN encoder + label graph + bi-LSTM predictor
- **Loss:** multi-label classification loss
- **Headline metric:** improved performance over prior RS multi-label baselines
- **Why relevant:** shows how to combine image, label graph, and sequence decoding in RS
- **Code:** not stated in abstract

### 10) DCA-GCN: A Dual-Branching Channel Attention and Graph Convolution Network for Multi-Label Remote Sensing Image Classification
- **Authors:** Minhang Yang, Hui Liu, Liang Gao et al.
- **Venue / year:** JARS, 2021
- **ID:** DOI 10.1117/1.JRS.15.044519
- **Task:** multi-label remote sensing image classification
- **Datasets:** three public multi-label RS datasets
- **Architecture:** SE-ResNet dual-branch backbone + channel attention + label GCN
- **Loss:** multi-label classification loss
- **Headline metric:** strong results on three RS multi-label datasets
- **Why relevant:** good RS-native template for label-dependency modeling
- **Code:** not stated in abstract

### 11) COMIC: Multi-Label Classification with Long-Tailed Distribution and Partial Labels
- **Authors:** Wenqiao Zhang, Changshuo Liu, Lingze Zeng et al.
- **Venue / year:** ICCV, 2023
- **ID:** arXiv 2304.10539 / DOI 10.1109/ICCV51070.2023.00137
- **Task:** multi-label classification under long-tail and partial labels
- **Datasets:** new PLT-MLC benchmarks
- **Architecture:** correction -> modification -> balance pipeline
- **Loss:** multi-focal modifier loss + balanced classifier
- **Headline metric:** significantly outperforms LT-MLC and PL-MLC baselines
- **Why relevant:** your labels are likely incomplete or noisy if decomposed from captions; this paper addresses that regime directly
- **Code:** https://github.com/wannature/COMIC

### 12) GradNorm, Kendall uncertainty weighting, and PCGrad
- **GradNorm:** Chen et al., ICML 2018; adaptive loss balancing for multitask training
- **Kendall uncertainty weighting:** Kendall, Gal, Cipolla, CVPR 2018; learns task weights from homoscedastic uncertainty
- **PCGrad / Gradient Surgery:** Yu et al., NeurIPS 2020; projects conflicting task gradients
- **Why relevant:** these are the three most defensible loss-balancing baselines for your 3-family heads
- **Code:** each has public reference implementations or widely used reimplementations

## Must-cite papers

1. Remote Sensing Image Change Captioning With Dual-Branch Transformers (LEVIR-CC)
2. TransWCD
3. AdvCP
4. DISep
5. Semantic-CC
6. Change-Agent
7. ChangeFormer
8. ChangeMamba
9. Query2Label
10. Asymmetric Loss For Multi-Label Classification

## Papers to compare against numerically

1. **TransWCD** - strongest direct scene-level weak supervision baseline
2. **AdvCP** - best match for noisy scene-level supervision
3. **DISep** - good if your labels are sparse and instances are crowded

## State of the field

Remote sensing change understanding has split into three partially overlapping lines: binary/semantic change detection, change captioning, and interactive vision-language change interpretation. The strongest recent gains come from better bitemporal interaction rather than deeper decoders: transformer fusion, SAM/CLIP-assisted semantic priors, and now Mamba/state-space encoders. At the same time, weakly supervised CD papers show that image-level supervision is usable but noisy, which makes class prompting, pseudo-label cleanup, and instance separation important. For your task, the literature gap is not model capacity; it is the lack of a standard benchmark for **structured scene-level multi-label change triples** and the lack of loss designs that jointly handle label imbalance, label co-occurrence, and missing labels.

## Gaps to position against

- No public paper surfaced that directly learns **object / event / attribute** multi-label triples from LEVIR-CC-style captions.
- Most CD papers are still pixel-mask first, not scene-label first.
- Most RSICC papers optimize caption metrics, not structured label recovery.
- Label families are usually modeled independently; hierarchical coupling across object/event/attribute is underexplored.
- Threshold calibration for per-class multi-label outputs is rarely treated as a first-class problem in remote sensing.
