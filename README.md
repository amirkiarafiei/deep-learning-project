# deep-learning-project

Semester project for **BLM5135 — Neural Networks and Deep Learning**
(Yıldız Technical University, Spring 2026).

## Task

Scene-level **multi-label classification of bi-temporal remote-sensing
image pairs**. Given two co-registered RGB images of the same scene at
times *t₁* (before) and *t₂* (after), predict three independent
multi-label outputs that describe what changed between them.

| Family    | # classes | Example labels                       |
|-----------|----------:|--------------------------------------|
| Object    |        13 | `building`, `road`, `tree`, `water`  |
| Event     |        13 | `build`, `remove`, `replace`, `turn` |
| Attribute |        25 | `green`, `dark`, `large`, `residential` |

Each family includes a `none` label that fires when nothing changed in
that family. `none` across all three families ≡ `changeflag == 0`.

The dataset is derived from LEVIR-CC (Liu et al., TGRS 2022): 10,855
image pairs at 256×256, split 78/11/11 train/val/test. See
[`results/eda/eda_summary.md`](./results/eda/eda_summary.md) for the
full distributional picture.

## Project plan

The course rubric splits the work into two phases:

- **Phase 1 (75% of the grade).** Three independent models, one per
  label family, each trained from scratch. Compare per-family
  difficulty, micro/macro F1, precision, recall.
- **Phase 2 (25% of the grade).** One shared-encoder multi-task model
  with three classification heads. Compare against Phase 1 and
  discuss which families benefit (or suffer) from joint training.

The implementation spec for both phases lives in
[`docs/track1.md`](./docs/track1.md): Siamese ConvNeXt-Tiny backbone +
concat / difference / abs-difference fusion + per-family heads + an
auxiliary `changeflag` head. `BCEWithLogitsLoss` with per-class
`pos_weight` against the severe class imbalance documented in EDA.

## Current status

| Stage | Status | Output |
|---|---|---|
| Literature survey | ✅ done | [`docs/literature/`](./docs/literature/) — three independent reviews |
| EDA | ✅ done | [`results/eda/`](./results/eda/) — vocab, per-split stats, 8 figure sets |
| Dataset / dataloader | ⏳ pending | `scripts/dataset.py` (to come) |
| Phase 1 training (Object / Event / Attribute) | ⏳ pending | `scripts/train.py` |
| Phase 2 training (shared encoder, multi-head) | ⏳ pending | `scripts/train.py` |
| Evaluation + IEEE report | ⏳ pending | `scripts/eval.py`, `report/` |

## Key EDA findings (informs modeling)

- **Severe imbalance.** Object max/min frequency ratio is **270×**
  (`building` 64.7% vs. `plant` 0.24%). Attribute is 67×, Event is
  28×. ASL or distribution-balanced loss is mandatory; per-class
  thresholds at inference.
- **`none` dominance.** All three families share the same 28.25%
  `none` rate (= the 3,067 no-change samples). The `changeflag`
  auxiliary head is essentially free supervision.
- **Drift warning.** Attribute `dark` has 0% prevalence in val (vs.
  ~0.6% in train/test) — unevaluable on val. Report per-class
  metrics, not just averaged mAP.
- **Heavy A↔B appearance variation even on no-change pairs.** Mean
  absolute gray-diff is 0.09–0.23 (normalized) on `changeflag==0`
  samples. Strong illumination/seasonal drift even when nothing
  structurally changed — illumination-robust fusion (concat + diff
  + |diff| or FDAF-style alignment) is justified.
- **Uniform 256×256 resolution.** Safe to hardcode in the dataloader.
- **Test split contains no `_random_augment` samples** (only
  originals and `_ters_` reversals). The test distribution is the
  cleanest of the three splits.

## Repository layout

```
.
├── README.md                       # this file
├── AGENTS.md                       # context for AI coding agents
├── requirements.txt                # pinned deps (currently EDA-only)
├── LICENSE
├── scripts/
│   └── run_eda.py                  # exploratory data analysis pipeline
├── docs/
│   ├── track1.md                   # Track-1 implementation spec
│   ├── clasroom/                   # course-provided materials
│   │   ├── BLM5135-ProjeAçıklaması.pdf
│   │   ├── classroom.md
│   │   └── dataset_short.json
│   └── literature/                 # our three independent literature reviews
│       ├── literature_codex.md
│       ├── literature_flash.md
│       └── literature_opus.md
├── results/
│   └── eda/                        # produced by scripts/run_eda.py
│       ├── eda_summary.md
│       ├── label_vocab.json        # canonical class ordering (used everywhere)
│       ├── per_split_stats.json
│       ├── extras.json
│       └── figures/                # 300 DPI PNG + PDF (gitignored)
└── dataset/                        # local-only; provided by course (gitignored)
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate           # Linux / macOS
# .venv\Scripts\activate            # Windows
pip install --upgrade pip
pip install -r requirements.txt

# Reproduce the EDA — needs dataset/ to be present locally.
python scripts/run_eda.py
```

The EDA writes everything to `results/eda/`. Figures are PNG + PDF at
300 DPI, ready for the IEEE report.

## Submission artifacts (per course rubric)

The course rubric requires a `ReadMe.txt` and `requirements.txt` at
submission time (10-point penalty per missing file). `requirements.txt`
already exists. `ReadMe.txt` will be generated at submission with the
final run instructions for the grader.

## License

[MIT](./LICENSE).
