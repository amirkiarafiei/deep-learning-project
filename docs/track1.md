# Track 1 — Implementation Spec

A working, well-tested baseline that satisfies the course rubric. Track 2 (novel methods aimed at paper publication, likely transformer-heavy, will be planned separately after the deadline) is intentionally out of scope here.

This spec describes *what* to build and *why* the choices are what they are, but does not prescribe every implementation detail. Use judgment on naming, formatting, error handling, and structure within the bounds described.

> **Read `AGENTS.md` first.** It has the project background, the task, the data, the non-goals, and how to work with the user. This document is just the technical spec on top of that context.

---

## Phase structure of Track 1

Track 1 progresses through controlled phases. Each phase ships a tagged release; v2 always builds on v1's data + analysis, never replaces it.

| Phase | Version | Status | Scope |
|---|---|---|---|
| **Phase 1** | **v1.0.0** | ✅ shipped 2026-05-26 (commit `f16d610`, tag `v1.0.0`) | Clean baseline as originally specified. ConvNeXt-Tiny + concat-diff fusion + BCE+pos_weight(50). Both course-stages (single-task + multi-task) run end-to-end. |
| **Phase 2** | **v2** | 📋 planned (this revision) | Diagnose Phase-1 failure modes, apply targeted regularization fixes + val-set threshold optimization. Same architecture; hyperparameters and one head-side change only. |
| (Track 2) | post-deadline | future | Different architecture family (likely transformer-based fusion or foundation-backbone). Not covered here. |

> **Terminology disambiguation:** "Phase 1 / Phase 2" in this document refers to the *Track-1 iteration* (v1 vs v2). The Turkish course rubric also uses "Aşama 1 / Aşama 2" to mean *single-task models vs multi-task model*. Both senses appear in every phase of Track 1 — when this doc says "Phase 1" without further qualification it means the iteration; when it says "single-task" or "multi-task" it means the course-stage.

---

# ─────────────────────────────────────────────────────────
# Phase 1 — v1.0.0 baseline
# ─────────────────────────────────────────────────────────

The rest of this section documents the **Phase 1 spec as originally written and as actually implemented in v1.0.0**. Phase 2's revisions are layered on top later in this document; do not edit the Phase 1 spec retroactively.

## Goals (Phase 1)

- Three single-task models, one per label family (Object, Event, Attribute), each trained independently.
- One multi-task model with a shared encoder and three heads, trained jointly.
- For each: train logs, val/test metrics (micro & macro F1, precision, recall, per-class breakdown), loss curves, and ≥20 qualitative example figures showing both images, ground truth, and predicted labels with their sigmoid scores.

---

## Architecture (Track 1)

A standard Siamese-encoder change-classification model:

- **Backbone:** ConvNeXt-Tiny (`convnext_tiny.fb_in1k` via timm), ImageNet-pretrained, shared weights between A and B. Use `features_only=True, out_indices=(3,)` to get the last-stage feature map (`B × 768 × 7 × 7` at 224 input). Don't pool inside the backbone — the fusion module handles that.
- **Fusion:** Concatenate `[feat_A, feat_B, feat_A − feat_B, |feat_A − feat_B|]` along the channel dimension (`B × 3072 × 7 × 7`), then a 1×1 conv projecting back to 768 channels, followed by BatchNorm + GELU + global average pool. Output: `B × 768`.
- **Heads:** Three linear heads (`nn.Linear(768, num_classes)`), one per family, with `num_classes` = 12, 12, 24 respectively. Plus one binary `changeflag` head (`nn.Linear(768, 1)`). The changeflag head is always present.
- **Phase 1 vs Phase 2:** The same `ChangeClassifier` class, parameterized by which family heads to instantiate. Phase 1 → one family head. Phase 2 → all three family heads. The backbone, fusion, and changeflag head are always present in both.

The model returns a dict of logits keyed by family name (`'object'`, `'event'`, `'attribute'`) plus `'changeflag'`. Sigmoid is applied at loss/inference time, never inside the model.

**Why this fusion:** It's the simplest defensible choice that all three literature reviews independently identify as a baseline. It's not state-of-the-art — that's intentional. Track 1's job is to be obviously correct, not impressive.

---

## Data Pipeline

The dataset has 10,855 samples in `dataset/dataset.json` with fields including `split`, `rgb_A`, `rgb_B`, `changeflag`, `object_labels`, `event_labels`, `attribute_labels`, and `sample_id`. Use the splits as given (don't reshuffle across splits).

**Three pieces:**

1. **Label encoder.** A class that converts a label list like `["building", "tree"]` into a fixed-order multi-hot tensor (shape `(12,)` for object) using the vocabulary in `results/eda/label_vocab.json`. The `none` label (index 0 in the vocab file) is *never* part of the output vector — if the input contains `none`, the output is all zeros. The encoder needs a `decode` method too (inverse mapping, used for visualization).

2. **Dataset class.** A `torch.utils.data.Dataset` subclass that returns a dict per sample with keys: `image_A`, `image_B`, `changeflag`, `object_labels`, `event_labels`, `attribute_labels`, `sample_id`. Images are `float32 (3, 224, 224)`, ImageNet-normalized. Labels are `float32` multi-hot tensors. The `changeflag` value is derived from the JSON (it should match whether the label lists are non-`none`).

3. **Paired transforms.** Apply the same transform pipeline to A and B (same flip decision, same resize). Train-time: resize to 224×224 → horizontal flip with p=0.5 → ToTensor → ImageNet normalize. Val/test-time: resize → ToTensor → normalize. No color jitter, no rotation, no random crop — the data is already heavily pre-augmented by the professor.

---

## Loss and Optimization

**Loss:** `BCEWithLogitsLoss` with per-class `pos_weight`. Compute `pos_weight[c] = num_negatives[c] / num_positives[c]` on the training set, then clamp to `[1.0, 50.0]` so ultra-rare classes (e.g., `plant` with 22 training samples) don't blow up gradients. One loss instance per family (Phase 2 has three).

Also compute a `pos_weight` (scalar) for the `changeflag` head and use it the same way.

**Loss aggregation in Phase 2:** Naive sum across heads. Auxiliary `changeflag` loss is weighted 0.5; family losses are weighted 1.0 each. No GradNorm, no Kendall uncertainty. We want a clean baseline to compare against in Track 2.

**Optimizer:** AdamW with different learning rates for backbone vs head:
- Backbone: `lr = 1e-5`
- Fusion + heads: `lr = 1e-4`
- Weight decay: `1e-4`
- Gradient clip: 1.0
- Schedule: Cosine annealing over the full epoch budget, `eta_min = 1e-6`

**Training duration:** 30 epochs with early stopping on val macro-F1 (averaged across active families), patience = 10 epochs.

**Precision:** bf16 autocast on A100. fp32 elsewhere.

**Batch size:** Start with 64. If A100 runs out of memory, drop to 32.

---

## Repository Layout

The current tree has `dataset/`, `docs/`, `results/eda/`, `scripts/`, and root files. Build the rest like this:

```
src/
├── data/           # dataset, label encoder, transforms
├── models/         # backbone, fusion, heads, classifier
├── losses/         # BCE with pos_weight
├── training/       # trainer loop, metrics, callbacks
├── utils/          # config loader, seeding, text logger
└── scripts/        # internal helpers (not CLI)

configs/
├── track1_object.yaml
├── track1_event.yaml
├── track1_attribute.yaml
└── track1_multitask.yaml

results/track1/
├── object/         # one folder per run, each with checkpoints/ logs/ plots/ metrics/ qualitative/
├── event/
├── attribute/
└── multitask/

train.py            # thin CLI: parse args → load config → call into src/
eval.py
visualize.py
```

The CLI scripts at the root are thin — they're not where logic lives. They parse args, load YAML, and call into `src/`.

Each experiment writes to its own `results/track1/<run_name>/` folder. Train logs, eval logs, plots, metrics CSVs, and qualitative examples all live there.

---

## Configs

Each experiment is one YAML file. The four configs differ only in `model.families` and `run_name`. Use a dataclass-based config loader in `src/utils/config.py` so configs can be validated and accessed with attribute syntax.

Example fields the config should include:

```yaml
run_name: track1_object
output_dir: results/track1/object
seed: 42

data:
  json_path: dataset/dataset.json
  dataset_root: dataset
  image_size: 224
  batch_size: 64
  num_workers: 4

model:
  families: [object]                # or [event], [attribute], [object, event, attribute]
  include_changeflag: true
  pretrained_backbone: true

training:
  epochs: 30
  lr_backbone: 1.0e-5
  lr_head: 1.0e-4
  weight_decay: 1.0e-4
  changeflag_weight: 0.5
  grad_clip: 1.0
  early_stop_patience: 10

evaluation:
  threshold: 0.5
```

---

## Metrics and Reporting

For every evaluation pass:

- Micro and macro F1, precision, recall — per family.
- Per-class precision / recall / F1 / support → save as CSV.
- Save the full prediction matrix (`logits` and `targets` for the eval split) so we can re-compute metrics later without re-running inference.

Use `sklearn.metrics.precision_recall_fscore_support` rather than reinventing the math.

Plot train and val loss curves per epoch, plus per-family val macro-F1 curves. Save as one combined PNG (`results/track1/<run>/plots/train_curves.png`).

The qualitative visualization should match the format the professor showed (see Image 2 in the project PDF): two images side by side, right-side text block with sample metadata, ground truth labels, and predicted labels with their sigmoid scores formatted as `label (0.xxx)`. Save 20+ examples per run as PNGs, plus one combined PDF.

---

## CLI Entry Points

Three top-level scripts:

```bash
python train.py --config configs/track1_object.yaml
python eval.py --config configs/track1_object.yaml --ckpt results/track1/object/checkpoints/best.pt --split test
python visualize.py --config configs/track1_object.yaml --ckpt results/track1/object/checkpoints/best.pt --num-samples 20
```

`eval.py` runs on the test set by default, but `--split val` should also work.

---

## Suggested Implementation Order

Each step should run cleanly before moving to the next:

1. Label encoder + a quick round-trip test in a notebook or `if __name__ == "__main__"` block.
2. Dataset class + smoke test (fetch a sample, print shapes).
3. Paired transforms + smoke test (verify A and B get the same flip).
4. Model components → full `ChangeClassifier` → smoke test on dummy input.
5. Loss with `pos_weight` + smoke test.
6. Metrics module + smoke test on a known-easy case.
7. Trainer loop → smoke test with `epochs=2, batch_size=16` and a small subset of data (~200 train, ~100 val samples).
8. Full training run of `track1_object.yaml` on Colab A100.
9. Repeat for event, attribute, multitask.
10. Run `eval.py` and `visualize.py` for each.

If any step takes more than an hour of debugging, stop and ask the user. Sunk-cost fallacy kills these projects.

---

## Sanity Expectations

These are *lower bounds for sanity*, not targets. If actual numbers come in much lower, something is wrong; if much higher, great.

- Object macro-F1 on test ≥ 0.40 (the `building` class alone should carry this)
- Event macro-F1 on test ≥ 0.35
- Attribute macro-F1 on test ≥ 0.30
- Phase 2 macro-F1 within ±5% of best Phase 1 per family

Per-class F1 will vary a lot:
- `building`: 0.85–0.90 expected
- `road`, `tree`, common events: 0.5–0.7 expected
- Long-tail classes (`plant`, `green`, `asphalt`, `dark`, `same`): F1 ≈ 0 is statistically expected and not a model failure

Report micro-F1 (good because of `building`), macro-F1 (the honest metric for imbalanced multi-label), and per-class breakdown. The honest framing in the report: "the macro-F1 figure understates the model's value on common classes and overstates its failures on classes too rare to learn from the available samples."

---

## Course Compliance Notes

The course rubric is explicit: **missing `ReadMe.txt` or `requirements.txt` costs 10 points each**. Both files must exist with these exact names.

- `ReadMe.txt` — plain text, exact filename (capital R and M). Written in Turkish. Covers description, dependencies, and how to run training / evaluation / visualization. Keep it concise; the longer narrative version goes in `README.md`. The course PDF includes a template at the bottom of the project description — follow that structure.
- `requirements.txt` — pinned versions. Generate by `pip freeze` after the environment is set up; don't hand-write.
- `README.md` — keep this too, for repo discoverability. Same content as `ReadMe.txt` but in English and more detailed.
- All experiments must produce text logs (.txt) of training output. Use a logger that writes to both console and file.

---

## Defaults If Something Isn't Specified

- Type hints on public function signatures. Docstrings, one line minimum.
- No bare `print` in `src/`. Use the project logger. `print` is OK in the top-level CLI scripts.
- Pass config explicitly. No global state.
- `pathlib.Path` over `os.path.join` for new code.
- `argparse` only in the three top-level scripts, never in `src/`.
- Plain PyTorch over Lightning. The training loop should fit in one screen.

---

## When to Stop (Phase 1)

Phase 1 is done when:

- All four configs train without errors.
- Each training run completes in under 2 hours on a Colab A100.
- All expected outputs (logs, checkpoints, plots, metrics, qualitative examples) exist.
- A smoke test of `eval.py` and `visualize.py` works end-to-end.

Submission files (`ReadMe.txt`, `requirements.txt`, IEEE report, presentation) are **deferred to whichever phase produces the final winning model** — do not write them on Phase 1 if Phase 2 is planned.

## Phase 1 status — completed 2026-05-26

Shipped as commit `f16d610`, tag `v1.0.0`. Headline test metrics (`best.pt`, threshold=0.5, full held-out test set n=1227):

| family | course-stage | best_ep | macro_F1 | micro_F1 | precision | recall |
|---|---|---:|---:|---:|---:|---:|
| object | single-task | 3 | 0.2045 | 0.5244 | 0.3930 | 0.7879 |
| event | single-task | 8 | 0.2796 | 0.3883 | 0.2911 | 0.5831 |
| attribute | single-task | 10 | 0.2335 | 0.3692 | 0.2728 | 0.5708 |
| object | multi-task | 15 | **0.2850** | **0.6381** | 0.5439 | 0.7717 |
| event | multi-task | 15 | 0.2861 | 0.4036 | 0.3209 | 0.5437 |
| attribute | multi-task | 15 | 0.2404 | 0.3477 | 0.2482 | 0.5802 |
| changeflag | both | * | * | ~0.95 | * | * |

Outputs are at `results/track1_v1/`. The macro_F1 lands below the spec's own sanity bounds (object ≥ 0.40, event ≥ 0.35, attribute ≥ 0.30) but the multi-task variant outperforms the single-task variant on every family, which is the Phase 1's main scientific finding.

Full forensic analysis + cross-agent diagnosis (Copilot CLI + Gemini CLI) lives at `results/track1_v1/v1_diagnosis_and_v2_plan.md`. The Phase 2 spec below is the operationalization of those findings.

---

# ─────────────────────────────────────────────────────────
# Phase 2 — v2 (regularization + threshold optimization)
# ─────────────────────────────────────────────────────────

Phase 2's job is to fix Phase 1's diagnosed failure modes **without changing the architecture family**. We stay inside Track 1's "obviously correct" framing — no asymmetric losses, no transformer fusion, no foundation backbone, no Query2Label/ML-Decoder. Just targeted hyperparameter changes + a small head-side regularizer + proper threshold protocol.

## Hypotheses (drives every Phase 2 change)

Two convergent diagnoses from Copilot CLI and Gemini CLI on the v1 data:

1. **`pos_weight=50` is over-rewarding positive predictions.** Recall ≫ precision in every family (e.g., object P1 prec=0.39 / rec=0.79). Lowering the clamp ceiling rebalances the loss landscape.
2. **Classical overfitting present.** Train_loss 1.15 → 0.13 while val_loss climbs after epoch 3 (object) / 8 (event) / 10 (attribute). Best_epoch arrives very early and val macro-F1 degrades afterward — confirms train-set memorization. Stronger weight decay + head dropout slows this trajectory.

A third lever — **per-class threshold optimization on the val set** — is free (no retrain) and is the standard professional protocol for multi-label deployment. Phase 1 evaluated everything at a flat 0.5 threshold; that's a defensible "obvious baseline" choice but leaves measurable gains on the table.

## Architectural delta from Phase 1

Exactly one structural change to the model code, and three config-level changes. Nothing else.

**Model change:**
- `ChangeClassifier` family heads gain an optional `nn.Dropout(p=head_dropout)` immediately before the final `nn.Linear`. The changeflag head stays plain.

**Config changes** (applied to all four YAMLs `configs/track1_*.yaml`):

```yaml
model:
  head_dropout: 0.3              # NEW field; Phase 1 default = 0.0
training:
  pos_weight_clamp_max: 10.0     # was 50.0
  weight_decay: 1.0e-3           # was 1.0e-4 (10×)
```

Everything else — backbone, fusion, optimizer LRs, schedule, epochs, batch size, early-stop policy, seed, bf16 autocast — stays identical to Phase 1 so the comparison is controlled.

## Phase 2 evaluation protocol

A second change, applied at evaluation time only:

1. **Train v2 to convergence** (early stop on val avg macro-F1, same as Phase 1).
2. After `best.pt` is selected, run a **per-class threshold sweep on the VAL set** (grid `0.05 … 0.95 step 0.05` per class). Pick per-class thresholds that maximize per-class F1 on val. Save the chosen thresholds alongside the checkpoint as `metrics/thresholds_val.json`.
3. **Apply the val-optimized thresholds to the TEST set** for the final metrics. Report both the 0.5-flat baseline and the val-tuned numbers side by side.

Tuning thresholds on val (not test) is the protocol both reviewers explicitly flagged as the correct one — Phase 1's test-set-optimistic threshold ablation was diagnostic only, not deployable.

## Implementation tasks (Phase 2)

In order, each independently verifiable:

1. **Add `head_dropout` to `ModelConfig`** in `src/utils/config.py`. Default 0.0 (keeps Phase 1 reproducible).
2. **Thread `head_dropout` through `ChangeClassifier`**: replace each family `nn.Linear` with `nn.Sequential(nn.Dropout(p), nn.Linear(...))` when `p > 0`.
3. **Bump config values** in the four YAMLs as above. Bump `run_name` and `output_dir` from `track1_<fam>` to `track1_<fam>_v2` (or similar) so v2 outputs don't overwrite v1.
4. **Threshold-sweep script** at `src/scripts/tune_thresholds.py`: loads `predictions_val.pt`, sweeps per-class thresholds, writes `thresholds_val.json` next to the checkpoint.
5. **Eval CLI extension**: `eval.py` accepts `--thresholds <path.json>` to apply per-class thresholds at inference instead of 0.5. Existing 0.5-baseline path stays default.
6. **Add a val-predictions save step**: `eval.py` already saves `predictions_<split>.pt`; ensure it also runs on val for v2 so the threshold sweep has data. Either re-run `eval.py --split val` after train, or extend the trainer to dump val predictions at best_epoch.
7. **Run v2 on Colab A100**: same plan as v1 (notebook `colab_runbook.ipynb` already auto-resumes; just point output dirs to v2).
8. **Tag the result** as `v2.0.0` once metrics are in.

## Phase 2 sanity targets

If the hypotheses are correct, v2 should improve over v1 by **+0.05 to +0.10 macro_F1** across families, and per-class threshold tuning should add another **+0.02 to +0.07** on top. Stacked, that gets multi-task object plausibly to ≥0.40 — clearing the spec's sanity bound. Single-task event/attribute may still fall short; honest framing in the report accepts that.

**Decision tree after v2 completes:**

- Multi-task macro_F1 ≥ sanity bounds across all three families → ship v2 as the final, write report.
- Significant improvement but still below sanity for one family → still write report on v2; argue spec's bounds were aspirational; consider v3 only if time allows.
- Marginal improvement (≤ +0.03) → hypotheses were wrong; reopen diagnosis and consider Track-2 escalation (ASL was Copilot's top pick if v2 fails; Gemini accepted it as second line).

## What Phase 2 does NOT do

To keep the contrast clean and the deadline reachable, the following remain reserved for Track 2 and must not be added in v2:

- Asymmetric Loss (ASL), focal loss, distribution-balanced loss
- ML-Decoder, Query2Label, or any transformer head
- DINOv2, RemoteCLIP, SAM, or other foundation backbones
- Cross-attention or any transformer-style fusion
- GradNorm / PCGrad / Kendall uncertainty for multi-task balancing
- TTA, EMA, SWA
- Architectural changes to the fusion module
- Re-splitting the dataset to remove `_ters_` leakage (the course rubric mandates "splits used as given")

If a candidate change is in the above list and seems necessary, that's the signal to escalate to Track 2 planning, not to bend Phase 2's scope.

---

## When to stop (overall Track 1)

Track 1 is done when **either** Phase 2 produces the final shippable model (and submission deliverables are written for it), **or** Phase 2 fails its decision tree and Track 2 planning begins. Tag the winning artifact and only then move to the IEEE report / `ReadMe.txt` / `requirements.txt` / presentation deliverables.