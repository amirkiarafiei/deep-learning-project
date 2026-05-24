# Track 1 — Implementation Spec

A working, well-tested baseline that satisfies the course rubric. Track 2 (novel methods aimed at paper publication) is a separate document.

This spec describes *what* to build and *why* the choices are what they are, but does not prescribe every implementation detail. Use judgment on naming, formatting, error handling, and structure within the bounds described.

> **Read `AGENTS.md` first.** It has the project background, the task, the data, the non-goals, and how to work with the user. This document is just the technical spec on top of that context.

---

## Goals

- Three Phase-1 models, one per label family (Object, Event, Attribute), each trained independently.
- One Phase-2 model with a shared encoder and three heads, trained jointly.
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

The course rubric specifies a few things that must be present:

- `requirements.txt` with pinned versions. Generate by `pip freeze` after the environment is set up; don't guess versions.
- A `README.md` covering: description, install steps, how to run training / evaluation / visualization, and the file layout. The course PDF mentions a `ReadMe.txt`; we're using `README.md` instead (same content, more conventional name). The −10 penalty in the rubric is for missing content, not the extension.
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

## When to Stop

Track 1 is done when:

- All four configs train without errors.
- Each training run completes in under 2 hours on a Colab A100.
- All expected outputs (logs, checkpoints, plots, metrics, qualitative examples) exist.
- The README is complete.
- A smoke test of `eval.py` and `visualize.py` works end-to-end.

Then hand back for review. If you finish faster than expected, the answer is "do nothing extra." Track 2 is where extra effort goes.