# Track 2 — A3 Hybrid Long-Tail Pipeline (v3) + A2 Isolation Experiment (v4)

**Status:** completed 2026-05-27 (results in `results/track2_v3/`, `results/track2_v4/`, IEEE report in `report/track2_report/`).
**Last revised:** 2026-05-28 (backfilled from execution history).
**Companion docs:** [`track1.md`](track1.md) (v1+v2 baseline), [`track3.md`](track3.md) (v5 planned), [`track4.md`](track4.md) (v6 planned).

> **Read `AGENTS.md` first.** This doc is a retrospective spec, written after Track 2 ran. The IEEE report at `report/track2_report/track2_report.pdf` is the primary written deliverable; this doc captures the engineering plan and the cross-agent diagnostic story behind the negative result.

---

## Phase summary

| Track / version | Status | Headline |
|---|---|---|
| Track 1 v1.0.0 | ✅ shipped 2026-05-26 | CNN baseline, test avg macro F1 **0.270** (headline) |
| Track 1 v2.0.0 | ✅ shipped 2026-05-26 | + regularization, 0.246 (tuned), slowed overfitting but lower peak |
| **Track 2 v3** | ✅ shipped 2026-05-27 (tag `v3.0.0`) | A3 Hybrid — failed, **0.175** with LA τ=1.0 |
| **Track 2 v4** | ✅ shipped 2026-05-27 (tag `v4.0.0`) | A2 isolation — tied with v1, **0.268** |
| Track 3 v5 | 📋 planned (`track3.md`) | DINOv2 + cross-attention + ML-Decoder |
| Track 4 v6 | 📋 planned (`track4.md`) | VMamba + interleaved SSM + RAL |

---

## Motivation

Track 1 hit a 0.270 average test macro-F1 ceiling. v2's regularization variant underperformed v1. Peer evidence (`docs/wp.md`) showed 10+ classmates across ResNet, ConvNeXt-Tiny, Swin, ViT-Small, EfficientNet all plateauing in 0.20–0.30. The natural next step was to deploy techniques specifically designed for the long-tail multi-label regime — which is what Track 2 v3 attempted.

Track 2 was deliberately *not* an architecture swap (those are reserved for Tracks 3 and 4). It kept v1's ConvNeXt-Tiny Siamese + concat-diff fusion + per-family linear heads, and changed five orthogonal pieces:

1. **Data-side cleanup** — Cleanlab-style noisy-label ranker + Gemini-2.5-Flash relabel + conservative AND-rule.
2. **Loss-side rebalancing** — LDAM-DRW (margin-based long-tail loss).
3. **Two-stage classifier decoupling** — classifier re-training (cRT) phase with class-aware sampling.
4. **Inference-time logit adjustment** — multi-label LA at τ ∈ {0.0, 0.3, 0.5, 1.0}.
5. **Test-time augmentation + per-class thresholds** — 4-way dihedral TTA + val-tuned thresholds.

The plan was: ship v3 (A3 hybrid = all five) first; if it failed, decompose into A1 (loss path only) and A2 (data path only) to isolate which components contributed.

---

## Phase structure

| Version | Tag | What changed vs v1 | Result |
|---|---|---|---|
| **v3 (A3 Hybrid)** | `v3.0.0` | All five interventions stacked | **FAIL**: 0.236 flat / 0.175 with LA τ=1.0 |
| **v4 (A2 isolation)** | `v4.0.0` | *Only* the cleaned dataset; v1's exact training recipe otherwise | **TIED**: 0.268 (v1 was 0.270) |

A1 (loss-only) was never trained — by the time we had v4's result, the cross-agent diagnosis had already explained the v3 failure mode without ambiguity (see below).

---

# ─────────────────────────────────────────────────────────
# v3 — A3 Hybrid
# ─────────────────────────────────────────────────────────

## Pipeline

Five components, in execution order:

1. **Predict v1 train.** `src/scripts/predict_for_cleanlab.py` dumps v1 multitask logits + targets on the train split → `results/track2_v3/cleanup/predictions_train.pt`.
2. **Rank by Cleanlab disagreement.** `src/scripts/find_noisy_labels.py` produces `noisy_candidates.json` with the top-K most suspect samples per family.
3. **Gemini relabel top-500.** `src/scripts/gemini_relabel.py` queries Gemini-2.5-Flash with the image pair + a Pydantic-typed JSON schema enforcing the closed label vocabulary via `google.genai`. Append-only JSONL with `--resume` for idempotence. Writes to `results/track2_v3/cleanup/gemini_relabels_train.jsonl`.
4. **Reconcile via AND-rule.** `src/scripts/reconcile_labels.py` flips a label set only when (cleanlab flag) AND (Gemini confidence=`high`) AND (Gemini labels differ from original). Writes `dataset_v3_clean.json` + summary. **Result: 108 of 8438 train samples flipped (1.3%).** Val and test untouched.
5. **Train on cleaned data with LDAM-DRW + cRT.** Config `configs/track2_v3_multitask.yaml`. Phase A is 30 epochs LDAM with DRW activated at epoch 5 (early-stop patience 15). cRT phase is 5 additional epochs of head-only retraining on plain BCE with class-aware sampling.

Inference (per `eval.py` flags):

- `--logit-adjust log_priors.json --logit-adjust-tau 1.0` — subtract τ·log π_c from each logit
- `--tta 4` — 4-way dihedral (id, hflip, vflip, hflip+vflip), averages sigmoid probabilities
- `--thresholds thresholds_val.json` — per-class thresholds tuned on original (uncleaned) val

## Code additions in v3

```
src/losses/ldam.py                 LDAMMultiLabelLoss with DRW switch
src/training/logit_adjustment.py   compute / save / load / apply log priors
src/data/tta.py                    4-way dihedral TTA
src/data/class_balanced_sampler.py per-sample weight = Σ_c 1/n_c
src/scripts/predict_for_cleanlab.py   v1 train-split predictions
src/scripts/find_noisy_labels.py      Cleanlab-style ranking
src/scripts/gemini_relabel.py         Gemini 2.5 Flash relabel pipeline
src/scripts/reconcile_labels.py       AND-rule label reconciliation
src/scripts/smoke_test_gemini.py      5-sample sanity check before $0.30 API spend
```

Trainer additions (in `src/training/trainer.py`):
- LDAM loss switch via `cfg.training.loss_type = "ldam"`
- DRW transition with patience reset at `ldam_drw_epoch`
- `fit_crt()` method swapping to plain BCE during cRT phase
- `save_log_priors()` for downstream LA inference
- Atomic checkpoint writes

## v3 headline (test split)

| Variant | object | event | attribute | **Avg** | Δ vs v1 |
|---|---|---|---|---|---|
| v1 multitask (flat 0.5) | 0.285 | 0.286 | 0.240 | **0.270** | — |
| v3 flat 0.5 (no LA/TTA) | 0.225 | 0.259 | 0.224 | 0.236 | −0.034 |
| v3 + TTA + thr (τ=0.0) | 0.207 | 0.272 | 0.214 | 0.231 | −0.039 |
| v3 + LA τ=0.3 + TTA + thr | 0.209 | 0.261 | 0.202 | 0.224 | −0.046 |
| v3 + LA τ=0.5 + TTA + thr | 0.190 | 0.246 | 0.180 | 0.205 | −0.065 |
| v3 + LA τ=1.0 + TTA + thr | 0.161 | 0.219 | 0.144 | **0.175** | −0.095 |

LA monotonically degraded performance. Per-class breakdown: tail classes (support ≤ 25) all collapsed to F1=0 at flat 0.5; LA at τ=1.0 flipped them into over-prediction (recall 0.91–0.97, precision 0.08–0.14).

## Cross-agent diagnosis (Copilot gpt-5.2 + Gemini 3.1-pro)

Both agents, briefed independently, refuted the original convention-drift hypothesis (the cleanup is too small at 1.3% to drive the regression; over-prediction under LA is the opposite signature from cleanup over-labeling). They converged on three independent failure modes:

- **F1 — Weakened class-balance signal vs v1 (training time).** v3 dropped pos_weight clamp from `[1, 50]` (v1) to `[1, 10]`, expecting LDAM to make up the difference. LDAM with `ldam_s = 1.0` (we disabled the scaling to avoid sigmoid saturation on plain `nn.Linear` heads) only subtracts margins of `m_max = 0.3` from positive logits — a much weaker push than v1's 50× positive weighting. Net effect: v3 Phase A trained a *strictly weaker* long-tail signal than v1.

- **F2 — Logit-adjustment magnitude mismatch (inference time).** `log_priors.json` has values as low as −5.95. At τ=1 this *adds* ~+6 to tail-class logits, pushing them past the sigmoid decision boundary regardless of input. LA was derived for ERM training (no class weights, no resampling); stacking it on DRW (positive weighting) + cRT (class-aware sampling) is a triple-correction of imbalance.

- **F3 — Multi-label incompatibility of class-aware sampling in cRT.** Sampling by `sum(1/n_c for c in positive_labels)` up-weights samples with rare labels, but those samples *also* contain head labels (e.g. *building* nearly always co-occurs with whatever the rare-class label is). cRT therefore over-samples head classes through the back door, destroying the label-correlation structure v1 had implicitly learned. Direct evidence: val macro F1 collapsed from 0.234 → 0.13 within one cRT epoch.

# ─────────────────────────────────────────────────────────
# v4 — A2 isolation
# ─────────────────────────────────────────────────────────

After the v3 failure + cross-agent diagnosis, one ambiguity remained: did the 108-flip Gemini cleanup itself help or hurt, independent of the broken loss stack? v4 resolved this by holding the cleaned dataset fixed and reverting the loss recipe to v1's plain BCE.

## Pipeline (single training)

- Architecture: identical to v1 (ConvNeXt-Tiny Siamese + concat-diff fusion + 4 heads).
- Dataset: `results/track2_v3/cleanup/dataset_v3_clean.json` (the cleaned dataset from v3.6).
- Loss: plain `BCEWithLogitsLoss` with `pos_weight clamp_max=50` (v1's setting, **not** v3's clamp=10).
- Optimizer: AdamW, lr_backbone=1e-5, lr_head=1e-4, wd=1e-4 (v1 settings).
- **No** LDAM, **no** cRT, **no** LA, **no** TTA, **no** per-class thresholds at first eval.

Config: `configs/track2_v4_clean_v1recipe_multitask.yaml`.

## v4 result (test split)

| Variant | object | event | attribute | **Avg** | Δ vs v1 |
|---|---|---|---|---|---|
| v1 multitask (flat 0.5) | 0.285 | 0.286 | 0.240 | **0.270** | — |
| **v4 flat 0.5 (cleanup + v1 recipe)** | 0.279 | 0.280 | **0.244** | **0.268** | −0.003 |
| v4 + val-tuned thresholds | 0.233 | 0.276 | 0.228 | 0.246 | −0.024 |

**Tied with v1 within run-to-run variance, beats v1 on Attribute by +0.004.** Tail classes that v3 collapsed to zero all recover under v4: `roof` 0.000→0.129 (v1=0.211), `parking` 0.050→0.160 (v1=0.164), `green` 0.000→0.667 (v4 best in column).

This **definitively isolates the v3 regression to the loss-side stack**, exactly as the cross-agent diagnosis (F1+F2+F3) predicted before v4 was trained. The Gemini cleanup is methodologically sound but not material at 1.3% flip rate.

---

# ─────────────────────────────────────────────────────────
# What we learned
# ─────────────────────────────────────────────────────────

1. **Track 1's data-ceiling hypothesis is real and survives every loss-side intervention.** Confirmed by peer evidence (`docs/wp.md`), v2 underperforming v1, v3 catastrophically failing, and v4 recovering to v1.

2. **Cross-agent diagnostic studies are high-EV.** Two AI agents independently refuted my initial cleanup-convention-drift hypothesis and provided a sharper alternative (F1+F2+F3) which v4 then directly confirmed. This is documented as a methodology contribution in the Track 2 IEEE report's *Cross-Agent Diagnostic Study* section.

3. **LDAM-DRW and cRT have multi-label-incompatible failure modes** (F3, plus the LDAM-scale issue with non-feature-normed heads). These should not be applied naively from the single-label literature to multi-label settings.

4. **Logit Adjustment is fragile in stacks.** LA assumes ERM training; combining with DRW or cRT triples the imbalance correction and overshoots.

5. **The remaining moves are architecture-side or data-side.** *Architecture-side* = Tracks 3 (DINOv2 + cross-attention + ML-Decoder) and 4 (VMamba + interleaved SSM + RAL). *Data-side* = re-annotation under a sparse convention, or test-set sanitization — out of scope for the course timeline.

---

## Deliverables

- **Code.** All v3 additions (losses, scripts, sampler, TTA, LA) live on `main`.
- **Tags.** `v3.0.0` (commit `0c326d4`, v3 results landed) and `v4.0.0` (commit `dc39c05`, v4 confirmed v3 diagnosis).
- **Configs.** `configs/track2_v3_multitask.yaml`, `configs/track2_v4_clean_v1recipe_multitask.yaml`.
- **Results.** `results/track2_v3/`, `results/track2_v4/` (metric JSONs, per-class CSVs, training history, train curves; heavy artifacts gitignored).
- **Notebook.** `colab_runbook.ipynb` cells `v3.1–v3.8` + v4 train + v4 eval cells.
- **IEEE report.** `report/track2_report/track2_report.pdf` (5 pages, IEEE journal format).

## References

- **Cao et al. 2019.** *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss (LDAM).* NeurIPS.
- **Kang et al. 2020.** *Decoupling Representation and Classifier for Long-Tailed Recognition (cRT).* ICLR.
- **Menon et al. 2021.** *Long-Tail Learning via Logit Adjustment.* ICLR.
- **Northcutt et al. 2021.** *Confident Learning: Estimating Uncertainty in Dataset Labels (Cleanlab).* JAIR.
- **Ridnik et al. 2021.** *Asymmetric Loss for Multi-Label Classification.* ICCV.
- **Wang et al. 2019.** *Aleatoric Uncertainty Estimation with Test-Time Augmentation.* Neurocomputing.
