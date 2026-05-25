# v1 Diagnosis & v2 Experiment Plan

Compiled 2026-05-26 from per-class forensics + threshold-tuning ablation + parallel diagnoses by Copilot CLI (gpt-5.2, code-review agent) and Gemini CLI (gemini-2.5-pro). Both agents reviewed the same brief in isolation.

## v1 per-class forensic findings (local data analysis)

- **Object Phase 2 (multi-task) wins on 10/12 classes vs Phase 1.** `green` revives from F1=0 to F1=0.500. Multi-task acts as a regularizer for rare classes.
- **3 "dead" F1=0 classes in object** — `asphalt`, `plant`, `green` with test_support of 6, 2, 2 respectively. Tiny test supports = statistical noise, not model failure. macro_F1 excluding dead classes = 0.27 (much closer to the spec's lower bound than the raw 0.20).
- **Recall ≫ precision everywhere** (e.g., object P1 prec=0.39, rec=0.79). Direct evidence the pos_weight=50 is over-rewarding positive predictions.
- **Threshold tuning ablation gives only +0.02 to +0.07 macro_F1** even on test-set-optimistic tuning. Not enough to clear sanity bounds alone — the model itself needs improvement.

## Cross-agent diagnosis (Copilot + Gemini in parallel)

### Convergent findings

Both agents independently agree on:

1. **Overfitting is the dominant failure mode.** Verify by tracking val macro-F1 over epochs (peaks then degrades), not val_loss alone. pos_weight=50 inflates loss magnitudes but doesn't cause the *trend*.
2. **pos_weight clamp [1, 50] → [1, 10] is the #1 highest-leverage, lowest-risk fix.** Both rank this in their top 2.
3. **Per-class threshold optimization belongs AFTER retraining, tuned on the VAL set** (not test). v1's test-set-optimistic sweep was the wrong protocol — redo properly post-v2.
4. **MTL rare-class gains need val-set verification.** Tiny test supports (2–6 samples) are too noisy for confident claims. Compare per-class val_F1 P1 vs P2 to confirm.
5. **Don't pursue backbone swap (DINOv2/RemoteCLIP), ML-Decoder head, or cross-attention fusion under 5-day deadline.** Both flag these as high integration risk.

### Divergent recommendations

- **Copilot's bolder top-3:** ASL (Asymmetric Loss) > pos_weight clamp 10 > val threshold opt.
- **Gemini's safer top-3:** pos_weight clamp 10 > dropout+weight_decay > val threshold opt.

ASL is a known winning technique on long-tailed multi-label, but the literature reports it needs hyperparameter tuning. Gemini's view is the conservative one for 5-day execution; Copilot's is the maximum-grade ceiling. We adopt Gemini's recommendation for v2 (conservative) and reserve ASL for v3 if v2 falls short.

### Convergent minimum experiment plan

- **E0 baseline** = v1 (already done — 4 configs, single seed each).
- **E1 v2 retrain** = pos_weight [1,10] + head dropout 0.3 + weight_decay 1e-3.
- **E2 threshold opt** = per-class threshold on VAL applied to E1 → test.
- **(Optional) E3 v3 ASL** = if E1+E2 don't clear sanity bounds.
- **(Optional) E4 leakage sanity** = re-eval after removing `_ters_` overlapping scenes; one config only.

Copilot recommends 3 seeds per config for statistical rigor; Gemini accepts single-seed for time efficiency. Hybrid: run single-seed first to see direction, add seeds only if results are borderline or claims need defense.

## Proposed v2 config (Track 1 spirit, hyperparameter-only change)

```yaml
# Single change set, applied to all 4 configs:
model:
  head_dropout: 0.3               # NEW field; Phase 1 default = 0.0
training:
  pos_weight_clamp_max: 10.0      # was 50.0  (highest-leverage change per both agents)
  weight_decay:        1.0e-3     # was 1.0e-4  (10×)
# All other hyperparameters unchanged for controlled comparison
```

**Estimated cost:** ~2–3 hours A100 across 4 configs (similar to v1, possibly faster since regularization can slow overfitting onset → may not early-stop as quickly, but may also reach better optimum before patience exhausts).

## Decision tree

```
v2 trains →
  ├─ macro_F1 ≥ sanity bounds across all 3 families   → write report on v2, done
  ├─ macro_F1 +0.05 over v1 (significant but partial) → add val-threshold opt, then decide on v3
  └─ macro_F1 ≈ v1 (no real improvement)              → escalate to v3 with ASL (Track-2 territory)
```

## Why we are NOT changing more in v2

For a controlled experiment that tells a clean scientific story in the IEEE report:

- **One change set, one comparison.** v1 vs v2 must be interpretable as "what does regularization buy us?". If we also swap the backbone, change fusion, and switch losses, we have nothing to say about any individual choice.
- **Track 2 reservations from `AGENTS.md` and `docs/track1.md` § Phase 2 § What Phase 2 does NOT do** must hold. ASL / ML-Decoder / DINOv2 / cross-attention / GradNorm / TTA / EMA / SWA are reserved.
- **The deadline allows ~2–3 iterations**, not unbounded sweeps. Each iteration must answer one well-defined question.
