# Dataset EDA Summary

_Source: `dataset/dataset.json` (n = 10,855 samples)._

## 1. Overall scale

- **Total samples:** 10,855
- **Per-split counts:**
  - `train`: 8,438 (77.73%)
  - `val`: 1,190 (10.96%)
  - `test`: 1,227 (11.30%)
- **Changeflag distribution:**
  - `changeflag=1` (changed): 7,788 (71.75%)
  - `changeflag=0` (no-change): 3,067 (28.25%)
- **Per-split changeflag:**
  - `train`: changed=6,052 (71.7%), no-change=2,386 (28.3%)
  - `val`: changed=850 (71.4%), no-change=340 (28.6%)
  - `test`: changed=886 (72.2%), no-change=341 (27.8%)

## 2. Label vocabularies

### `object` — 13 unique labels ✅ (expected 13)

| idx | label | count | % |
|---:|---|---:|---:|
| 0 | `none` | 3,067 | 28.25% |
| 1 | `building` | 7,022 | 64.69% |
| 2 | `tree` | 1,082 | 9.97% |
| 3 | `road` | 1,025 | 9.44% |
| 4 | `field` | 463 | 4.27% |
| 5 | `vegetation` | 280 | 2.58% |
| 6 | `water` | 279 | 2.57% |
| 7 | `parking` | 219 | 2.02% |
| 8 | `land` | 114 | 1.05% |
| 9 | `roof` | 87 | 0.80% |
| 10 | `asphalt` | 54 | 0.50% |
| 11 | `green` | 32 | 0.29% |
| 12 | `plant` | 26 | 0.24% |

### `event` — 13 unique labels ✅ (expected 13)

| idx | label | count | % |
|---:|---|---:|---:|
| 0 | `none` | 3,067 | 28.25% |
| 1 | `build` | 3,272 | 30.14% |
| 2 | `remove` | 2,188 | 20.16% |
| 3 | `turn` | 1,306 | 12.03% |
| 4 | `appear` | 1,293 | 11.91% |
| 5 | `replace` | 1,029 | 9.48% |
| 6 | `change` | 659 | 6.07% |
| 7 | `destroy` | 589 | 5.43% |
| 8 | `increase` | 372 | 3.43% |
| 9 | `vegetate` | 301 | 2.77% |
| 10 | `add` | 236 | 2.17% |
| 11 | `surround` | 154 | 1.42% |
| 12 | `remain` | 115 | 1.06% |

### `attribute` — 25 unique labels ✅ (expected 25)

| idx | label | count | % |
|---:|---|---:|---:|
| 0 | `none` | 3,067 | 28.25% |
| 1 | `blue` | 3,679 | 33.89% |
| 2 | `gray` | 2,294 | 21.13% |
| 3 | `green` | 1,768 | 16.29% |
| 4 | `large` | 1,415 | 13.04% |
| 5 | `huge` | 1,173 | 10.81% |
| 6 | `black` | 1,170 | 10.78% |
| 7 | `white` | 718 | 6.61% |
| 8 | `more` | 548 | 5.05% |
| 9 | `small` | 528 | 4.86% |
| 10 | `brown` | 426 | 3.92% |
| 11 | `empty` | 424 | 3.91% |
| 12 | `bare` | 378 | 3.48% |
| 13 | `lush` | 351 | 3.23% |
| 14 | `middle` | 288 | 2.65% |
| 15 | `red` | 221 | 2.04% |
| 16 | `residential` | 213 | 1.96% |
| 17 | `long` | 210 | 1.93% |
| 18 | `industrial` | 194 | 1.79% |
| 19 | `adjacent` | 145 | 1.34% |
| 20 | `sparse` | 122 | 1.12% |
| 21 | `dense` | 90 | 0.83% |
| 22 | `paved` | 70 | 0.64% |
| 23 | `same` | 57 | 0.53% |
| 24 | `dark` | 55 | 0.51% |

## 3. Class imbalance

### `object`
- Most frequent (non-none): `building` — 7,022 samples (64.69%)
- Least frequent (non-none): `plant` — 26 samples (0.24%)
- Max/min frequency ratio: **270.1×**
- Suggested treatment: Asymmetric Loss (ASL) + class-aware reweighting; consider Distribution-Balanced loss

### `event`
- Most frequent (non-none): `build` — 3,272 samples (30.14%)
- Least frequent (non-none): `remain` — 115 samples (1.06%)
- Max/min frequency ratio: **28.5×**
- Suggested treatment: ASL or focal loss on positives; per-class thresholds at inference

### `attribute`
- Most frequent (non-none): `blue` — 3,679 samples (33.89%)
- Least frequent (non-none): `dark` — 55 samples (0.51%)
- Max/min frequency ratio: **66.9×**
- Suggested treatment: Asymmetric Loss (ASL) + class-aware reweighting; consider Distribution-Balanced loss

Also note the dominance of `none` (no-change) labels in every family — the model must learn an implicit "no change" prior, and changeflag prediction is essentially an auxiliary task already.

## 4. Per-split distribution drift

Drift score = (max-min)/mean across the three splits, where each value is the percentage of samples in a split that contain the label. Labels with mean < 0.05% are excluded (too rare to be reliable).

### `object` — top 3 drifted labels

| label | rel_spread | train% | val% | test% | flag |
|---|---:|---:|---:|---:|:---:|
| `asphalt` | 90.7% | 0.43 | 1.01 | 0.49 | ⚠️ |
| `land` | 74.4% | 1.00 | 1.68 | 0.81 | ⚠️ |
| `green` | 64.3% | 0.31 | 0.34 | 0.16 | ⚠️ |

### `event` — top 3 drifted labels

| label | rel_spread | train% | val% | test% | flag |
|---|---:|---:|---:|---:|:---:|
| `surround` | 53.2% | 1.23 | 2.18 | 1.96 | ⚠️ |
| `remain` | 33.5% | 0.97 | 1.34 | 1.39 | OK |
| `destroy` | 31.3% | 5.59 | 4.03 | 5.62 | OK |

### `attribute` — top 3 drifted labels

| label | rel_spread | train% | val% | test% | flag |
|---|---:|---:|---:|---:|:---:|
| `dark` | 150.2% | 0.57 | 0.00 | 0.57 | ⚠️ |
| `dense` | 103.6% | 0.85 | 0.34 | 1.14 | ⚠️ |
| `long` | 94.4% | 2.18 | 1.34 | 0.81 | ⚠️ |

**Verdict:** at least one label has >50% relative spread across splits — use stratified resampling for hyperparameter tuning, and report per-class metrics rather than relying on global mAP.

## 5. Multi-label intensity

Number of non-`none` labels per sample, per family. `all` includes no-change samples (forced to zero), `changed` restricts to `changeflag=1`.

| family | scope | mean | median | max | std |
|---|---|---:|---:|---:|---:|
| `object` | all | 0.984 | 1 | 4 | 0.812 |
| `object` | changed | 1.372 | 1 | 4 | 0.622 |
| `event` | all | 1.061 | 1 | 4 | 0.959 |
| `event` | changed | 1.478 | 1 | 4 | 0.815 |
| `attribute` | all | 1.523 | 1 | 4 | 1.282 |
| `attribute` | changed | 2.123 | 2 | 4 | 1.008 |

## 6. Co-occurrences

### `object` — top 5 co-occurring pairs

| label A | label B | # samples |
|---|---|---:|
| `building` | `road` | 812 |
| `building` | `tree` | 721 |
| `building` | `field` | 336 |
| `tree` | `road` | 217 |
| `building` | `vegetation` | 199 |

### `event` — top 5 co-occurring pairs

| label A | label B | # samples |
|---|---|---:|
| `remove` | `replace` | 415 |
| `build` | `appear` | 367 |
| `build` | `remove` | 322 |
| `build` | `replace` | 320 |
| `appear` | `replace` | 302 |

### `attribute` — top 5 co-occurring pairs

| label A | label B | # samples |
|---|---|---:|
| `blue` | `gray` | 1,239 |
| `blue` | `large` | 610 |
| `blue` | `black` | 538 |
| `gray` | `huge` | 495 |
| `blue` | `green` | 488 |

Strong off-diagonal mass indicates exploitable label correlations — candidate for ML-GCN / C-Tran / Query2Label-style heads that model dependencies.

## 7. Image properties

Sampled 500 images for resolution and file-size measurement.

**Resolution distribution (top entries):**

| resolution (WxH) | # images (in sample) |
|---|---:|
| 256x256 | 500 |

**File-size stats (bytes):** mean = 114879, median = 114862, min = 49,205, max = 156,518.

All inspected images share a single resolution — safe to assume a fixed input size in the data loader.

## 8. Augmentation breakdown

| split | original | random_augment | ters | ters+aug | total |
|---|---:|---:|---:|---:|---:|
| `train` | 3,625 | 3,619 | 598 | 596 | 8,438 |
| `val` | 509 | 509 | 87 | 85 | 1,190 |
| `test` | 1,062 | 0 | 165 | 0 | 1,227 |
| **total** | **5,196** | **4,128** | **850** | **681** | **10,855** |

`random_augment` and `_ters_` are pre-baked augmentations in the dataset folder. If your training pipeline also applies on-the-fly augmentation, ensure you don't double-augment. The `_ters_` reversed pairs are particularly important to keep in the same split as their originals to avoid information leakage; verify this with the split assignment script before training.

## 9. Implications for modeling

- TODO: discuss with Claude after EDA
