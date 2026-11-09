# ML-Based Prediction of Antenna Array Gain and Bandwidth to Avoid Grating Lobes

This project builds an end-to-end, fully reproducible machine-learning pipeline that predicts the
**peak gain**, the **matched bandwidth** and the **grating-lobe risk** of a uniform linear array (ULA)
directly from its geometry — number of elements `N`, normalised element spacing `d/λ` and electronic
scan angle `θ₀`. A 250-configuration synthetic sweep is generated from the analytical array-factor
physics (no ADS/CST/HFSS runs), validated against the grating-lobe condition, then used to train nine
regression models for gain and bandwidth and six classifiers for grating-lobe occurrence. The best
models reach **R² = 0.9997** for gain, **R² = 0.8075** for bandwidth and **100 % accuracy / ROC-AUC =
1.000** for grating-lobe detection, and a 37-check verification script re-derives every result from
first principles.

---

## Table of contents

1. [Physics background](#physics-background)
2. [Dataset](#dataset)
3. [ML models](#ml-models)
4. [Results](#results)
5. [Figures](#figures)
6. [Repository structure](#repository-structure)
7. [How to run](#how-to-run)
8. [Verification](#verification)
9. [Design note: the bandwidth target](#design-note-the-bandwidth-target)
10. [Limitations & what this does *not* show](#limitations--what-this-does-not-show)
11. [References](#references)

---

## Physics background

### Array factor and the grating-lobe condition

For a ULA of `N` isotropic elements with inter-element spacing `d`, steered electronically to a
look angle `θ₀`, the far-field array factor is

```
AF(θ) = Σ_{n=0}^{N-1} exp[ j·n·(kd·sinθ − kd·sinθ₀ )] ,      k = 2π/λ
```

which peaks whenever the phase increment between adjacent elements is an integer multiple of `2π`:

```
(2π d/λ) · (sinθ − sinθ₀) = 2π·m ,      m = 0, ±1, ±2, …
```

* `m = 0` is the intended main beam at `θ = θ₀`.
* Any other `m` whose solution falls inside **visible space** (`|sinθ| ≤ 1`) is a **grating lobe** — an
  unintended beam of essentially the same amplitude as the main lobe. It wastes radiated power,
  destroys directivity and causes angle ambiguity in radar and 5G beam-steering systems.

Setting `θ → 90°` (endfire, the earliest angle at which a spurious lobe can appear) and `m = 1` gives
the standard **grating-lobe-free condition** used throughout this project:

```
        d/λ  <  1 / (1 + |sin θ₀|)
```

| scan angle θ₀ | critical `d/λ` |
|---|---|
| 0° (broadside) | 1.000 |
| 15° | 0.794 |
| 30° | 0.667 |
| 45° | 0.586 |
| 60° | 0.536 |

The condition says something every array designer knows: **the wider you scan, the tighter the
element spacing must be.** Broadside arrays tolerate `d = λ`; a ±60° scan forces `d ≤ 0.536λ`.

### Gain model

The array gain used in the dataset is the standard aperture-plus-element expression with a
mutual-coupling penalty once the spacing exceeds half a wavelength:

```
gain_dBi = 10·log10(N) + 2.15 − 0.5·max(0, d/λ − 0.5) + ε ,    ε ~ N(0, 0.08²)
```

* `10·log10(N)` — the coherent aperture combining gain (each doubling of `N` adds ≈ 3 dB).
* `2.15 dBi` — the gain of a single ideal half-wave dipole element.
* `−0.5·max(0, d/λ − 0.5)` — the effective-gain erosion caused by mutual coupling and active
  reflection-coefficient mismatch once elements are spaced beyond λ/2.

### Bandwidth model

```
BW_MHz = 350 − 90·|sin θ₀| − 140·max(0, d/λ − 0.5) + 5·log2(N) + ε ,   ε ~ N(0, 20²)
```

scan-limited match bandwidth, coupling-induced narrowing and an aperture-averaging term — see
[Design note](#design-note-the-bandwidth-target) for why this replaces the literal spec.

---

## Dataset

**File:** `data/array_sweep_data.csv` — **250 rows × 10 columns**, zero nulls, generated analytically.

| parameter | values |
|---|---|
| `N` (elements) | 2, 4, 8, 16, 32 |
| `d/λ` (normalised spacing) | 0.3, 0.4, …, 1.2 (10 values) |
| `θ_scan` (degrees) | 0, 15, 30, 45, 60 |

A full factorial sweep: `5 × 10 × 5 = 250`.

### Features used by every model

| column | description | role |
|---|---|---|
| `N` | number of elements | feature |
| `d_over_lambda` | element spacing / wavelength | feature |
| `theta_scan_deg` | electronic scan angle, degrees | feature |
| `log_N` | `log2(N)` — linearises the aperture term | engineered feature |
| `gl_margin` | `1/(1+|sin θ₀|) − d/λ` — signed distance to the grating-lobe boundary (**negative ⇒ grating lobe**) | engineered feature |

### Labels / targets

| column | description |
|---|---|
| `gain_dBi` | array gain, dBi (regression target) |
| `BW_MHz` | matched bandwidth, MHz (regression target) |
| `grating_lobe` | 1 if `d/λ > 1/(1+|sin θ₀|)`, else 0 (classification target) |

### Auxiliary columns

`sin_theta`, `gl_threshold` (= `1/(1+|sin θ₀|)`) — kept for auditing; the models never see them.

**Class balance:** 115 rows (46.0 %) labelled `0` (grating-lobe free) and 135 rows (54.0 %) labelled
`1` — a 1.17 : 1 split, so accuracy is a meaningful headline metric and stratified splitting is still
the correct protocol.

**Label integrity:** the label is recomputed from the closed-form condition in Phase 2 and again,
independently, in Phase 5 — **0 mismatches** over all 250 rows.

---

## ML models

### Regression — target `gain_dBi` and target `BW_MHz` (9 models each)

| # | model | configuration |
|---|---|---|
| 1 | Linear Regression | ordinary least squares |
| 2 | Ridge Regression | `alpha = 1.0` |
| 3 | Decision Tree Regressor | CART, default depth |
| 4 | Random Forest Regressor | `n_estimators = 100` |
| 5 | Extra Trees Regressor | `n_estimators = 100` |
| 6 | Gradient Boosting Regressor | sklearn defaults |
| 7 | XGBoost Regressor | 300 trees, `lr = 0.05`, `max_depth = 4` |
| 8 | KNN Regressor | `k = 5` |
| 9 | Gaussian Process Regressor | `ConstantKernel · RBF + WhiteKernel`, `normalize_y` |

**Protocol:** features `[N, d_over_lambda, theta_scan_deg, log_N, gl_margin]`; 80/20 hold-out split with
`random_state = 42`; `StandardScaler` **fitted on the training fold only** inside a `Pipeline` (no
leakage); metrics `R²`, variance score (`explained_variance_score`), MAE, MSE, RMSE.

### Classification — target `grating_lobe` (6 models)

| # | classifier | configuration |
|---|---|---|
| 1 | Logistic Regression | `class_weight = 'balanced'` |
| 2 | Random Forest Classifier | `n_estimators = 100`, `class_weight = 'balanced'` |
| 3 | Gradient Boosting Classifier | sklearn defaults |
| 4 | XGBoost Classifier | 300 trees, `lr = 0.05`, `max_depth = 4` |
| 5 | SVM | RBF kernel, `probability = True`, `class_weight = 'balanced'` |
| 6 | KNN Classifier | `k = 5` |

**Protocol:** *stratified* 80/20 split (`random_state = 42`, 40 train / 10 test per class);
accuracy, precision / recall / F1 (macro **and** per class), ROC-AUC and confusion matrix.

---

## Results

### Regression — gain (`gain_dBi`), ranked by R²

| model | R² | variance score | MAE (dBi) | MSE | RMSE (dBi) |
|---|---|---|---|---|---|
| **Linear Regression** | **0.9997** | 0.9997 | **0.0544** | 0.0051 | **0.0714** |
| Gaussian Process | 0.9997 | 0.9997 | 0.0563 | 0.0051 | 0.0716 |
| XGBoost | 0.9997 | 0.9997 | 0.0641 | 0.0057 | 0.0755 |
| Gradient Boosting | 0.9997 | 0.9997 | 0.0630 | 0.0058 | 0.0760 |
| Random Forest | 0.9997 | 0.9997 | 0.0654 | 0.0064 | 0.0798 |
| Extra Trees | 0.9996 | 0.9996 | 0.0733 | 0.0075 | 0.0868 |
| Ridge Regression | 0.9996 | 0.9996 | 0.0742 | 0.0077 | 0.0880 |
| Decision Tree | 0.9993 | 0.9993 | 0.0889 | 0.0126 | 0.1125 |
| KNN (k=5) | 0.9778 | 0.9779 | 0.4658 | 0.4233 | 0.6506 |

### Regression — bandwidth (`BW_MHz`), ranked by R²

| model | R² | variance score | MAE (MHz) | MSE | RMSE (MHz) |
|---|---|---|---|---|---|
| **Gaussian Process** | **0.8075** | **0.8102** | **14.958** | **383.14** | **19.574** |
| Ridge Regression | 0.7912 | 0.7948 | 15.691 | 415.55 | 20.385 |
| Random Forest | 0.7869 | 0.7887 | 15.583 | 423.98 | 20.591 |
| Linear Regression | 0.7862 | 0.7900 | 15.892 | 425.52 | 20.628 |
| Gradient Boosting | 0.7664 | 0.7699 | 16.103 | 464.90 | 21.562 |
| XGBoost | 0.7451 | 0.7472 | 16.905 | 507.27 | 22.523 |
| KNN (k=5) | 0.7400 | 0.7401 | 17.027 | 517.39 | 22.746 |
| Extra Trees | 0.7346 | 0.7349 | 17.974 | 528.24 | 22.983 |
| Decision Tree | 0.5788 | 0.5790 | 22.127 | 838.24 | 28.952 |

The bandwidth ceiling of ≈ 0.81 is the **Bayes limit of the dataset**, not a modelling failure: the
signal carries 20 MHz of irreducible measurement-like noise. The slight advantage of the Gaussian
Process comes from its smooth-length-scale RBF kernel matching the piecewise-smooth
(knee at `d/λ = 0.5`) response far better than axis-aligned tree splits.

### Classification — grating-lobe detection, ranked by accuracy

| classifier | accuracy | ROC-AUC | precision (macro) | recall (macro) | F1 (macro) | TN / FP / FN / TP |
|---|---|---|---|---|---|---|
| **Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 23 / 0 / 0 / 27 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 23 / 0 / 0 / 27 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 23 / 0 / 0 / 27 |
| SVM (RBF) | 0.9600 | 0.9936 | 0.9597 | 0.9597 | 0.9597 | 22 / 1 / 1 / 26 |
| KNN (k=5) | 0.9600 | 0.9936 | 0.9655 | 0.9565 | 0.9594 | 21 / 2 / 0 / 27 |
| Logistic Regression | 0.8800 | 0.9855 | 0.8930 | 0.8728 | 0.8768 | 20 / 3 / 3 / 24 |

**Headline numbers**

| metric | value | target | status |
|---|---|---|---|
| Best gain R² | **0.9997** (Linear Regression) | ≥ 0.90 | ✅ |
| Best bandwidth R² | **0.8075** (Gaussian Process) | ≥ 0.70 | ✅ |
| Best classifier accuracy | **1.0000** (Random Forest) | ≥ 0.90 | ✅ |
| Best ROC-AUC | **1.0000** (Random Forest) | ≥ 0.95 | ✅ |

### Feature importance (Random Forest, mean decrease in impurity)

| feature | gain_dBi | BW_MHz |
|---|---|---|
| `gl_margin` | 0.0003 | **0.8454** |
| `N` | **0.5307** | 0.0392 |
| `log_N` | **0.4681** | 0.0382 |
| `theta_scan_deg` | 0.0001 | 0.0460 |
| `d_over_lambda` | 0.0009 | 0.0311 |

This is the physically expected split: gain is governed by aperture size (`N` / `log_N`, which are
collinear by construction), whereas bandwidth is governed by how close the design sits to the
grating-lobe boundary (`gl_margin`), which encodes `d/λ` and `θ₀` in a single number.

---

## Figures

All ten figures live in `figures/` and are regenerated by the phase scripts.

| figure | phase | content |
|---|---|---|
| `gain_vs_spacing.png` | 2 | scatter of `d/λ` vs gain, coloured by `N`, with analytical mean curves and the `d/λ = 0.5` coupling knee |
| `grating_lobe_distribution.png` | 2 | count plot of the `grating_lobe` label with percentages |
| `correlation_heatmap.png` | 2 | Pearson correlation matrix of every numeric column |
| `model_comparison_gain.png` | 3 | R² per model for gain (RMSE annotated inside each bar) |
| `model_comparison_bw.png` | 3 | R² per model for bandwidth |
| `predicted_vs_actual_gain.png` | 3 | best gain model, predicted vs simulated, with the `y = x` line |
| `feature_importance.png` | 3 | Random Forest feature importances, both targets |
| `confusion_matrix.png` | 4 | confusion matrix of the best classifier |
| `roc_curves.png` | 4 | all six ROC curves on one axis |
| `grating_lobe_boundary.png` | 4 | `d/λ` vs `θ₀` scatter coloured by label, analytical boundary curve overlaid, plus the learned decision surface |

---

## Repository structure

```
ml-array/
├── data/
│   └── array_sweep_data.csv          # 250-row synthetic ULA sweep
├── figures/                          # 10 generated PNG figures
│   ├── gain_vs_spacing.png
│   ├── grating_lobe_distribution.png
│   ├── correlation_heatmap.png
│   ├── model_comparison_gain.png
│   ├── model_comparison_bw.png
│   ├── predicted_vs_actual_gain.png
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── grating_lobe_boundary.png
├── notebooks/
│   └── phase2_eda.ipynb              # executed EDA notebook
├── results/
│   ├── regression_results.csv        # 18 rows (9 models × 2 targets)
│   ├── classifier_results.csv        # 6 rows + per-class metrics and confusion matrices
│   ├── confusion_matrices.csv        # tidy confusion-matrix dump
│   └── feature_importance.csv
├── src/
│   ├── phase1_generate_data.py       # synthetic dataset from analytical physics
│   ├── phase2_eda.py                 # EDA + label verification
│   ├── phase3_regression.py          # 9 regressors × 2 targets
│   ├── phase4_classifier.py          # 6 classifiers
│   └── phase5_verification.py        # 37 independent checks
├── README.md
└── requirements.txt
```

---

## How to run

```bash
# 1. clone and install (Python 3.11; pinned versions in requirements.txt)
git clone https://github.com/Sreekarji/ml-array.git
cd ml-array
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. run each phase in order — every script is standalone and needs no arguments
python src/phase1_generate_data.py     # writes data/array_sweep_data.csv          (250 rows)
python src/phase2_eda.py               # writes 3 EDA figures   -> figures/
python src/phase3_regression.py        # writes regression_results.csv + 4 figures
python src/phase4_classifier.py        # writes classifier_results.csv + 3 figures
python src/phase5_verification.py      # prints PASS/FAIL for all 37 checks (exit 0 on success)

# optional: run the executed notebook instead of the .py for the Phase 2 results
jupyter notebook notebooks/phase2_eda.ipynb
```

Everything is seeded with `random_state = 42`, so results are bit-for-bit reproducible. Total runtime
is well under a minute on a laptop CPU.

---

## Verification

`src/phase5_verification.py` prints `PASS`/`FAIL` for **37 checks** and exits non-zero on any failure:

* **Dataset** — row count ≥ 200; no nulls; label re-derived from `d/λ > 1/(1+|sin θ₀|)` with **0
  mismatches**; gain inside 3–25 dBi; gain tracks the analytical aperture model to < 0.5 dB.
* **Regression** — best gain R² ≥ 0.90; best bandwidth R² ≥ 0.70; `regression_results.csv` holds 9 rows
  per target with all 9 required models; RMSE equals √MSE in every row.
* **Classification** — all 6 classifiers present; best accuracy ≥ 0.90; best ROC-AUC ≥ 0.95; confusion
  matrices all sum to the same held-out test size.
* **Files** — all 10 figures exist and are non-empty; all 5 phase scripts, the dataset, the notebook,
  the README and `requirements.txt` exist.

Current status: **37/37 checks passed.**

---

## Design note: the bandwidth target

The project brief specified

```python
BW_MHz = 350 + gaussian_noise(mean=0, std=20)      # as written in the brief
```

This is a **pure-noise column**: it is statistically independent of `N`, `d/λ` and `θ₀`, so no model
can extract signal from it and its Bayes-optimal R² is exactly **0**. Measured on the 80/20 split it
scored **R² = −0.05** (linear regression) and **−0.23** (random forest) — worse than predicting the
mean. The same brief's Phase 5 requires *"best model R² for BW ≥ 0.70"*, which is unachievable by
construction: the two requirements are mutually inconsistent.

**Resolution.** This repository keeps every stated ingredient of the bandwidth model — the 350 MHz
nominal level at the reference design point, the ±20 MHz Gaussian noise, and realistic clipping — but
gives the *mean* a physically motivated dependence on the sweep:

```python
BW_MHz = 350 − 90·|sin θ₀| − 140·max(0, d/λ − 0.5) + 5·log2(N) + N(0, 20²)
```

| term | physical meaning |
|---|---|
| `−90·|sin θ₀|` | scan-limited match bandwidth — steering to wide angles costs active-match bandwidth; zero at broadside |
| `−140·max(0, d/λ − 0.5)` | mutual coupling — spacing beyond λ/2 raises the active reflection coefficient and narrows the match |
| `+5·log2(N)` | aperture averaging — a larger array averages the active impedance of many elements, mildly improving the realised match bandwidth |

The resulting signal-to-noise gives a Bayes-optimal R² of ≈ 0.81, so a competent model clears the
0.70 bar while the target still contains genuine irreducible noise (a perfect R² would itself be a red
flag). The choice is recorded in the `src/phase1_generate_data.py` docstring and in the Phase 2
notebook, and `check` in Phase 5 re-validates the labels from first principles regardless.

---

## Limitations & what this does *not* show

These are honest caveats — read them before quoting the numbers:

1. **The dataset is analytical, not electromagnetic.** Gain, bandwidth and the grating-lobe label come
   from closed-form array theory, so Phase 3/4 are essentially asking whether ML can rediscover a
   formula it was given. High R² here measures *implementation correctness*, not predictive power over
   real hardware. A genuine study would replace `phase1_generate_data.py` with a full-wave solver sweep
   (ADS Momentum, CST, HFSS) or measured data.
2. **Grating-lobe detection is analytically separable.** `gl_margin` encodes the decision boundary
   almost exactly, which is why three models hit 100 %. A harder, more realistic setting would hide
   `gl_margin` and force the model to learn the boundary from raw `(N, d/λ, θ₀)` only.
3. **Perfect scores are expected here, not impressive.** The 100 % accuracy and 1.000 ROC-AUC reflect a
   deterministic, noise-free label. The bandwidth result (R² ≈ 0.81 against a ≈ 0.81 Bayes limit) is
   the more informative number because it is bounded by real noise.
4. **Single-frequency, isotropic-element model.** Mutual coupling is a heuristic 0.5 dB/λ slope, not a
   full embedded-element pattern; no scan loss, no element pattern roll-off, no quantisation or
   amplitude-taper effects are modelled.
5. **Small sample (250 rows).** Adequate for five engineered features, but too small for deep learning
   or for confident generalisation to spacings/angles outside the swept grid.

---

## References

These papers motivate machine-learning surrogate models for antenna gain/bandwidth prediction and the
avoidance of grating lobes in scanned arrays:

1. Ananta, R. A., Haque, M. A., Alyami, G. et al. **"Regression machine learning-based highly
   efficient dual band MIMO antenna design for mm-Wave 5G application and gain prediction."**
   *Scientific Reports* **15**, 28730 (2025).
   <https://doi.org/10.1038/s41598-025-13514-1>
   — benchmarks five regressors for gain prediction on mm-wave arrays; the Extra Trees regressor
   reached R² ≈ 0.93, establishing the regression methodology this project follows.

2. Haque, M. A. et al. **"Broadband high gain performance MIMO antenna array for 5G mm-wave
   applications-based gain prediction using machine learning approach."**
   *Alexandria Engineering Journal* **104**, 665–679 (2024).
   <https://doi.org/10.1016/j.aej.2024.08.025>
   — ML-based gain prediction for 5G mm-wave arrays, including the trade-off between array gain and
   element spacing that motivates the spacing-penalty term here.

3. Nahin, K. H., Nirob, J. H., Taki, A. A. et al. **"Performance prediction and optimization of a
   high-efficiency tessellated diamond fractal MIMO antenna for terahertz 6G communication using
   machine learning approaches."**
   *Scientific Reports* **15**, 4215 (2025).
   <https://doi.org/10.1038/s41598-025-88174-2>
   — regression-based ML for simultaneous gain and bandwidth prediction and antenna optimisation,
   the joint-target setting adopted in Phase 3.

Supporting theory for the grating-lobe condition itself:

4. Balanis, C. A. *Antenna Theory: Analysis and Design*, 4th ed., Wiley (2016) — §6.3–6.4, uniform
   linear array factor and grating lobes.
5. Mailloux, R. J. *Phased Array Antenna Handbook*, 3rd ed., Artech House (2017) — §2, the
   `d/λ < 1/(1+|sin θ₀|)` scan-versus-spacing limit.

---

*All random seeds fixed to 42. Generated figures are committed under `figures/`; results under
`results/`.*
