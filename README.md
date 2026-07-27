<div align="center">

# Mineral Scale Prediction — Thermodynamic Baseline vs. ML Ensemble

*A reproducible machine-learning pipeline for binary scale-risk classification from*
*produced-water chemistry, benchmarked against a physics-based thermodynamic saturation model.*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-337AB7?style=flat-square)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-latest-9ACD32?style=flat-square)](https://lightgbm.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-6DB3F2?style=flat-square)](https://optuna.org/)
[![SHAP](https://img.shields.io/badge/SHAP-explainability-8A2BE2?style=flat-square)](https://shap.readthedocs.io/)

</div>

---

## Overview

Mineral scaling — the precipitation of calcite, barite, and celestite from produced water — is a persistent flow-assurance hazard that can choke wellbores, foul surface facilities, and drive costly workover campaigns. Conventional risk screening relies on **thermodynamic saturation modeling** (Saturation Index / Saturation Ratio), which is physically grounded but often brittle in the field, since it depends on accurate activity-coefficient corrections and can miss risk regimes not captured by simple Ksp thresholds.

This repository implements a full binary-classification pipeline that predicts scale occurrence (`Scale` vs. `No Scale`) directly from produced-water composition and P/T conditions, and benchmarks five tuned ML classifiers against a **naive thermodynamic saturation-index baseline** computed from first principles (ionic strength, Davies activity coefficients, temperature-dependent Ksp for calcite, barite, and celestite).

The pipeline is self-contained end-to-end: it trains and tunes every model, evaluates everything on a held-out test set, **saves every fitted model to disk**, then reloads them from disk and reproduces every table and figure a second time — proving the saved artifacts alone are sufficient to regenerate the paper's results without repeating hyperparameter search.

---

## Methodology

### Thermodynamic Baseline

A naive Saturation Index (calcite) / Saturation Ratio (barite, celestite) model is computed per sample from ionic concentrations, pH, and temperature:

- Ionic strength `I = 0.5 · Σ cᵢzᵢ²` from all major ions (Ca²⁺, Na⁺, Mg²⁺, Fe²⁺, HCO₃⁻, SO₄²⁻, Cl⁻, CO₃²⁻, Ba²⁺, Sr²⁺)
- Carbonate concentration back-calculated from bicarbonate and pH
- A sample is flagged `Scale` if calcite SI > 0 **or** barite/celestite SR > 1

The Davies activity-coefficient correction and temperature-dependent Ksp machinery (calcite, barite, celestite) are retained in the code for methodological completeness but are excluded from the reported results table, per the paper's final scope.

### ML Classifiers

Five classifiers are independently tuned with **Optuna** (40 trials each, 5-fold stratified CV, ROC-AUC objective), each wrapped in an imbalanced-learn `Pipeline` with `RobustScaler` + `ADASYN` oversampling:

<div align="center">

| Model | Library |
|:---:|:---:|
| Random Forest | `scikit-learn` |
| Extra Trees | `scikit-learn` |
| XGBoost | `xgboost` |
| LightGBM | `lightgbm` |
| MLP | `scikit-learn` |

</div>

Each model is trained twice — with and without ADASYN — to isolate the resampling effect. All final metrics are computed on a single untouched, stratified 25% test split (`random_state=5`).

### Reproducibility Design

- **Part G** saves every fitted pipeline (with & without ADASYN, 10 models total) plus the train/test split, feature names, and thermodynamic-model outputs to `./saved_models/` via `joblib`.
- **Part H** (in the same script) deletes all in-memory models/results, reloads everything from `./saved_models/`, and regenerates every table and figure a second time — confirming bit-for-bit reproducibility from disk alone.
- `reproduce.py` is the standalone companion script: run the main pipeline **once**, then use this script any time afterward to regenerate all tables/figures (plus bootstrap 95% confidence intervals) without rerunning Optuna or retraining.

---

## Results

All models are evaluated on the held-out test set across seven metrics: Accuracy, Precision, Recall, Specificity, F1, MCC, and ROC-AUC.

<div align="center">

| Model | Accuracy | Precision | Recall | Specificity | F1 | MCC | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | — | — | — | — | — | — | — |
| LightGBM | — | — | — | — | — | — | — |
| Extra Trees | — | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — | — |
| MLP | — | — | — | — | — | — | — |
| Thermodynamic | — | — | — | — | — | — | — |

*(populate from `final_results_table.csv` / `classification_reports.xlsx` after running the pipeline)*

</div>

The full per-class classification report for every model (precision/recall/F1/support) is exported to `classification_reports.xlsx`, one sheet per model, alongside a formatted summary sheet.

---

## Visualisations

The pipeline produces the following figures (each saved twice: once from the first training pass, once reproduced from the reloaded saved models, with a `_reloaded` / `_reproduced` suffix):

### A2 · AI Models vs. Thermodynamic Baseline
Horizontal bar comparison across Accuracy, Precision, Recall, and F1 for all six models side by side.
`A2_ai_vs_thermo_all_metrics.png`

### B · Combined Confusion Matrices
One panel per model (all 6), showing predicted vs. actual scale classification on the test set.
`B_confusion_matrices_combined.png`

### C · ADASYN Resampling Effect
- **C1** — grouped bar chart of Accuracy / F1 / ROC-AUC, with vs. without ADASYN, for all 5 ML models.
- **C2** — confusion-matrix comparison for the single best-performing model, with vs. without ADASYN.

`C1_adasyn_comparison.png`, `C2_best_model_cm_adasyn_vs_none.png`

### D · Combined ROC-AUC Curves
ROC curves for all 6 models (5 ML + thermodynamic) overlaid on one axis with AUC in the legend.
`D_roc_curves_all_models.png`

### E · Permutation Importance
Top-10 permutation importance (mean ROC-AUC drop) for each of the 5 ML models, one combined multi-panel figure.
`E_permutation_importance_all_models.png`

### F · SHAP Analysis (Best Model Only)
SHAP summary (beeswarm) and bar-importance plots for the single best-performing ML model, restricted to the `Scale` class.
`F_shap_summary_best_model.png`, `F_shap_bar_best_model.png`

---

## Repository Structure

```
.
├── scale_prediction_results_pipeline.py   # main pipeline: train, tune, evaluate, save, reload, verify
├── reproduce.py                           # standalone: reload saved models -> regenerate all figures + bootstrap CIs
├── Images/                          # created on first run — fitted pipelines + results bundle
│   ├── *.png                             # all figures listed above
```

---

## How to Run

**1. Install dependencies**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm optuna imbalanced-learn shap joblib openpyxl
```

**2. Point the script at your dataset**

Edit `DATA_PATH` in `scale_prediction_results_pipeline.py` to your Excel file. Expected columns include T, P, pH, and major ion concentrations (Ca²⁺, Na⁺, Mg²⁺, Fe²⁺, HCO₃⁻, SO₄²⁻, Cl⁻, CO₃²⁻, Ba²⁺, Sr²⁺), plus a binary target column named `Inspection Result`.

**3. Run the full pipeline (tunes, trains, evaluates, saves, and self-verifies)**

```bash
python scale_prediction_results_pipeline.py
```

This trains and Optuna-tunes all 5 ML models, evaluates everything against the thermodynamic baseline, saves every model to `./saved_models/`, then reloads them from disk and regenerates every result a second time to confirm reproducibility.

**4. Regenerate figures later without retraining**

```bash
python reproduce.py
```

Loads the saved pipelines and results bundle, recomputes all metrics/figures from the fitted models, and additionally computes 2000-sample bootstrap 95% confidence intervals for Accuracy, Precision, Recall, F1, and ROC-AUC (`bootstrap_ci_table.csv`).

---

<div align="center">
  <sub>Developed as part of a research pipeline in Petroleum Engineering — flow assurance / scale prediction</sub>
</div>
