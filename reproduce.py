# -*- coding: utf-8 -*-
"""
Reproduce Plots From Saved Models — CORRECTED THERMODYNAMIC MODEL VERSION
=============================================================================
Companion script to `scale_prediction_results_pipeline.py`.

WHAT CHANGED vs. the previous reproduce script (per author's request):
  - The thermodynamic baseline is now computed using the ACTIVITY-CORRECTED
    (Davies equation) saturation index / saturation ratio, with
    temperature-dependent Ksp for calcite, barite, and celestite — i.e. the
    functions that were already written in the main pipeline
    (`calcite_SI_corrected`, `barite_SR_corrected`, `celestite_SR_corrected`,
    `logKsp_calcite/barite/celestite`, `activity_coefficient_davies`) but were
    previously left unused ("kept for Methods narrative only"). They are now
    the ones actually driving Table 3 / Fig. 5 / Fig. 9, matching Eq. (4) in
    the manuscript (SR = a_Me*a_An / Ksp(P,T,I)) literally, not just in name.
  - The continuous risk score used to compute ROC-AUC for the thermodynamic
    model (previously an arbitrary 0.4/0.6-weighted, /10-scaled heuristic)
    is replaced with a principled, threshold-anchored score: a logistic
    (sigmoid) transform of each mineral's own SI / log10(SR), anchored at
    the physical equilibrium point (SI=0 or SR=1), and the OVERALL score
    is the maximum across the three minerals — i.e. "the well is at risk if
    ANY one of the three minerals is supersaturated," which mirrors the
    OR-logic already used for the binary thermo_decision() label.

NO ML MODEL IS RETRAINED HERE. All 5 ML pipelines (with & without ADASYN)
are loaded from ./saved_models/ exactly as before; only the thermodynamic
baseline is recomputed from the raw X_test columns already stored in
results_bundle.joblib.

Requires: pip install shap joblib
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import joblib
import shap

from sklearn.metrics import (confusion_matrix, roc_auc_score, f1_score, roc_curve,
                              accuracy_score, matthews_corrcoef, precision_score, recall_score)
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# GLOBAL STYLE  (identical to the main pipeline)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     10,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8,
    "legend.frameon":     True,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.7",
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.grid":          True,
    "grid.color":         "0.88",
    "grid.linewidth":     0.5,
    "grid.linestyle":     "--",
})

MODEL_NAMES  = ['Random Forest', 'Extra Trees', 'XGBoost', 'LightGBM', 'MLP']
COLORS       = ['#D55E00', '#E69F00', '#0072B2', '#009E73', '#999999']
MARKERS      = ['s', 'D', 'o', '^', 'X']
MODEL_COLOR  = dict(zip(MODEL_NAMES, COLORS))
MODEL_MARKER = dict(zip(MODEL_NAMES, MARKERS))
THERMO_COLOR = '#B2182B'
THERMO_LABEL = 'Thermodynamic'   # corrected model now stands as THE baseline

def bar_color(model_name):
    if model_name == THERMO_LABEL:
        return THERMO_COLOR
    return MODEL_COLOR.get(model_name, '#7F7F7F')

# =============================================================================
# 1) LOAD SAVED ML MODELS + RAW TEST DATA (no retraining of any ML model)
# =============================================================================

SAVE_DIR = 'saved_models'

bundle = joblib.load(os.path.join(SAVE_DIR, 'results_bundle.joblib'))
X_train        = bundle['X_train']     # raw (unscaled) features, needed to recompute thermo
X_test         = bundle['X_test']      # raw (unscaled) features, needed to recompute thermo
y_train        = bundle['y_train']
y_test         = bundle['y_test']
feature_names  = bundle['feature_names']
best_ml_name   = bundle['best_ml_name']
y_true         = y_test.values

fitted_pipelines    = {}
no_adasyn_pipelines = {}
for name in MODEL_NAMES:
    fitted_pipelines[name] = joblib.load(
        os.path.join(SAVE_DIR, f"{name.replace(' ', '_')}_pipeline.joblib"))
    no_adasyn_pipelines[name] = joblib.load(
        os.path.join(SAVE_DIR, f"{name.replace(' ', '_')}_no_adasyn_pipeline.joblib"))

print(f"Loaded {len(MODEL_NAMES)} ML models (with & without ADASYN) from ./{SAVE_DIR}/ "
      f"— NOT retrained.")
print(f"Best ML model (from saved bundle): {best_ml_name}")

# -----------------------------------------------------------------------
# Display-name mapping for figures only (does not touch the actual column
# names the ML models were fit on)
# -----------------------------------------------------------------------
FEATURE_DISPLAY_MAP = {
    'T':                  'T (F)',
    'P':                  'P (psia)',
    'pH':                 'pH',
    'Ca2+ (ppm)':         'Ca²⁺ (ppm)',
    'Na+ (ppm)':          'Na⁺ (ppm)',
    'Mg2+ (ppm)':         'Mg²⁺ (ppm)',
    'Fe2+ (ppm)':         'Fe²⁺ (ppm)',
    'HCO30 (ppm)':        'HCO₃⁻ (ppm)',
    'SO4 2-(ppm)':        'SO₄²⁻ (ppm)',
    'Cl- (ppm)':          'Cl⁻ (ppm)',
    'CO3 20 (ppm)':       'CO₃²⁻ (ppm)',
    'Ba2+ (ppm)':         'Ba²⁺ (ppm)',
    'Sr2+':               'Sr²⁺ (ppm)',
}

def to_display(names):
    return [FEATURE_DISPLAY_MAP.get(n, n) for n in names]

feature_names_display = to_display(feature_names)

# =============================================================================
# 2) CORRECTED THERMODYNAMIC MODEL  (activity-based SI/SR, T-dependent Ksp)
#    Recomputed here from RAW X_test columns — no ML retraining involved.
# =============================================================================

def find_col(possible_names, columns):
    for p in possible_names:
        for c in columns:
            if p.lower().replace(" ", "") in c.lower().replace(" ", ""):
                return c
    return None

col_T    = find_col(["T"], X_test.columns)
col_P    = find_col(["P"], X_test.columns)
col_pH   = find_col(["pH"], X_test.columns)
col_Ca   = find_col(["Ca2+", "Ca"], X_test.columns)
col_Na   = find_col(["Na+", "Na"], X_test.columns)
col_Mg   = find_col(["Mg2+", "Mg"], X_test.columns)
col_Fe   = find_col(["Fe2+", "Fe"], X_test.columns)
col_HCO3 = find_col(["HCO30", "HCO3"], X_test.columns)
col_SO4  = find_col(["SO4"], X_test.columns)
col_Cl   = find_col(["Cl-", "Cl"], X_test.columns)
col_CO3  = find_col(["CO320", "CO3"], X_test.columns)
col_Ba   = find_col(["Ba2+", "Ba"], X_test.columns)
col_Sr   = find_col(["Sr2+", "Sr"], X_test.columns)
col_TDS  = find_col(["TDS"], X_test.columns)

MW = {
    'Ca': 40.078, 'Na': 22.990, 'Mg': 24.305, 'Fe': 55.845,
    'HCO3': 61.016, 'SO4': 96.060, 'Cl': 35.453, 'CO3': 60.008,
    'Ba': 137.327, 'Sr': 87.620,
}

def get_molar(row, col, mw):
    if col is None or pd.isna(row.get(col, np.nan)):
        return 0.0
    return max(row[col], 0.0) / (mw * 1000.0)

def ionic_strength(row):
    ions = [
        (get_molar(row, col_Ca, MW['Ca']), 2),
        (get_molar(row, col_Na, MW['Na']), 1),
        (get_molar(row, col_Mg, MW['Mg']), 2),
        (get_molar(row, col_Fe, MW['Fe']), 2),
        (get_molar(row, col_HCO3, MW['HCO3']), 1),
        (get_molar(row, col_SO4, MW['SO4']), 2),
        (get_molar(row, col_Cl, MW['Cl']), 1),
        (get_molar(row, col_CO3, MW['CO3']), 2),
        (get_molar(row, col_Ba, MW['Ba']), 2),
        (get_molar(row, col_Sr, MW['Sr']), 2),
    ]
    I = 0.5 * sum(c * (z ** 2) for c, z in ions)
    return max(I, 1e-8)

def debye_huckel_A(T_K):
    T_C = T_K - 273.15
    return 0.4918 + 6.0435e-4 * T_C + 1.3132e-6 * T_C ** 2

def activity_coefficient_davies(z, I, T_K):
    A = debye_huckel_A(T_K)
    sqI = np.sqrt(I)
    log_gamma = -A * (z ** 2) * (sqI / (1 + sqI) - 0.3 * I)
    return 10 ** log_gamma

def to_kelvin(T_F):
    return (T_F - 32) * 5 / 9 + 273.15

def logKsp_calcite(T_K):
    return (-171.9065 - 0.077993 * T_K + 2839.319 / T_K + 71.595 * np.log10(T_K))

def logKsp_barite(T_K):
    T_C = T_K - 273.15
    return -9.97 - 0.00028 * T_C + 0.0000068 * (T_C ** 2)

def logKsp_celestite(T_K):
    T_C = T_K - 273.15
    return -6.63 - 0.0022 * T_C + 0.0000091 * (T_C ** 2)

def estimate_carbonate_from_ph(hco3_mol, ph):
    hco3_mol = max(hco3_mol, 1e-10)
    ratio = 10 ** (ph - 10.3)
    return max(hco3_mol * ratio, 1e-10)

def calcite_SI_corrected(row):
    try:
        T_K = to_kelvin(row[col_T])
        pH = row[col_pH]
        I = ionic_strength(row)
        Ca_mol = get_molar(row, col_Ca, MW['Ca'])
        HCO3_mol = get_molar(row, col_HCO3, MW['HCO3'])
        CO3_mol = estimate_carbonate_from_ph(HCO3_mol, pH)
        gamma_Ca = activity_coefficient_davies(2, I, T_K)
        gamma_CO3 = activity_coefficient_davies(2, I, T_K)
        IAP = (gamma_Ca * max(Ca_mol, 1e-10)) * (gamma_CO3 * CO3_mol)
        logKsp = logKsp_calcite(T_K)
        return np.log10(IAP) - logKsp
    except Exception:
        return np.nan

def barite_SR_corrected(row):
    try:
        T_K = to_kelvin(row[col_T])
        I = ionic_strength(row)
        Ba_mol = get_molar(row, col_Ba, MW['Ba'])
        SO4_mol = get_molar(row, col_SO4, MW['SO4'])
        gamma_Ba = activity_coefficient_davies(2, I, T_K)
        gamma_SO4 = activity_coefficient_davies(2, I, T_K)
        IAP = (gamma_Ba * max(Ba_mol, 1e-10)) * (gamma_SO4 * max(SO4_mol, 1e-10))
        Ksp = 10 ** logKsp_barite(T_K)
        return IAP / Ksp
    except Exception:
        return np.nan

def celestite_SR_corrected(row):
    try:
        T_K = to_kelvin(row[col_T])
        I = ionic_strength(row)
        Sr_mol = get_molar(row, col_Sr, MW['Sr'])
        SO4_mol = get_molar(row, col_SO4, MW['SO4'])
        gamma_Sr = activity_coefficient_davies(2, I, T_K)
        gamma_SO4 = activity_coefficient_davies(2, I, T_K)
        IAP = (gamma_Sr * max(Sr_mol, 1e-10)) * (gamma_SO4 * max(SO4_mol, 1e-10))
        Ksp = 10 ** logKsp_celestite(T_K)
        return IAP / Ksp
    except Exception:
        return np.nan

print("\nRecomputing CORRECTED (activity-based) thermodynamic model on X_test...")
thermo_df = X_test.copy()
thermo_df["I"]                      = thermo_df.apply(ionic_strength, axis=1)
thermo_df["Calcite_SI_corrected"]   = thermo_df.apply(calcite_SI_corrected, axis=1)
thermo_df["Barite_SR_corrected"]    = thermo_df.apply(barite_SR_corrected, axis=1)
thermo_df["Celestite_SR_corrected"] = thermo_df.apply(celestite_SR_corrected, axis=1)

# -----------------------------------------------------------------------
# Binary decision (unchanged logic): scale-forming if SI>0 for calcite OR
# SR>1 for barite/celestite -- now driven by the corrected values.
# -----------------------------------------------------------------------

def thermo_decision(df_, si_col, sr_cols, threshold_si=0.0, threshold_sr=1.0):
    calcite_pred = (df_[si_col] > threshold_si).astype(int)
    other_preds = [(df_[c] > threshold_sr).astype(int) for c in sr_cols]
    combined = calcite_pred.copy()
    for p in other_preds:
        combined = combined | p
    return combined

y_pred_thermo = thermo_decision(
    thermo_df, "Calcite_SI_corrected", ["Barite_SR_corrected", "Celestite_SR_corrected"]
).values

# -----------------------------------------------------------------------
# Continuous risk score for ROC-AUC — principled replacement for the old
# arbitrary 0.4/0.6-weighted, /10-scaled heuristic.
#
# Each mineral's own equilibrium departure is mapped through a logistic
# (sigmoid) function, anchored at its own physical equilibrium point:
#   - calcite:   sigmoid(SI)                     [SI=0 is equilibrium]
#   - barite:    sigmoid(log10(SR))               [SR=1 -> log10(SR)=0]
#   - celestite: sigmoid(log10(SR))               [SR=1 -> log10(SR)=0]
# The OVERALL risk score is the MAXIMUM across the three minerals, mirroring
# the OR-logic already used for the binary label above ("the well is at risk
# if ANY one of the three minerals is supersaturated").
# -----------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def thermo_risk_score(df_, si_col, sr_cols):
    calcite_risk = sigmoid(df_[si_col].values)
    sr_risks = [sigmoid(np.log10(np.clip(df_[c].values, 1e-6, None))) for c in sr_cols]
    return np.maximum.reduce([calcite_risk] + sr_risks)

proba_thermo = thermo_risk_score(
    thermo_df, "Calcite_SI_corrected", ["Barite_SR_corrected", "Celestite_SR_corrected"]
)

def safe_auc(y_true_, proba_):
    try:
        return roc_auc_score(y_true_, proba_)
    except Exception:
        return np.nan

def compute_thermo_metrics(y_te, y_pred, y_proba):
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {
        'Accuracy':    accuracy_score(y_te, y_pred),
        'Precision':   precision_score(y_te, y_pred, zero_division=0),
        'Recall':      recall_score(y_te, y_pred, zero_division=0),
        'Specificity': specificity,
        'F1':          f1_score(y_te, y_pred, average='weighted'),
        'MCC':         matthews_corrcoef(y_te, y_pred),
        'ROC-AUC':     safe_auc(y_te, y_proba),
        'cm':          cm,
    }

metrics_thermo = compute_thermo_metrics(y_true, y_pred_thermo, proba_thermo)

print("\n" + "=" * 80)
print("Thermodynamic Model (CORRECTED — activity-based, T-dependent Ksp)")
print("=" * 80)
print(f"Thermodynamic -> Acc: {metrics_thermo['Accuracy']:.4f} | "
      f"Prec: {metrics_thermo['Precision']:.4f} | Rec: {metrics_thermo['Recall']:.4f} | "
      f"Spec: {metrics_thermo['Specificity']:.4f} | F1: {metrics_thermo['F1']:.4f} | "
      f"MCC: {metrics_thermo['MCC']:.4f} | AUC: {metrics_thermo['ROC-AUC']:.4f}")

# =============================================================================
# 3) RECOMPUTE PREDICTIONS/METRICS FROM THE LOADED ML MODELS (unchanged —
#    same fitted pipelines, same test set -> identical numbers as before)
# =============================================================================

def compute_full_metrics(y_te, y_pred, y_proba):
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {
        'Accuracy':    accuracy_score(y_te, y_pred),
        'Precision':   precision_score(y_te, y_pred, zero_division=0),
        'Recall':      recall_score(y_te, y_pred, zero_division=0),
        'Specificity': specificity,
        'F1':          f1_score(y_te, y_pred, average='weighted'),
        'MCC':         matthews_corrcoef(y_te, y_pred),
        'ROC-AUC':     roc_auc_score(y_te, y_proba),
        'cm':          cm,
    }

results = {}
for name in MODEL_NAMES:
    pipe = fitted_pipelines[name]
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    m = compute_full_metrics(y_test, y_pred, y_proba)
    m['y_pred'] = y_pred
    m['y_proba'] = y_proba
    results[name] = m

no_adasyn_results = {}
for name in MODEL_NAMES:
    pipe_no = no_adasyn_pipelines[name]
    y_pred_no = pipe_no.predict(X_test)
    y_proba_no = pipe_no.predict_proba(X_test)[:, 1]
    no_adasyn_results[name] = compute_full_metrics(y_test, y_pred_no, y_proba_no)

ALL_METRICS = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'MCC', 'ROC-AUC']

final_rows = [
    {'Model': THERMO_LABEL, **{k: metrics_thermo[k] for k in ALL_METRICS}},
]
for name in MODEL_NAMES:
    final_rows.append({'Model': name, **{k: results[name][k] for k in ALL_METRICS}})
final_comparison = pd.DataFrame(final_rows)

# =============================================================================
# (A) FINAL RESULTS TABLE
# =============================================================================

print("\n" + "=" * 80)
print("(A) FINAL RESULTS TABLE — Thermodynamic (Corrected) vs ML MODELS — ALL METRICS")
print("=" * 80)
print(final_comparison.round(4).to_string(index=False))
final_comparison.round(4).to_csv('final_results_table_corrected_thermo.csv', index=False)

# =============================================================================
# (A2) COMBINED METRIC COMPARISON CHART — AI models + Thermodynamic baseline
# =============================================================================

A2_METRICS = ['Accuracy', 'Precision', 'Recall', 'F1']

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.45, hspace=0.4)

for i, metric in enumerate(A2_METRICS):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    data = final_comparison.sort_values(metric)
    colors = [bar_color(m) for m in data['Model']]
    ax.barh(data['Model'], data[metric], color=colors, edgecolor='white', linewidth=0.6, height=0.65)
    ax.set_title(f'({chr(97+i)}) {metric}', loc='left', fontweight='bold', fontsize=10)
    lo = min(0, data[metric].min() - 0.05)
    ax.set_xlim(lo, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis='y', labelsize=9)
    for idx, value in enumerate(data[metric]):
        ax.text(value + 0.015, idx, f"{value:.3f}", va='center', fontsize=7.5)

fig.suptitle('AI Models vs. Thermodynamic Baseline — Key Metrics Compared',
             y=1.0, fontsize=12, fontweight='bold')
plt.savefig('A2_ai_vs_thermo_all_metrics_corrected.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (B) COMBINED CONFUSION-MATRIX FIGURE — all 6 models
# =============================================================================

cm_thermo = confusion_matrix(y_true, y_pred_thermo)

all_panels = [(cm_thermo, THERMO_LABEL, THERMO_COLOR)] + \
             [(results[name]['cm'], name, MODEL_COLOR[name]) for name in MODEL_NAMES]

fig = plt.figure(figsize=(13, 8))
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.5)

for i, (cm, title, color) in enumerate(all_panels):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    cmap = sns.light_palette(color, as_cmap=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False, square=True,
                annot_kws={'fontsize': 10, 'fontweight': 'bold'}, linewidths=0.5, linecolor='white',
                xticklabels=['No Scale', 'Scale'], yticklabels=['No Scale', 'Scale'], ax=ax)
    ax.set_title(f'({chr(97+i)}) {title}', loc='left', fontweight='bold', fontsize=9.5)
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('Actual' if i % 3 == 0 else '', fontsize=8)
    ax.tick_params(length=0, labelsize=8)

fig.suptitle('Confusion Matrices — Thermodynamic Baseline vs. Machine-Learning Models',
             y=1.0, fontsize=12, fontweight='bold')
plt.savefig('B_confusion_matrices_combined_corrected.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (C1) RESAMPLING EFFECT — grouped bar chart, ADASYN vs No-ADASYN
#      (unchanged — ML models only, loaded not retrained)
# =============================================================================

model_order = MODEL_NAMES
adasyn_rows = []
for name in model_order:
    adasyn_rows.append({'Model': f'{name} (ADASYN)', 'Accuracy': results[name]['Accuracy'],
                         'F1': results[name]['F1'], 'ROC-AUC': results[name]['ROC-AUC']})
    adasyn_rows.append({'Model': f'{name} (No ADASYN)', 'Accuracy': no_adasyn_results[name]['Accuracy'],
                         'F1': no_adasyn_results[name]['F1'], 'ROC-AUC': no_adasyn_results[name]['ROC-AUC']})
adasyn_comparison = pd.DataFrame(adasyn_rows)

metrics_ = ['Accuracy', 'F1', 'ROC-AUC']
fig = plt.figure(figsize=(11, 9))
gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55, top=0.90)

for idx, metric in enumerate(metrics_):
    ax = fig.add_subplot(gs[idx, 0])
    x = np.arange(len(model_order))
    width = 0.36
    vals_ada = [adasyn_comparison.loc[adasyn_comparison['Model'] == f'{m} (ADASYN)', metric].values[0]
                for m in model_order]
    vals_no = [adasyn_comparison.loc[adasyn_comparison['Model'] == f'{m} (No ADASYN)', metric].values[0]
               for m in model_order]
    bars1 = ax.bar(x - width / 2, vals_ada, width, label='With ADASYN',
                    color='#2166AC', edgecolor='white', linewidth=0.6)
    bars2 = ax.bar(x + width / 2, vals_no, width, label='Without ADASYN',
                    color='#B2182B', edgecolor='white', linewidth=0.6)
    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.015, f'{h:.3f}',
                     ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=9)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_title(f'({chr(97+idx)}) {metric}', loc='left', fontweight='bold', fontsize=10)
    if idx == 0:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.18), ncol=2, fontsize=9, frameon=True)

fig.suptitle('Effect of ADASYN Resampling on ML Model Performance', y=0.995, fontsize=11, fontweight='bold')
plt.savefig('C1_adasyn_comparison_corrected.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (C2) CONFUSION-MATRIX COMPARISON — BEST model, with vs without ADASYN
#      (unchanged — ML models only)
# =============================================================================

cm_best_with = results[best_ml_name]['cm']
cm_best_without = no_adasyn_results[best_ml_name]['cm']

fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
for ax, cm, subtitle in zip(
        axes,
        [cm_best_with, cm_best_without],
        [f'{best_ml_name} — With ADASYN', f'{best_ml_name} — Without ADASYN']):
    cmap = sns.light_palette(MODEL_COLOR[best_ml_name], as_cmap=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False, square=True,
                annot_kws={'fontsize': 12, 'fontweight': 'bold'}, linewidths=0.6, linecolor='white',
                xticklabels=['No Scale', 'Scale'], yticklabels=['No Scale', 'Scale'], ax=ax)
    ax.set_title(subtitle, fontweight='bold', fontsize=10)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.tick_params(length=0)

fig.suptitle('Effect of Resampling on the Best Model\'s Confusion Matrix', y=1.03,
             fontsize=11, fontweight='bold')
plt.savefig('C2_best_model_cm_adasyn_vs_none_corrected.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (D) COMBINED ROC-AUC CURVE — all 6 models
# =============================================================================

fig = plt.figure(figsize=(7, 6.5))
ax = fig.add_subplot(1, 1, 1)

fpr_n, tpr_n, _ = roc_curve(y_true, proba_thermo)
ax.plot(fpr_n, tpr_n, color=THERMO_COLOR, linestyle=':', linewidth=1.8,
        label=f"{THERMO_LABEL} (AUC={metrics_thermo['ROC-AUC']:.3f})")

for name in MODEL_NAMES:
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_proba'])
    ax.plot(fpr, tpr, color=MODEL_COLOR[name], marker=MODEL_MARKER[name],
             markevery=0.1, markersize=5, linewidth=1.5,
             label=f"{name} (AUC={results[name]['ROC-AUC']:.3f})")

ax.plot([0, 1], [0, 1], color='0.6', linestyle='--', linewidth=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — ML Models vs. Thermodynamic Baseline', fontweight='bold', fontsize=10)
ax.legend(loc='lower right', fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
plt.savefig('D_roc_curves_all_models_corrected.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (E) PERMUTATION IMPORTANCE — 5 ML models, one combined figure (unchanged)
# =============================================================================

perm_importances = {}
print("\nComputing permutation importance for all 5 ML models (test set)...")
for name in MODEL_NAMES:
    perm_result = permutation_importance(
        fitted_pipelines[name], X_test, y_test,
        n_repeats=20, random_state=42, scoring='roc_auc', n_jobs=-1
    )
    perm_df = pd.DataFrame({
        'Feature': feature_names_display,
        'Importance': perm_result.importances_mean,
        'Std': perm_result.importances_std,
    }).sort_values('Importance', ascending=False)
    perm_importances[name] = perm_df

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.55, hspace=0.5)

for i, name in enumerate(MODEL_NAMES):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    top = perm_importances[name].sort_values('Importance', ascending=True).tail(10)
    ax.barh(top['Feature'], top['Importance'], xerr=top['Std'],
            color=MODEL_COLOR[name], edgecolor='white', linewidth=0.6,
            error_kw=dict(elinewidth=0.8, capsize=2))
    ax.set_title(f'({chr(97+i)}) {name}', loc='left', fontweight='bold', fontsize=9.5)
    ax.set_xlabel('Permutation Importance (ROC-AUC drop)', fontsize=8)
    ax.tick_params(axis='y', labelsize=7.5)
    ax.xaxis.set_major_locator(MaxNLocator(4))

if len(MODEL_NAMES) < 6:
    ax_off = fig.add_subplot(gs[1, 2])
    ax_off.axis('off')

fig.suptitle('Permutation Importance — All ML Models (Test Set)', y=1.0, fontsize=12, fontweight='bold')
plt.savefig('E_permutation_importance_all_models_corrected.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (F) SHAP ANALYSIS — BEST ML model only (unchanged)
# =============================================================================

print(f"\nRunning SHAP analysis for the best model: {best_ml_name}")

best_pipeline = fitted_pipelines[best_ml_name]
scaler_final = best_pipeline.named_steps['scaler']
clf_final = best_pipeline.named_steps['clf']

X_train_scaled = pd.DataFrame(scaler_final.transform(X_train), columns=feature_names)
X_test_scaled = pd.DataFrame(scaler_final.transform(X_test), columns=feature_names)

TREE_MODELS = ['Random Forest', 'Extra Trees', 'XGBoost', 'LightGBM']

if best_ml_name in TREE_MODELS:
    explainer = shap.TreeExplainer(clf_final)
    shap_values = explainer(X_test_scaled)
else:
    background = shap.sample(X_train_scaled, min(100, len(X_train_scaled)), random_state=42)
    explainer = shap.Explainer(clf_final.predict_proba, background)
    shap_values = explainer(X_test_scaled)

sv = shap_values
if hasattr(sv, "values") and sv.values.ndim == 3:
    sv_pos_values = sv.values[:, :, 1]
    base_val = sv.base_values[:, 1] if np.ndim(sv.base_values) > 1 else sv.base_values
    sv_pos = shap.Explanation(values=sv_pos_values, base_values=base_val,
                               data=sv.data, feature_names=feature_names_display)
else:
    sv_pos = sv
    sv_pos.feature_names = feature_names_display

X_test_scaled_display = X_test_scaled.copy()
X_test_scaled_display.columns = feature_names_display

plt.figure(figsize=(8, 7))
shap.summary_plot(sv_pos, X_test_scaled_display, show=False, plot_size=(8, 7))
plt.title(f'SHAP Summary — {best_ml_name} (Test Set, class = Scale)', fontweight='bold', fontsize=11)
plt.savefig('F_shap_summary_best_model_corrected.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(7, 6))
shap.summary_plot(sv_pos, X_test_scaled_display, plot_type='bar', show=False, plot_size=(7, 6))
plt.title(f'SHAP Feature Importance — {best_ml_name}', fontweight='bold', fontsize=11)
plt.savefig('F_shap_bar_best_model_corrected.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("ALL FIGURES REPRODUCED — ML models loaded (not retrained),")
print("thermodynamic baseline recomputed with activity-corrected SI/SR.")
print("=" * 80)
print(final_comparison.round(4).to_string(index=False))

# =============================================================================
# 2.5) BOOTSTRAP CONFIDENCE INTERVALS (95% CI via resampling)
# =============================================================================

def bootstrap_metrics(y_true_, y_pred_, y_proba_, n_bootstrap=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true_)
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'ROC-AUC': []}
    y_true_ = np.array(y_true_)
    y_pred_ = np.array(y_pred_)
    y_proba_ = np.array(y_proba_)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true_[idx], y_pred_[idx], y_proba_[idx]
        if len(np.unique(yt)) < 2:
            continue
        metrics['Accuracy'].append(accuracy_score(yt, yp))
        metrics['Precision'].append(precision_score(yt, yp, zero_division=0))
        metrics['Recall'].append(recall_score(yt, yp, zero_division=0))
        metrics['F1'].append(f1_score(yt, yp, average='weighted'))
        metrics['ROC-AUC'].append(roc_auc_score(yt, ypr))
    summary = {}
    for k, v in metrics.items():
        v = np.array(v)
        summary[k] = (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5))
    return summary

bootstrap_results = {}
for name in MODEL_NAMES:
    bootstrap_results[name] = bootstrap_metrics(
        y_test.values, results[name]['y_pred'], results[name]['y_proba']
    )
bootstrap_results[THERMO_LABEL] = bootstrap_metrics(
    y_true, y_pred_thermo, proba_thermo
)

print("\n" + "=" * 80)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS (n_bootstrap=2000)")
print("=" * 80)
for name, m in bootstrap_results.items():
    print(f"\n{name}")
    for metric, (mean, lo, hi) in m.items():
        print(f"  {metric}: {mean:.4f} ({lo:.4f}-{hi:.4f})")

ci_rows = []
for name, m in bootstrap_results.items():
    row = {'Model': name}
    for metric, (mean, lo, hi) in m.items():
        row[f'{metric}_mean'] = mean
        row[f'{metric}_CI_low'] = lo
        row[f'{metric}_CI_high'] = hi
    ci_rows.append(row)
pd.DataFrame(ci_rows).round(4).to_csv('bootstrap_ci_table_corrected.csv', index=False)
