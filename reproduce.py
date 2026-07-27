# -*- coding: utf-8 -*-
"""
Reproduce Plots From Saved Models
=============================================================================
Companion script to `scale_prediction_results_pipeline.py`.

Run the main pipeline ONCE (it saves every trained model + the data needed
to rebuild all figures into ./saved_models/). After that, run THIS script
any time you want to regenerate the exact same tables/figures for the paper
— no Optuna tuning, no retraining, just loading + plotting.

Reproduces, in the same order as the main script:
  (A)  Final results table  (Thermo-Naive + 5 ML models, all metrics)
  (A2) AI models vs Thermodynamic baseline — combined metric comparison chart
  (B)  Combined confusion-matrix figure (all 6 models)
  (C1) ADASYN vs No-ADASYN grouped bar chart (5 ML models)
  (C2) Confusion-matrix comparison: best model, with vs without ADASYN
  (D)  Combined ROC-AUC curve (all 6 models)
  (E)  Permutation importance — 5 ML models, one combined figure
  (F)  SHAP analysis — best ML model only

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
THERMO_COLOR_NAIVE = '#B2182B'

def bar_color(model_name):
    if model_name == 'Thermodynamic (Naive)':
        return THERMO_COLOR_NAIVE
    return MODEL_COLOR.get(model_name, '#7F7F7F')

# =============================================================================
# 1) LOAD EVERYTHING
# =============================================================================

SAVE_DIR = 'saved_models'

bundle = joblib.load(os.path.join(SAVE_DIR, 'results_bundle.joblib'))
X_train              = bundle['X_train']
X_test               = bundle['X_test']
y_train              = bundle['y_train']
y_test               = bundle['y_test']
feature_names        = bundle['feature_names']
best_ml_name         = bundle['best_ml_name']
y_pred_thermo_naive  = bundle['y_pred_thermo_naive']
proba_thermo_naive   = bundle['proba_thermo_naive']
metrics_thermo_naive = bundle['metrics_thermo_naive']

# -----------------------------------------------------------------------
# نگاشت نام نمایشی فیچرها فقط برای نمودارها (بدون تغییر در نام واقعی ستون‌ها
# که مدل‌ها با آن‌ها fit شده‌اند — این mapping هیچ تاثیری روی محاسبات ندارد)
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# نگاشت نام نمایشی فیچرها فقط برای نمودارها (بدون تغییر در نام واقعی ستون‌ها
# که مدل‌ها با آن‌ها fit شده‌اند — این mapping هیچ تاثیری روی محاسبات ندارد)
# بارهای یون‌ها با mathtext متلب‌پلات‌لیب به صورت superscript واقعی رسم می‌شوند
# -----------------------------------------------------------------------
FEATURE_DISPLAY_MAP = {
    'T':      'T (F)',
    'P':      'P (psia)',
    'pH':     'pH',
    'Ca2+ (ppm)':         'Ca²⁺ (ppm)',
    'Na+ (ppm)':          'Na⁺ (ppm)',
    'Mg2+ (ppm)':         'Mg²⁺ (ppm)',
    'Fe2+ (ppm)':         'Fe²⁺ (ppm)',
    'HCO30 (ppm)':        'HCO₃⁻ (ppm)',
    'SO4 2-(ppm)':        'SO₄²⁻ (ppm)',
    'Cl- (ppm)':          'Cl⁻ (ppm)',
    'CO3 20 (ppm)':       'CO₃²⁻ (ppm)',
    'Ba2+ (ppm)':         'Ba²⁺ (ppm)',
    'Sr2+':         'Sr²⁺ (ppm)',
}

def to_display(names):
    return [FEATURE_DISPLAY_MAP.get(n, n) for n in names]

feature_names_display = to_display(feature_names)



def to_display(names):
    return [FEATURE_DISPLAY_MAP.get(n, n) for n in names]

feature_names_display = to_display(feature_names)

y_true = y_test.values

fitted_pipelines   = {}
no_adasyn_pipelines = {}
for name in MODEL_NAMES:
    fitted_pipelines[name] = joblib.load(
        os.path.join(SAVE_DIR, f"{name.replace(' ', '_')}_pipeline.joblib"))
    no_adasyn_pipelines[name] = joblib.load(
        os.path.join(SAVE_DIR, f"{name.replace(' ', '_')}_no_adasyn_pipeline.joblib"))

print(f"Loaded {len(MODEL_NAMES)} ML models (with & without ADASYN) from ./{SAVE_DIR}/")
print(f"Best ML model (from saved bundle): {best_ml_name}")

# =============================================================================
# 2) RECOMPUTE PREDICTIONS/METRICS FROM THE LOADED MODELS (deterministic —
#    same test set, same fitted models -> identical numbers as the original run)
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

ALL_METRICS = ['Accuracy', 'Precision', 'Recall']

final_rows = [
    {'Model': 'Thermodynamic (Naive)', **{k: metrics_thermo_naive[k] for k in ALL_METRICS}},
]
for name in MODEL_NAMES:
    final_rows.append({'Model': name, **{k: results[name][k] for k in ALL_METRICS}})
final_comparison = pd.DataFrame(final_rows)

# =============================================================================
# (A) FINAL RESULTS TABLE
# =============================================================================

print("\n" + "=" * 80)
print("(A) FINAL RESULTS TABLE — Thermodynamic (Naive) vs ML MODELS — ALL METRICS")
print("=" * 80)
print(final_comparison.round(4).to_string(index=False))
final_comparison.round(4).to_csv('final_results_table_reproduced.csv', index=False)

# =============================================================================
# (A2) COMBINED METRIC COMPARISON CHART — AI models + Thermodynamic baseline
# =============================================================================

fig = plt.figure(figsize=(13, 9))
gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.5, hspace=0.55)

for i, metric in enumerate(ALL_METRICS):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    data = final_comparison.sort_values(metric)
    colors = [bar_color(m) for m in data['Model']]
    ax.barh(data['Model'], data[metric], color=colors, edgecolor='white', linewidth=0.6, height=0.65)
    ax.set_title(f'({chr(97+i)}) {metric}', loc='left', fontweight='bold', fontsize=9.5)
    lo = min(0, data[metric].min() - 0.05)
    ax.set_xlim(lo, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis='y', labelsize=8)
    for idx, value in enumerate(data[metric]):
        ax.text(value + 0.015, idx, f"{value:.3f}", va='center', fontsize=6.5)

fig.suptitle('AI Models vs. Thermodynamic Baseline — All Metrics Compared',
             y=1.0, fontsize=12, fontweight='bold')
plt.savefig('A2_ai_vs_thermo_all_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (B) COMBINED CONFUSION-MATRIX FIGURE — all 6 models
# =============================================================================

cm_thermo_naive = confusion_matrix(y_true, y_pred_thermo_naive)

all_panels = [(cm_thermo_naive, 'Thermodynamic (Naive)', THERMO_COLOR_NAIVE)] + \
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
plt.savefig('B_confusion_matrices_combined_reproduced.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (C1) RESAMPLING EFFECT — grouped bar chart, ADASYN vs No-ADASYN
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
fig = plt.figure(figsize=(11, 8.5))
gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

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
        ax.legend(loc='upper right', ncol=2)

fig.suptitle('Effect of ADASYN Resampling on ML Model Performance', y=0.995, fontsize=11, fontweight='bold')
plt.savefig('C1_adasyn_comparison_reproduced.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (C2) CONFUSION-MATRIX COMPARISON — BEST model, with vs without ADASYN
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
plt.savefig('C2_best_model_cm_adasyn_vs_none_reproduced.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (D) COMBINED ROC-AUC CURVE — all 6 models
# =============================================================================

fig = plt.figure(figsize=(7, 6.5))
ax = fig.add_subplot(1, 1, 1)

fpr_n, tpr_n, _ = roc_curve(y_true, proba_thermo_naive)
ax.plot(fpr_n, tpr_n, color=THERMO_COLOR_NAIVE, linestyle=':', linewidth=1.8,
        label=f"Thermo–Naive (AUC={metrics_thermo_naive['ROC-AUC']:.3f})")

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
plt.savefig('D_roc_curves_all_models_reproduced.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (E) PERMUTATION IMPORTANCE — 5 ML models, one combined figure
# =============================================================================

perm_importances = {}
print("\nComputing permutation importance for all 5 ML models (test set)...")
for name in MODEL_NAMES:
    perm_result = permutation_importance(
        fitted_pipelines[name], X_test, y_test,
        n_repeats=20, random_state=42, scoring='roc_auc', n_jobs=-1
    )
    perm_df = pd.DataFrame({
        'Feature': feature_names_display,   # نام‌های نمایشی فقط برای نمودار
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
plt.savefig('E_permutation_importance_all_models_reproduced.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (F) SHAP ANALYSIS — BEST ML model only
# =============================================================================

print(f"\nRunning SHAP analysis for the best model: {best_ml_name}")

best_pipeline = fitted_pipelines[best_ml_name]
scaler_final = best_pipeline.named_steps['scaler']
clf_final = best_pipeline.named_steps['clf']

# مقادیر عددی همیشه با نام واقعی ستون‌ها (feature_names) که مدل با آن fit شده
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
    # از همین‌جا نام‌های نمایشی را به Explanation می‌دهیم تا نمودارها برچسب درست را نشان دهند
    sv_pos = shap.Explanation(values=sv_pos_values, base_values=base_val,
                               data=sv.data, feature_names=feature_names_display)
else:
    sv_pos = sv
    sv_pos.feature_names = feature_names_display

# نسخه‌ی نمایشی X_test_scaled فقط برای رسم (محاسبات SHAP قبلاً با نام اصلی انجام شده)
X_test_scaled_display = X_test_scaled.copy()
X_test_scaled_display.columns = feature_names_display

plt.figure(figsize=(8, 7))
shap.summary_plot(sv_pos, X_test_scaled_display, show=False, plot_size=(8, 7))
plt.title(f'SHAP Summary — {best_ml_name} (Test Set, class = Scale)', fontweight='bold', fontsize=11)
plt.savefig('F_shap_summary_best_model_reproduced.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(7, 6))
shap.summary_plot(sv_pos, X_test_scaled_display, plot_type='bar', show=False, plot_size=(7, 6))
plt.title(f'SHAP Feature Importance — {best_ml_name}', fontweight='bold', fontsize=11)
plt.savefig('F_shap_bar_best_model_reproduced.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("ALL FIGURES REPRODUCED FROM SAVED MODELS")
print("=" * 80)
print(final_comparison.round(4).to_string(index=False))


# =============================================================================
# 2.5) BOOTSTRAP CONFIDENCE INTERVALS (95% CI via resampling)
# =============================================================================

def bootstrap_metrics(y_true, y_pred, y_proba, n_bootstrap=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'ROC-AUC': []}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
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
bootstrap_results['Thermodynamic (Naive)'] = bootstrap_metrics(
    y_true, y_pred_thermo_naive, proba_thermo_naive
)

print("\n" + "=" * 80)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS (n_bootstrap=2000)")
print("=" * 80)
for name, m in bootstrap_results.items():
    print(f"\n{name}")
    for metric, (mean, lo, hi) in m.items():
        print(f"  {metric}: {mean:.4f} ({lo:.4f}–{hi:.4f})")

ci_rows = []
for name, m in bootstrap_results.items():
    row = {'Model': name}
    for metric, (mean, lo, hi) in m.items():
        row[f'{metric}_mean'] = mean
        row[f'{metric}_CI_low'] = lo
        row[f'{metric}_CI_high'] = hi
    ci_rows.append(row)
pd.DataFrame(ci_rows).round(4).to_csv('bootstrap_ci_table.csv', index=False)