# -*- coding: utf-8 -*-
"""
Data Description & Preprocessing Visualization Suite
=============================================================================
Standalone script (independent of the modeling pipeline) that produces the
figures recommended for the "Dataset Description" and "Data Preprocessing"
sections of the paper. Visual style (fonts, colors, grid, spines, dpi) is
matched to the main modeling script so that all figures look consistent
across the manuscript.

Sections:
  DATA DESCRIPTION
    D1) Class distribution bar chart (Scale vs No Scale)
    D2) Grouped box plots of every feature, split by class
    D3) Correlation heatmap (Pearson) of all features + TDS
    D4) Histogram + KDE distribution of every feature
    D5) Scatter matrix (pair plot) of key scale-forming ions

  PREPROCESSING
    P1) Before/after Robust Scaler comparison (selected features)
    P2) Class distribution before vs after ADASYN (side-by-side bars)
    P3) 2D PCA projection of original vs ADASYN-synthesized samples
    P4) Preprocessing pipeline flowchart (schematic, matplotlib-only)

Requires: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn
Author: pejma (visualization companion script)
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN

# ---------------------------------------------------------------------------
# GLOBAL STYLE  (matches the main modeling script exactly)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9,
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

# ---------------------------------------------------------------------------
# PALETTE (colorblind-safe, consistent with the main script)
# ---------------------------------------------------------------------------
CLASS_COLORS   = {'No Scale': '#0072B2', 'Scale': '#D55E00'}   # blue / vermillion
ADASYN_COLORS  = {'Original': '#2166AC', 'Synthetic (ADASYN)': '#B2182B'}
FEATURE_COLOR  = '#009E73'
HEATMAP_CMAP   = 'RdBu_r'

# =============================================================================
# 1) LOAD DATA  (same loading/cleaning logic as the main pipeline)
# =============================================================================

DATA_PATH = r"C:\python\scale_PT\scaledata_1.xlsx"
df = pd.read_excel(DATA_PATH)
df = (df
      .dropna()
      .drop_duplicates()
      .drop(columns=['Scale Type', 'Well No'], errors='ignore'))

TARGET_COL = 'Inspection Result'
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in data!")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

CLASS_LABELS_MAP = {0: 'No Scale', 1: 'Scale'}
y_named = y.map(CLASS_LABELS_MAP)

# ---------------------------------------------------------------------------
# Column mapping (same fuzzy-matching helper as the main script)
# ---------------------------------------------------------------------------

def find_col(possible_names, columns):
    for p in possible_names:
        for c in columns:
            if p.lower().replace(" ", "") in c.lower().replace(" ", ""):
                return c
    return None

col_T    = find_col(["T"], X.columns)
col_P    = find_col(["P"], X.columns)
col_pH   = find_col(["pH"], X.columns)
col_Ca   = find_col(["Ca2+", "Ca"], X.columns)
col_Na   = find_col(["Na+", "Na"], X.columns)
col_Mg   = find_col(["Mg2+", "Mg"], X.columns)
col_Fe   = find_col(["Fe2+", "Fe"], X.columns)
col_HCO3 = find_col(["HCO30", "HCO3"], X.columns)
col_SO4  = find_col(["SO4"], X.columns)
col_Cl   = find_col(["Cl-", "Cl"], X.columns)
col_CO3  = find_col(["CO320", "CO3"], X.columns)
col_Ba   = find_col(["Ba2+", "Ba"], X.columns)
col_Sr   = find_col(["Sr2+", "Sr"], X.columns)
col_TDS  = find_col(["TDS"], X.columns)

feature_names = X.columns.tolist()

# Same stratified 75:25 split as the main modeling pipeline. All preprocessing
# figures below (scaling, ADASYN) use ONLY X_train / y_train, since scaling
# and ADASYN are fit exclusively on the training set in the real pipeline.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5, stratify=y
)

print("Detected columns:")
for name, val in [("T", col_T), ("P", col_P), ("pH", col_pH), ("Ca", col_Ca),
                   ("Na", col_Na), ("Mg", col_Mg), ("Fe", col_Fe), ("HCO3", col_HCO3),
                   ("SO4", col_SO4), ("Cl", col_Cl), ("CO3", col_CO3), ("Ba", col_Ba),
                   ("Sr", col_Sr), ("TDS", col_TDS)]:
    print(f"  {name:6s}: {val}")

# -----------------------------------------------------------------------
# نگاشت نام نمایشی فیچرها فقط برای نمودارها (بدون تغییر نام واقعی ستون‌ها).
# T ،P و TDS همان‌طور که هستند باقی می‌مانند (نیازی به تغییر ندارند).
# بارهای یون‌ها با mathtext متلب‌پلات‌لیب به صورت superscript واقعی رسم می‌شوند.
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# نگاشت نام نمایشی فیچرها فقط برای نمودارها (بدون تغییر نام واقعی ستون‌ها).
# کلیدها دقیقاً همان نام ستون‌های اکسل هستند. T ،P و TDS تغییر نمی‌کنند.
# بارهای یون‌ها با mathtext متلب‌پلات‌لیب به صورت superscript واقعی رسم می‌شوند.
# -----------------------------------------------------------------------
FEATURE_DISPLAY_MAP = {
    'pH':                'pH',
    'Ca2+ (ppm)':         'Ca²⁺ (ppm)',
    'Na+ (ppm)':          'Na⁺ (ppm)',
    'Mg2+ (ppm)':         'Mg²⁺ (ppm)',
    'Fe2+ (ppm)':         'Fe²⁺ (ppm)',
    'HCO30 (ppm)':        'HCO₃⁻ (ppm)',
    'SO4 2-(ppm)':        'SO₄²⁻ (ppm)',
    'Cl- (ppm)':          'Cl⁻ (ppm)',
    'CO3 20 (ppm)':       'CO₃²⁻ (ppm)',
    'Ba2+ (ppm)':         'Ba²⁺ (ppm)',
    'Sr2+ (ppm)':         'Sr²⁺ (ppm)',
}

def to_display(name):
    return FEATURE_DISPLAY_MAP.get(name, name)

# =============================================================================
# (D1) CLASS DISTRIBUTION BAR CHART
# =============================================================================

counts = y_named.value_counts().reindex(['No Scale', 'Scale'])
pct = counts / counts.sum() * 100

fig, ax = plt.subplots(figsize=(5.5, 5))
bars = ax.bar(counts.index, counts.values,
              color=[CLASS_COLORS[c] for c in counts.index],
              edgecolor='white', linewidth=0.8, width=0.55)
for b, c, p in zip(bars, counts.values, pct.values):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(counts.values) * 0.015,
            f'{c}\n({p:.1f}%)', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax.set_ylabel('Number of Records')
ax.set_title('Class Distribution — Mineral Scale Dataset', fontweight='bold', fontsize=11)
ax.set_ylim(0, max(counts.values) * 1.2)
ax.yaxis.set_major_locator(MaxNLocator(6))
plt.savefig('D1_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# =============================================================================
# (D2) GROUPED BOX PLOTS — every feature, split by class
# =============================================================================

n_feat = len(feature_names)
n_cols = 4
n_rows = int(np.ceil(n_feat / n_cols))

fig = plt.figure(figsize=(4 * n_cols, 3.4 * n_rows))
gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.4, hspace=0.55)

plot_df = X.copy()
plot_df['Class'] = y_named.values

for i, feat in enumerate(feature_names):
    ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
    sns.boxplot(data=plot_df, x='Class', y=feat, order=['No Scale', 'Scale'],
                palette=CLASS_COLORS, ax=ax, width=0.5,
                fliersize=2.5, linewidth=1.1)
    ax.set_title(f'{to_display(feat)}', loc='left', fontweight='bold', fontsize=11)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=9.5)
    ax.yaxis.set_major_locator(MaxNLocator(4))

fig.suptitle('Feature Distributions by Class — Box Plots', y=1.0, fontsize=15, fontweight='bold')
plt.savefig('D2_boxplots_by_class.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (D3) CORRELATION HEATMAP (Pearson)
# =============================================================================

corr = X.corr(method='pearson')
corr_display = corr.copy()
corr_display.index = [to_display(c) for c in corr.index]
corr_display.columns = [to_display(c) for c in corr.columns]

fig, ax = plt.subplots(figsize=(11, 9.5))
mask = np.triu(np.ones_like(corr_display, dtype=bool), k=1)
sns.heatmap(corr_display, mask=mask, cmap=HEATMAP_CMAP, vmin=-1, vmax=1, center=0,
            annot=True, fmt='.2f', annot_kws={'fontsize': 8.5, 'fontweight': 'bold'},
            square=True, linewidths=0.8, linecolor='white',
            cbar_kws={'shrink': 0.8, 'label': 'Pearson correlation coefficient'}, ax=ax)
ax.set_title('Correlation Heatmap of Input Features', fontweight='bold', fontsize=13, pad=14)
ax.tick_params(axis='x', labelsize=10, rotation=45)
ax.tick_params(axis='y', labelsize=10, rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), ha='right')
ax.grid(False)
plt.savefig('D3_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (D4) HISTOGRAM + KDE DISTRIBUTION — every feature
# =============================================================================

fig = plt.figure(figsize=(4 * n_cols, 3.2 * n_rows))
gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.4, hspace=0.55)

for i, feat in enumerate(feature_names):
    ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
    sns.histplot(X[feat], kde=True, color=FEATURE_COLOR, ax=ax,
                 edgecolor='white', linewidth=0.4, alpha=0.75)
    ax.set_title(f'({chr(97 + i) if i < 26 else i}) {to_display(feat)}', loc='left', fontweight='bold', fontsize=11)
    ax.set_xlabel('')
    ax.set_ylabel('Count', fontsize=9.5)
    ax.tick_params(axis='both', labelsize=9.5)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))

fig.suptitle('Feature Distributions — Histograms with Kernel Density Estimate', y=1.0,
             fontsize=15, fontweight='bold')
plt.savefig('D4_histograms_kde.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (D5) SCATTER MATRIX (PAIR PLOT) — key scale-forming ions
# =============================================================================

key_ions = [c for c in [col_Ca, col_SO4, col_Ba, col_Sr, col_HCO3] if c is not None]

if len(key_ions) >= 2:
    pair_df = X[key_ions].copy()
    pair_df['Class'] = y_named.values
    pair_df_display = pair_df.rename(columns={c: to_display(c) for c in key_ions})

    g = sns.pairplot(pair_df_display, hue='Class', palette=CLASS_COLORS, corner=True,
                      plot_kws=dict(alpha=0.6, s=18, edgecolor='white', linewidth=0.3),
                      diag_kws=dict(alpha=0.6, linewidth=0.8))
    g.fig.suptitle('Pairwise Relationships Among Key Scale-Forming Ions', y=1.02,
                    fontsize=13, fontweight='bold')
    g.fig.set_size_inches(11, 10)
    g.savefig('D5_pairplot_key_ions.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("Not enough key-ion columns detected for pair plot; skipping D5.")

# =============================================================================
# (P1) BEFORE / AFTER ROBUST SCALER — selected features with very different scales
# =============================================================================

# Fit RobustScaler on the TRAINING set only (as in the real pipeline)
scaler_demo = RobustScaler()
X_train_scaled_demo = pd.DataFrame(scaler_demo.fit_transform(X_train), columns=feature_names)

demo_feats = [c for c in [col_Cl, col_TDS, col_pH, col_CO3] if c is not None][:4]
if len(demo_feats) < 2:
    demo_feats = feature_names[:4]

fig = plt.figure(figsize=(5 * len(demo_feats), 7.5))
gs = gridspec.GridSpec(2, len(demo_feats), figure=fig, wspace=0.45, hspace=0.5)

for i, feat in enumerate(demo_feats):
    ax_top = fig.add_subplot(gs[0, i])
    sns.histplot(X_train[feat], kde=True, color='#B2182B', ax=ax_top,
                 edgecolor='white', linewidth=0.4, alpha=0.75)
    ax_top.set_title(f'{to_display(feat)}\n(Before Scaling — Train Set)', fontweight='bold', fontsize=10)
    ax_top.set_xlabel('')
    ax_top.tick_params(axis='both', labelsize=9)
    ax_top.xaxis.set_major_locator(MaxNLocator(4))

    ax_bot = fig.add_subplot(gs[1, i])
    sns.histplot(X_train_scaled_demo[feat], kde=True, color='#2166AC', ax=ax_bot,
                 edgecolor='white', linewidth=0.4, alpha=0.75)
    ax_bot.set_title(f'{to_display(feat)}\n(After Robust Scaling — Train Set)', fontweight='bold', fontsize=10)
    ax_bot.set_xlabel('')
    ax_bot.tick_params(axis='both', labelsize=9)
    ax_bot.xaxis.set_major_locator(MaxNLocator(4))

fig.suptitle('Effect of Robust Scaling on Selected Features (Training Set Only)', y=1.0,
             fontsize=14, fontweight='bold')
plt.savefig('P1_before_after_scaling.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (P2) CLASS DISTRIBUTION BEFORE vs AFTER ADASYN
# =============================================================================

# Scale on TRAIN only, then apply ADASYN on the scaled TRAIN set only
# (the test set is never touched by scaling-fit or resampling, exactly as
# in the real pipeline).
X_train_scaled_full = pd.DataFrame(RobustScaler().fit_transform(X_train), columns=feature_names)

adasyn = ADASYN(sampling_strategy='minority', random_state=42)
X_train_res, y_train_res = adasyn.fit_resample(X_train_scaled_full, y_train)

before_counts = y_train.value_counts().reindex(sorted(y_train.unique())).rename(index=CLASS_LABELS_MAP)
after_counts = pd.Series(y_train_res).value_counts().reindex(sorted(pd.Series(y_train_res).unique())).rename(index=CLASS_LABELS_MAP)

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
for ax, counts_, subtitle in zip(axes, [before_counts, after_counts],
                                  ['Before ADASYN (Train Set)', 'After ADASYN (Train Set)']):
    bars = ax.bar(counts_.index, counts_.values,
                   color=[CLASS_COLORS.get(c, '#7F7F7F') for c in counts_.index],
                   edgecolor='white', linewidth=0.8, width=0.55)
    for b, c in zip(bars, counts_.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(after_counts.values) * 0.015,
                f'{int(c)}', ha='center', va='bottom', fontsize=10.5, fontweight='bold')
    ax.set_title(subtitle, fontweight='bold', fontsize=11.5)
    ax.set_ylim(0, max(after_counts.values) * 1.2)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis='both', labelsize=10)

axes[0].set_ylabel('Number of Records (Training Set)')
fig.suptitle('Class Distribution Before and After ADASYN Resampling\n(Applied to the Training Set Only)',
             y=1.06, fontsize=13, fontweight='bold')
plt.savefig('P2_adasyn_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (P3) 2D PCA PROJECTION — original vs ADASYN-synthesized samples
# =============================================================================

n_original = len(X_train_scaled_full)
n_total = len(X_train_res)
is_synthetic = np.zeros(n_total, dtype=bool)
is_synthetic[n_original:] = True

pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_train_res)

fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.scatter(pca_coords[~is_synthetic, 0], pca_coords[~is_synthetic, 1],
           s=18, alpha=0.6, color=ADASYN_COLORS['Original'],
           edgecolor='white', linewidth=0.2, label='Original (Train Set)')
ax.scatter(pca_coords[is_synthetic, 0], pca_coords[is_synthetic, 1],
           s=18, alpha=0.6, color=ADASYN_COLORS['Synthetic (ADASYN)'],
           edgecolor='white', linewidth=0.2, label='Synthetic (ADASYN)')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)', fontsize=11)
ax.set_title('PCA Projection — Original vs. ADASYN-Synthesized Samples\n(Training Set Only)',
             fontweight='bold', fontsize=12)
ax.tick_params(axis='both', labelsize=10)
ax.legend(loc='best', fontsize=10)
plt.savefig('P3_pca_adasyn_original_vs_synthetic.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# (P4) PREPROCESSING PIPELINE FLOWCHART (schematic)
# =============================================================================

steps_labels = [
    "Raw Dataset\n(n = 419)",
    "Remove Missing\nValues & Duplicates",
    "Stratified Train/Test\nSplit (75:25)",
    "Robust Scaler\n(fit on train only)",
    "ADASYN Resampling\n(train set only)",
    "Model Training\n& Evaluation",
]

fig, ax = plt.subplots(figsize=(14, 3.2))
ax.set_xlim(0, len(steps_labels))
ax.set_ylim(0, 1)
ax.axis('off')

box_color = '#0072B2'
box_edge = '#034673'

for i, label in enumerate(steps_labels):
    x0 = i + 0.06
    box = FancyBboxPatch((x0, 0.28), 0.88, 0.44,
                          boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=1.2, edgecolor=box_edge,
                          facecolor=box_color, alpha=0.85)
    ax.add_patch(box)
    ax.text(x0 + 0.44, 0.5, label, ha='center', va='center',
            fontsize=8.8, fontweight='bold', color='white')
    if i < len(steps_labels) - 1:
        arrow = FancyArrowPatch((x0 + 0.94, 0.5), (x0 + 1.06, 0.5),
                                 arrowstyle='-|>', mutation_scale=14,
                                 linewidth=1.3, color='0.25')
        ax.add_patch(arrow)

ax.set_title('Data Preprocessing Pipeline', fontweight='bold', fontsize=12, pad=14)
plt.savefig('P4_preprocessing_flowchart.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("ALL DATA DESCRIPTION & PREPROCESSING FIGURES GENERATED")
print("=" * 80)
print("Saved: D1_class_distribution.png, D2_boxplots_by_class.png,")
print("       D3_correlation_heatmap.png, D4_histograms_kde.png,")
print("       D5_pairplot_key_ions.png, P1_before_after_scaling.png,")
print("       P2_adasyn_class_distribution.png,")
print("       P3_pca_adasyn_original_vs_synthetic.png,")
print("       P4_preprocessing_flowchart.png")