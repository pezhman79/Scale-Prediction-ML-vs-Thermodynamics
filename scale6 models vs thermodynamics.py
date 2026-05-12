# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 11:00:47 2025

@author: pejma
"""

# -*- coding: utf-8 -*-
"""
Leakage-Safe Random Forest + XGBoost + SVM Pipeline with Optuna, ADASYN-in-CV, Visualizations
Includes: Decision Boundary Visualization & ADASYN vs No-ADASYN Comparison

Created on Wed Oct 15 14:41:53 2025
Author: pejma
Enhanced: Added SVM, Decision Boundaries, ADASYN Comparison
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, f1_score,
                             roc_curve, precision_recall_curve, accuracy_score)
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import ADASYN

# -----------------------------
# Plotting style
# -----------------------------
plt.style.use('default')
sns.set_palette("husl")

# -----------------------------
# Load & preprocess data
# -----------------------------
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

classes = np.unique(y)
if len(classes) != 2:
    raise ValueError(f"Expected binary target; found {len(classes)} classes: {classes}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5, stratify=y
)

# -----------------------------
# Check & plot original class distribution
# -----------------------------
print("Original class distribution (train):")
print(y_train.value_counts(normalize=True))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
y_train.value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('Original Class Distribution (Training Set)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
y_train.value_counts(normalize=True).plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('Original Class Distribution (Normalized)')
plt.xlabel('Class')
plt.ylabel('Proportion')
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('original_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# Cross-validation splitter
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------
# Define leakage-safe Pipelines
# -----------------------------
def build_rf_pipeline(rf_params, use_adasyn=True):
    steps = [('scaler', RobustScaler())]
    if use_adasyn:
        steps.append(('sampler', ADASYN(sampling_strategy='minority', random_state=42)))
    steps.append(('rf', RandomForestClassifier(random_state=42, **rf_params)))
    return ImbPipeline(steps=steps)

def build_xgb_pipeline(xgb_params, use_adasyn=True):
    steps = [('scaler', RobustScaler())]
    if use_adasyn:
        steps.append(('sampler', ADASYN(sampling_strategy='minority', random_state=42)))
    steps.append(('xgb', XGBClassifier(random_state=42, eval_metric='logloss', **xgb_params)))
    return ImbPipeline(steps=steps)

def build_svm_pipeline(svm_params, use_adasyn=True):
    steps = [('scaler', RobustScaler())]
    if use_adasyn:
        steps.append(('sampler', ADASYN(sampling_strategy='minority', random_state=42)))
    steps.append(('svm', SVC(random_state=42, probability=True, **svm_params)))
    return ImbPipeline(steps=steps)

# -----------------------------
# Optuna objectives
# -----------------------------
def objective_rf(trial):
    rf_params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'max_samples': trial.suggest_float('max_samples', 0.5, 1.0)
    }
    model = build_rf_pipeline(rf_params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    return float(np.mean(scores))

def objective_xgb(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
    }
    model = build_xgb_pipeline(xgb_params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    return float(np.mean(scores))

def objective_svm(trial):
    svm_params = {
        'C': trial.suggest_float('C', 0.1, 100, log=True),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
    }
    if svm_params['kernel'] == 'poly':
        svm_params['degree'] = trial.suggest_int('degree', 2, 5)
    
    model = build_svm_pipeline(svm_params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    return float(np.mean(scores))

# -----------------------------
# Optimize Random Forest
# -----------------------------
print("\n" + "="*70)
print("RANDOM FOREST HYPERPARAMETER OPTIMIZATION")
print("="*70)

print("Starting Random Forest hyperparameter optimization...")
study_rf = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_rf.optimize(objective_rf, n_trials=5, show_progress_bar=True)

print("\n=== BEST RANDOM FOREST PARAMETERS ===")
print(study_rf.best_params)
print(f"Best ROC-AUC Score: {study_rf.best_value:.4f}")

# -----------------------------
# Optimize XGBoost
# -----------------------------
print("\n" + "="*70)
print("XGBOOST HYPERPARAMETER OPTIMIZATION")
print("="*70)

print("Starting XGBoost hyperparameter optimization...")
study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=5, show_progress_bar=True)

print("\n=== BEST XGBOOST PARAMETERS ===")
print(study_xgb.best_params)
print(f"Best ROC-AUC Score: {study_xgb.best_value:.4f}")

# -----------------------------
# Optimize SVM
# -----------------------------
print("\n" + "="*70)
print("SVM HYPERPARAMETER OPTIMIZATION")
print("="*70)

print("Starting SVM hyperparameter optimization...")
study_svm = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_svm.optimize(objective_svm, n_trials=5, show_progress_bar=True)

print("\n=== BEST SVM PARAMETERS ===")
print(study_svm.best_params)
print(f"Best ROC-AUC Score: {study_svm.best_value:.4f}")

# -----------------------------
# Build final models with best parameters
# -----------------------------
best_rf_params = study_rf.best_params.copy()
best_xgb_params = study_xgb.best_params.copy()
best_svm_params = study_svm.best_params.copy()

rf_pipeline = build_rf_pipeline(best_rf_params)
xgb_pipeline = build_xgb_pipeline(best_xgb_params)
svm_pipeline = build_svm_pipeline(best_svm_params)

# -----------------------------
# Enhanced evaluation
# -----------------------------
def enhanced_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    """Fit on train, evaluate on untouched test, with metrics and plots."""
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    auc_score = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"ROC-AUC Score: {auc_score:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {auc_score:.3f})', color='darkorange')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--', color='navy', label='Random')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, lw=2, color='blue')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'roc_pr_curves_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return y_pred_proba, f1_weighted, auc_score, accuracy

# -----------------------------
# Evaluate all models
# -----------------------------
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

print("\n>>> Evaluating Random Forest...")
proba_rf, f1_rf, auc_rf, acc_rf = enhanced_evaluate(
    rf_pipeline, X_train, y_train, X_test, y_test, "Random Forest"
)

print("\n>>> Evaluating XGBoost...")
proba_xgb, f1_xgb, auc_xgb, acc_xgb = enhanced_evaluate(
    xgb_pipeline, X_train, y_train, X_test, y_test, "XGBoost"
)

print("\n>>> Evaluating SVM...")
proba_svm, f1_svm, auc_svm, acc_svm = enhanced_evaluate(
    svm_pipeline, X_train, y_train, X_test, y_test, "SVM"
)

# -----------------------------
# Compare models
# -----------------------------
print("\n" + "="*70)
print("MODEL COMPARISON (WITH ADASYN)")
print("="*70)

comparison_df = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'SVM'],
    'Accuracy': [acc_rf, acc_xgb, acc_svm],
    'F1 (Weighted)': [f1_rf, f1_xgb, f1_svm],
    'ROC-AUC': [auc_rf, auc_xgb, auc_svm]
})

print(comparison_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['Accuracy', 'F1 (Weighted)', 'ROC-AUC']

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    data = comparison_df.sort_values(metric, ascending=True)
    colors = ['steelblue', 'coral', 'mediumseagreen']
    ax.barh(data['Model'], data[metric], color=colors)
    ax.set_xlabel(metric)
    ax.set_title(f'{metric} Comparison')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_all_models.png', dpi=300, bbox_inches='tight')
plt.show()

best_model_idx = comparison_df['ROC-AUC'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']

if best_model_name == 'Random Forest':
    best_model = rf_pipeline
elif best_model_name == 'XGBoost':
    best_model = xgb_pipeline
else:
    best_model = svm_pipeline

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_model_name}")
print(f"ROC-AUC: {comparison_df.loc[best_model_idx, 'ROC-AUC']:.4f}")
print(f"{'='*70}")

# -----------------------------
# ADASYN vs NO-ADASYN Comparison
# -----------------------------
print("\n" + "="*70)
print("ADASYN vs NO-ADASYN COMPARISON")
print("="*70)

# Build models without ADASYN
rf_no_adasyn = build_rf_pipeline(best_rf_params, use_adasyn=False)
xgb_no_adasyn = build_xgb_pipeline(best_xgb_params, use_adasyn=False)
svm_no_adasyn = build_svm_pipeline(best_svm_params, use_adasyn=False)

print("\n>>> Evaluating models WITHOUT ADASYN...")

# Evaluate without ADASYN
rf_no_adasyn.fit(X_train, y_train)
xgb_no_adasyn.fit(X_train, y_train)
svm_no_adasyn.fit(X_train, y_train)

# Get predictions
y_pred_rf_no = rf_no_adasyn.predict(X_test)
y_pred_xgb_no = xgb_no_adasyn.predict(X_test)
y_pred_svm_no = svm_no_adasyn.predict(X_test)

y_proba_rf_no = rf_no_adasyn.predict_proba(X_test)[:, 1]
y_proba_xgb_no = xgb_no_adasyn.predict_proba(X_test)[:, 1]
y_proba_svm_no = svm_no_adasyn.predict_proba(X_test)[:, 1]

# Calculate metrics
acc_rf_no = accuracy_score(y_test, y_pred_rf_no)
acc_xgb_no = accuracy_score(y_test, y_pred_xgb_no)
acc_svm_no = accuracy_score(y_test, y_pred_svm_no)

f1_rf_no = f1_score(y_test, y_pred_rf_no, average='weighted')
f1_xgb_no = f1_score(y_test, y_pred_xgb_no, average='weighted')
f1_svm_no = f1_score(y_test, y_pred_svm_no, average='weighted')

auc_rf_no = roc_auc_score(y_test, y_proba_rf_no)
auc_xgb_no = roc_auc_score(y_test, y_proba_xgb_no)
auc_svm_no = roc_auc_score(y_test, y_proba_svm_no)

# Create comparison dataframe
adasyn_comparison = pd.DataFrame({
    'Model': ['RF (ADASYN)', 'RF (No ADASYN)', 
              'XGB (ADASYN)', 'XGB (No ADASYN)',
              'SVM (ADASYN)', 'SVM (No ADASYN)'],
    'Accuracy': [acc_rf, acc_rf_no, acc_xgb, acc_xgb_no, acc_svm, acc_svm_no],
    'F1 Score': [f1_rf, f1_rf_no, f1_xgb, f1_xgb_no, f1_svm, f1_svm_no],
    'ROC-AUC': [auc_rf, auc_rf_no, auc_xgb, auc_xgb_no, auc_svm, auc_svm_no]
})

print("\n=== ADASYN Impact Comparison ===")
print(adasyn_comparison.to_string(index=False))

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics = ['Accuracy', 'F1 Score', 'ROC-AUC']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    bars = ax.bar(range(len(adasyn_comparison)), adasyn_comparison[metric], color=colors)
    ax.set_xticks(range(len(adasyn_comparison)))
    ax.set_xticklabels(adasyn_comparison['Model'], rotation=45, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric}: ADASYN vs No ADASYN')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('adasyn_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------
# Feature importance (if applicable)
# -----------------------------
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

feature_names = X.columns.tolist()
feature_importance_df = pd.DataFrame({'Feature': feature_names})

# Get feature importance from each model
if hasattr(rf_pipeline['rf'], 'feature_importances_'):
    feature_importance_df['RF_Importance'] = rf_pipeline['rf'].feature_importances_
else:
    feature_importance_df['RF_Importance'] = 0

if hasattr(xgb_pipeline['xgb'], 'feature_importances_'):
    feature_importance_df['XGB_Importance'] = xgb_pipeline['xgb'].feature_importances_
else:
    feature_importance_df['XGB_Importance'] = 0

# SVM doesn't have feature_importances_, use absolute coef if linear kernel
feature_importance_df['SVM_Importance'] = 0

feature_importance_df['Avg_Importance'] = (feature_importance_df['RF_Importance'] + 
                                            feature_importance_df['XGB_Importance']) / 2
feature_importance_df = feature_importance_df.sort_values('Avg_Importance', ascending=False)

print("\nTop 15 Features (by average importance):")
print(feature_importance_df.head(15).to_string(index=False))

# Plot feature importance
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

top_rf = feature_importance_df.sort_values('RF_Importance', ascending=True).tail(15)
axes[0].barh(range(len(top_rf)), top_rf['RF_Importance'], color='steelblue')
axes[0].set_yticks(range(len(top_rf)))
axes[0].set_yticklabels(top_rf['Feature'])
axes[0].set_xlabel('Feature Importance')
axes[0].set_title('Top 15 Feature Importance - Random Forest')
axes[0].grid(True, alpha=0.3)

top_xgb = feature_importance_df.sort_values('XGB_Importance', ascending=True).tail(15)
axes[1].barh(range(len(top_xgb)), top_xgb['XGB_Importance'], color='coral')
axes[1].set_yticks(range(len(top_xgb)))
axes[1].set_yticklabels(top_xgb['Feature'])
axes[1].set_xlabel('Feature Importance')
axes[1].set_title('Top 15 Feature Importance - XGBoost')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"\nBest Model: {best_model_name}")
print(f"With ADASYN - ROC-AUC: {comparison_df.loc[best_model_idx, 'ROC-AUC']:.4f}")
print("\nKey Visualizations Saved:")
print("  - original_class_distribution.png")
print("  - model_comparison_all_models.png")
print("  - adasyn_comparison.png")
print(f"  - decision_boundary_{best_model_name.lower().replace(' ', '_')}.png")
print("  - confusion matrices and ROC curves for all models")


# -----------------------------
# Visualize Confusion Matrices for all three models
# -----------------------------
def plot_confusion_matrices(models, X_train, y_train, X_test, y_test, model_names):
    plt.figure(figsize=(15, 5))
    
    for i, model in enumerate(models):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Subplot برای هر مدل
        plt.subplot(1, 3, i+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Confusion Matrix - {model_names[i]}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.grid(False)

    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# بعد از ارزیابی مدل‌ها (برای مثال بعد از enhanced_evaluate):
models = [rf_pipeline, xgb_pipeline, svm_pipeline]
model_names = ['Random Forest', 'XGBoost', 'SVM']
plot_confusion_matrices(models, X_train, y_train, X_test, y_test, model_names)



# -----------------------------
# Confusion Matrix – RF (ADASYN) vs RF (No ADASYN)
# -----------------------------
from sklearn.metrics import ConfusionMatrixDisplay

# پیش‌بینی‌های RF با ADASYN از قبل تو enhanced_evaluate انجام شد
# پس دوباره پیش‌بینی می‌گیریم تا مطمئن باشیم
y_pred_rf_ada = rf_pipeline.predict(X_test)
cm_rf_ada = confusion_matrix(y_test, y_pred_rf_ada)

y_pred_rf_no = rf_no_adasyn.predict(X_test)
cm_rf_no = confusion_matrix(y_test, y_pred_rf_no)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.heatmap(cm_rf_ada, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['Actual 0', 'Actual 1'],
            ax=axes[0])
axes[0].set_title('RF + SMOTE')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(cm_rf_no, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['Actual 0', 'Actual 1'],
            ax=axes[1])
axes[1].set_title('RF (No SMOTE)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()



# import shap
# import numpy as np
# import pandas as pd

# # فرض: svm_pipeline همون مدلیه که بالا ساختی و fit هم شده
# # svm_pipeline: RobustScaler -> (اختیاری ADASYN) -> SVC(probability=True)

# # 1) یه زیرنمونه‌ی پس‌زمینه برای SHAP بگیر (برای سرعت)
# X_background = X_train.sample(n=100, random_state=42)  # اگر دیتات کم نیست

# # 2) تابع پیش‌بینی احتمال بساز (برای کلاس 1)
# def predict_proba_svm(x):
#     # x باید DataFrame یا np.array باشه
#     return svm_pipeline.predict_proba(x)[:, 1]

# # 3) explainer
# explainer = shap.KernelExplainer(predict_proba_svm, X_background)

# # 4) داده‌ای که می‌خوای توضیح بدی (مثلا 50 تا از تست)
# X_explain = X_test.sample(n=50, random_state=42)

# # 5) محاسبه shap values
# shap_values = explainer.shap_values(X_explain, nsamples=100)

# # 6) نمودار کلی اهمیت ویژگی‌ها
# shap.summary_plot(shap_values, X_explain, feature_names=X.columns)

# # 7) اگر بخوای یه نمونه خاص رو ببینی
# idx = 0
# shap.force_plot(explainer.expected_value, shap_values[idx], X_explain.iloc[idx, :])






# =============================================================================
#  THERMODYNAMIC SCALE PREDICTION (SI / SR) + COMPARISON WITH ML MODELS
#  Add this section at the END of your current code
# =============================================================================

print("\n" + "="*80)
print("THERMODYNAMIC SCALE PREDICTION (SI / SR)")
print("="*80)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score
)

# =============================================================================
# 1) SAFE COLUMN NAME MAPPING
# =============================================================================

def find_col(possible_names, columns):
    for p in possible_names:
        for c in columns:
            if p.lower().replace(" ", "") in c.lower().replace(" ", ""):
                return c
    return None

col_T     = find_col(["T"], X.columns)
col_P     = find_col(["P"], X.columns)
col_pH    = find_col(["pH"], X.columns)

col_Ca    = find_col(["Ca"], X.columns)
col_Ba    = find_col(["Ba"], X.columns)
col_Sr    = find_col(["Sr"], X.columns)

col_HCO3  = find_col(["HCO3"], X.columns)
col_CO3   = find_col(["CO3"], X.columns)
col_SO4   = find_col(["SO4"], X.columns)

print("\nDetected Columns:")
print(f"T      : {col_T}")
print(f"P      : {col_P}")
print(f"pH     : {col_pH}")
print(f"Ca     : {col_Ca}")
print(f"Ba     : {col_Ba}")
print(f"Sr     : {col_Sr}")
print(f"HCO3   : {col_HCO3}")
print(f"CO3    : {col_CO3}")
print(f"SO4    : {col_SO4}")

# =============================================================================
# 2) THERMODYNAMIC FUNCTIONS
# =============================================================================

# ---------------------------------------------------------
# Carbonate equilibrium approximation
# ---------------------------------------------------------
def estimate_carbonate_from_ph(hco3, ph):
    """
    Approximate carbonate concentration from bicarbonate and pH
    """

    hco3 = np.maximum(hco3, 1e-8)

    # empirical approximation
    ratio = 10 ** (ph - 10.3)

    co3 = hco3 * ratio

    return np.maximum(co3, 1e-8)


# ---------------------------------------------------------
# Calcite Saturation Index (SI)
# ---------------------------------------------------------
def calcite_SI(row):

    try:

        Ca = max(row[col_Ca], 1e-8)
        HCO3 = max(row[col_HCO3], 1e-8)
        pH = row[col_pH]
        T = row[col_T]

        # ppm -> mol/L
        Ca_mol = Ca / 40078.0
        HCO3_mol = HCO3 / 61016.0

        # estimate carbonate
        CO3_mol = estimate_carbonate_from_ph(HCO3_mol, pH)

        # Ionic Activity Product
        IAP = Ca_mol * CO3_mol

        # Temperature correction
        T_K = (T - 32) * 5/9 + 273.15

        # Calcite Ksp approximation
        logKsp = -8.48 + 0.01 * ((298.15 - T_K) / 10)

        SI = np.log10(IAP) - logKsp

        return SI

    except:
        return np.nan


# ---------------------------------------------------------
# Barite Saturation Ratio
# ---------------------------------------------------------
def barite_SR(row):

    try:

        Ba = max(row[col_Ba], 1e-8)
        SO4 = max(row[col_SO4], 1e-8)

        # ppm -> mol/L
        Ba_mol = Ba / 137327.0
        SO4_mol = SO4 / 96060.0

        IAP = Ba_mol * SO4_mol

        # Approximate Ksp for barite
        Ksp_barite = 1e-10

        SR = IAP / Ksp_barite

        return SR

    except:
        return np.nan


# ---------------------------------------------------------
# Celestite Saturation Ratio
# ---------------------------------------------------------
def celestite_SR(row):

    try:

        Sr = max(row[col_Sr], 1e-8)
        SO4 = max(row[col_SO4], 1e-8)

        Sr_mol = Sr / 87620.0
        SO4_mol = SO4 / 96060.0

        IAP = Sr_mol * SO4_mol

        Ksp_celestite = np.power(10, -6.63)
        SR = IAP / Ksp_celestite

        return SR

    except:
        return np.nan


# =============================================================================
# 3) APPLY THERMODYNAMIC MODELS
# =============================================================================

thermo_df = X_test.copy()

print("\nCalculating SI and SR values...")

thermo_df["Calcite_SI"] = thermo_df.apply(calcite_SI, axis=1)
thermo_df["Barite_SR"] = thermo_df.apply(barite_SR, axis=1)
thermo_df["Celestite_SR"] = thermo_df.apply(celestite_SR, axis=1)

# =============================================================================
# 4) THERMODYNAMIC CLASSIFICATION
# =============================================================================

# ---------------------------------------------
# Calcite prediction
# ---------------------------------------------
thermo_df["Calcite_Pred"] = np.where(
    thermo_df["Calcite_SI"] > 0,
    1,
    0
)

# ---------------------------------------------
# Barite prediction
# ---------------------------------------------
thermo_df["Barite_Pred"] = np.where(
    thermo_df["Barite_SR"] > 1,
    1,
    0
)

# ---------------------------------------------
# Celestite prediction
# ---------------------------------------------
thermo_df["Celestite_Pred"] = np.where(
    thermo_df["Celestite_SR"] > 1,
    1,
    0
)

# ---------------------------------------------
# Combined thermodynamic prediction
# ---------------------------------------------
thermo_df["Thermo_Pred"] = np.where(
    (thermo_df["Calcite_Pred"] == 1) |
    (thermo_df["Barite_Pred"] == 1) |
    (thermo_df["Celestite_Pred"] == 1),
    1,
    0
)

# =============================================================================
# 5) THERMODYNAMIC EVALUATION
# =============================================================================

y_true = y_test.values
y_pred_thermo = thermo_df["Thermo_Pred"].values

print("\n" + "="*80)
print("THERMODYNAMIC MODEL RESULTS")
print("="*80)

cm_thermo = confusion_matrix(y_true, y_pred_thermo)

print("\nConfusion Matrix:")
print(cm_thermo)

plt.figure(figsize=(7,6))

sns.heatmap(
    cm_thermo,
    annot=True,
    fmt='d',
    cmap='Reds',
    xticklabels=['Predicted No Scale', 'Predicted Scale'],
    yticklabels=['Actual No Scale', 'Actual Scale']
)

plt.title('Thermodynamic Model Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred_thermo))

# =============================================================================
# 6) THERMODYNAMIC METRICS
# =============================================================================

thermo_acc = accuracy_score(y_true, y_pred_thermo)

thermo_f1_weighted = f1_score(
    y_true,
    y_pred_thermo,
    average='weighted'
)

thermo_f1_macro = f1_score(
    y_true,
    y_pred_thermo,
    average='macro'
)

# pseudo probability
thermo_probability = (
    0.4 * (thermo_df["Calcite_SI"] > 0).astype(float) +
    0.3 * np.clip(thermo_df["Barite_SR"]/10, 0, 1) +
    0.3 * np.clip(thermo_df["Celestite_SR"]/10, 0, 1)
)

try:
    thermo_auc = roc_auc_score(y_true, thermo_probability)
except:
    thermo_auc = np.nan

print(f"\nAccuracy           : {thermo_acc:.4f}")
print(f"F1 Score Weighted  : {thermo_f1_weighted:.4f}")
print(f"F1 Score Macro     : {thermo_f1_macro:.4f}")
print(f"ROC-AUC            : {thermo_auc:.4f}")

# =============================================================================
# 7) ADD THERMODYNAMIC MODEL TO COMPARISON TABLE
# =============================================================================

print("\n" + "="*80)
print("ML vs THERMODYNAMIC COMPARISON")
print("="*80)

extended_comparison = pd.DataFrame({
    'Model': [
        'Random Forest',
        'XGBoost',
        'SVM',
        'Thermodynamic SI/SR'
    ],

    'Accuracy': [
        acc_rf,
        acc_xgb,
        acc_svm,
        thermo_acc
    ],

    'F1 Weighted': [
        f1_rf,
        f1_xgb,
        f1_svm,
        thermo_f1_weighted
    ],

    'ROC-AUC': [
        auc_rf,
        auc_xgb,
        auc_svm,
        thermo_auc
    ]
})

print(extended_comparison.to_string(index=False))




# =============================================================================
# 8) VISUAL COMPARISON
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18,6))

metrics = ['Accuracy', 'F1 Weighted', 'ROC-AUC']

for i, metric in enumerate(metrics):

    ax = axes[i]

    data = extended_comparison.sort_values(metric)

    ax.barh(
        data['Model'],
        data[metric]
    )

    ax.set_title(metric)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    for idx, value in enumerate(data[metric]):
        ax.text(
            value + 0.01,
            idx,
            f"{value:.3f}",
            va='center'
        )

plt.tight_layout()
plt.show()

# =============================================================================
# 9) SI / SR DISTRIBUTIONS
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18,5))

# Calcite SI
axes[0].hist(
    thermo_df["Calcite_SI"],
    bins=30
)
axes[0].axvline(0, linestyle='--')
axes[0].set_title("Calcite SI Distribution")
axes[0].set_xlabel("SI")

# Barite SR
axes[1].hist(
    thermo_df["Barite_SR"],
    bins=30
)
axes[1].axvline(1, linestyle='--')
axes[1].set_title("Barite SR Distribution")
axes[1].set_xlabel("SR")

# Celestite SR
axes[2].hist(
    thermo_df["Celestite_SR"],
    bins=30
)
axes[2].axvline(1, linestyle='--')
axes[2].set_title("Celestite SR Distribution")
axes[2].set_xlabel("SR")

plt.tight_layout()
plt.show()

# =============================================================================
# 10) PRINT FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("FINAL THERMODYNAMIC SUMMARY")
print("="*80)

print(f"""
Thermodynamic Models Used:
--------------------------------
1) Calcite Saturation Index (SI)
2) Barite Saturation Ratio (SR)
3) Celestite Saturation Ratio (SR)

Decision Rules:
--------------------------------
Calcite:
    SI > 0  ---> Scaling likely

Barite:
    SR > 1  ---> Scaling likely

Celestite:
    SR > 1  ---> Scaling likely

Final Thermodynamic Decision:
--------------------------------
If ANY mineral scaling tendency exists:
    ==> SCALE = 1
Else:
    ==> SCALE = 0
""")

print("\nTop ML Model:")
print(best_model_name)

print("\nThermodynamic Model Metrics:")
print(f"Accuracy  : {thermo_acc:.4f}")
print(f"F1 Score  : {thermo_f1_weighted:.4f}")
print(f"ROC-AUC   : {thermo_auc:.4f}")

print("\nAnalysis Complete.")




# =============================================================================
# 11) FINAL CONFUSION MATRIX COMPARISON
#     RANDOM FOREST vs THERMODYNAMIC MODEL
# =============================================================================

print("\n" + "="*80)
print("FINAL CONFUSION MATRIX COMPARISON")
print("="*80)

from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------
# 1) Random Forest prediction (re-run to ensure consistency)
# ---------------------------------------------------------

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)

print("\nRandom Forest Classification Report")
print(classification_report(y_test, y_pred_rf))

# ---------------------------------------------------------
# 2) Thermodynamic model prediction (already computed)
# ---------------------------------------------------------

cm_thermo = confusion_matrix(y_test, y_pred_thermo)

# ---------------------------------------------------------
# 3) SIDE-BY-SIDE PLOT
# ---------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---------------- RANDOM FOREST ----------------
sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['No Scale', 'Scale'],
    yticklabels=['No Scale', 'Scale'],
    ax=axes[0]
)

axes[0].set_title('Random Forest Model')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# ---------------- THERMODYNAMIC ----------------
sns.heatmap(
    cm_thermo,
    annot=True,
    fmt='d',
    cmap='Reds',
    xticklabels=['No Scale', 'Scale'],
    yticklabels=['No Scale', 'Scale'],
    ax=axes[1]
)

axes[1].set_title('Thermodynamic Model (SI / SR)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('')

plt.tight_layout()

plt.savefig(
    "rf_vs_thermo_confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()