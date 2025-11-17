"""
Test Set Evaluation for Clinical-Grade Model

This script evaluates the best clinical model (LR with class_weight=3.0)
on the held-out test set to verify real-world performance.

Author: Claude
Date: 2025-11-17
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed
np.random.seed(42)

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR = RESULTS_DIR / 'models'
FIGURES_DIR = RESULTS_DIR / 'figures'

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

print("="*80)
print("TEST SET EVALUATION - CLINICAL-GRADE MODEL")
print("="*80)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n1. Loading data...")

# Load full datasets
X_train_full = pd.read_csv(PROCESSED_DIR / 'X_train.csv', index_col=0)
X_val_full = pd.read_csv(PROCESSED_DIR / 'X_val.csv', index_col=0)
X_test_full = pd.read_csv(PROCESSED_DIR / 'X_test.csv', index_col=0)

# Load labels (already 0/1 encoded)
y_train = pd.read_csv(PROCESSED_DIR / 'y_train.csv')['label'].values
y_val = pd.read_csv(PROCESSED_DIR / 'y_val.csv')['label'].values
y_test = pd.read_csv(PROCESSED_DIR / 'y_test.csv')['label'].values

# Load selected biomarker panel
biomarkers = pd.read_csv(RESULTS_DIR / 'final_biomarker_panel.csv')
selected_genes = biomarkers['gene'].values

# Filter to only genes available in processed data
available_genes = [g for g in selected_genes if g in X_train_full.columns]
missing_genes = [g for g in selected_genes if g not in X_train_full.columns]

print(f"Selected biomarkers: {len(selected_genes)} genes")
print(f"Available in processed data: {len(available_genes)} genes")
if missing_genes:
    print(f"Missing genes (removed during preprocessing): {missing_genes}")
print(f"Full datasets - Train: {X_train_full.shape}, Val: {X_val_full.shape}, Test: {X_test_full.shape}")

# Select only available biomarker features
X_train = X_train_full[available_genes]
X_val = X_val_full[available_genes]
X_test = X_test_full[available_genes]

print(f"\nAfter feature selection:")
print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
print(f"Test set class distribution: Control={np.sum(y_test==0)}, Diabetes={np.sum(y_test==1)}")

# ============================================================================
# 2. Train Clinical Model
# ============================================================================
print("\n2. Training clinical-grade model...")

# Best model from clinical optimization: LR with class_weight=3.0
clinical_model = LogisticRegression(
    C=0.01,
    penalty='l2',
    max_iter=1000,
    class_weight={0: 1.0, 1: 3.0},  # Prioritize diabetes detection
    random_state=42
)

clinical_model.fit(X_train, y_train)
print("✓ Model trained successfully")

# ============================================================================
# 3. Evaluate on Test Set
# ============================================================================
print("\n3. Evaluating on test set...")

# Predictions
y_test_pred = clinical_model.predict(X_test)
y_test_proba = clinical_model.predict_proba(X_test)[:, 1]

# Calculate metrics
test_metrics = {
    'roc_auc': roc_auc_score(y_test, y_test_proba),
    'accuracy': accuracy_score(y_test, y_test_pred),
    'precision': precision_score(y_test, y_test_pred),
    'recall': recall_score(y_test, y_test_pred),  # Sensitivity
    'f1': f1_score(y_test, y_test_pred)
}

# Confusion matrix
cm_test = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm_test.ravel()

# Clinical metrics
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

test_metrics['specificity'] = specificity
test_metrics['sensitivity'] = sensitivity
test_metrics['ppv'] = ppv
test_metrics['npv'] = npv

print("\nTEST SET PERFORMANCE:")
print("-" * 50)
print(f"ROC-AUC:      {test_metrics['roc_auc']:.4f}")
print(f"Accuracy:     {test_metrics['accuracy']:.4f}")
print(f"Sensitivity:  {test_metrics['sensitivity']:.4f} ({tp}/{tp+fn})")
print(f"Specificity:  {test_metrics['specificity']:.4f} ({tn}/{tn+fp})")
print(f"Precision:    {test_metrics['precision']:.4f}")
print(f"F1-Score:     {test_metrics['f1']:.4f}")
print(f"PPV:          {test_metrics['ppv']:.4f}")
print(f"NPV:          {test_metrics['npv']:.4f}")

# ============================================================================
# 4. Compare with Validation Performance
# ============================================================================
print("\n4. Comparing test vs validation performance...")

# Validation predictions
y_val_pred = clinical_model.predict(X_val)
y_val_proba = clinical_model.predict_proba(X_val)[:, 1]

# Calculate validation metrics
val_metrics = {
    'roc_auc': roc_auc_score(y_val, y_val_proba),
    'accuracy': accuracy_score(y_val, y_val_pred),
    'precision': precision_score(y_val, y_val_pred),
    'recall': recall_score(y_val, y_val_pred),
    'f1': f1_score(y_val, y_val_pred)
}

cm_val = confusion_matrix(y_val, y_val_pred)
tn_val, fp_val, fn_val, tp_val = cm_val.ravel()
val_metrics['specificity'] = tn_val / (tn_val + fp_val)
val_metrics['sensitivity'] = tp_val / (tp_val + fn_val)

# Create comparison DataFrame (transpose to have metrics as rows)
comparison = pd.DataFrame({
    'Validation': val_metrics,
    'Test': test_metrics
}).T

print("\nVALIDATION vs TEST COMPARISON:")
print(comparison[['roc_auc', 'accuracy', 'sensitivity', 'specificity', 'f1']])

# Save comparison
comparison.to_csv(RESULTS_DIR / 'test_vs_validation_performance.csv')
print(f"\n✓ Saved comparison to {RESULTS_DIR / 'test_vs_validation_performance.csv'}")

# ============================================================================
# 5. Detailed Classification Report
# ============================================================================
print("\n5. Generating detailed classification report...")

print("\nCLASSIFICATION REPORT (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['Control', 'Diabetes']))

# ============================================================================
# 6. Generate Visualizations
# ============================================================================
print("\n6. Generating visualizations...")

fig = plt.figure(figsize=(16, 10))

# 6.1 ROC Curve (Test vs Validation)
plt.subplot(2, 3, 1)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)

plt.plot(fpr_test, tpr_test, label=f'Test (AUC={test_metrics["roc_auc"]:.3f})', linewidth=2)
plt.plot(fpr_val, tpr_val, label=f'Validation (AUC={val_metrics["roc_auc"]:.3f})', linewidth=2, linestyle='--')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Test vs Validation')
plt.legend()
plt.grid(alpha=0.3)

# 6.2 Confusion Matrix (Test)
plt.subplot(2, 3, 2)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Control', 'Diabetes'],
            yticklabels=['Control', 'Diabetes'])
plt.title(f'Confusion Matrix (Test)\nSensitivity: {sensitivity:.2%}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 6.3 Confusion Matrix (Validation)
plt.subplot(2, 3, 3)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Control', 'Diabetes'],
            yticklabels=['Control', 'Diabetes'])
plt.title(f'Confusion Matrix (Validation)\nSensitivity: {val_metrics["sensitivity"]:.2%}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 6.4 Metrics Comparison Bar Plot
plt.subplot(2, 3, 4)
metrics_to_plot = ['sensitivity', 'specificity', 'precision', 'f1']
x = np.arange(len(metrics_to_plot))
width = 0.35

plt.bar(x - width/2, [val_metrics[m] for m in metrics_to_plot], width, label='Validation', alpha=0.8)
plt.bar(x + width/2, [test_metrics[m] for m in metrics_to_plot], width, label='Test', alpha=0.8)

plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('Performance Metrics: Validation vs Test')
plt.xticks(x, [m.capitalize() for m in metrics_to_plot], rotation=45)
plt.legend()
plt.ylim([0, 1])
plt.grid(axis='y', alpha=0.3)

# 6.5 Prediction Distribution (Test)
plt.subplot(2, 3, 5)
plt.hist(y_test_proba[y_test == 0], bins=20, alpha=0.6, label='Control', color='blue')
plt.hist(y_test_proba[y_test == 1], bins=20, alpha=0.6, label='Diabetes', color='red')
plt.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Prediction Distribution (Test Set)')
plt.legend()
plt.grid(alpha=0.3)

# 6.6 Misclassification Analysis
plt.subplot(2, 3, 6)
misclassified = y_test != y_test_pred
categories = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
counts = [tn, fp, fn, tp]
colors = ['green', 'orange', 'red', 'darkgreen']

plt.bar(categories, counts, color=colors, alpha=0.7)
plt.ylabel('Count')
plt.title('Test Set Classification Breakdown')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add counts on bars
for i, v in enumerate(counts):
    plt.text(i, v + 0.1, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'test_set_evaluation.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to {FIGURES_DIR / 'test_set_evaluation.png'}")
plt.close()

# ============================================================================
# 7. Save Final Model
# ============================================================================
print("\n7. Saving final clinical model...")

with open(MODELS_DIR / 'clinical_model_final.pkl', 'wb') as f:
    pickle.dump(clinical_model, f)
print(f"✓ Saved model to {MODELS_DIR / 'clinical_model_final.pkl'}")

# ============================================================================
# 8. Generate Final Summary
# ============================================================================
print("\n" + "="*80)
print("FINAL EVALUATION SUMMARY")
print("="*80)

print(f"""
Model: Logistic Regression (class_weight={{0: 1.0, 1: 3.0}})
Goal: Early diabetes detection with high sensitivity

TEST SET RESULTS:
- ROC-AUC:      {test_metrics['roc_auc']:.4f}
- Sensitivity:  {test_metrics['sensitivity']:.2%} ({'✓ ACHIEVED' if test_metrics['sensitivity'] >= 0.90 else '✗ BELOW TARGET'})
- Specificity:  {test_metrics['specificity']:.2%}
- F1-Score:     {test_metrics['f1']:.4f}
- Accuracy:     {test_metrics['accuracy']:.2%}

CLINICAL INTERPRETATION:
- {tp} out of {tp+fn} diabetes patients correctly identified
- {fn} diabetes patient(s) missed (false negatives)
- {fp} healthy individuals incorrectly flagged (false positives)

VALIDATION vs TEST:
- Sensitivity: {val_metrics['sensitivity']:.2%} (val) → {test_metrics['sensitivity']:.2%} (test)
- Specificity: {val_metrics['specificity']:.2%} (val) → {test_metrics['specificity']:.2%} (test)
- ROC-AUC:     {val_metrics['roc_auc']:.4f} (val) → {test_metrics['roc_auc']:.4f} (test)

RECOMMENDATION:
{'✓ Model suitable for 1st-stage screening in clinical settings' if test_metrics['sensitivity'] >= 0.85 else '✗ Model requires further optimization before clinical deployment'}
{'  Positive cases should be confirmed with 2nd-stage confirmatory tests' if test_metrics['sensitivity'] >= 0.85 else ''}
""")

print("="*80)
print("Evaluation complete!")
print("="*80)
