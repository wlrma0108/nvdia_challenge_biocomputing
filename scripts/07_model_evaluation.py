"""
Step 7: Comprehensive Model Evaluation

Evaluate all trained models on the test set:
1. Load all trained models
2. Generate predictions on test set
3. Calculate metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
4. Create ROC curves
5. Generate confusion matrices
6. Feature importance analysis
7. SHAP values for interpretability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

# Try to import SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️  SHAP not available")

# Set up paths
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def load_data():
    """
    Load test data with selected biomarkers
    """
    print("📂 Loading test data...")

    X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')
    y_test = pd.read_csv(PROCESSED_DIR / 'y_test.csv')['label'].values

    # Load selected biomarkers
    biomarkers = pd.read_csv(RESULTS_DIR / 'final_biomarker_panel.csv')
    selected_genes = biomarkers['gene'].tolist()

    X_test = X_test[selected_genes]

    print(f"   Test set: {X_test.shape}")
    print(f"   Biomarkers: {len(selected_genes)}")

    return X_test, y_test, selected_genes

def load_models():
    """
    Load all trained models
    """
    print("\n📦 Loading trained models...")

    models = {}
    model_files = list(MODELS_DIR.glob("*.pkl"))

    for model_file in model_files:
        model_name = model_file.stem.replace('_', ' ').title()
        with open(model_file, 'rb') as f:
            models[model_name] = pickle.load(f)
        print(f"   ✅ {model_name}")

    return models

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

    return metrics, y_pred, y_proba

def plot_roc_curves(models, X_test, y_test):
    """
    Plot ROC curves for all models
    """
    print("\n📈 Creating ROC curves...")

    plt.figure(figsize=(10, 8))

    for model_name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Test Set', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'roc_curves_test.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ ROC curves saved")

def plot_confusion_matrices(models, X_test, y_test):
    """
    Plot confusion matrices for all models
    """
    print("\n🔲 Creating confusion matrices...")

    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (model_name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar=False, square=True,
                   xticklabels=['Control', 'Diabetes'],
                   yticklabels=['Control', 'Diabetes'])

        axes[idx].set_title(f'{model_name}\nAccuracy: {(cm[0,0]+cm[1,1])/cm.sum():.3f}',
                           fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    # Hide empty subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Confusion Matrices - Test Set', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'confusion_matrices_test.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ Confusion matrices saved")

def plot_metrics_comparison(results_df):
    """
    Plot comprehensive metrics comparison
    """
    print("\n📊 Creating metrics comparison...")

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))

    for idx, metric in enumerate(metrics):
        axes[idx].barh(results_df['model'], results_df[metric],
                      color='steelblue', edgecolor='black')
        axes[idx].set_xlabel(metric.replace('_', ' ').upper())
        axes[idx].set_xlim([0, 1])
        axes[idx].grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, val in enumerate(results_df[metric]):
            axes[idx].text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=8)

    plt.suptitle('Model Performance Metrics - Test Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'metrics_comparison_test.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ Metrics comparison saved")

def analyze_feature_importance(models, feature_names):
    """
    Analyze feature importance for tree-based models
    """
    print("\n🌳 Analyzing feature importance...")

    tree_models = ['Random Forest', 'Gradient Boosting', 'Xgboost', 'Lightgbm']
    tree_models_found = {name: model for name, model in models.items()
                        if any(tm in name for tm in tree_models)}

    if not tree_models_found:
        print("   ⚠️  No tree-based models found")
        return

    # Plot feature importance
    n_models = len(tree_models_found)
    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))

    if n_models == 1:
        axes = [axes]

    for idx, (model_name, model) in enumerate(tree_models_found.items()):
        # Get feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20

        # Plot
        axes[idx].barh(range(len(indices)),
                      importances[indices],
                      color='forestgreen', edgecolor='black')
        axes[idx].set_yticks(range(len(indices)))
        axes[idx].set_yticklabels([feature_names[i] for i in indices], fontsize=8)
        axes[idx].set_xlabel('Importance')
        axes[idx].set_title(f'{model_name}\nTop 20 Features', fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_importance_test.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ Feature importance saved")

    # Save feature importance to CSV
    for model_name, model in tree_models_found.items():
        importances_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        safe_name = model_name.lower().replace(' ', '_')
        importances_df.to_csv(RESULTS_DIR / f'feature_importance_{safe_name}.csv', index=False)

def create_classification_reports(models, X_test, y_test):
    """
    Create detailed classification reports
    """
    print("\n📝 Creating classification reports...")

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred,
                                      target_names=['Control', 'Diabetes'],
                                      digits=4)

        # Save report
        safe_name = model_name.lower().replace(' ', '_')
        with open(RESULTS_DIR / f'classification_report_{safe_name}.txt', 'w') as f:
            f.write(f"Classification Report: {model_name}\n")
            f.write("="*60 + "\n\n")
            f.write(report)

    print("   ✅ Classification reports saved")

def main():
    """
    Main evaluation pipeline
    """
    print("="*80)
    print("🔬 Step 7: Comprehensive Model Evaluation")
    print("="*80)

    # Load data
    X_test, y_test, feature_names = load_data()

    # Load models
    models = load_models()

    if not models:
        print("❌ No trained models found!")
        return

    # Evaluate all models
    print("\n" + "="*80)
    print("Evaluating Models on Test Set")
    print("="*80)

    results = []
    for model_name, model in models.items():
        print(f"\n📊 {model_name}:")
        metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test, model_name)

        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")

        results.append(metrics)

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('roc_auc', ascending=False)

    # Save results
    results_df.to_csv(RESULTS_DIR / 'test_set_evaluation.csv', index=False)

    # Visualizations
    plot_roc_curves(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    plot_metrics_comparison(results_df)
    analyze_feature_importance(models, feature_names)
    create_classification_reports(models, X_test, y_test)

    # Final summary
    best_model = results_df.iloc[0]

    print("\n" + "="*80)
    print("📊 TEST SET EVALUATION SUMMARY")
    print("="*80)
    print(f"Models evaluated: {len(models)}")
    print(f"\nBest Model: {best_model['model']}")
    print(f"  • Accuracy:  {best_model['accuracy']:.4f}")
    print(f"  • Precision: {best_model['precision']:.4f}")
    print(f"  • Recall:    {best_model['recall']:.4f}")
    print(f"  • F1-Score:  {best_model['f1']:.4f}")
    print(f"  • ROC-AUC:   {best_model['roc_auc']:.4f}")

    print(f"\n📊 Complete Results Table:")
    print(results_df.to_string(index=False))

    print("\n✅ Model evaluation complete!")
    print("\n🎯 Next step: Biological Interpretation")

if __name__ == "__main__":
    main()
