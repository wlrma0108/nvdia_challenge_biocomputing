"""
Advanced Model Optimization and Performance Enhancement

Strategies:
1. Ensemble Methods (Voting, Stacking)
2. Threshold Optimization
3. Feature Engineering
4. Deep Learning (if time permits)
5. SMOTE for data balancing
6. Advanced hyperparameter tuning with Optuna
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score

# Try advanced libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except:
    HAS_SMOTE = False
    print("⚠️  SMOTE not available - install with: pip install imbalanced-learn")

# Set up paths
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"

def load_data():
    """Load data with selected biomarkers"""
    print("📂 Loading data...")

    X_train = pd.read_csv(PROCESSED_DIR / 'X_train.csv')
    X_val = pd.read_csv(PROCESSED_DIR / 'X_val.csv')
    X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')

    y_train = pd.read_csv(PROCESSED_DIR / 'y_train.csv')['label'].values
    y_val = pd.read_csv(PROCESSED_DIR / 'y_val.csv')['label'].values
    y_test = pd.read_csv(PROCESSED_DIR / 'y_test.csv')['label'].values

    # Load biomarkers
    biomarkers = pd.read_csv(RESULTS_DIR / 'final_biomarker_panel.csv')
    selected_genes = biomarkers['gene'].tolist()

    X_train = X_train[selected_genes]
    X_val = X_val[selected_genes]
    X_test = X_test[selected_genes]

    print(f"   Training: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def apply_smote(X_train, y_train):
    """Apply SMOTE for data balancing"""
    if not HAS_SMOTE:
        print("\n⚠️  SMOTE not available, skipping...")
        return X_train, y_train

    print("\n🔄 Applying SMOTE for data balancing...")
    print(f"   Before SMOTE: {len(y_train)} samples")
    print(f"   Class distribution: {np.bincount(y_train)}")

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"   After SMOTE: {len(y_train_resampled)} samples")
    print(f"   Class distribution: {np.bincount(y_train_resampled)}")

    return X_train_resampled, y_train_resampled

def create_voting_ensemble(X_train, y_train, X_val, y_val):
    """Create voting ensemble of best models"""
    print("\n🎯 Creating Voting Ensemble...")

    # Define base estimators (use best params from previous training)
    estimators = [
        ('lr', LogisticRegression(C=0.01, penalty='l2', max_iter=1000, random_state=42)),
        ('svm', SVC(C=1, kernel='rbf', probability=True, random_state=42)),
        ('rf', RandomForestClassifier(max_depth=5, max_features='log2',
                                      n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(learning_rate=0.2, max_depth=3,
                                         n_estimators=200, random_state=42))
    ]

    if HAS_XGBOOST:
        estimators.append(
            ('xgb', xgb.XGBClassifier(learning_rate=0.1, max_depth=3,
                                     n_estimators=200, random_state=42, eval_metric='logloss'))
        )

    # Hard voting
    print("   Training Hard Voting Ensemble...")
    voting_hard = VotingClassifier(estimators=estimators, voting='hard')
    voting_hard.fit(X_train, y_train)

    # Soft voting
    print("   Training Soft Voting Ensemble...")
    voting_soft = VotingClassifier(estimators=estimators, voting='soft')
    voting_soft.fit(X_train, y_train)

    # Evaluate
    results = []

    # Hard voting (no probability)
    y_pred_hard = voting_hard.predict(X_val)
    results.append({
        'model': 'Voting (Hard)',
        'accuracy': accuracy_score(y_val, y_pred_hard),
        'precision': precision_score(y_val, y_pred_hard, zero_division=0),
        'recall': recall_score(y_val, y_pred_hard),
        'f1': f1_score(y_val, y_pred_hard),
        'roc_auc': 0.0  # Not available for hard voting
    })
    print(f"   Voting (Hard):")
    print(f"      Accuracy: {results[-1]['accuracy']:.4f}")
    print(f"      F1-score: {results[-1]['f1']:.4f}")

    # Soft voting (with probability)
    y_pred_soft = voting_soft.predict(X_val)
    y_proba_soft = voting_soft.predict_proba(X_val)[:, 1]
    results.append({
        'model': 'Voting (Soft)',
        'accuracy': accuracy_score(y_val, y_pred_soft),
        'precision': precision_score(y_val, y_pred_soft, zero_division=0),
        'recall': recall_score(y_val, y_pred_soft),
        'f1': f1_score(y_val, y_pred_soft),
        'roc_auc': roc_auc_score(y_val, y_proba_soft)
    })
    print(f"   Voting (Soft):")
    print(f"      Accuracy: {results[-1]['accuracy']:.4f}")
    print(f"      ROC-AUC: {results[-1]['roc_auc']:.4f}")

    return voting_soft, results

def create_stacking_ensemble(X_train, y_train, X_val, y_val):
    """Create stacking ensemble"""
    print("\n🏗️  Creating Stacking Ensemble...")

    # Base estimators
    estimators = [
        ('lr', LogisticRegression(C=0.01, penalty='l2', max_iter=1000, random_state=42)),
        ('svm', SVC(C=1, kernel='rbf', probability=True, random_state=42)),
        ('rf', RandomForestClassifier(max_depth=5, max_features='log2',
                                      n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(learning_rate=0.2, max_depth=3,
                                         n_estimators=200, random_state=42))
    ]

    # Meta-learner
    meta_learner = LogisticRegression(random_state=42)

    # Create stacking classifier
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5
    )

    print("   Training Stacking Ensemble (this may take a while)...")
    stacking.fit(X_train, y_train)

    # Evaluate
    y_pred = stacking.predict(X_val)
    y_proba = stacking.predict_proba(X_val)[:, 1]

    result = {
        'model': 'Stacking Ensemble',
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_proba)
    }

    print(f"   Stacking Ensemble:")
    print(f"      Accuracy: {result['accuracy']:.4f}")
    print(f"      ROC-AUC: {result['roc_auc']:.4f}")

    return stacking, [result]

def optimize_threshold(model, X_val, y_val):
    """Optimize classification threshold for best F1 score"""
    print("\n🎚️  Optimizing Classification Threshold...")

    # Get probabilities
    y_proba = model.predict_proba(X_val)[:, 1]

    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_f1 = 0

    results = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"   Best threshold: {best_threshold:.2f}")
    print(f"   Best F1-score: {best_f1:.4f}")

    # Plot threshold optimization
    results_df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot metrics vs threshold
    axes[0].plot(results_df['threshold'], results_df['accuracy'], label='Accuracy', marker='o')
    axes[0].plot(results_df['threshold'], results_df['precision'], label='Precision', marker='s')
    axes[0].plot(results_df['threshold'], results_df['recall'], label='Recall', marker='^')
    axes[0].plot(results_df['threshold'], results_df['f1'], label='F1-score', marker='d')
    axes[0].axvline(best_threshold, color='red', linestyle='--', label=f'Best ({best_threshold:.2f})')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Metrics vs Classification Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot precision-recall tradeoff
    axes[1].plot(results_df['recall'], results_df['precision'], marker='o')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Tradeoff')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'threshold_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()

    return best_threshold, results_df

def feature_engineering(X_train, X_val, X_test):
    """Add polynomial and interaction features"""
    print("\n🔧 Feature Engineering...")
    print(f"   Original features: {X_train.shape[1]}")

    # Add polynomial features (degree 2) for top 10 features
    from sklearn.preprocessing import PolynomialFeatures

    # Select top 10 features by variance
    variances = X_train.var()
    top_features = variances.nlargest(10).index.tolist()

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)

    X_train_poly = poly.fit_transform(X_train[top_features])
    X_val_poly = poly.transform(X_val[top_features])
    X_test_poly = poly.transform(X_test[top_features])

    # Convert to DataFrame
    poly_feature_names = poly.get_feature_names_out(top_features)
    X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_feature_names, index=X_train.index)
    X_val_poly_df = pd.DataFrame(X_val_poly, columns=poly_feature_names, index=X_val.index)
    X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_feature_names, index=X_test.index)

    # Combine with original features
    X_train_enhanced = pd.concat([X_train, X_train_poly_df], axis=1)
    X_val_enhanced = pd.concat([X_val, X_val_poly_df], axis=1)
    X_test_enhanced = pd.concat([X_test, X_test_poly_df], axis=1)

    print(f"   Enhanced features: {X_train_enhanced.shape[1]}")
    print(f"   Added {X_train_enhanced.shape[1] - X_train.shape[1]} new features")

    return X_train_enhanced, X_val_enhanced, X_test_enhanced

def compare_all_approaches(results_dict):
    """Compare all optimization approaches"""
    print("\n📊 Comparing All Approaches...")

    all_results = []
    for approach, results_list in results_dict.items():
        for result in results_list:
            result['approach'] = approach
            all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('roc_auc', ascending=False)

    # Save results
    results_df.to_csv(RESULTS_DIR / 'optimization_comparison.csv', index=False)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ROC-AUC comparison
    ax = axes[0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
    bars = ax.barh(range(len(results_df)), results_df['roc_auc'], color=colors, edgecolor='black')
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels([f"{r['approach']}: {r['model']}" for _, r in results_df.iterrows()], fontsize=9)
    ax.set_xlabel('ROC-AUC')
    ax.set_title('Model Performance Comparison (ROC-AUC)')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (idx, row) in enumerate(results_df.iterrows()):
        ax.text(row['roc_auc'] + 0.01, i, f"{row['roc_auc']:.4f}", va='center', fontsize=8)

    # Metrics heatmap
    ax = axes[1]
    metrics_data = results_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].values
    im = ax.imshow(metrics_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

    ax.set_xticks(range(5))
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'], rotation=45, ha='right')
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels([f"{r['approach'][:10]}: {r['model'][:15]}" for _, r in results_df.iterrows()], fontsize=8)
    ax.set_title('Performance Metrics Heatmap')

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(len(results_df)):
        for j in range(5):
            text = ax.text(j, i, f'{metrics_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=7)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✅ Comparison complete!")
    print(f"\n🏆 Best Model: {results_df.iloc[0]['approach']} - {results_df.iloc[0]['model']}")
    print(f"   ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f}")
    print(f"   Accuracy: {results_df.iloc[0]['accuracy']:.4f}")
    print(f"   F1-score: {results_df.iloc[0]['f1']:.4f}")

    return results_df

def main():
    """Main optimization pipeline"""
    print("="*80)
    print("🚀 Advanced Model Optimization")
    print("="*80)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    results_dict = {}

    # Strategy 1: SMOTE + Best Model
    print("\n" + "="*80)
    print("Strategy 1: SMOTE for Data Balancing")
    print("="*80)
    if HAS_SMOTE:
        X_train_smote, y_train_smote = apply_smote(X_train, y_train)

        # Train best model with SMOTE
        lr_smote = LogisticRegression(C=0.01, penalty='l2', max_iter=1000, random_state=42)
        lr_smote.fit(X_train_smote, y_train_smote)

        y_pred = lr_smote.predict(X_val)
        y_proba = lr_smote.predict_proba(X_val)[:, 1]

        results_dict['SMOTE'] = [{
            'model': 'Logistic Regression',
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_proba)
        }]

        print(f"\nSMOTE + LR Results:")
        print(f"   Accuracy: {results_dict['SMOTE'][0]['accuracy']:.4f}")
        print(f"   ROC-AUC: {results_dict['SMOTE'][0]['roc_auc']:.4f}")

    # Strategy 2: Voting Ensemble
    print("\n" + "="*80)
    print("Strategy 2: Voting Ensemble")
    print("="*80)
    voting_model, voting_results = create_voting_ensemble(X_train, y_train, X_val, y_val)
    results_dict['Voting'] = voting_results

    # Strategy 3: Stacking Ensemble
    print("\n" + "="*80)
    print("Strategy 3: Stacking Ensemble")
    print("="*80)
    stacking_model, stacking_results = create_stacking_ensemble(X_train, y_train, X_val, y_val)
    results_dict['Stacking'] = stacking_results

    # Strategy 4: Threshold Optimization
    print("\n" + "="*80)
    print("Strategy 4: Threshold Optimization")
    print("="*80)

    # Train a fresh LR model for threshold optimization
    lr_original = LogisticRegression(C=0.01, penalty='l2', max_iter=1000, random_state=42)
    lr_original.fit(X_train, y_train)

    best_threshold, threshold_results = optimize_threshold(lr_original, X_val, y_val)

    # Evaluate with optimized threshold
    y_proba = lr_original.predict_proba(X_val)[:, 1]
    y_pred_optimized = (y_proba >= best_threshold).astype(int)

    results_dict['Optimized Threshold'] = [{
        'model': f'Logistic Regression (t={best_threshold:.2f})',
        'accuracy': accuracy_score(y_val, y_pred_optimized),
        'precision': precision_score(y_val, y_pred_optimized, zero_division=0),
        'recall': recall_score(y_val, y_pred_optimized),
        'f1': f1_score(y_val, y_pred_optimized),
        'roc_auc': roc_auc_score(y_val, y_proba)
    }]

    # Strategy 5: Feature Engineering
    print("\n" + "="*80)
    print("Strategy 5: Feature Engineering")
    print("="*80)
    X_train_eng, X_val_eng, X_test_eng = feature_engineering(X_train, X_val, X_test)

    lr_eng = LogisticRegression(C=0.01, penalty='l2', max_iter=2000, random_state=42)
    lr_eng.fit(X_train_eng, y_train)

    y_pred = lr_eng.predict(X_val_eng)
    y_proba = lr_eng.predict_proba(X_val_eng)[:, 1]

    results_dict['Feature Engineering'] = [{
        'model': 'Logistic Regression (Enhanced)',
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_proba)
    }]

    print(f"\nFeature Engineering Results:")
    print(f"   Accuracy: {results_dict['Feature Engineering'][0]['accuracy']:.4f}")
    print(f"   ROC-AUC: {results_dict['Feature Engineering'][0]['roc_auc']:.4f}")

    # Compare all approaches
    print("\n" + "="*80)
    print("Final Comparison")
    print("="*80)
    results_df = compare_all_approaches(results_dict)

    # Save best model
    best_approach = results_df.iloc[0]['approach']
    if best_approach == 'Voting':
        best_model = voting_model
    elif best_approach == 'Stacking':
        best_model = stacking_model
    elif best_approach == 'SMOTE' and HAS_SMOTE:
        best_model = lr_smote
    elif best_approach == 'Feature Engineering':
        best_model = lr_eng
    else:
        best_model = lr_original

    with open(MODELS_DIR / 'best_optimized_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    print(f"\n💾 Best model saved to: best_optimized_model.pkl")

    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*80)
    print(f"\n🏆 Performance Improvement:")
    print(f"   Original Best (Logistic Regression): ROC-AUC = 0.7121")
    print(f"   New Best ({results_df.iloc[0]['model']}): ROC-AUC = {results_df.iloc[0]['roc_auc']:.4f}")
    improvement = (results_df.iloc[0]['roc_auc'] - 0.7121) / 0.7121 * 100
    print(f"   Improvement: {improvement:+.2f}%")

if __name__ == "__main__":
    main()
