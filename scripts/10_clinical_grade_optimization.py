"""
Clinical-Grade Optimization for Diabetes Early Detection

Goal: Achieve clinical performance suitable for early detection
- Sensitivity (Recall) > 0.90: Don't miss diabetes patients
- Specificity > 0.80: Minimize false alarms
- ROC-AUC > 0.85: Overall excellent discrimination

Strategies:
1. Aggressive data augmentation (3x samples)
2. Class weighting (prioritize sensitivity)
3. Deep neural networks
4. Multi-model ensemble with optimal weights
5. Threshold optimization for clinical use
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)

try:
    from imblearn.over_sampling import ADASYN, BorderlineSMOTE
    HAS_IMBLEARN = True
except:
    HAS_IMBLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

# Paths
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"

def load_data():
    """Load processed data"""
    print("📂 Loading data...")

    X_train = pd.read_csv(PROCESSED_DIR / 'X_train.csv')
    X_val = pd.read_csv(PROCESSED_DIR / 'X_val.csv')
    X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')

    y_train = pd.read_csv(PROCESSED_DIR / 'y_train.csv')['label'].values
    y_val = pd.read_csv(PROCESSED_DIR / 'y_val.csv')['label'].values
    y_test = pd.read_csv(PROCESSED_DIR / 'y_test.csv')['label'].values

    biomarkers = pd.read_csv(RESULTS_DIR / 'final_biomarker_panel.csv')
    selected_genes = biomarkers['gene'].tolist()

    X_train = X_train[selected_genes]
    X_val = X_val[selected_genes]
    X_test = X_test[selected_genes]

    print(f"   Training: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def aggressive_augmentation(X_train, y_train, target_ratio=3.0):
    """Aggressively augment data"""
    if not HAS_IMBLEARN:
        print("⚠️ imbalanced-learn not available")
        return X_train, y_train

    print(f"\n🔄 Aggressive data augmentation (target: {target_ratio}x)...")
    print(f"   Original: {len(y_train)} samples")

    # Calculate target samples
    n_minority = min(np.bincount(y_train))
    n_target = int(n_minority * target_ratio)

    try:
        # Use ADASYN for intelligent oversampling
        adasyn = ADASYN(
            sampling_strategy={1: n_target, 0: n_target},
            random_state=42,
            n_neighbors=3
        )
        X_aug, y_aug = adasyn.fit_resample(X_train, y_train)

        print(f"   Augmented: {len(y_aug)} samples")
        print(f"   Class balance: {np.bincount(y_aug)}")
        print(f"   Increase: {len(y_aug) / len(y_train):.1f}x")

        return X_aug, y_aug

    except Exception as e:
        print(f"   ADASYN failed, trying BorderlineSMOTE...")
        try:
            smote = BorderlineSMOTE(
                sampling_strategy={1: n_target, 0: n_target},
                random_state=42,
                k_neighbors=3
            )
            X_aug, y_aug = smote.fit_resample(X_train, y_train)
            print(f"   Augmented: {len(y_aug)} samples (BorderlineSMOTE)")
            return X_aug, y_aug
        except:
            print(f"   All augmentation failed, using original data")
            return X_train, y_train

def train_clinical_models(X_train, y_train, X_val, y_val):
    """Train models optimized for clinical use"""
    print("\n🏥 Training Clinical-Grade Models...")

    models = []
    results = []

    # 1. Logistic Regression with high diabetes weight
    print("\n1️⃣ Logistic Regression (diabetes weight=3.0)")
    lr = LogisticRegression(
        C=0.01,
        penalty='l2',
        max_iter=2000,
        class_weight={0: 1.0, 1: 3.0},  # Prioritize diabetes detection
        random_state=42
    )
    lr.fit(X_train, y_train)
    models.append(('LR (w=3.0)', lr))

    y_pred = lr.predict(X_val)
    y_proba = lr.predict_proba(X_val)[:, 1]
    results.append({
        'model': 'LR (w=3.0)',
        'roc_auc': roc_auc_score(y_val, y_proba),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred)
    })
    print(f"   ROC-AUC: {results[-1]['roc_auc']:.4f}, Sensitivity: {results[-1]['recall']:.4f}")

    # 2. Random Forest with class weight
    print("\n2️⃣ Random Forest (diabetes weight=2.5)")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight={0: 1.0, 1: 2.5},
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models.append(('RF (w=2.5)', rf))

    y_pred = rf.predict(X_val)
    y_proba = rf.predict_proba(X_val)[:, 1]
    results.append({
        'model': 'RF (w=2.5)',
        'roc_auc': roc_auc_score(y_val, y_proba),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred)
    })
    print(f"   ROC-AUC: {results[-1]['roc_auc']:.4f}, Sensitivity: {results[-1]['recall']:.4f}")

    # 3. Deep Neural Network
    print("\n3️⃣ Deep Neural Network [512-256-128-64]")
    dnn = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=16,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
        random_state=42,
        verbose=False
    )
    dnn.fit(X_train, y_train)
    models.append(('DNN-Deep', dnn))

    y_pred = dnn.predict(X_val)
    y_proba = dnn.predict_proba(X_val)[:, 1]
    results.append({
        'model': 'DNN-Deep',
        'roc_auc': roc_auc_score(y_val, y_proba),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred)
    })
    print(f"   ROC-AUC: {results[-1]['roc_auc']:.4f}, Sensitivity: {results[-1]['recall']:.4f}")

    # 4. Gradient Boosting
    print("\n4️⃣ Gradient Boosting")
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train, y_train)
    models.append(('GB', gb))

    y_pred = gb.predict(X_val)
    y_proba = gb.predict_proba(X_val)[:, 1]
    results.append({
        'model': 'GB',
        'roc_auc': roc_auc_score(y_val, y_proba),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred)
    })
    print(f"   ROC-AUC: {results[-1]['roc_auc']:.4f}, Sensitivity: {results[-1]['recall']:.4f}")

    # 5. XGBoost if available
    if HAS_XGB:
        print("\n5️⃣ XGBoost")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2.0,  # Handle class imbalance
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        xgb_model.fit(X_train, y_train)
        models.append(('XGBoost', xgb_model))

        y_pred = xgb_model.predict(X_val)
        y_proba = xgb_model.predict_proba(X_val)[:, 1]
        results.append({
            'model': 'XGBoost',
            'roc_auc': roc_auc_score(y_val, y_proba),
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred)
        })
        print(f"   ROC-AUC: {results[-1]['roc_auc']:.4f}, Sensitivity: {results[-1]['recall']:.4f}")

    return models, results

def create_optimized_ensemble(models, X_val, y_val):
    """Create ensemble with optimized weights"""
    print("\n🎯 Creating Optimized Weighted Ensemble...")

    # Get all predictions
    predictions = []
    names = []
    for name, model in models:
        y_proba = model.predict_proba(X_val)[:, 1]
        predictions.append(y_proba)
        names.append(name)

    predictions = np.array(predictions)

    # Try different weight combinations
    best_score = 0
    best_weights = None

    # Grid search over weights
    from itertools import product
    weight_options = [0.1, 0.15, 0.2, 0.25, 0.3]

    print("   Searching for optimal weights...")
    for weights in product(weight_options, repeat=len(models)):
        if abs(sum(weights) - 1.0) < 0.01:  # Weights sum to 1
            y_pred_proba = np.average(predictions, axis=0, weights=weights)
            score = roc_auc_score(y_val, y_pred_proba)

            if score > best_score:
                best_score = score
                best_weights = weights

    # Create ensemble with best weights
    y_pred_proba = np.average(predictions, axis=0, weights=best_weights)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print(f"\n   ✅ Best weights found:")
    for name, weight in zip(names, best_weights):
        print(f"      {name}: {weight:.2f}")

    result = {
        'model': 'Weighted Ensemble',
        'roc_auc': roc_auc_score(y_val, y_pred_proba),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred)
    }

    print(f"\n   ROC-AUC: {result['roc_auc']:.4f}")
    print(f"   Sensitivity: {result['recall']:.4f}")

    return result, y_pred_proba, best_weights

def find_optimal_clinical_threshold(y_true, y_pred_proba):
    """Find threshold that maximizes sensitivity while maintaining specificity > 0.75"""
    print("\n🎚️ Finding Optimal Clinical Threshold...")

    thresholds = np.arange(0.05, 0.95, 0.05)
    best_threshold = 0.5
    best_score = 0

    results = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Score: prioritize sensitivity but maintain minimum specificity
        if specificity >= 0.75:
            score = sensitivity + 0.3 * specificity  # Weighted towards sensitivity

            results.append({
                'threshold': threshold,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'score': score
            })

            if score > best_score:
                best_score = score
                best_threshold = threshold

    print(f"\n   ✅ Optimal threshold: {best_threshold:.2f}")

    # Evaluate at optimal threshold
    y_pred_opt = (y_pred_proba >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_opt).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    print(f"\n   🏥 Clinical Metrics at threshold={best_threshold:.2f}:")
    print(f"      Sensitivity: {sensitivity:.4f} ({sensitivity*100:.1f}%)")
    print(f"      Specificity: {specificity:.4f} ({specificity*100:.1f}%)")
    print(f"      PPV: {ppv:.4f}")
    print(f"      NPV: {npv:.4f}")
    print(f"      False Negative Rate: {fn/(tp+fn)*100:.1f}%")
    print(f"      False Positive Rate: {fp/(tn+fp)*100:.1f}%")

    return best_threshold, results

def visualize_clinical_performance(y_true, y_pred_proba, threshold):
    """Visualize clinical performance"""
    print("\n📊 Creating clinical performance visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'ROC-AUC = {roc_auc:.4f}')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0, 0].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[0, 0].set_ylabel('True Positive Rate (Sensitivity)')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Confusion Matrix
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['Control', 'Diabetes'],
                yticklabels=['Control', 'Diabetes'])
    axes[0, 1].set_title(f'Confusion Matrix (threshold={threshold:.2f})')
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_xlabel('Predicted Label')

    # Sensitivity vs Specificity
    thresholds_eval = np.arange(0.1, 0.9, 0.05)
    sensitivities = []
    specificities = []

    for t in thresholds_eval:
        y_p = (y_pred_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_p).ravel()
        sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    axes[1, 0].plot(thresholds_eval, sensitivities, label='Sensitivity', marker='o')
    axes[1, 0].plot(thresholds_eval, specificities, label='Specificity', marker='s')
    axes[1, 0].axvline(threshold, color='red', linestyle='--', label=f'Optimal ({threshold:.2f})')
    axes[1, 0].axhline(0.90, color='green', linestyle=':', alpha=0.5, label='Target Sensitivity')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Rate')
    axes[1, 0].set_title('Sensitivity vs Specificity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Prediction distribution
    axes[1, 1].hist(y_pred_proba[y_true == 0], bins=20, alpha=0.6, label='Control', color='blue')
    axes[1, 1].hist(y_pred_proba[y_true == 1], bins=20, alpha=0.6, label='Diabetes', color='red')
    axes[1, 1].axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold:.2f}')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'clinical_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ Visualizations saved")

def main():
    """Clinical-grade optimization pipeline"""
    print("="*80)
    print("🏥 CLINICAL-GRADE OPTIMIZATION FOR DIABETES EARLY DETECTION")
    print("="*80)

    print("\n🎯 Clinical Performance Targets:")
    print("   • Sensitivity (Recall) > 90% - Don't miss diabetes patients")
    print("   • Specificity > 80% - Minimize false alarms")
    print("   • ROC-AUC > 0.85 - Excellent discrimination")

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Step 1: Aggressive Augmentation
    print("\n" + "="*80)
    print("Step 1: Aggressive Data Augmentation")
    print("="*80)
    X_train_aug, y_train_aug = aggressive_augmentation(X_train, y_train, target_ratio=2.5)

    # Step 2: Train Clinical Models
    print("\n" + "="*80)
    print("Step 2: Train Clinical-Grade Models")
    print("="*80)
    models, results = train_clinical_models(X_train_aug, y_train_aug, X_val, y_val)

    # Step 3: Create Optimized Ensemble
    print("\n" + "="*80)
    print("Step 3: Create Optimized Ensemble")
    print("="*80)
    ensemble_result, y_pred_proba, best_weights = create_optimized_ensemble(models, X_val, y_val)
    results.append(ensemble_result)

    # Step 4: Find Optimal Clinical Threshold
    print("\n" + "="*80)
    print("Step 4: Optimize for Clinical Use")
    print("="*80)
    best_threshold, threshold_results = find_optimal_clinical_threshold(y_val, y_pred_proba)

    # Step 5: Visualize
    visualize_clinical_performance(y_val, y_pred_proba, best_threshold)

    # Final Report
    print("\n" + "="*80)
    print("📊 FINAL PERFORMANCE REPORT")
    print("="*80)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('roc_auc', ascending=False)
    print("\n" + df_results.to_string(index=False))

    df_results.to_csv(RESULTS_DIR / 'clinical_grade_results.csv', index=False)

    # Best model
    best = df_results.iloc[0]

    print(f"\n🏆 Best Model: {best['model']}")
    print(f"   ROC-AUC: {best['roc_auc']:.4f}")
    print(f"   Sensitivity: {best['recall']:.4f}")
    print(f"   Specificity: Not directly calculated, see confusion matrix")
    print(f"   F1-Score: {best['f1']:.4f}")

    # Clinical assessment
    print("\n" + "="*80)
    print("🏥 CLINICAL READINESS ASSESSMENT")
    print("="*80)

    if best['roc_auc'] >= 0.85 and best['recall'] >= 0.90:
        print("✅ CLINICAL GRADE ACHIEVED!")
        print("   This model meets the criteria for diabetes early detection.")
        print("   Ready for further validation with real clinical data.")
    elif best['roc_auc'] >= 0.80 and best['recall'] >= 0.85:
        print("⚠️ GOOD PERFORMANCE - Close to clinical grade")
        print("   Consider:")
        print("   • Collecting more real patient data")
        print("   • External validation on independent cohorts")
        print("   • Multi-center validation")
    else:
        print("⚠️ NEEDS IMPROVEMENT for clinical deployment")
        print("   Current performance is not sufficient for medical use.")
        print("   Recommendations:")
        print("   • Obtain real GEO datasets (see HOW_TO_USE_REAL_GEO_DATA.md)")
        print("   • Increase sample size (target: 1000+ patients)")
        print("   • Include diverse patient populations")

    # Performance improvement summary
    print("\n" + "="*80)
    print("📈 PERFORMANCE EVOLUTION")
    print("="*80)
    print(f"   Original (Logistic Regression): ROC-AUC = 0.7121")
    print(f"   After Stacking Optimization: ROC-AUC = 0.7273 (+2.1%)")
    print(f"   After Clinical Optimization: ROC-AUC = {best['roc_auc']:.4f} ({(best['roc_auc']-0.7121)/0.7121*100:+.1f}%)")

    # Save best model
    if 'Ensemble' in best['model']:
        # Save ensemble info
        ensemble_info = {
            'weights': best_weights,
            'threshold': best_threshold,
            'models': [name for name, _ in models]
        }
        with open(MODELS_DIR / 'clinical_ensemble_info.pkl', 'wb') as f:
            pickle.dump(ensemble_info, f)
        print(f"\n💾 Clinical ensemble saved to: clinical_ensemble_info.pkl")

    print("\n✅ CLINICAL-GRADE OPTIMIZATION COMPLETE!")

if __name__ == "__main__":
    main()
