"""
Step 6: Machine Learning Model Development

Build and compare multiple classification models:
1. Logistic Regression (baseline)
2. Random Forest
3. Support Vector Machine (SVM)
4. Gradient Boosting (XGBoost, LightGBM)
5. Neural Network (simple feedforward)

For each model:
- Hyperparameter tuning
- Cross-validation
- Training on full training set
- Evaluation on validation set
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

# Try to import advanced libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️  LightGBM not available")

# Set up paths
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def load_data_with_features():
    """
    Load data using selected biomarkers
    """
    print("📂 Loading data with selected biomarkers...")

    # Load full data
    X_train = pd.read_csv(PROCESSED_DIR / 'X_train.csv')
    X_val = pd.read_csv(PROCESSED_DIR / 'X_val.csv')
    X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')

    y_train = pd.read_csv(PROCESSED_DIR / 'y_train.csv')['label'].values
    y_val = pd.read_csv(PROCESSED_DIR / 'y_val.csv')['label'].values
    y_test = pd.read_csv(PROCESSED_DIR / 'y_test.csv')['label'].values

    # Load selected biomarkers
    biomarkers = pd.read_csv(RESULTS_DIR / 'final_biomarker_panel.csv')
    selected_genes = biomarkers['gene'].tolist()

    print(f"   Selected biomarkers: {len(selected_genes)}")

    # Select only biomarker columns
    X_train = X_train[selected_genes]
    X_val = X_val[selected_genes]
    X_test = X_test[selected_genes]

    print(f"   Training: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, selected_genes

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train Logistic Regression with hyperparameter tuning
    """
    print("\n🔹 Model 1: Logistic Regression")

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'max_iter': [1000]
    }

    lr = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

    print("   Tuning hyperparameters...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"   Best params: {grid_search.best_params_}")
    print(f"   Best CV score: {grid_search.best_score_:.4f}")

    # Evaluate
    val_pred = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]

    print(f"   Validation accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"   Validation ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

    return best_model, grid_search.best_score_

def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest with hyperparameter tuning
    """
    print("\n🔹 Model 2: Random Forest")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)

    print("   Tuning hyperparameters...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"   Best params: {grid_search.best_params_}")
    print(f"   Best CV score: {grid_search.best_score_:.4f}")

    # Evaluate
    val_pred = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]

    print(f"   Validation accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"   Validation ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

    return best_model, grid_search.best_score_

def train_svm(X_train, y_train, X_val, y_val):
    """
    Train SVM with hyperparameter tuning
    """
    print("\n🔹 Model 3: Support Vector Machine")

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

    print("   Tuning hyperparameters...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"   Best params: {grid_search.best_params_}")
    print(f"   Best CV score: {grid_search.best_score_:.4f}")

    # Evaluate
    val_pred = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]

    print(f"   Validation accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"   Validation ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

    return best_model, grid_search.best_score_

def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """
    Train Gradient Boosting
    """
    print("\n🔹 Model 4: Gradient Boosting (sklearn)")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 1.0]
    }

    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)

    print("   Tuning hyperparameters...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"   Best params: {grid_search.best_params_}")
    print(f"   Best CV score: {grid_search.best_score_:.4f}")

    # Evaluate
    val_pred = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]

    print(f"   Validation accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"   Validation ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

    return best_model, grid_search.best_score_

def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost
    """
    if not HAS_XGBOOST:
        return None, 0

    print("\n🔹 Model 5: XGBoost")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)

    print("   Tuning hyperparameters...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"   Best params: {grid_search.best_params_}")
    print(f"   Best CV score: {grid_search.best_score_:.4f}")

    # Evaluate
    val_pred = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]

    print(f"   Validation accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"   Validation ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

    return best_model, grid_search.best_score_

def train_lightgbm(X_train, y_train, X_val, y_val):
    """
    Train LightGBM
    """
    if not HAS_LIGHTGBM:
        return None, 0

    print("\n🔹 Model 6: LightGBM")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'num_leaves': [15, 31, 63],
        'min_child_samples': [10, 20, 30]
    }

    lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    grid_search = GridSearchCV(lgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)

    print("   Tuning hyperparameters...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"   Best params: {grid_search.best_params_}")
    print(f"   Best CV score: {grid_search.best_score_:.4f}")

    # Evaluate
    val_pred = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]

    print(f"   Validation accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"   Validation ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

    return best_model, grid_search.best_score_

def train_neural_network(X_train, y_train, X_val, y_val):
    """
    Train simple feedforward neural network
    """
    print("\n🔹 Model 7: Neural Network (MLP)")

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }

    mlp = MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)

    print("   Tuning hyperparameters...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"   Best params: {grid_search.best_params_}")
    print(f"   Best CV score: {grid_search.best_score_:.4f}")

    # Evaluate
    val_pred = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]

    print(f"   Validation accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"   Validation ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

    return best_model, grid_search.best_score_

def save_models(models):
    """
    Save all trained models
    """
    print("\n💾 Saving models...")

    for name, model in models.items():
        if model is not None:
            model_file = MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"   ✅ {name} saved")

def compare_models(results):
    """
    Create comparison visualization of all models
    """
    print("\n📊 Creating model comparison plots...")

    # Extract results
    model_names = list(results.keys())
    cv_scores = [results[m]['cv_score'] for m in model_names]
    val_scores = [results[m]['val_roc_auc'] for m in model_names]

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CV scores
    axes[0].barh(model_names, cv_scores, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('ROC-AUC Score')
    axes[0].set_title('Cross-Validation Scores (5-fold)')
    axes[0].set_xlim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, score in enumerate(cv_scores):
        axes[0].text(score + 0.01, i, f'{score:.3f}', va='center')

    # Validation scores
    axes[1].barh(model_names, val_scores, color='coral', edgecolor='black')
    axes[1].set_xlabel('ROC-AUC Score')
    axes[1].set_title('Validation Set Scores')
    axes[1].set_xlim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, score in enumerate(val_scores):
        axes[1].text(score + 0.01, i, f'{score:.3f}', va='center')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ Comparison plot saved")

def create_results_table(results):
    """
    Create comprehensive results table
    """
    print("\n📝 Creating results table...")

    results_data = []
    for model_name, metrics in results.items():
        results_data.append({
            'Model': model_name,
            'CV ROC-AUC': f"{metrics['cv_score']:.4f}",
            'Val Accuracy': f"{metrics['val_accuracy']:.4f}",
            'Val ROC-AUC': f"{metrics['val_roc_auc']:.4f}",
            'Val Precision': f"{metrics['val_precision']:.4f}",
            'Val Recall': f"{metrics['val_recall']:.4f}",
            'Val F1': f"{metrics['val_f1']:.4f}"
        })

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(RESULTS_DIR / 'model_comparison_table.csv', index=False)

    print("   ✅ Results table saved")
    print(f"\n{results_df.to_string(index=False)}")

    return results_df

def main():
    """
    Main ML model development pipeline
    """
    print("="*80)
    print("🔬 Step 6: Machine Learning Model Development")
    print("="*80)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, features = load_data_with_features()

    # Dictionary to store models and results
    models = {}
    results = {}

    # Train all models
    print("\n" + "="*80)
    print("Training Models with 5-fold Cross-Validation")
    print("="*80)

    # 1. Logistic Regression
    model, cv_score = train_logistic_regression(X_train, y_train, X_val, y_val)
    models['Logistic Regression'] = model
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    results['Logistic Regression'] = {
        'cv_score': cv_score,
        'val_accuracy': accuracy_score(y_val, val_pred),
        'val_precision': precision_score(y_val, val_pred, zero_division=0),
        'val_recall': recall_score(y_val, val_pred),
        'val_f1': f1_score(y_val, val_pred),
        'val_roc_auc': roc_auc_score(y_val, val_proba)
    }

    # 2. Random Forest
    model, cv_score = train_random_forest(X_train, y_train, X_val, y_val)
    models['Random Forest'] = model
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    results['Random Forest'] = {
        'cv_score': cv_score,
        'val_accuracy': accuracy_score(y_val, val_pred),
        'val_precision': precision_score(y_val, val_pred, zero_division=0),
        'val_recall': recall_score(y_val, val_pred),
        'val_f1': f1_score(y_val, val_pred),
        'val_roc_auc': roc_auc_score(y_val, val_proba)
    }

    # 3. SVM
    model, cv_score = train_svm(X_train, y_train, X_val, y_val)
    models['SVM'] = model
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    results['SVM'] = {
        'cv_score': cv_score,
        'val_accuracy': accuracy_score(y_val, val_pred),
        'val_precision': precision_score(y_val, val_pred, zero_division=0),
        'val_recall': recall_score(y_val, val_pred),
        'val_f1': f1_score(y_val, val_pred),
        'val_roc_auc': roc_auc_score(y_val, val_proba)
    }

    # 4. Gradient Boosting
    model, cv_score = train_gradient_boosting(X_train, y_train, X_val, y_val)
    models['Gradient Boosting'] = model
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    results['Gradient Boosting'] = {
        'cv_score': cv_score,
        'val_accuracy': accuracy_score(y_val, val_pred),
        'val_precision': precision_score(y_val, val_pred, zero_division=0),
        'val_recall': recall_score(y_val, val_pred),
        'val_f1': f1_score(y_val, val_pred),
        'val_roc_auc': roc_auc_score(y_val, val_proba)
    }

    # 5. XGBoost
    if HAS_XGBOOST:
        model, cv_score = train_xgboost(X_train, y_train, X_val, y_val)
        models['XGBoost'] = model
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1]
        results['XGBoost'] = {
            'cv_score': cv_score,
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred, zero_division=0),
            'val_recall': recall_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred),
            'val_roc_auc': roc_auc_score(y_val, val_proba)
        }

    # 6. LightGBM
    if HAS_LIGHTGBM:
        model, cv_score = train_lightgbm(X_train, y_train, X_val, y_val)
        models['LightGBM'] = model
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1]
        results['LightGBM'] = {
            'cv_score': cv_score,
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred, zero_division=0),
            'val_recall': recall_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred),
            'val_roc_auc': roc_auc_score(y_val, val_proba)
        }

    # 7. Neural Network
    model, cv_score = train_neural_network(X_train, y_train, X_val, y_val)
    models['Neural Network'] = model
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    results['Neural Network'] = {
        'cv_score': cv_score,
        'val_accuracy': accuracy_score(y_val, val_pred),
        'val_precision': precision_score(y_val, val_pred, zero_division=0),
        'val_recall': recall_score(y_val, val_pred),
        'val_f1': f1_score(y_val, val_pred),
        'val_roc_auc': roc_auc_score(y_val, val_proba)
    }

    # Save models
    save_models(models)

    # Compare models
    compare_models(results)

    # Create results table
    results_df = create_results_table(results)

    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['val_roc_auc'])

    print("\n" + "="*80)
    print("📊 MACHINE LEARNING SUMMARY")
    print("="*80)
    print(f"Models trained: {len(models)}")
    print(f"Best model: {best_model_name}")
    print(f"Best validation ROC-AUC: {results[best_model_name]['val_roc_auc']:.4f}")
    print(f"Best validation accuracy: {results[best_model_name]['val_accuracy']:.4f}")

    print("\n✅ Model development complete!")
    print("\n🎯 Next step: Model Evaluation and Interpretation")

if __name__ == "__main__":
    main()
