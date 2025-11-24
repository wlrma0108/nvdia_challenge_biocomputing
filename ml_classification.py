import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gzip
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            roc_auc_score, roc_curve, precision_recall_curve, 
                            f1_score, matthews_corrcoef, cohen_kappa_score)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              VotingClassifier, StackingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Advanced ML
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class AdvancedMLClassifier:
    def __init__(self, expression_data, metadata, output_dir='./ml_advanced_results'):
        self.expr = expression_data
        self.metadata = metadata
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.best_features = None
        
        print("="*80)
        print("🤖 ADVANCED ML CLASSIFICATION PIPELINE")
        print("="*80)
        print(f"\n📊 Data:")
        print(f"   Expression: {self.expr.shape[0]} genes × {self.expr.shape[1]} samples")
        print(f"   Metadata: {self.metadata.shape[0]} samples")
    
    def prepare_data(self, group_column='diabetes_label', exclude_classes=['T3cD'], 
                    n_genes=5000, test_size=0.2):
        print(f"\n🔧 DATA PREPARATION")
        print("-"*80)
        
        if 'title' in self.metadata.columns:
            import re
            def extract_sample_id(title):
                match = re.search(r'(DP\d+)', str(title))
                return match.group(1) if match else None
            
            self.metadata['sample_id'] = self.metadata['title'].apply(extract_sample_id)
            valid_mask = self.metadata['sample_id'].notna()
            self.metadata = self.metadata[valid_mask]
            self.metadata = self.metadata.set_index('sample_id', drop=False)
        
        common_samples = list(set(self.expr.columns) & set(self.metadata.index))
        self.expr = self.expr[common_samples]
        self.metadata = self.metadata.loc[common_samples]
        
        if exclude_classes:
            mask = ~self.metadata[group_column].isin(exclude_classes)
            self.metadata = self.metadata[mask]
            filtered_samples = self.metadata.index.tolist()
            self.expr = self.expr[filtered_samples]
        
        print(f"✅ Filtered samples: {len(filtered_samples)}")
        print(self.metadata[group_column].value_counts())
        
        # Variance-based feature selection
        gene_vars = self.expr.var(axis=1).sort_values(ascending=False)
        top_genes = gene_vars.head(n_genes).index
        self.expr = self.expr.loc[top_genes]
        
        print(f"\n🧬 Selected top {len(top_genes)} variable genes")
        
        # Transpose and clean
        self.expr = self.expr.T
        self.expr = self.expr.fillna(self.expr.mean())
        self.expr = self.expr.replace([np.inf, -np.inf], 0)
        
        # Prepare X, y
        self.X = self.expr.values
        self.y = self.metadata[group_column].values
        
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_encoded,
            test_size=test_size,
            random_state=42,
            stratify=self.y_encoded
        )
        
        # Scaling
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\n✅ Data splits:")
        print(f"   Train: {len(self.X_train)} ({len(self.X_train)/len(self.X)*100:.1f}%)")
        print(f"   Test: {len(self.X_test)} ({len(self.X_test)/len(self.X)*100:.1f}%)")
        print(f"   Features: {self.X.shape[1]}")
        print(f"   Classes: {self.label_encoder.classes_}")
    
    def advanced_feature_selection(self, methods=['variance', 'mutual_info', 'f_test'], 
                                   n_features=1000):
        print(f"\n🎯 ADVANCED FEATURE SELECTION")
        print("-"*80)
        
        feature_scores = pd.DataFrame(index=self.expr.columns)
        
        # Method 1: Variance (already done, but re-calculate for ranking)
        if 'variance' in methods:
            variances = self.expr.var(axis=0)
            feature_scores['variance'] = variances
            print(f"   ✓ Variance-based scoring")
        
        # Method 2: Mutual Information
        if 'mutual_info' in methods:
            mi_scores = mutual_info_classif(self.X_train_scaled, self.y_train, random_state=42)
            feature_scores['mutual_info'] = mi_scores
            print(f"   ✓ Mutual Information scoring")
        
        # Method 3: F-test
        if 'f_test' in methods:
            f_scores, _ = f_classif(self.X_train_scaled, self.y_train)
            feature_scores['f_test'] = f_scores
            print(f"   ✓ F-test scoring")
        
        # Method 4: Random Forest importance
        if 'rf_importance' in methods:
            rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_temp.fit(self.X_train_scaled, self.y_train)
            feature_scores['rf_importance'] = rf_temp.feature_importances_
            print(f"   ✓ Random Forest importance")
        
        # Normalize and combine
        for col in feature_scores.columns:
            feature_scores[col] = (feature_scores[col] - feature_scores[col].min()) / \
                                 (feature_scores[col].max() - feature_scores[col].min())
        
        feature_scores['combined_score'] = feature_scores.mean(axis=1)
        feature_scores = feature_scores.sort_values('combined_score', ascending=False)
        
        self.best_features = feature_scores.head(n_features).index.tolist()
        
        print(f"\n✅ Selected top {n_features} features")
        print(f"   Combined scoring from {len(methods)} methods")
        
        feature_scores.to_csv(f'{self.output_dir}/feature_selection_scores.csv')
        
        # Update data with selected features
        feature_indices = [self.expr.columns.get_loc(f) for f in self.best_features]
        self.X_train_selected = self.X_train_scaled[:, feature_indices]
        self.X_test_selected = self.X_test_scaled[:, feature_indices]
        
        return feature_scores
    
    def train_baseline_models(self):
        print(f"\n🤖 TRAINING BASELINE MODELS")
        print("="*80)
        
        baseline_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        
        for name, model in baseline_models.items():
            print(f"\n📊 Training {name}...")
            model.fit(self.X_train_selected, self.y_train)
            
            train_score = model.score(self.X_train_selected, self.y_train)
            test_score = model.score(self.X_test_selected, self.y_test)
            
            print(f"   Train Accuracy: {train_score:.3f}")
            print(f"   Test Accuracy: {test_score:.3f}")
            
            self.models[name] = model
            self.results[name] = {
                'model': model,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'predictions': model.predict(self.X_test_selected)
            }
    
    def train_ensemble_models(self):
        print(f"\n🚀 TRAINING ADVANCED ENSEMBLE MODELS")
        print("="*80)
        
        # Random Forest with tuning
        print(f"\n📊 Random Forest (Hyperparameter Tuning)...")
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf_grid = GridSearchCV(rf_base, rf_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
        rf_grid.fit(self.X_train_selected, self.y_train)
        
        print(f"   Best params: {rf_grid.best_params_}")
        print(f"   Train Accuracy: {rf_grid.score(self.X_train_selected, self.y_train):.3f}")
        print(f"   Test Accuracy: {rf_grid.score(self.X_test_selected, self.y_test):.3f}")
        
        self.models['Random Forest (Tuned)'] = rf_grid.best_estimator_
        self.results['Random Forest (Tuned)'] = {
            'model': rf_grid.best_estimator_,
            'train_accuracy': rf_grid.score(self.X_train_selected, self.y_train),
            'test_accuracy': rf_grid.score(self.X_test_selected, self.y_test),
            'predictions': rf_grid.predict(self.X_test_selected),
            'best_params': rf_grid.best_params_
        }
        
        # XGBoost
        print(f"\n📊 XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        xgb_model.fit(self.X_train_selected, self.y_train)
        
        train_score = xgb_model.score(self.X_train_selected, self.y_train)
        test_score = xgb_model.score(self.X_test_selected, self.y_test)
        
        print(f"   Train Accuracy: {train_score:.3f}")
        print(f"   Test Accuracy: {test_score:.3f}")
        
        self.models['XGBoost'] = xgb_model
        self.results['XGBoost'] = {
            'model': xgb_model,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'predictions': xgb_model.predict(self.X_test_selected)
        }
        
        # LightGBM
        print(f"\n📊 LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(self.X_train_selected, self.y_train)
        
        train_score = lgb_model.score(self.X_train_selected, self.y_train)
        test_score = lgb_model.score(self.X_test_selected, self.y_test)
        
        print(f"   Train Accuracy: {train_score:.3f}")
        print(f"   Test Accuracy: {test_score:.3f}")
        
        self.models['LightGBM'] = lgb_model
        self.results['LightGBM'] = {
            'model': lgb_model,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'predictions': lgb_model.predict(self.X_test_selected)
        }
        
        # Extra Trees
        print(f"\n📊 Extra Trees...")
        et_model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        et_model.fit(self.X_train_selected, self.y_train)
        
        train_score = et_model.score(self.X_train_selected, self.y_train)
        test_score = et_model.score(self.X_test_selected, self.y_test)
        
        print(f"   Train Accuracy: {train_score:.3f}")
        print(f"   Test Accuracy: {test_score:.3f}")
        
        self.models['Extra Trees'] = et_model
        self.results['Extra Trees'] = {
            'model': et_model,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'predictions': et_model.predict(self.X_test_selected)
        }
    
    def create_voting_ensemble(self):
        print(f"\n🗳️ CREATING VOTING ENSEMBLE")
        print("-"*80)
        
        estimators = [
            ('rf', self.models['Random Forest (Tuned)']),
            ('xgb', self.models['XGBoost']),
            ('lgb', self.models['LightGBM']),
            ('et', self.models['Extra Trees'])
        ]
        
        voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        voting_clf.fit(self.X_train_selected, self.y_train)
        
        train_score = voting_clf.score(self.X_train_selected, self.y_train)
        test_score = voting_clf.score(self.X_test_selected, self.y_test)
        
        print(f"   Train Accuracy: {train_score:.3f}")
        print(f"   Test Accuracy: {test_score:.3f}")
        
        self.models['Voting Ensemble'] = voting_clf
        self.results['Voting Ensemble'] = {
            'model': voting_clf,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'predictions': voting_clf.predict(self.X_test_selected)
        }
    
    def create_stacking_ensemble(self):
        print(f"\n📚 CREATING STACKING ENSEMBLE")
        print("-"*80)
        
        estimators = [
            ('rf', self.models['Random Forest (Tuned)']),
            ('xgb', self.models['XGBoost']),
            ('lgb', self.models['LightGBM']),
            ('et', self.models['Extra Trees'])
        ]
        
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5,
            n_jobs=-1
        )
        stacking_clf.fit(self.X_train_selected, self.y_train)
        
        train_score = stacking_clf.score(self.X_train_selected, self.y_train)
        test_score = stacking_clf.score(self.X_test_selected, self.y_test)
        
        print(f"   Train Accuracy: {train_score:.3f}")
        print(f"   Test Accuracy: {test_score:.3f}")
        
        self.models['Stacking Ensemble'] = stacking_clf
        self.results['Stacking Ensemble'] = {
            'model': stacking_clf,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'predictions': stacking_clf.predict(self.X_test_selected)
        }
    
    def comprehensive_evaluation(self):
        print(f"\n📊 COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        
        eval_results = []
        
        for name, result in self.results.items():
            print(f"\n{'='*80}")
            print(f"MODEL: {name}")
            print('='*80)
            
            y_pred = result['predictions']
            
            # Basic metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1_micro = f1_score(self.y_test, y_pred, average='micro')
            f1_macro = f1_score(self.y_test, y_pred, average='macro')
            f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
            
            # Advanced metrics
            mcc = matthews_corrcoef(self.y_test, y_pred)
            kappa = cohen_kappa_score(self.y_test, y_pred)
            
            print(f"\n📈 Performance Metrics:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   F1-Score (Micro): {f1_micro:.4f}")
            print(f"   F1-Score (Macro): {f1_macro:.4f}")
            print(f"   F1-Score (Weighted): {f1_weighted:.4f}")
            print(f"   Matthews Correlation Coefficient: {mcc:.4f}")
            print(f"   Cohen's Kappa: {kappa:.4f}")
            
            # Classification report
            print(f"\n📋 Classification Report:")
            print(classification_report(self.y_test, y_pred, 
                                       target_names=self.label_encoder.classes_))
            
            # ROC-AUC (if probability available)
            try:
                if hasattr(result['model'], 'predict_proba'):
                    y_pred_proba = result['model'].predict_proba(self.X_test_selected)
                    roc_auc = roc_auc_score(self.y_test, y_pred_proba, 
                                           multi_class='ovr', average='weighted')
                    print(f"   ROC-AUC (OvR, Weighted): {roc_auc:.4f}")
                else:
                    roc_auc = None
            except:
                roc_auc = None
            
            eval_results.append({
                'Model': name,
                'Accuracy': accuracy,
                'F1-Micro': f1_micro,
                'F1-Macro': f1_macro,
                'F1-Weighted': f1_weighted,
                'MCC': mcc,
                'Kappa': kappa,
                'ROC-AUC': roc_auc,
                'Train_Acc': result['train_accuracy'],
                'Test_Acc': result['test_accuracy'],
                'Overfit': result['train_accuracy'] - result['test_accuracy']
            })
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            self.plot_confusion_matrix(cm, name)
        
        # Save results
        eval_df = pd.DataFrame(eval_results)
        eval_df = eval_df.sort_values('Test_Acc', ascending=False)
        eval_df.to_csv(f'{self.output_dir}/model_comparison.csv', index=False)
        
        print(f"\n{'='*80}")
        print("📊 MODEL COMPARISON SUMMARY")
        print('='*80)
        print(eval_df.to_string(index=False))
        
        return eval_df
    
    def plot_confusion_matrix(self, cm, model_name):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {model_name}', fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrix_{model_name.replace(" ", "_")}.png', 
                   dpi=300)
        plt.close()
    
    def plot_model_comparison(self, eval_df):
        print(f"\n📊 Creating comparison visualizations...")
        
        # 1. Accuracy comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy
        ax = axes[0, 0]
        x = range(len(eval_df))
        ax.bar(x, eval_df['Test_Acc'], alpha=0.7, label='Test')
        ax.bar(x, eval_df['Train_Acc'], alpha=0.4, label='Train')
        ax.set_xticks(x)
        ax.set_xticklabels(eval_df['Model'], rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # F1 Scores
        ax = axes[0, 1]
        width = 0.25
        x = np.arange(len(eval_df))
        ax.bar(x - width, eval_df['F1-Micro'], width, label='F1-Micro', alpha=0.8)
        ax.bar(x, eval_df['F1-Macro'], width, label='F1-Macro', alpha=0.8)
        ax.bar(x + width, eval_df['F1-Weighted'], width, label='F1-Weighted', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(eval_df['Model'], rotation=45, ha='right')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # MCC and Kappa
        ax = axes[1, 0]
        x = range(len(eval_df))
        ax.plot(x, eval_df['MCC'], marker='o', linewidth=2, label='MCC')
        ax.plot(x, eval_df['Kappa'], marker='s', linewidth=2, label='Kappa')
        ax.set_xticks(x)
        ax.set_xticklabels(eval_df['Model'], rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('MCC and Kappa Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Overfitting analysis
        ax = axes[1, 1]
        colors = ['green' if x < 0.1 else 'orange' if x < 0.2 else 'red' 
                 for x in eval_df['Overfit']]
        ax.bar(range(len(eval_df)), eval_df['Overfit'], color=colors, alpha=0.7)
        ax.set_xticks(range(len(eval_df)))
        ax.set_xticklabels(eval_df['Model'], rotation=45, ha='right')
        ax.set_ylabel('Train - Test Accuracy')
        ax.set_title('Overfitting Analysis', fontweight='bold')
        ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Caution')
        ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Overfit')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison_comprehensive.png', dpi=300)
        plt.close()
        
        print(f"   💾 Saved comprehensive comparison")
    
    def feature_importance_analysis(self):
        print(f"\n🎯 FEATURE IMPORTANCE ANALYSIS")
        print("-"*80)
        
        importance_data = {}
        
        # Random Forest
        if 'Random Forest (Tuned)' in self.models:
            rf_imp = self.models['Random Forest (Tuned)'].feature_importances_
            importance_data['RF'] = rf_imp
        
        # XGBoost
        if 'XGBoost' in self.models:
            xgb_imp = self.models['XGBoost'].feature_importances_
            importance_data['XGB'] = xgb_imp
        
        # LightGBM
        if 'LightGBM' in self.models:
            lgb_imp = self.models['LightGBM'].feature_importances_
            importance_data['LGB'] = lgb_imp
        
        # Extra Trees
        if 'Extra Trees' in self.models:
            et_imp = self.models['Extra Trees'].feature_importances_
            importance_data['ET'] = et_imp
        
        # Combine
        importance_df = pd.DataFrame(importance_data, index=self.best_features)
        importance_df['Mean'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('Mean', ascending=False)
        
        importance_df.to_csv(f'{self.output_dir}/feature_importance_combined.csv')
        
        print(f"✅ Top 20 most important features:")
        for i, (gene, row) in enumerate(importance_df.head(20).iterrows(), 1):
            print(f"   {i}. {gene}: {row['Mean']:.4f}")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top 20
        top_20 = importance_df.head(20)
        ax = axes[0]
        y_pos = range(len(top_20))
        ax.barh(y_pos, top_20['Mean'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([gene[:20] for gene in top_20.index], fontsize=8)
        ax.set_xlabel('Mean Importance')
        ax.set_title('Top 20 Most Important Features', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Cumulative importance
        ax = axes[1]
        cumsum = importance_df['Mean'].sort_values(ascending=False).cumsum()
        cumsum_norm = cumsum / cumsum.iloc[-1]
        ax.plot(range(len(cumsum_norm)), cumsum_norm, linewidth=2)
        ax.axhline(y=0.8, color='red', linestyle='--', label='80% variance')
        ax.axhline(y=0.9, color='orange', linestyle='--', label='90% variance')
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Cumulative Importance')
        ax.set_title('Cumulative Feature Importance', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance_analysis.png', dpi=300)
        plt.close()
        
        print(f"   💾 Saved feature importance analysis")
        
        return importance_df
    
    def cross_validation_analysis(self, n_folds=5):
        print(f"\n🔄 CROSS-VALIDATION ANALYSIS ({n_folds}-Fold)")
        print("="*80)
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        cv_results = {}
        
        for name, model in self.models.items():
            if 'Stacking' in name or 'Voting' in name:
                continue  # Skip ensemble models (too slow)
            
            print(f"\n📊 {name}...")
            
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train_selected, 
                                                                   self.y_train), 1):
                X_fold_train = self.X_train_selected[train_idx]
                X_fold_val = self.X_train_selected[val_idx]
                y_fold_train = self.y_train[train_idx]
                y_fold_val = self.y_train[val_idx]
                
                # Clone and train
                from sklearn.base import clone
                model_clone = clone(model)
                model_clone.fit(X_fold_train, y_fold_train)
                
                score = model_clone.score(X_fold_val, y_fold_val)
                fold_scores.append(score)
                
                print(f"   Fold {fold}: {score:.4f}")
            
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            print(f"   Mean: {mean_score:.4f} ± {std_score:.4f}")
            
            cv_results[name] = {
                'scores': fold_scores,
                'mean': mean_score,
                'std': std_score
            }
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(cv_results.keys())
        means = [cv_results[m]['mean'] for m in models]
        stds = [cv_results[m]['std'] for m in models]
        
        x = range(len(models))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{n_folds}-Fold Cross-Validation Results', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/cross_validation_results.png', dpi=300)
        plt.close()
        
        print(f"\n   💾 Saved cross-validation results")
        
        return cv_results
    
    def generate_comprehensive_report(self, eval_df, importance_df, cv_results):
        print(f"\n📋 GENERATING COMPREHENSIVE REPORT")
        print("-"*80)
        
        report = []
        
        report.append("="*100)
        report.append("ADVANCED MACHINE LEARNING CLASSIFICATION REPORT")
        report.append("="*100)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("="*100)
        report.append("1. DATASET SUMMARY")
        report.append("="*100)
        report.append(f"Total Samples: {len(self.X)}")
        report.append(f"Training Samples: {len(self.X_train)}")
        report.append(f"Test Samples: {len(self.X_test)}")
        report.append(f"Features Used: {len(self.best_features)}")
        report.append(f"Classes: {', '.join(self.label_encoder.classes_)}")
        report.append("")
        
        report.append("="*100)
        report.append("2. MODEL PERFORMANCE RANKING")
        report.append("="*100)
        report.append(eval_df.to_string(index=False))
        report.append("")
        
        report.append("="*100)
        report.append("3. BEST MODEL")
        report.append("="*100)
        best_model = eval_df.iloc[0]
        report.append(f"Model: {best_model['Model']}")
        report.append(f"Test Accuracy: {best_model['Test_Acc']:.4f}")
        report.append(f"F1-Score (Weighted): {best_model['F1-Weighted']:.4f}")
        report.append(f"Matthews Correlation: {best_model['MCC']:.4f}")
        report.append(f"Cohen's Kappa: {best_model['Kappa']:.4f}")
        if best_model['ROC-AUC'] is not None:
            report.append(f"ROC-AUC: {best_model['ROC-AUC']:.4f}")
        report.append("")
        
        report.append("="*100)
        report.append("4. TOP 10 MOST IMPORTANT FEATURES")
        report.append("="*100)
        for i, (gene, row) in enumerate(importance_df.head(10).iterrows(), 1):
            report.append(f"{i}. {gene}: {row['Mean']:.4f}")
        report.append("")
        
        report.append("="*100)
        report.append("5. CROSS-VALIDATION RESULTS")
        report.append("="*100)
        for name, results in cv_results.items():
            report.append(f"{name}: {results['mean']:.4f} ± {results['std']:.4f}")
        report.append("")
        
        report.append("="*100)
        report.append("6. RECOMMENDATIONS")
        report.append("="*100)
        
        # Overfitting check
        if best_model['Overfit'] < 0.1:
            report.append("- Model shows good generalization (overfitting < 10%)")
        elif best_model['Overfit'] < 0.2:
            report.append("- Moderate overfitting detected (10-20%), consider regularization")
        else:
            report.append("- Significant overfitting detected (>20%), reduce model complexity")
        
        # Performance check
        if best_model['Test_Acc'] > 0.85:
            report.append("- Excellent classification performance (>85%)")
        elif best_model['Test_Acc'] > 0.75:
            report.append("- Good classification performance (75-85%)")
        else:
            report.append("- Moderate performance (<75%), consider more data or features")
        
        # Feature check
        n_features_80 = (importance_df['Mean'].cumsum() / importance_df['Mean'].sum() > 0.8).argmax()
        report.append(f"- Top {n_features_80} features capture 80% of importance")
        report.append(f"- Consider reducing to {n_features_80} features for efficiency")
        
        report.append("")
        report.append("="*100)
        
        report_text = "\n".join(report)
        print(report_text)
        
        with open(f'{self.output_dir}/comprehensive_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n💾 Saved comprehensive report")
    
    def run_full_pipeline(self):
        print("\n" + "="*80)
        print("🚀 RUNNING FULL ADVANCED ML PIPELINE")
        print("="*80)
        
        # Step 1: Data preparation
        self.prepare_data()
        
        # Step 2: Feature selection
        feature_scores = self.advanced_feature_selection(
            methods=['variance', 'mutual_info', 'f_test', 'rf_importance'],
            n_features=1000
        )
        
        # Step 3: Train baseline models
        self.train_baseline_models()
        
        # Step 4: Train ensemble models
        self.train_ensemble_models()
        
        # Step 5: Create voting ensemble
        self.create_voting_ensemble()
        
        # Step 6: Create stacking ensemble
        self.create_stacking_ensemble()
        
        # Step 7: Comprehensive evaluation
        eval_df = self.comprehensive_evaluation()
        
        # Step 8: Plot comparisons
        self.plot_model_comparison(eval_df)
        
        # Step 9: Feature importance
        importance_df = self.feature_importance_analysis()
        
        # Step 10: Cross-validation
        cv_results = self.cross_validation_analysis(n_folds=5)
        
        # Step 11: Generate report
        self.generate_comprehensive_report(eval_df, importance_df, cv_results)
        
        print("\n" + "="*80)
        print("✅ ADVANCED ML PIPELINE COMPLETE!")
        print("="*80)
        print(f"\n📁 All results saved in: {self.output_dir}/")
        print("\nGenerated files:")
        print("   - model_comparison.csv")
        print("   - feature_importance_combined.csv")
        print("   - feature_selection_scores.csv")
        print("   - Confusion matrices for all models")
        print("   - Comprehensive visualizations")
        print("   - comprehensive_report.txt")
        print("="*80)
        
        return eval_df, importance_df, cv_results


def load_gse164416_data():
    print("="*80)
    print("📥 LOADING GSE164416 DATA")
    print("="*80)
    
    print("\n🧬 Loading expression data...")
    expr_path_gz = './suppl_data/GSE164416/GSE164416_DP_htseq_counts.txt.gz'
    
    if not os.path.exists(expr_path_gz):
        raise FileNotFoundError(f"Expression file not found: {expr_path_gz}")
    
    with gzip.open(expr_path_gz, 'rt') as f:
        expression_data = pd.read_csv(f, sep='\t', index_col=0)
    
    print(f"✅ Loaded: {expression_data.shape[0]} genes × {expression_data.shape[1]} samples")
    
    print("\n📋 Loading metadata...")
    metadata_path = './outputdata/GSE164416_metadata.csv'
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata = pd.read_csv(metadata_path, index_col=0)
    print(f"✅ Loaded: {metadata.shape[0]} samples")
    
    return expression_data, metadata


if __name__ == '__main__':
    
    expression_data, metadata = load_gse164416_data()
    
    classifier = AdvancedMLClassifier(expression_data, metadata)
    
    eval_df, importance_df, cv_results = classifier.run_full_pipeline()
    
    print("\n" + "="*80)
    print("🎉 ALL DONE!")
    print("="*80)