"""
Step 5: Feature Selection

Apply multiple feature selection methods to identify the most predictive biomarkers:
1. Variance threshold (filter method)
2. Correlation analysis with target
3. Univariate feature selection (chi2, f_classif)
4. Recursive Feature Elimination (RFE) - wrapper method
5. LASSO / Elastic Net - embedded method
6. Tree-based feature importance
7. Combine all methods to create final biomarker panel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, chi2,
    RFE, SelectFromModel
)
from sklearn.linear_model import LogisticRegression, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def load_data():
    """
    Load processed data
    """
    print("📂 Loading data...")

    X_train = pd.read_csv(PROCESSED_DIR / 'X_train.csv')
    X_val = pd.read_csv(PROCESSED_DIR / 'X_val.csv')
    y_train = pd.read_csv(PROCESSED_DIR / 'y_train.csv')['label'].values
    y_val = pd.read_csv(PROCESSED_DIR / 'y_val.csv')['label'].values
    gene_names = pd.read_csv(PROCESSED_DIR / 'gene_names.csv')['gene'].values

    print(f"   Training: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Features: {len(gene_names)}")

    return X_train, X_val, y_train, y_val, gene_names

def variance_threshold_selection(X_train, gene_names, threshold_percentile=75):
    """
    Select features based on variance threshold
    """
    print(f"\n🔍 Method 1: Variance Threshold (top {100-threshold_percentile}%)...")

    # Calculate variance for each feature
    variances = X_train.var(axis=0).values

    # Select threshold using percentile
    threshold = np.percentile(variances, threshold_percentile)

    # For normalized data, threshold might be very close to 1
    # Select features above threshold manually
    mask = variances > threshold
    selected_features = gene_names[mask]

    print(f"   Selected {len(selected_features)} features (variance > {threshold:.4f})")
    print(f"   Variance range: {variances.min():.4f} to {variances.max():.4f}")

    return selected_features, variances

def correlation_selection(X_train, y_train, gene_names, top_n=500):
    """
    Select features based on correlation with target
    """
    print(f"\n🔍 Method 2: Correlation with Target (top {top_n})...")

    # Calculate correlation with target
    correlations = np.array([np.corrcoef(X_train.iloc[:, i], y_train)[0, 1]
                            for i in range(X_train.shape[1])])

    # Get absolute correlations
    abs_correlations = np.abs(correlations)

    # Select top N
    top_indices = np.argsort(abs_correlations)[-top_n:]
    selected_features = gene_names[top_indices]

    print(f"   Selected {len(selected_features)} features")
    print(f"   Max correlation: {abs_correlations.max():.4f}")
    print(f"   Min correlation (selected): {abs_correlations[top_indices].min():.4f}")

    return selected_features, correlations

def univariate_selection(X_train, y_train, gene_names, top_n=500):
    """
    Select features using univariate statistical tests (ANOVA F-value)
    """
    print(f"\n🔍 Method 3: Univariate Feature Selection (top {top_n})...")

    # Use F-statistic
    selector = SelectKBest(f_classif, k=top_n)
    selector.fit(X_train, y_train)

    selected_features = gene_names[selector.get_support()]
    scores = selector.scores_

    print(f"   Selected {len(selected_features)} features")
    print(f"   Max F-score: {scores.max():.2f}")
    print(f"   Min F-score (selected): {scores[selector.get_support()].min():.2f}")

    return selected_features, scores

def lasso_selection(X_train, y_train, gene_names, n_features=100):
    """
    Select features using LASSO regularization
    """
    print(f"\n🔍 Method 4: LASSO Regularization...")

    # Use LassoCV for automatic alpha selection
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000, n_jobs=-1)
    lasso.fit(X_train, y_train)

    # Get non-zero coefficients
    non_zero_mask = lasso.coef_ != 0
    selected_features = gene_names[non_zero_mask]

    # If too few features, select top N by absolute coefficient
    if len(selected_features) < n_features:
        top_indices = np.argsort(np.abs(lasso.coef_))[-n_features:]
        selected_features = gene_names[top_indices]

    print(f"   Selected alpha: {lasso.alpha_:.4f}")
    print(f"   Non-zero coefficients: {non_zero_mask.sum()}")
    print(f"   Selected features: {len(selected_features)}")

    return selected_features, lasso.coef_

def elasticnet_selection(X_train, y_train, gene_names, n_features=100):
    """
    Select features using Elastic Net regularization
    """
    print(f"\n🔍 Method 5: Elastic Net Regularization...")

    # Use ElasticNetCV
    elastic = ElasticNetCV(cv=5, random_state=42, max_iter=5000, n_jobs=-1, l1_ratio=[0.1, 0.5, 0.7, 0.9])
    elastic.fit(X_train, y_train)

    # Get top features by absolute coefficient
    top_indices = np.argsort(np.abs(elastic.coef_))[-n_features:]
    selected_features = gene_names[top_indices]

    print(f"   Selected alpha: {elastic.alpha_:.4f}")
    print(f"   Selected l1_ratio: {elastic.l1_ratio_:.2f}")
    print(f"   Non-zero coefficients: {(elastic.coef_ != 0).sum()}")
    print(f"   Selected features: {len(selected_features)}")

    return selected_features, elastic.coef_

def rfe_selection(X_train, y_train, gene_names, n_features=50):
    """
    Recursive Feature Elimination with Logistic Regression
    """
    print(f"\n🔍 Method 6: Recursive Feature Elimination (target: {n_features} features)...")

    # Use a subset of features for faster computation
    # First, select top 1000 by variance
    variances = X_train.var(axis=0)
    top_var_indices = np.argsort(variances)[-1000:]
    X_train_subset = X_train.iloc[:, top_var_indices]
    gene_subset = gene_names[top_var_indices]

    # RFE with Logistic Regression
    estimator = LogisticRegression(penalty='l2', max_iter=1000, random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features, step=50)
    selector.fit(X_train_subset, y_train)

    selected_features = gene_subset[selector.support_]
    rankings = selector.ranking_

    print(f"   Selected {len(selected_features)} features")

    return selected_features, rankings

def random_forest_selection(X_train, y_train, gene_names, n_features=100):
    """
    Select features using Random Forest feature importance
    """
    print(f"\n🔍 Method 7: Random Forest Feature Importance (top {n_features})...")

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Get feature importances
    importances = rf.feature_importances_

    # Select top N
    top_indices = np.argsort(importances)[-n_features:]
    selected_features = gene_names[top_indices]

    print(f"   Selected {len(selected_features)} features")
    print(f"   Max importance: {importances.max():.4f}")
    print(f"   Min importance (selected): {importances[top_indices].min():.4f}")

    return selected_features, importances

def combine_feature_selections(all_selections, gene_names, top_n=50):
    """
    Combine all feature selection methods using voting
    """
    print(f"\n🎯 Combining all feature selection methods...")

    # Count how many methods selected each gene
    gene_votes = {gene: 0 for gene in gene_names}

    for method_name, selected_features in all_selections.items():
        print(f"   {method_name}: {len(selected_features)} features")
        for gene in selected_features:
            gene_votes[gene] += 1

    # Sort by votes
    sorted_genes = sorted(gene_votes.items(), key=lambda x: x[1], reverse=True)

    # Get distribution of votes
    vote_counts = {}
    for gene, votes in sorted_genes:
        vote_counts[votes] = vote_counts.get(votes, 0) + 1

    print(f"\n   Vote distribution:")
    for votes in sorted(vote_counts.keys(), reverse=True):
        print(f"      {votes} methods: {vote_counts[votes]} genes")

    # Select top genes by votes
    # First, get all genes with maximum votes
    max_votes = sorted_genes[0][1]
    top_genes = [gene for gene, votes in sorted_genes if votes >= max(max_votes - 2, 3)][:top_n]

    print(f"\n   ✅ Selected {len(top_genes)} final biomarkers")
    print(f"   Minimum votes: {min([gene_votes[g] for g in top_genes])}")
    print(f"   Maximum votes: {max([gene_votes[g] for g in top_genes])}")

    return top_genes, gene_votes

def evaluate_feature_set(X_train, X_val, y_train, y_val, feature_indices):
    """
    Evaluate a feature set using logistic regression
    """
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train.iloc[:, feature_indices], y_train)

    train_score = lr.score(X_train.iloc[:, feature_indices], y_train)
    val_score = lr.score(X_val.iloc[:, feature_indices], y_val)

    return train_score, val_score

def plot_feature_selection_results(all_selections, gene_votes, top_genes, gene_names):
    """
    Visualize feature selection results
    """
    print("\n📊 Creating feature selection visualizations...")

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Vote distribution
    ax1 = fig.add_subplot(gs[0, 0])
    vote_counts = {}
    for votes in gene_votes.values():
        vote_counts[votes] = vote_counts.get(votes, 0) + 1
    ax1.bar(vote_counts.keys(), vote_counts.values(), color='steelblue', edgecolor='black')
    ax1.set_xlabel('Number of Methods')
    ax1.set_ylabel('Number of Genes')
    ax1.set_title('Feature Selection Vote Distribution')
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Method overlap
    ax2 = fig.add_subplot(gs[0, 1])
    method_sizes = [len(features) for features in all_selections.values()]
    ax2.barh(list(all_selections.keys()), method_sizes, color='coral', edgecolor='black')
    ax2.set_xlabel('Number of Features Selected')
    ax2.set_title('Features Selected by Each Method')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Top genes by votes
    ax3 = fig.add_subplot(gs[0, 2])
    top_genes_sorted = sorted([(g, gene_votes[g]) for g in top_genes[:20]],
                              key=lambda x: x[1], reverse=True)
    genes, votes = zip(*top_genes_sorted)
    ax3.barh(range(len(genes)), votes, color='seagreen', edgecolor='black')
    ax3.set_yticks(range(len(genes)))
    ax3.set_yticklabels(genes, fontsize=8)
    ax3.set_xlabel('Number of Methods')
    ax3.set_title('Top 20 Genes by Vote Count')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()

    # 4-9. Venn-like comparison (overlap matrix)
    ax4 = fig.add_subplot(gs[1:, :])
    method_names = list(all_selections.keys())
    n_methods = len(method_names)

    # Create overlap matrix
    overlap_matrix = np.zeros((n_methods, n_methods))
    for i, method1 in enumerate(method_names):
        for j, method2 in enumerate(method_names):
            set1 = set(all_selections[method1])
            set2 = set(all_selections[method2])
            overlap = len(set1 & set2)
            overlap_matrix[i, j] = overlap

    # Plot heatmap
    im = ax4.imshow(overlap_matrix, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(n_methods))
    ax4.set_yticks(range(n_methods))
    ax4.set_xticklabels(method_names, rotation=45, ha='right')
    ax4.set_yticklabels(method_names)
    ax4.set_title('Feature Selection Method Overlap', fontsize=12, pad=20)

    # Add text annotations
    for i in range(n_methods):
        for j in range(n_methods):
            text = ax4.text(j, i, int(overlap_matrix[i, j]),
                          ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=ax4, label='Number of Shared Features')

    plt.savefig(FIGURES_DIR / 'feature_selection_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ Visualizations saved")

def compare_with_known_genes(top_genes, gene_votes):
    """
    Compare selected features with known diabetes genes
    """
    print("\n🔍 Comparing with known diabetes biomarkers...")

    # Load known genes
    known_genes_df = pd.read_csv(RAW_DIR / 'known_diabetes_genes.csv')
    known_genes = set(known_genes_df['gene'].tolist())

    # Check overlap
    selected_set = set(top_genes)
    overlap = known_genes & selected_set

    print(f"   Known diabetes genes: {len(known_genes)}")
    print(f"   Selected features: {len(top_genes)}")
    print(f"   Overlap: {len(overlap)} ({len(overlap)/len(known_genes)*100:.1f}% recovery)")

    if overlap:
        print(f"\n   ✅ Recovered known genes:")
        for gene in sorted(overlap):
            print(f"      • {gene} (selected by {gene_votes[gene]} methods)")

def save_feature_selection_results(top_genes, gene_votes, all_selections, gene_names):
    """
    Save all feature selection results
    """
    print("\n💾 Saving feature selection results...")

    # Save final biomarker panel
    biomarker_df = pd.DataFrame({
        'gene': top_genes,
        'vote_count': [gene_votes[g] for g in top_genes],
        'rank': range(1, len(top_genes) + 1)
    })
    biomarker_df.to_csv(RESULTS_DIR / 'final_biomarker_panel.csv', index=False)

    # Save all votes
    votes_df = pd.DataFrame([
        {'gene': gene, 'votes': votes}
        for gene, votes in sorted(gene_votes.items(), key=lambda x: x[1], reverse=True)
        if votes > 0
    ])
    votes_df.to_csv(RESULTS_DIR / 'all_feature_votes.csv', index=False)

    # Save method-specific selections
    for method_name, features in all_selections.items():
        method_df = pd.DataFrame({'gene': features})
        safe_name = method_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        method_df.to_csv(RESULTS_DIR / f'features_{safe_name}.csv', index=False)

    print("   ✅ Feature selection results saved")

def main():
    """
    Main feature selection pipeline
    """
    print("="*80)
    print("🔬 Step 5: Feature Selection")
    print("="*80)

    # Load data
    X_train, X_val, y_train, y_val, gene_names = load_data()

    # Apply all feature selection methods
    all_selections = {}

    # 1. Variance threshold
    selected, _ = variance_threshold_selection(X_train, gene_names, threshold_percentile=90)
    all_selections['Variance (top 10%)'] = selected

    # 2. Correlation
    selected, _ = correlation_selection(X_train, y_train, gene_names, top_n=200)
    all_selections['Correlation (top 200)'] = selected

    # 3. Univariate
    selected, _ = univariate_selection(X_train, y_train, gene_names, top_n=200)
    all_selections['Univariate (top 200)'] = selected

    # 4. LASSO
    selected, _ = lasso_selection(X_train, y_train, gene_names, n_features=100)
    all_selections['LASSO (top 100)'] = selected

    # 5. Elastic Net
    selected, _ = elasticnet_selection(X_train, y_train, gene_names, n_features=100)
    all_selections['ElasticNet (top 100)'] = selected

    # 6. RFE
    selected, _ = rfe_selection(X_train, y_train, gene_names, n_features=50)
    all_selections['RFE (50 features)'] = selected

    # 7. Random Forest
    selected, _ = random_forest_selection(X_train, y_train, gene_names, n_features=100)
    all_selections['Random Forest (top 100)'] = selected

    # Combine all methods
    top_genes, gene_votes = combine_feature_selections(all_selections, gene_names, top_n=50)

    # Evaluate feature set
    print("\n📊 Evaluating final biomarker panel...")
    top_indices = [list(gene_names).index(g) for g in top_genes]
    train_score, val_score = evaluate_feature_set(X_train, X_val, y_train, y_val, top_indices)
    print(f"   Training accuracy: {train_score:.4f}")
    print(f"   Validation accuracy: {val_score:.4f}")

    # Visualize results
    plot_feature_selection_results(all_selections, gene_votes, top_genes, gene_names)

    # Compare with known genes
    compare_with_known_genes(top_genes, gene_votes)

    # Save results
    save_feature_selection_results(top_genes, gene_votes, all_selections, gene_names)

    # Final summary
    print("\n" + "="*80)
    print("📊 FEATURE SELECTION SUMMARY")
    print("="*80)
    print(f"Original features: {len(gene_names)}")
    print(f"Final biomarkers: {len(top_genes)}")
    print(f"Feature reduction: {(1 - len(top_genes)/len(gene_names))*100:.2f}%")
    print(f"\nTop 10 biomarkers:")
    for i, gene in enumerate(top_genes[:10], 1):
        print(f"   {i}. {gene} (selected by {gene_votes[gene]} methods)")

    print("\n✅ Feature selection complete!")
    print("\n🎯 Next step: Machine Learning Model Development")

if __name__ == "__main__":
    main()
