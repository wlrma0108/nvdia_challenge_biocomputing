"""
Step 4: Differential Expression Analysis

Identify significantly differentially expressed genes (DEGs):
1. Perform t-tests for each gene
2. Calculate fold changes
3. Apply multiple testing correction (FDR)
4. Generate volcano plots
5. Create ranked list of top DEGs
6. Compare with known diabetes genes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
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
sns.set_palette("husl")

def load_data():
    """
    Load processed data for DEG analysis
    """
    print("📂 Loading data...")

    X_train = pd.read_csv(PROCESSED_DIR / 'X_train.csv')
    y_train = pd.read_csv(PROCESSED_DIR / 'y_train.csv')['label'].values
    gene_names = pd.read_csv(PROCESSED_DIR / 'gene_names.csv')['gene'].values

    print(f"   Samples: {X_train.shape[0]}")
    print(f"   Genes: {X_train.shape[1]}")
    print(f"   Control: {np.sum(y_train == 0)}, Diabetes: {np.sum(y_train == 1)}")

    return X_train, y_train, gene_names

def calculate_differential_expression(X, y, gene_names):
    """
    Calculate fold changes and p-values for each gene
    """
    print("\n🧬 Calculating differential expression...")

    # Separate control and diabetes groups
    X_control = X[y == 0]
    X_diabetes = X[y == 1]

    results = []

    for i, gene in enumerate(gene_names):
        control_expr = X_control.iloc[:, i]
        diabetes_expr = X_diabetes.iloc[:, i]

        # Calculate means
        mean_control = control_expr.mean()
        mean_diabetes = diabetes_expr.mean()

        # Calculate fold change (log2 scale)
        # Since data is already normalized, we can use the difference
        log2_fc = mean_diabetes - mean_control

        # Calculate actual fold change
        fold_change = 2 ** log2_fc if log2_fc > 0 else -(2 ** abs(log2_fc))

        # Perform t-test
        stat, pval = stats.ttest_ind(diabetes_expr, control_expr)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_expr) - 1) * control_expr.std()**2 +
                             (len(diabetes_expr) - 1) * diabetes_expr.std()**2) /
                            (len(control_expr) + len(diabetes_expr) - 2))
        cohens_d = (mean_diabetes - mean_control) / pooled_std if pooled_std > 0 else 0

        results.append({
            'gene': gene,
            'mean_control': mean_control,
            'mean_diabetes': mean_diabetes,
            'log2_fold_change': log2_fc,
            'fold_change': fold_change,
            'pvalue': pval,
            'cohens_d': cohens_d,
            'abs_log2_fc': abs(log2_fc)
        })

    df_results = pd.DataFrame(results)

    # Apply multiple testing correction (FDR)
    print("   Applying FDR correction...")
    rejected, pvals_corrected, _, _ = multipletests(df_results['pvalue'],
                                                     alpha=0.05,
                                                     method='fdr_bh')

    df_results['pvalue_adj'] = pvals_corrected
    df_results['significant'] = rejected

    # Calculate -log10(p-value)
    df_results['neg_log10_pval'] = -np.log10(df_results['pvalue'] + 1e-300)
    df_results['neg_log10_pval_adj'] = -np.log10(df_results['pvalue_adj'] + 1e-300)

    # Sort by adjusted p-value
    df_results = df_results.sort_values('pvalue_adj')

    print(f"   ✅ Differential expression calculated")
    print(f"   Significant genes (FDR < 0.05): {df_results['significant'].sum()}")

    return df_results

def create_volcano_plot(df_results):
    """
    Create volcano plot
    """
    print("\n🌋 Creating volcano plot...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Define significance thresholds
    fc_threshold = 0.5  # log2 fold change
    pval_threshold = 0.05  # adjusted p-value

    # Categorize genes
    df_results['category'] = 'Not significant'
    df_results.loc[(df_results['log2_fold_change'] > fc_threshold) &
                   (df_results['pvalue_adj'] < pval_threshold), 'category'] = 'Upregulated'
    df_results.loc[(df_results['log2_fold_change'] < -fc_threshold) &
                   (df_results['pvalue_adj'] < pval_threshold), 'category'] = 'Downregulated'

    # Count categories
    n_up = (df_results['category'] == 'Upregulated').sum()
    n_down = (df_results['category'] == 'Downregulated').sum()
    n_ns = (df_results['category'] == 'Not significant').sum()

    # Plot points
    colors = {'Not significant': 'lightgray', 'Upregulated': '#e74c3c', 'Downregulated': '#3498db'}

    for category in ['Not significant', 'Upregulated', 'Downregulated']:
        mask = df_results['category'] == category
        ax.scatter(df_results.loc[mask, 'log2_fold_change'],
                  df_results.loc[mask, 'neg_log10_pval_adj'],
                  c=colors[category], label=f'{category} ({mask.sum()})',
                  alpha=0.6, s=20)

    # Add threshold lines
    ax.axhline(-np.log10(pval_threshold), color='black', linestyle='--',
              linewidth=1, alpha=0.5, label=f'FDR = {pval_threshold}')
    ax.axvline(fc_threshold, color='black', linestyle='--',
              linewidth=1, alpha=0.5)
    ax.axvline(-fc_threshold, color='black', linestyle='--',
              linewidth=1, alpha=0.5, label=f'|log2FC| = {fc_threshold}')

    # Label top genes
    top_genes = df_results.nsmallest(15, 'pvalue_adj')
    for _, row in top_genes.iterrows():
        if abs(row['log2_fold_change']) > fc_threshold and row['pvalue_adj'] < pval_threshold:
            ax.annotate(row['gene'],
                       xy=(row['log2_fold_change'], row['neg_log10_pval_adj']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax.set_xlabel('log₂ Fold Change (Diabetes vs Control)', fontsize=12)
    ax.set_ylabel('-log₁₀ Adjusted P-value', fontsize=12)
    ax.set_title('Volcano Plot: Differentially Expressed Genes', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'volcano_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Volcano plot saved")
    print(f"   Upregulated: {n_up}")
    print(f"   Downregulated: {n_down}")
    print(f"   Not significant: {n_ns}")

def create_ma_plot(df_results):
    """
    Create MA plot (mean vs fold change)
    """
    print("\n📊 Creating MA plot...")

    # Calculate average expression
    df_results['avg_expression'] = (df_results['mean_control'] + df_results['mean_diabetes']) / 2

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot points
    colors = {'Not significant': 'lightgray', 'Upregulated': '#e74c3c', 'Downregulated': '#3498db'}

    for category in ['Not significant', 'Upregulated', 'Downregulated']:
        mask = df_results['category'] == category
        ax.scatter(df_results.loc[mask, 'avg_expression'],
                  df_results.loc[mask, 'log2_fold_change'],
                  c=colors[category], label=category,
                  alpha=0.6, s=20)

    # Add reference line
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(-0.5, color='blue', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Average Expression', fontsize=12)
    ax.set_ylabel('log₂ Fold Change', fontsize=12)
    ax.set_title('MA Plot: Mean Expression vs Fold Change', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ma_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ MA plot saved")

def compare_with_known_genes(df_results):
    """
    Compare DEG results with known diabetes genes
    """
    print("\n🔍 Comparing with known diabetes biomarkers...")

    # Load known genes
    known_genes_df = pd.read_csv(RAW_DIR / 'known_diabetes_genes.csv')
    known_genes = set(known_genes_df['gene'].tolist())

    # Check which known genes are significant
    sig_genes = set(df_results[df_results['significant']]['gene'].tolist())
    known_and_sig = known_genes & sig_genes

    print(f"   Known diabetes genes: {len(known_genes)}")
    print(f"   Significant DEGs: {len(sig_genes)}")
    print(f"   Known genes that are significant: {len(known_and_sig)}")
    print(f"   Recovery rate: {len(known_and_sig)/len(known_genes)*100:.1f}%")

    if known_and_sig:
        print(f"\n   ✅ Recovered known genes:")
        for gene in sorted(known_and_sig)[:10]:
            row = df_results[df_results['gene'] == gene].iloc[0]
            print(f"      • {gene}: log2FC={row['log2_fold_change']:.3f}, "
                  f"p-adj={row['pvalue_adj']:.2e}")

    # Save comparison
    comparison = []
    for gene in known_genes:
        if gene in df_results['gene'].values:
            row = df_results[df_results['gene'] == gene].iloc[0]
            known_info = known_genes_df[known_genes_df['gene'] == gene].iloc[0]
            comparison.append({
                'gene': gene,
                'expected_fc': known_info['fold_change'],
                'observed_log2fc': row['log2_fold_change'],
                'pvalue_adj': row['pvalue_adj'],
                'significant': row['significant'],
                'rank': list(df_results['gene']).index(gene) + 1
            })

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(RESULTS_DIR / 'known_genes_comparison.csv', index=False)

    return comparison_df

def plot_top_degs(X, y, gene_names, df_results, n_genes=20):
    """
    Plot expression of top DEGs
    """
    print(f"\n📈 Plotting top {n_genes} DEGs...")

    top_genes = df_results.nsmallest(n_genes, 'pvalue_adj')

    n_cols = 5
    n_rows = (n_genes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(top_genes.iterrows()):
        gene = row['gene']
        gene_idx = list(gene_names).index(gene)
        gene_expr = X.iloc[:, gene_idx]

        # Create box plot
        data_to_plot = [gene_expr[y == 0], gene_expr[y == 1]]
        bp = axes[idx].boxplot(data_to_plot, labels=['Control', 'Diabetes'],
                               patch_artist=True, widths=0.6)

        # Color boxes
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')

        axes[idx].set_title(f"{gene}\nlog2FC={row['log2_fold_change']:.2f}, p={row['pvalue_adj']:.1e}",
                           fontsize=9)
        axes[idx].set_ylabel('Expression')
        axes[idx].grid(True, alpha=0.3, axis='y')

    # Hide empty subplots
    for idx in range(len(top_genes), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Top {n_genes} Differentially Expressed Genes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'top_degs_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ Boxplots saved")

def create_deg_summary(df_results):
    """
    Create summary table of DEGs
    """
    print("\n📝 Creating DEG summary...")

    # Top upregulated genes
    top_up = df_results[df_results['log2_fold_change'] > 0].nsmallest(50, 'pvalue_adj')
    top_up.to_csv(RESULTS_DIR / 'top_50_upregulated_genes.csv', index=False)

    # Top downregulated genes
    top_down = df_results[df_results['log2_fold_change'] < 0].nsmallest(50, 'pvalue_adj')
    top_down.to_csv(RESULTS_DIR / 'top_50_downregulated_genes.csv', index=False)

    # All significant genes
    sig_genes = df_results[df_results['significant']]
    sig_genes.to_csv(RESULTS_DIR / 'all_significant_degs.csv', index=False)

    # Top 100 genes by adjusted p-value
    top_100 = df_results.nsmallest(100, 'pvalue_adj')
    top_100.to_csv(RESULTS_DIR / 'top_100_degs.csv', index=False)

    print(f"   ✅ Saved DEG lists:")
    print(f"      • top_50_upregulated_genes.csv")
    print(f"      • top_50_downregulated_genes.csv")
    print(f"      • all_significant_degs.csv ({len(sig_genes)} genes)")
    print(f"      • top_100_degs.csv")

def plot_deg_statistics(df_results):
    """
    Plot DEG statistics
    """
    print("\n📊 Creating DEG statistics plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. P-value distribution
    axes[0, 0].hist(df_results['pvalue'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('P-value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('P-value Distribution')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Fold change distribution
    axes[0, 1].hist(df_results['log2_fold_change'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('log₂ Fold Change')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Fold Change Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Effect size distribution
    axes[1, 0].hist(df_results['cohens_d'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel("Cohen's d")
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Effect Size Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Top categories
    sig_genes = df_results[df_results['significant']]
    categories = sig_genes['category'].value_counts()

    axes[1, 1].bar(categories.index, categories.values, color=['#e74c3c', '#3498db'])
    axes[1, 1].set_xlabel('Category')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Significant DEGs by Category')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'deg_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ Statistics plots saved")

def main():
    """
    Main differential expression analysis pipeline
    """
    print("="*80)
    print("🔬 Step 4: Differential Expression Analysis")
    print("="*80)

    # Load data
    X_train, y_train, gene_names = load_data()

    # Calculate differential expression
    df_results = calculate_differential_expression(X_train, y_train, gene_names)

    # Create volcano plot
    create_volcano_plot(df_results)

    # Create MA plot
    create_ma_plot(df_results)

    # Compare with known genes
    comparison_df = compare_with_known_genes(df_results)

    # Plot top DEGs
    plot_top_degs(X_train, y_train, gene_names, df_results, n_genes=20)

    # Create DEG summary
    create_deg_summary(df_results)

    # Plot statistics
    plot_deg_statistics(df_results)

    # Save full results
    df_results.to_csv(RESULTS_DIR / 'differential_expression_results.csv', index=False)

    # Final summary
    print("\n" + "="*80)
    print("📊 DIFFERENTIAL EXPRESSION ANALYSIS SUMMARY")
    print("="*80)

    n_sig = df_results['significant'].sum()
    n_up = (df_results['category'] == 'Upregulated').sum()
    n_down = (df_results['category'] == 'Downregulated').sum()

    print(f"Total genes analyzed: {len(df_results)}")
    print(f"Significant DEGs (FDR < 0.05): {n_sig} ({n_sig/len(df_results)*100:.2f}%)")
    print(f"  • Upregulated: {n_up}")
    print(f"  • Downregulated: {n_down}")

    print(f"\nTop 5 most significant genes:")
    for idx, row in df_results.head(5).iterrows():
        direction = "↑" if row['log2_fold_change'] > 0 else "↓"
        print(f"  {direction} {row['gene']}: log2FC={row['log2_fold_change']:.3f}, "
              f"p-adj={row['pvalue_adj']:.2e}")

    print("\n✅ Differential expression analysis complete!")
    print("\n🎯 Next step: Feature Selection")

if __name__ == "__main__":
    main()
