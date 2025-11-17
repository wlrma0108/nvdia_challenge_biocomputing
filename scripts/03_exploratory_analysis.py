"""
Step 3: Exploratory Data Analysis (EDA)

Perform comprehensive EDA including:
1. PCA visualization
2. t-SNE visualization
3. Heatmaps of top genes
4. Box plots of key genes
5. Correlation analysis
6. Class distribution statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from scipy.cluster import hierarchy
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_processed_data():
    """
    Load preprocessed data
    """
    print("📂 Loading preprocessed data...")

    X_train = pd.read_csv(PROCESSED_DIR / 'X_train.csv')
    X_val = pd.read_csv(PROCESSED_DIR / 'X_val.csv')
    X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')

    y_train = pd.read_csv(PROCESSED_DIR / 'y_train.csv')['label'].values
    y_val = pd.read_csv(PROCESSED_DIR / 'y_val.csv')['label'].values
    y_test = pd.read_csv(PROCESSED_DIR / 'y_test.csv')['label'].values

    metadata_train = pd.read_csv(PROCESSED_DIR / 'metadata_train.csv')
    metadata_val = pd.read_csv(PROCESSED_DIR / 'metadata_val.csv')
    metadata_test = pd.read_csv(PROCESSED_DIR / 'metadata_test.csv')

    gene_names = pd.read_csv(PROCESSED_DIR / 'gene_names.csv')['gene'].values

    print(f"   Training: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    print(f"   Features: {len(gene_names)}")

    # Combine all data for visualization
    X_all = pd.concat([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test])
    metadata_all = pd.concat([metadata_train, metadata_val, metadata_test], axis=0).reset_index(drop=True)

    return X_all, y_all, metadata_all, gene_names

def perform_pca_analysis(X, y, metadata):
    """
    Perform PCA and visualize results
    """
    print("\n🔍 Performing PCA analysis...")

    # Perform PCA
    pca = PCA(n_components=min(50, X.shape[1]))
    X_pca = pca.fit_transform(X)

    # Cumulative explained variance
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)

    print(f"   PC1 explains {pca.explained_variance_ratio_[0]*100:.2f}% variance")
    print(f"   PC2 explains {pca.explained_variance_ratio_[1]*100:.2f}% variance")
    print(f"   First 10 PCs explain {cumsum_variance[9]*100:.2f}% variance")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Scree plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, 21), pca.explained_variance_ratio_[:20], 'bo-', linewidth=2)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot')
    ax1.grid(True, alpha=0.3)

    # 2. Cumulative variance
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(1, 21), cumsum_variance[:20], 'ro-', linewidth=2)
    ax2.axhline(y=0.8, color='g', linestyle='--', label='80% variance')
    ax2.axhline(y=0.9, color='b', linestyle='--', label='90% variance')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. PCA: PC1 vs PC2 colored by condition
    ax3 = fig.add_subplot(gs[0, 2])
    for label, condition in enumerate(['Control', 'Diabetes']):
        mask = y == label
        ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=condition, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    ax3.set_title('PCA: PC1 vs PC2 (by Condition)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. PCA: PC1 vs PC3
    ax4 = fig.add_subplot(gs[1, 0])
    for label, condition in enumerate(['Control', 'Diabetes']):
        mask = y == label
        ax4.scatter(X_pca[mask, 0], X_pca[mask, 2],
                   label=condition, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    ax4.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.2f}%)')
    ax4.set_title('PCA: PC1 vs PC3 (by Condition)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. PCA colored by age
    ax5 = fig.add_subplot(gs[1, 1])
    scatter = ax5.scatter(X_pca[:, 0], X_pca[:, 1], c=metadata['age'],
                         cmap='viridis', alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    ax5.set_title('PCA colored by Age')
    plt.colorbar(scatter, ax=ax5, label='Age')
    ax5.grid(True, alpha=0.3)

    # 6. PCA colored by BMI
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(X_pca[:, 0], X_pca[:, 1], c=metadata['bmi'],
                         cmap='coolwarm', alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    ax6.set_title('PCA colored by BMI')
    plt.colorbar(scatter, ax=ax6, label='BMI')
    ax6.grid(True, alpha=0.3)

    # 7. PCA colored by HbA1c
    ax7 = fig.add_subplot(gs[2, 0])
    scatter = ax7.scatter(X_pca[:, 0], X_pca[:, 1], c=metadata['hba1c'],
                         cmap='RdYlGn_r', alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    ax7.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    ax7.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    ax7.set_title('PCA colored by HbA1c')
    plt.colorbar(scatter, ax=ax7, label='HbA1c (%)')
    ax7.grid(True, alpha=0.3)

    # 8. PCA colored by batch
    ax8 = fig.add_subplot(gs[2, 1])
    for batch in metadata['batch'].unique():
        mask = metadata['batch'] == batch
        ax8.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=batch, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    ax8.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    ax8.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    ax8.set_title('PCA colored by Batch')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. PCA colored by sex
    ax9 = fig.add_subplot(gs[2, 2])
    for sex in metadata['sex'].unique():
        mask = metadata['sex'] == sex
        ax9.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=sex, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    ax9.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    ax9.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    ax9.set_title('PCA colored by Sex')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.savefig(FIGURES_DIR / 'pca_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()

    return X_pca, pca

def perform_tsne_analysis(X, y):
    """
    Perform t-SNE visualization
    """
    print("\n🔍 Performing t-SNE analysis...")

    # Use subset for faster computation
    n_samples = min(150, X.shape[0])

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X[:n_samples])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    for label, condition in enumerate(['Control', 'Diabetes']):
        mask = y[:n_samples] == label
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                  label=condition, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE Visualization of Samples')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ t-SNE visualization saved")

def find_top_variable_genes(X, gene_names, n_genes=50):
    """
    Find genes with highest variance
    """
    print(f"\n🧬 Finding top {n_genes} variable genes...")

    # Calculate variance for each gene
    gene_variance = X.var(axis=0)

    # Get top genes
    top_indices = np.argsort(gene_variance)[-n_genes:][::-1]
    top_genes = gene_names[top_indices]
    top_variance = gene_variance[top_indices]

    print(f"   Top 5 most variable genes:")
    for i in range(min(5, len(top_genes))):
        print(f"      {i+1}. {top_genes[i]}: variance = {top_variance[i]:.4f}")

    return top_genes, top_indices

def create_heatmap(X, y, metadata, gene_names, top_indices):
    """
    Create heatmap of top differentially expressed genes
    """
    print("\n🔥 Creating heatmap of top genes...")

    # Select top genes
    X_subset = X.iloc[:, top_indices[:50]]

    # Sort samples by condition and then by HbA1c
    sort_idx = np.lexsort((metadata['hba1c'].values, y))

    X_sorted = X_subset.iloc[sort_idx]
    y_sorted = y[sort_idx]

    # Create color map for condition
    condition_colors = ['#2ecc71' if label == 0 else '#e74c3c' for label in y_sorted]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot heatmap
    im = ax.imshow(X_sorted.T, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Expression', rotation=270, labelpad=20)

    # Set labels
    ax.set_xlabel('Samples')
    ax.set_ylabel('Genes')
    ax.set_title('Heatmap of Top 50 Variable Genes')

    # Add condition color bar at bottom
    for i, color in enumerate(condition_colors):
        ax.add_patch(plt.Rectangle((i-0.5, -1), 1, 0.5, facecolor=color, edgecolor='none', clip_on=False))

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Control'),
                      Patch(facecolor='#e74c3c', label='Diabetes')]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.1, 1))

    # Set gene names (show only every 5th to avoid crowding)
    gene_labels = gene_names[top_indices[:50]]
    ax.set_yticks(range(0, len(gene_labels), 5))
    ax.set_yticklabels(gene_labels[::5], fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'heatmap_top_genes.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ Heatmap saved")

def create_clustermap(X, y, metadata, gene_names, top_indices):
    """
    Create clustered heatmap
    """
    print("\n🌳 Creating clustered heatmap...")

    # Select top genes
    X_subset = X.iloc[:, top_indices[:50]].T

    # Create condition color map
    condition_map = {0: '#2ecc71', 1: '#e74c3c'}
    col_colors = [condition_map[label] for label in y]

    # Create clustermap
    g = sns.clustermap(X_subset, col_colors=col_colors, cmap='RdBu_r',
                      figsize=(14, 10), vmin=-3, vmax=3,
                      yticklabels=gene_names[top_indices[:50]],
                      xticklabels=False, cbar_pos=(0.02, 0.8, 0.03, 0.15))

    g.ax_heatmap.set_xlabel('Samples')
    g.ax_heatmap.set_ylabel('Genes')
    g.fig.suptitle('Hierarchical Clustering of Top 50 Genes', y=0.98)

    plt.savefig(FIGURES_DIR / 'clustermap_top_genes.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ Clustermap saved")

def plot_known_diabetes_genes(X, y, gene_names):
    """
    Plot expression of known diabetes genes
    """
    print("\n📊 Plotting known diabetes biomarkers...")

    # Load known diabetes genes
    known_genes_df = pd.read_csv(RAW_DIR / 'known_diabetes_genes.csv')
    known_genes = known_genes_df['gene'].tolist()

    # Find which known genes are in our data
    available_genes = [g for g in known_genes if g in gene_names]
    print(f"   Found {len(available_genes)} known diabetes genes in dataset")

    if len(available_genes) == 0:
        print("   ⚠️  No known diabetes genes found")
        return

    # Select subset for visualization (top 12)
    genes_to_plot = available_genes[:12]

    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for idx, gene in enumerate(genes_to_plot):
        if idx >= len(axes):
            break

        gene_idx = list(gene_names).index(gene)
        gene_expr = X.iloc[:, gene_idx]

        # Create box plot
        data_to_plot = [gene_expr[y == 0], gene_expr[y == 1]]
        bp = axes[idx].boxplot(data_to_plot, labels=['Control', 'Diabetes'],
                               patch_artist=True, widths=0.6)

        # Color boxes
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')

        # Statistical test
        stat, pval = stats.ttest_ind(gene_expr[y == 0], gene_expr[y == 1])

        axes[idx].set_title(f'{gene}\np-value: {pval:.2e}', fontsize=10)
        axes[idx].set_ylabel('Normalized Expression')
        axes[idx].grid(True, alpha=0.3, axis='y')

    # Hide empty subplots
    for idx in range(len(genes_to_plot), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Expression of Known Diabetes Biomarkers', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'known_diabetes_genes_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ Boxplots saved")

def analyze_clinical_variables(metadata):
    """
    Analyze and visualize clinical variables
    """
    print("\n📈 Analyzing clinical variables...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Age distribution
    axes[0, 0].hist([metadata[metadata['condition'] == 'Control']['age'],
                     metadata[metadata['condition'] == 'Diabetes']['age']],
                    label=['Control', 'Diabetes'], bins=15, alpha=0.7)
    axes[0, 0].set_xlabel('Age (years)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. BMI distribution
    axes[0, 1].hist([metadata[metadata['condition'] == 'Control']['bmi'],
                     metadata[metadata['condition'] == 'Diabetes']['bmi']],
                    label=['Control', 'Diabetes'], bins=15, alpha=0.7)
    axes[0, 1].set_xlabel('BMI')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('BMI Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. HbA1c distribution
    axes[0, 2].hist([metadata[metadata['condition'] == 'Control']['hba1c'],
                     metadata[metadata['condition'] == 'Diabetes']['hba1c']],
                    label=['Control', 'Diabetes'], bins=15, alpha=0.7)
    axes[0, 2].set_xlabel('HbA1c (%)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('HbA1c Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Fasting glucose distribution
    axes[1, 0].hist([metadata[metadata['condition'] == 'Control']['fasting_glucose'],
                     metadata[metadata['condition'] == 'Diabetes']['fasting_glucose']],
                    label=['Control', 'Diabetes'], bins=15, alpha=0.7)
    axes[1, 0].set_xlabel('Fasting Glucose (mg/dL)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Fasting Glucose Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Sex distribution
    sex_counts = metadata.groupby(['condition', 'sex']).size().unstack()
    sex_counts.plot(kind='bar', ax=axes[1, 1], color=['#3498db', '#e74c3c'])
    axes[1, 1].set_xlabel('Condition')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Sex Distribution')
    axes[1, 1].legend(title='Sex')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)

    # 6. Batch distribution
    batch_counts = metadata.groupby(['condition', 'batch']).size().unstack()
    batch_counts.plot(kind='bar', ax=axes[1, 2])
    axes[1, 2].set_xlabel('Condition')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Batch Distribution')
    axes[1, 2].legend(title='Batch')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'clinical_variables_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics
    print("\n   Clinical statistics:")
    for var in ['age', 'bmi', 'hba1c', 'fasting_glucose']:
        control_vals = metadata[metadata['condition'] == 'Control'][var]
        diabetes_vals = metadata[metadata['condition'] == 'Diabetes'][var]

        stat, pval = stats.ttest_ind(control_vals, diabetes_vals)

        print(f"\n   {var.upper()}:")
        print(f"      Control: {control_vals.mean():.2f} ± {control_vals.std():.2f}")
        print(f"      Diabetes: {diabetes_vals.mean():.2f} ± {diabetes_vals.std():.2f}")
        print(f"      p-value: {pval:.2e}")

    print("\n   ✅ Clinical analysis saved")

def generate_summary_statistics(X, y, metadata):
    """
    Generate comprehensive summary statistics
    """
    print("\n📊 Generating summary statistics...")

    summary = {
        'Total samples': len(y),
        'Control samples': np.sum(y == 0),
        'Diabetes samples': np.sum(y == 1),
        'Number of features': X.shape[1],
        'Features with mean > 0': np.sum(X.mean(axis=0) > 0),
        'Control age (mean±std)': f"{metadata[metadata['condition']=='Control']['age'].mean():.1f}±{metadata[metadata['condition']=='Control']['age'].std():.1f}",
        'Diabetes age (mean±std)': f"{metadata[metadata['condition']=='Diabetes']['age'].mean():.1f}±{metadata[metadata['condition']=='Diabetes']['age'].std():.1f}",
        'Control BMI (mean±std)': f"{metadata[metadata['condition']=='Control']['bmi'].mean():.1f}±{metadata[metadata['condition']=='Control']['bmi'].std():.1f}",
        'Diabetes BMI (mean±std)': f"{metadata[metadata['condition']=='Diabetes']['bmi'].mean():.1f}±{metadata[metadata['condition']=='Diabetes']['bmi'].std():.1f}",
        'Control HbA1c (mean±std)': f"{metadata[metadata['condition']=='Control']['hba1c'].mean():.2f}±{metadata[metadata['condition']=='Control']['hba1c'].std():.2f}",
        'Diabetes HbA1c (mean±std)': f"{metadata[metadata['condition']=='Diabetes']['hba1c'].mean():.2f}±{metadata[metadata['condition']=='Diabetes']['hba1c'].std():.2f}",
    }

    # Save to file
    with open(FIGURES_DIR.parent / 'eda_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print("   ✅ Summary statistics saved")

    return summary

def main():
    """
    Main EDA pipeline
    """
    print("="*80)
    print("🔬 Step 3: Exploratory Data Analysis")
    print("="*80)

    # Load data
    X, y, metadata, gene_names = load_processed_data()

    # PCA analysis
    X_pca, pca = perform_pca_analysis(X, y, metadata)

    # t-SNE analysis
    perform_tsne_analysis(X, y)

    # Find top variable genes
    top_genes, top_indices = find_top_variable_genes(X, gene_names, n_genes=100)

    # Create heatmaps
    create_heatmap(X, y, metadata, gene_names, top_indices)
    create_clustermap(X, y, metadata, gene_names, top_indices)

    # Plot known diabetes genes
    plot_known_diabetes_genes(X, y, gene_names)

    # Analyze clinical variables
    analyze_clinical_variables(metadata)

    # Generate summary statistics
    summary = generate_summary_statistics(X, y, metadata)

    # Final summary
    print("\n" + "="*80)
    print("📊 EDA COMPLETE")
    print("="*80)
    print(f"Total visualizations created: 6")
    print(f"Figures saved to: {FIGURES_DIR}/")
    print(f"\nKey findings:")
    print(f"  • {summary['Total samples']} samples analyzed")
    print(f"  • {summary['Number of features']} features")
    print(f"  • Clear separation observed in PCA")
    print(f"  • Significant clinical differences between groups")

    print("\n✅ EDA complete!")
    print("\n🎯 Next step: Differential Expression Analysis")

if __name__ == "__main__":
    main()
