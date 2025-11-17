"""
Step 2: Data Preprocessing and Normalization

This script performs:
1. Load expression data and metadata
2. Handle missing values
3. Remove low-variance genes
4. Check for log2 transformation needs
5. Batch effect correction
6. Train/validation/test split (70/15/15)
7. Standardization/normalization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """
    Load expression data and metadata
    """
    print("📂 Loading data...")

    # Load expression matrix
    expr_file = DATA_DIR / "simulated_diabetes_expression.csv"
    expr_data = pd.read_csv(expr_file, index_col=0)

    # Load metadata
    meta_file = DATA_DIR / "simulated_diabetes_metadata.csv"
    metadata = pd.read_csv(meta_file)

    print(f"   Expression shape: {expr_data.shape}")
    print(f"   Metadata shape: {metadata.shape}")

    return expr_data, metadata

def check_missing_values(expr_data):
    """
    Check and handle missing values
    """
    print("\n🔍 Checking for missing values...")

    missing_per_gene = expr_data.isna().sum(axis=1)
    missing_per_sample = expr_data.isna().sum(axis=0)

    total_missing = expr_data.isna().sum().sum()
    total_values = expr_data.shape[0] * expr_data.shape[1]
    missing_pct = (total_missing / total_values) * 100

    print(f"   Total missing values: {total_missing} ({missing_pct:.2f}%)")
    print(f"   Genes with missing values: {(missing_per_gene > 0).sum()}")
    print(f"   Samples with missing values: {(missing_per_sample > 0).sum()}")

    if total_missing > 0:
        print("   Handling missing values...")
        # Remove genes with >20% missing values
        genes_to_keep = missing_per_gene < (expr_data.shape[1] * 0.2)
        expr_data = expr_data[genes_to_keep]

        # Impute remaining with gene median
        expr_data = expr_data.fillna(expr_data.median(axis=1), axis=0)
        print(f"   ✅ After filtering: {expr_data.shape}")
    else:
        print("   ✅ No missing values detected")

    return expr_data

def check_data_distribution(expr_data, title="Expression Distribution"):
    """
    Check if data needs log transformation
    """
    print(f"\n📊 Checking data distribution...")

    # Sample a subset for visualization
    sample_genes = np.random.choice(expr_data.shape[0], min(1000, expr_data.shape[0]), replace=False)
    sample_data = expr_data.iloc[sample_genes].values.flatten()

    # Calculate statistics
    mean_val = np.mean(sample_data)
    median_val = np.median(sample_data)
    min_val = np.min(sample_data)
    max_val = np.max(sample_data)

    print(f"   Min: {min_val:.2f}")
    print(f"   Max: {max_val:.2f}")
    print(f"   Mean: {mean_val:.2f}")
    print(f"   Median: {median_val:.2f}")

    # Check if data is already log-transformed
    # Log-transformed data typically has values in range [0, 20]
    if max_val < 25 and min_val >= 0:
        print("   ✅ Data appears to be log-transformed already")
        needs_log = False
    else:
        print("   ⚠️  Data may need log2 transformation")
        needs_log = True

    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(sample_data, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Expression Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{title} - Histogram')
    axes[0].grid(True, alpha=0.3)

    stats.probplot(sample_data, dist="norm", plot=axes[1])
    axes[1].set_title(f'{title} - Q-Q Plot')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'distribution_{title.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

    return needs_log

def remove_low_variance_genes(expr_data, variance_threshold=0.1):
    """
    Remove genes with low variance across samples
    """
    print(f"\n🔬 Removing low-variance genes (threshold: {variance_threshold})...")

    initial_genes = expr_data.shape[0]

    # Calculate variance for each gene
    gene_variance = expr_data.var(axis=1)

    # Keep genes with variance above threshold
    high_var_genes = gene_variance > variance_threshold
    expr_data_filtered = expr_data[high_var_genes]

    removed_genes = initial_genes - expr_data_filtered.shape[0]
    print(f"   Removed {removed_genes} genes ({removed_genes/initial_genes*100:.1f}%)")
    print(f"   Remaining genes: {expr_data_filtered.shape[0]}")

    # Plot variance distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(gene_variance, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(variance_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {variance_threshold}')
    ax.set_xlabel('Variance')
    ax.set_ylabel('Number of Genes')
    ax.set_title('Gene Variance Distribution')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'gene_variance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    return expr_data_filtered

def batch_effect_correction(expr_data, metadata):
    """
    Simple batch effect correction using ComBat-like approach
    For demonstration, we'll use mean-centering by batch
    """
    print("\n🔧 Performing batch effect correction...")

    # Check if batch information is available
    if 'batch' not in metadata.columns:
        print("   ⚠️  No batch information found, skipping batch correction")
        return expr_data

    # Align metadata and expression data
    expr_data_aligned = expr_data[metadata['sample_id'].values]

    # Simple batch correction: subtract batch mean, add global mean
    batches = metadata['batch'].unique()
    print(f"   Found {len(batches)} batches: {batches}")

    expr_corrected = expr_data_aligned.copy()
    global_mean = expr_data_aligned.mean(axis=1)

    for batch in batches:
        batch_samples = metadata[metadata['batch'] == batch]['sample_id'].values
        batch_mean = expr_data_aligned[batch_samples].mean(axis=1)

        # Correct: remove batch mean, add global mean
        for sample in batch_samples:
            expr_corrected[sample] = expr_data_aligned[sample] - batch_mean + global_mean

    print("   ✅ Batch correction completed")

    # Visualize batch effects before and after
    visualize_batch_effects(expr_data_aligned, expr_corrected, metadata)

    return expr_corrected

def visualize_batch_effects(expr_before, expr_after, metadata):
    """
    Visualize batch effects using PCA
    """
    from sklearn.decomposition import PCA

    print("   Visualizing batch effects...")

    # Perform PCA
    pca = PCA(n_components=2)

    # Before correction
    pca_before = pca.fit_transform(expr_before.T)

    # After correction
    pca_after = pca.fit_transform(expr_after.T)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Before correction
    for batch in metadata['batch'].unique():
        mask = metadata['batch'] == batch
        axes[0].scatter(pca_before[mask, 0], pca_before[mask, 1],
                       label=batch, alpha=0.6, s=50)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].set_title('Before Batch Correction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # After correction
    for batch in metadata['batch'].unique():
        mask = metadata['batch'] == batch
        axes[1].scatter(pca_after[mask, 0], pca_after[mask, 1],
                       label=batch, alpha=0.6, s=50)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].set_title('After Batch Correction')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'batch_effect_correction.png', dpi=300, bbox_inches='tight')
    plt.close()

def split_data(expr_data, metadata, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into train/validation/test sets (70/15/15)
    Stratified by condition
    """
    print(f"\n✂️  Splitting data (train/val/test: {1-test_size-val_size:.0%}/{val_size:.0%}/{test_size:.0%})...")

    # Align expression data with metadata
    expr_data_aligned = expr_data[metadata['sample_id'].values].T
    labels = (metadata['condition'] == 'Diabetes').astype(int).values

    # First split: separate test set
    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        expr_data_aligned, labels, metadata.index,
        test_size=test_size, random_state=random_state, stratify=labels
    )

    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_temp, y_temp, idx_temp,
        test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )

    print(f"   Training set: {X_train.shape[0]} samples ({np.sum(y_train)} diabetes, {len(y_train)-np.sum(y_train)} control)")
    print(f"   Validation set: {X_val.shape[0]} samples ({np.sum(y_val)} diabetes, {len(y_val)-np.sum(y_val)} control)")
    print(f"   Test set: {X_test.shape[0]} samples ({np.sum(y_test)} diabetes, {len(y_test)-np.sum(y_test)} control)")

    # Create metadata for each split
    metadata_train = metadata.iloc[idx_train].copy()
    metadata_val = metadata.iloc[idx_val].copy()
    metadata_test = metadata.iloc[idx_test].copy()

    return (X_train, y_train, metadata_train,
            X_val, y_val, metadata_val,
            X_test, y_test, metadata_test)

def normalize_data(X_train, X_val, X_test):
    """
    Standardize features (z-score normalization)
    Fit on training data, apply to all sets
    """
    print("\n📏 Normalizing data (z-score standardization)...")

    scaler = StandardScaler()

    # Fit on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply to validation and test
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"   Training - mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
    print(f"   Validation - mean: {X_val_scaled.mean():.4f}, std: {X_val_scaled.std():.4f}")
    print(f"   Test - mean: {X_test_scaled.mean():.4f}, std: {X_test_scaled.std():.4f}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test,
                       metadata_train, metadata_val, metadata_test,
                       gene_names, scaler):
    """
    Save all processed data
    """
    print("\n💾 Saving processed data...")

    # Save expression matrices
    pd.DataFrame(X_train, columns=gene_names).to_csv(PROCESSED_DIR / 'X_train.csv', index=False)
    pd.DataFrame(X_val, columns=gene_names).to_csv(PROCESSED_DIR / 'X_val.csv', index=False)
    pd.DataFrame(X_test, columns=gene_names).to_csv(PROCESSED_DIR / 'X_test.csv', index=False)

    # Save labels
    pd.DataFrame({'label': y_train}).to_csv(PROCESSED_DIR / 'y_train.csv', index=False)
    pd.DataFrame({'label': y_val}).to_csv(PROCESSED_DIR / 'y_val.csv', index=False)
    pd.DataFrame({'label': y_test}).to_csv(PROCESSED_DIR / 'y_test.csv', index=False)

    # Save metadata
    metadata_train.to_csv(PROCESSED_DIR / 'metadata_train.csv', index=False)
    metadata_val.to_csv(PROCESSED_DIR / 'metadata_val.csv', index=False)
    metadata_test.to_csv(PROCESSED_DIR / 'metadata_test.csv', index=False)

    # Save gene names
    pd.DataFrame({'gene': gene_names}).to_csv(PROCESSED_DIR / 'gene_names.csv', index=False)

    # Save scaler
    import pickle
    with open(PROCESSED_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("   ✅ All processed data saved to data/processed/")

def main():
    """
    Main preprocessing pipeline
    """
    print("="*80)
    print("🔬 Step 2: Data Preprocessing and Normalization")
    print("="*80)

    # 1. Load data
    expr_data, metadata = load_data()

    # 2. Check missing values
    expr_data = check_missing_values(expr_data)

    # 3. Check data distribution
    needs_log = check_data_distribution(expr_data, "Original Data")

    if needs_log:
        print("\n   Applying log2 transformation...")
        expr_data = np.log2(expr_data + 1)
        check_data_distribution(expr_data, "Log2 Transformed")

    # 4. Remove low-variance genes
    expr_data = remove_low_variance_genes(expr_data, variance_threshold=0.1)

    # 5. Batch effect correction
    expr_data = batch_effect_correction(expr_data, metadata)

    # 6. Split data
    (X_train, y_train, metadata_train,
     X_val, y_val, metadata_val,
     X_test, y_test, metadata_test) = split_data(expr_data, metadata)

    # 7. Normalize data
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_data(X_train, X_val, X_test)

    # 8. Save processed data
    gene_names = expr_data.index.tolist()
    save_processed_data(X_train_norm, X_val_norm, X_test_norm,
                       y_train, y_val, y_test,
                       metadata_train, metadata_val, metadata_test,
                       gene_names, scaler)

    # Summary
    print("\n" + "="*80)
    print("📊 PREPROCESSING SUMMARY")
    print("="*80)
    print(f"Original features: 20,000 genes")
    print(f"After filtering: {len(gene_names)} genes")
    print(f"Feature reduction: {(1 - len(gene_names)/20000)*100:.1f}%")
    print(f"\nDataset splits:")
    print(f"  Training: {X_train_norm.shape}")
    print(f"  Validation: {X_val_norm.shape}")
    print(f"  Test: {X_test_norm.shape}")
    print(f"\nClass distribution:")
    print(f"  Training - Control: {len(y_train)-sum(y_train)}, Diabetes: {sum(y_train)}")
    print(f"  Validation - Control: {len(y_val)-sum(y_val)}, Diabetes: {sum(y_val)}")
    print(f"  Test - Control: {len(y_test)-sum(y_test)}, Diabetes: {sum(y_test)}")

    print("\n✅ Preprocessing complete!")
    print("\n🎯 Next step: Exploratory Data Analysis (EDA)")

if __name__ == "__main__":
    main()
