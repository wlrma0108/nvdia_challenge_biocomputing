"""
Generate Realistic Simulated Diabetes RNA Expression Data

Since we cannot access NCBI GEO from this environment, we'll generate
biologically realistic simulated data based on known diabetes biomarkers
and expression patterns from literature.

This simulated data will:
1. Include ~20,000 genes (typical RNA-seq)
2. Have 150 samples (75 diabetes, 75 controls)
3. Include known diabetes-related genes with differential expression
4. Add realistic noise and batch effects
5. Include biological variability
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

np.random.seed(42)  # For reproducibility

# Create data directory
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Known diabetes-related genes (from literature)
DIABETES_GENES = {
    # Insulin signaling pathway
    'INS': {'fold_change': -2.5, 'tissue': 'pancreas'},  # Insulin
    'INSR': {'fold_change': -1.5, 'tissue': 'muscle'},  # Insulin receptor
    'IRS1': {'fold_change': -1.8, 'tissue': 'liver'},  # Insulin receptor substrate 1
    'IRS2': {'fold_change': -1.6, 'tissue': 'liver'},
    'PIK3CA': {'fold_change': -1.4, 'tissue': 'multiple'},
    'AKT2': {'fold_change': -1.3, 'tissue': 'muscle'},

    # Glucose metabolism
    'GCK': {'fold_change': -2.0, 'tissue': 'pancreas'},  # Glucokinase
    'G6PC': {'fold_change': 1.8, 'tissue': 'liver'},  # Glucose-6-phosphatase
    'GLUT2': {'fold_change': -1.7, 'tissue': 'pancreas'},  # Glucose transporter 2
    'GLUT4': {'fold_change': -1.9, 'tissue': 'muscle'},  # Glucose transporter 4
    'HK2': {'fold_change': -1.5, 'tissue': 'muscle'},  # Hexokinase 2
    'PFKM': {'fold_change': -1.4, 'tissue': 'muscle'},

    # Inflammation markers (upregulated in diabetes)
    'TNF': {'fold_change': 2.3, 'tissue': 'blood'},  # Tumor necrosis factor
    'IL6': {'fold_change': 2.5, 'tissue': 'blood'},  # Interleukin 6
    'IL1B': {'fold_change': 2.1, 'tissue': 'blood'},  # Interleukin 1 beta
    'NFKB1': {'fold_change': 1.8, 'tissue': 'blood'},  # NF-kappa-B
    'CCL2': {'fold_change': 2.0, 'tissue': 'blood'},  # Chemokine
    'ICAM1': {'fold_change': 1.9, 'tissue': 'blood'},

    # Oxidative stress
    'SOD1': {'fold_change': -1.6, 'tissue': 'multiple'},  # Superoxide dismutase
    'SOD2': {'fold_change': -1.5, 'tissue': 'multiple'},
    'CAT': {'fold_change': -1.7, 'tissue': 'liver'},  # Catalase
    'GPX1': {'fold_change': -1.4, 'tissue': 'multiple'},

    # Beta cell function
    'PDX1': {'fold_change': -2.2, 'tissue': 'pancreas'},  # Pancreatic transcription factor
    'NEUROD1': {'fold_change': -1.9, 'tissue': 'pancreas'},
    'PAX6': {'fold_change': -1.7, 'tissue': 'pancreas'},
    'NKX6-1': {'fold_change': -2.0, 'tissue': 'pancreas'},

    # Lipid metabolism
    'PPARG': {'fold_change': -1.6, 'tissue': 'adipose'},  # Peroxisome proliferator-activated receptor gamma
    'ADIPOQ': {'fold_change': -2.1, 'tissue': 'adipose'},  # Adiponectin
    'LEP': {'fold_change': 1.8, 'tissue': 'adipose'},  # Leptin
    'FASN': {'fold_change': 1.5, 'tissue': 'liver'},  # Fatty acid synthase

    # Other diabetes-associated genes
    'TCF7L2': {'fold_change': 1.4, 'tissue': 'multiple'},  # Transcription factor 7-like 2
    'KCNJ11': {'fold_change': -1.5, 'tissue': 'pancreas'},  # Potassium channel
    'SLC30A8': {'fold_change': -1.6, 'tissue': 'pancreas'},  # Zinc transporter
    'HNF4A': {'fold_change': -1.5, 'tissue': 'liver'},  # Hepatocyte nuclear factor 4 alpha
    'GLP1R': {'fold_change': -1.7, 'tissue': 'pancreas'},  # Glucagon-like peptide 1 receptor

    # Additional biomarkers
    'RETN': {'fold_change': 2.2, 'tissue': 'blood'},  # Resistin
    'RBP4': {'fold_change': 1.9, 'tissue': 'blood'},  # Retinol binding protein 4
    'FGF21': {'fold_change': 1.6, 'tissue': 'liver'},  # Fibroblast growth factor 21
    'DPP4': {'fold_change': 1.5, 'tissue': 'multiple'},  # Dipeptidyl peptidase 4
}

def generate_gene_names(n_genes=20000, diabetes_genes=DIABETES_GENES):
    """
    Generate realistic gene names
    """
    # Start with diabetes genes
    gene_names = list(diabetes_genes.keys())

    # Add common housekeeping genes
    housekeeping = ['ACTB', 'GAPDH', 'TUBB', 'RPL13A', 'B2M', 'HPRT1', 'TBP', 'PPIA']
    gene_names.extend(housekeeping)

    # Generate remaining generic gene names
    remaining = n_genes - len(gene_names)
    gene_names.extend([f'GENE{i:05d}' for i in range(remaining)])

    return gene_names

def generate_expression_data(n_genes=20000, n_control=75, n_diabetes=75):
    """
    Generate realistic RNA expression data
    """
    print("🧬 Generating simulated RNA expression data...")
    print(f"   Genes: {n_genes}")
    print(f"   Control samples: {n_control}")
    print(f"   Diabetes samples: {n_diabetes}")

    # Generate gene names
    gene_names = generate_gene_names(n_genes)
    n_total_samples = n_control + n_diabetes

    # Initialize expression matrix with baseline expression
    # Log2-transformed expression values (typical for microarray/normalized RNA-seq)
    baseline_mean = 7.0  # Baseline expression level
    baseline_std = 2.0

    # Generate baseline expression for all genes
    expression = np.random.normal(baseline_mean, baseline_std, (n_genes, n_total_samples))

    # Add gene-specific mean expression levels (some genes are naturally high/low)
    gene_means = np.random.normal(0, 1.5, n_genes)
    expression = expression + gene_means[:, np.newaxis]

    # Add sample-specific effects (technical variation)
    sample_effects = np.random.normal(0, 0.3, n_total_samples)
    expression = expression + sample_effects

    # Add batch effects (simulate 3 batches)
    batch_size = n_total_samples // 3
    batch_effects = np.random.normal([0, 0.5, -0.3], 0.2, 3)
    for i in range(3):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size if i < 2 else n_total_samples
        expression[:, start_idx:end_idx] += batch_effects[i]

    # Apply differential expression to diabetes samples for known genes
    diabetes_indices = list(range(n_control, n_total_samples))

    for gene_name, info in DIABETES_GENES.items():
        if gene_name in gene_names:
            gene_idx = gene_names.index(gene_name)
            fold_change = info['fold_change']

            # Convert fold change to log2 difference
            log2_fc = np.log2(abs(fold_change)) * np.sign(fold_change)

            # Add the differential expression with some noise
            for sample_idx in diabetes_indices:
                # Add biological variability
                noise = np.random.normal(0, 0.3)
                expression[gene_idx, sample_idx] += log2_fc + noise

    # Add additional random differentially expressed genes (20-30)
    n_random_de = 25
    random_de_genes = np.random.choice(
        range(len(DIABETES_GENES), n_genes),
        size=n_random_de,
        replace=False
    )

    for gene_idx in random_de_genes:
        # Random fold change between 1.3 and 2.5
        log2_fc = np.random.uniform(0.4, 1.3) * np.random.choice([-1, 1])
        for sample_idx in diabetes_indices:
            noise = np.random.normal(0, 0.4)
            expression[gene_idx, sample_idx] += log2_fc + noise

    # Ensure non-negative values (clip at 0)
    expression = np.maximum(expression, 0)

    # Create DataFrame
    sample_names = [f'Control_{i:03d}' for i in range(n_control)] + \
                   [f'Diabetes_{i:03d}' for i in range(n_diabetes)]

    expr_df = pd.DataFrame(expression, index=gene_names, columns=sample_names)

    print(f"\n✅ Generated expression matrix: {expr_df.shape}")
    print(f"   Expression range: {expr_df.min().min():.2f} to {expr_df.max().max():.2f}")
    print(f"   Mean expression: {expr_df.mean().mean():.2f}")

    return expr_df

def generate_sample_metadata(n_control=75, n_diabetes=75):
    """
    Generate sample metadata
    """
    print("\n📋 Generating sample metadata...")

    # Control samples
    control_meta = []
    for i in range(n_control):
        control_meta.append({
            'sample_id': f'Control_{i:03d}',
            'condition': 'Control',
            'age': np.random.normal(45, 15),
            'sex': np.random.choice(['M', 'F']),
            'bmi': np.random.normal(24, 3),
            'fasting_glucose': np.random.normal(90, 10),  # mg/dL
            'hba1c': np.random.normal(5.2, 0.3),  # %
            'batch': np.random.choice(['Batch1', 'Batch2', 'Batch3']),
            'tissue': 'Blood',
        })

    # Diabetes samples
    diabetes_meta = []
    for i in range(n_diabetes):
        diabetes_meta.append({
            'sample_id': f'Diabetes_{i:03d}',
            'condition': 'Diabetes',
            'age': np.random.normal(52, 12),
            'sex': np.random.choice(['M', 'F']),
            'bmi': np.random.normal(30, 4),
            'fasting_glucose': np.random.normal(160, 30),  # mg/dL
            'hba1c': np.random.normal(8.5, 1.5),  # %
            'batch': np.random.choice(['Batch1', 'Batch2', 'Batch3']),
            'tissue': 'Blood',
        })

    metadata = pd.DataFrame(control_meta + diabetes_meta)

    # Round numeric columns
    metadata['age'] = metadata['age'].round().astype(int)
    metadata['bmi'] = metadata['bmi'].round(1)
    metadata['fasting_glucose'] = metadata['fasting_glucose'].round(1)
    metadata['hba1c'] = metadata['hba1c'].round(2)

    print(f"✅ Generated metadata for {len(metadata)} samples")
    print(f"\n📊 Condition distribution:")
    print(metadata['condition'].value_counts())

    return metadata

def generate_dataset_info():
    """
    Generate dataset description
    """
    info = {
        'dataset_id': 'SIMULATED_DIABETES_001',
        'title': 'Simulated RNA Expression Data for Type 2 Diabetes Detection',
        'description': '''
        This is a simulated RNA expression dataset designed to mimic real diabetes biology.
        The dataset includes known diabetes-related genes with realistic differential expression
        patterns based on published literature.

        Key features:
        - 20,000 genes (probes)
        - 150 samples (75 control, 75 diabetes)
        - Blood tissue (whole blood)
        - Known diabetes biomarkers included
        - Realistic technical and biological variation
        - Batch effects included

        Known differentially expressed genes include:
        - Insulin signaling: INS, INSR, IRS1, IRS2, AKT2
        - Glucose metabolism: GCK, GLUT2, GLUT4, HK2
        - Inflammation: TNF, IL6, IL1B, NFKB1
        - Beta cell function: PDX1, NEUROD1
        - And many more...
        ''',
        'platform': 'Simulated (based on Illumina HumanHT-12 design)',
        'organism': 'Homo sapiens',
        'tissue': 'Whole blood',
        'n_samples': 150,
        'n_features': 20000,
    }

    return info

def main():
    """
    Generate complete simulated dataset
    """
    print("="*80)
    print("🔬 Generating Simulated Diabetes RNA Expression Dataset")
    print("="*80)
    print("\n⚠️  Note: Due to NCBI access restrictions, we're generating biologically")
    print("realistic simulated data based on known diabetes biomarkers from literature.")
    print("This data will demonstrate the complete ML pipeline.\n")

    # Generate expression data
    expr_data = generate_expression_data(n_genes=20000, n_control=75, n_diabetes=75)

    # Generate metadata
    metadata = generate_sample_metadata(n_control=75, n_diabetes=75)

    # Generate dataset info
    info = generate_dataset_info()

    # Save all files
    print("\n💾 Saving files...")

    # Save expression matrix
    expr_file = DATA_DIR / "simulated_diabetes_expression.csv"
    expr_data.to_csv(expr_file)
    print(f"   ✅ {expr_file.name}")

    # Save metadata
    meta_file = DATA_DIR / "simulated_diabetes_metadata.csv"
    metadata.to_csv(meta_file, index=False)
    print(f"   ✅ {meta_file.name}")

    # Save dataset info
    info_file = DATA_DIR / "simulated_diabetes_info.txt"
    with open(info_file, 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    print(f"   ✅ {info_file.name}")

    # Save list of known diabetes genes
    diabetes_genes_file = DATA_DIR / "known_diabetes_genes.csv"
    diabetes_genes_df = pd.DataFrame([
        {'gene': gene, 'fold_change': info['fold_change'], 'tissue': info['tissue']}
        for gene, info in DIABETES_GENES.items()
    ])
    diabetes_genes_df.to_csv(diabetes_genes_file, index=False)
    print(f"   ✅ {diabetes_genes_file.name}")

    # Summary
    print("\n" + "="*80)
    print("📊 DATASET SUMMARY")
    print("="*80)
    print(f"Dataset ID: {info['dataset_id']}")
    print(f"Platform: {info['platform']}")
    print(f"Tissue: {info['tissue']}")
    print(f"Samples: {info['n_samples']} ({metadata['condition'].value_counts()['Control']} control, {metadata['condition'].value_counts()['Diabetes']} diabetes)")
    print(f"Features: {info['n_features']} genes")
    print(f"Known diabetes genes: {len(DIABETES_GENES)}")

    print(f"\n📈 Clinical characteristics:")
    print(f"   Control - Age: {metadata[metadata['condition']=='Control']['age'].mean():.1f}±{metadata[metadata['condition']=='Control']['age'].std():.1f}")
    print(f"   Diabetes - Age: {metadata[metadata['condition']=='Diabetes']['age'].mean():.1f}±{metadata[metadata['condition']=='Diabetes']['age'].std():.1f}")
    print(f"   Control - BMI: {metadata[metadata['condition']=='Control']['bmi'].mean():.1f}±{metadata[metadata['condition']=='Control']['bmi'].std():.1f}")
    print(f"   Diabetes - BMI: {metadata[metadata['condition']=='Diabetes']['bmi'].mean():.1f}±{metadata[metadata['condition']=='Diabetes']['bmi'].std():.1f}")
    print(f"   Control - HbA1c: {metadata[metadata['condition']=='Control']['hba1c'].mean():.2f}±{metadata[metadata['condition']=='Control']['hba1c'].std():.2f}%")
    print(f"   Diabetes - HbA1c: {metadata[metadata['condition']=='Diabetes']['hba1c'].mean():.2f}±{metadata[metadata['condition']=='Diabetes']['hba1c'].std():.2f}%")

    print("\n🎯 Next Steps:")
    print("1. Load and explore the simulated data")
    print("2. Verify known diabetes genes show differential expression")
    print("3. Proceed with preprocessing and normalization")
    print("4. Continue with ML pipeline")

    print("\n✅ Data generation complete!")

if __name__ == "__main__":
    main()
