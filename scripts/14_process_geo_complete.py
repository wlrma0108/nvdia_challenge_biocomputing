"""
Complete GEO Data Processing Pipeline

Processes 6 GEO series matrix files and prepares them for ML analysis.

Datasets:
- GSE164416: Pancreatic islets (primary)
- GSE76894: Blood/islets (206 samples)
- GSE25724: Pancreatic islets (validation)
- GSE81608: Cell-type specific
- GSE86468: Additional validation
- GSE86469: Cell-type specific

Author: Claude
Date: 2025-11-17
"""

import pandas as pd
import numpy as np
import gzip
import re
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
GEO_DIR = BASE_DIR / 'data' / 'geo_datasets'
OUTPUT_DIR = BASE_DIR / 'data' / 'real_geo_processed'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset priority (based on user recommendation)
DATASET_PRIORITY = {
    'GSE164416': 1,  # Primary - pancreatic islets
    'GSE76894': 2,   # Blood biomarkers - large cohort
    'GSE25724': 3,   # Validation
    'GSE86469': 4,   # Cell-type
    'GSE81608': 5,   # Cell-type
    'GSE86468': 6    # Additional validation
}

print("="*80)
print("GEO DATA PROCESSING PIPELINE")
print("="*80)

# ============================================================================
# 1. Parse Series Matrix Files
# ============================================================================

def parse_series_matrix_file(filepath):
    """
    Robust parser for GEO series matrix .txt or .txt.gz files

    Returns:
        series_meta: dict of series-level metadata
        sample_meta: DataFrame of sample-level metadata
        expression: DataFrame of expression data (probes × samples)
    """
    print(f"\nParsing: {filepath.name}")

    series_meta = {}
    sample_data = defaultdict(list)
    expression_lines = []
    in_table = False

    # Handle both .gz and plain text
    if filepath.suffix == '.gz':
        f = gzip.open(filepath, 'rt', encoding='utf-8', errors='replace')
    else:
        f = open(filepath, 'r', encoding='utf-8', errors='replace')

    try:
        for line in f:
            line = line.strip()

            # Series metadata
            if line.startswith('!Series_'):
                match = re.match(r'!Series_(\w+)\s+"?(.+?)"?$', line)
                if match:
                    key, value = match.groups()
                    series_meta[key] = value.strip('"')

            # Sample metadata
            elif line.startswith('!Sample_'):
                match = re.match(r'!Sample_(\w+)\s+(.+)$', line)
                if match:
                    key, values = match.groups()
                    # Split by tab, remove quotes
                    vals = [v.strip(' "') for v in values.split('\t')]
                    sample_data[key] = vals

            # Expression data table
            elif line == '!series_matrix_table_begin':
                in_table = True
                continue
            elif line == '!series_matrix_table_end':
                break
            elif in_table and line:
                expression_lines.append(line)

    finally:
        f.close()

    # Build sample metadata DataFrame
    sample_df = pd.DataFrame(sample_data)

    # Build expression DataFrame
    if expression_lines:
        # First line is header: ID_REF + sample IDs
        header = expression_lines[0].split('\t')
        header = [h.strip(' "') for h in header]

        # Rest are data
        data = []
        for line in expression_lines[1:]:
            vals = line.split('\t')
            vals = [v.strip(' "') for v in vals]
            data.append(vals)

        expr_df = pd.DataFrame(data, columns=header)
        expr_df = expr_df.set_index('ID_REF')

        # Convert to numeric
        for col in expr_df.columns:
            expr_df[col] = pd.to_numeric(expr_df[col], errors='coerce')

        print(f"  ✓ Expression: {expr_df.shape[0]} probes × {expr_df.shape[1]} samples")
        print(f"  ✓ Metadata: {len(sample_df)} samples × {len(sample_df.columns)} fields")
    else:
        expr_df = pd.DataFrame()
        print(f"  ✗ No expression data found!")

    return series_meta, sample_df, expr_df


# ============================================================================
# 2. Extract Diabetes Labels
# ============================================================================

def extract_diabetes_labels(sample_df, geo_id):
    """
    Intelligently extract diabetes/control labels from sample metadata

    Looks for patterns in:
    - Sample_title
    - Sample_characteristics_ch1
    - Sample_description
    """
    print(f"  Extracting labels for {geo_id}...")

    # Keywords for diabetes
    diabetes_patterns = [
        r'\bt2d\b', r'\bdiabetes\b', r'\bdiabetic\b',
        r'type\s*2', r'type\s*ii', r't2dm'
    ]

    # Keywords for control
    control_patterns = [
        r'\bcontrol\b', r'\bnon-diabetic\b', r'\bnondiabetic\b',
        r'\bnd\b', r'\bnormal\b', r'\bhealthy\b', r'\bnon\s*diabetic\b'
    ]

    labels = []

    for idx, row in sample_df.iterrows():
        # Combine all text fields
        text_fields = []
        for col in sample_df.columns:
            if col.startswith('Sample_'):
                val = row[col]
                if pd.notna(val):
                    text_fields.append(str(val).lower())

        combined_text = ' '.join(text_fields)

        # Check diabetes
        is_diabetes = any(re.search(p, combined_text, re.IGNORECASE)
                         for p in diabetes_patterns)

        # Check control
        is_control = any(re.search(p, combined_text, re.IGNORECASE)
                        for p in control_patterns)

        # Assign label
        if is_diabetes and not is_control:
            labels.append('Diabetes')
        elif is_control and not is_diabetes:
            labels.append('Control')
        elif 'igt' in combined_text:  # Impaired glucose tolerance
            labels.append('IGT')
        elif 't3c' in combined_text or 't3d' in combined_text:  # Type 3c
            labels.append('T3cD')
        else:
            labels.append('Unknown')

    sample_df['diabetes_status'] = labels

    # Summary
    counts = sample_df['diabetes_status'].value_counts()
    print(f"  Label distribution:")
    for status, count in counts.items():
        print(f"    {status}: {count}")

    return sample_df


# ============================================================================
# 3. Process All Datasets
# ============================================================================

print(f"\n{'='*80}")
print("STEP 1: LOADING ALL GEO DATASETS")
print(f"{'='*80}")

geo_files = sorted(GEO_DIR.glob('GSE*_series_matrix.txt*'))

if not geo_files:
    print(f"\n✗ ERROR: No GEO files found in {GEO_DIR}")
    print(f"  Expected: GSE164416_series_matrix.txt (or .txt.gz)")
    exit(1)

print(f"\nFound {len(geo_files)} files:")
for f in geo_files:
    geo_id = f.stem.split('_')[0]
    priority = DATASET_PRIORITY.get(geo_id, 99)
    print(f"  [{priority}] {f.name}")

# Parse all datasets
datasets = {}

for filepath in geo_files:
    geo_id = filepath.stem.split('_')[0]

    try:
        series_meta, sample_df, expr_df = parse_series_matrix_file(filepath)
        sample_df = extract_diabetes_labels(sample_df, geo_id)

        datasets[geo_id] = {
            'series_meta': series_meta,
            'sample_df': sample_df,
            'expr_df': expr_df,
            'priority': DATASET_PRIORITY.get(geo_id, 99)
        }

        # Save raw parsed data
        expr_df.to_csv(OUTPUT_DIR / f'{geo_id}_raw_expression.csv')
        sample_df.to_csv(OUTPUT_DIR / f'{geo_id}_raw_metadata.csv', index=False)

    except Exception as e:
        print(f"  ✗ Error parsing {filepath.name}: {e}")
        continue

print(f"\n✓ Successfully parsed {len(datasets)} datasets")


# ============================================================================
# 4. Select Primary Dataset
# ============================================================================

print(f"\n{'='*80}")
print("STEP 2: SELECTING PRIMARY DATASET")
print(f"{'='*80}")

# Sort by priority
sorted_datasets = sorted(datasets.items(),
                         key=lambda x: x[1]['priority'])

if not sorted_datasets:
    print("✗ No valid datasets!")
    exit(1)

primary_id = sorted_datasets[0][0]
primary_data = sorted_datasets[0][1]

print(f"\nPrimary dataset: {primary_id} (priority {primary_data['priority']})")
print(f"  Platform: {primary_data['series_meta'].get('platform_id', 'Unknown')}")
print(f"  Total samples: {len(primary_data['sample_df'])}")

# Filter to Diabetes and Control only
sample_df = primary_data['sample_df']
expr_df = primary_data['expr_df']

valid_mask = sample_df['diabetes_status'].isin(['Diabetes', 'Control'])
sample_df_filtered = sample_df[valid_mask].copy()

print(f"\nFiltered to Diabetes/Control:")
print(f"  Valid samples: {len(sample_df_filtered)}")
print(f"  Diabetes: {sum(sample_df_filtered['diabetes_status'] == 'Diabetes')}")
print(f"  Control: {sum(sample_df_filtered['diabetes_status'] == 'Control')}")

# Get sample IDs
if 'geo_accession' in sample_df_filtered.columns:
    sample_ids = sample_df_filtered['geo_accession'].values
else:
    # Extract from first Sample column
    sample_ids = sample_df_filtered.iloc[:, 0].values

# Filter expression data
expr_filtered = expr_df[sample_ids]

print(f"\nFiltered expression matrix: {expr_filtered.shape}")


# ============================================================================
# 5. Probe-to-Gene Mapping (GPL570 Annotation)
# ============================================================================

print(f"\n{'='*80}")
print("STEP 3: PROBE-TO-GENE MAPPING")
print(f"{'='*80}")

# For Affymetrix GPL570, we need to map probe IDs to gene symbols
# Since we can't download annotation files, we'll create a mapping strategy

print("\nProbe ID handling:")
print(f"  Total probe IDs: {len(expr_filtered)}")
print(f"  Sample probe IDs: {list(expr_filtered.index[:5])}")

# Option 1: Use probe IDs directly (works but not ideal)
# Option 2: Try to extract gene symbols from probe IDs (some contain gene names)
# Option 3: Manual annotation file (user needs to provide)

# For now, use probe IDs directly
# In production, you should download GPL570 annotation:
# ftp://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL570/GPL570.annot.gz

gene_expression = expr_filtered.copy()
gene_expression.index.name = 'Probe_ID'

print(f"  Using probe IDs as features (recommended: add GPL570 annotation)")
print(f"  Expression matrix: {gene_expression.shape}")


# ============================================================================
# 6. Quality Control
# ============================================================================

print(f"\n{'='*80}")
print("STEP 4: QUALITY CONTROL")
print(f"{'='*80}")

print("\nBefore QC:")
print(f"  Features: {gene_expression.shape[0]}")
print(f"  Samples: {gene_expression.shape[1]}")

# Remove probes with too many missing values
missing_pct = gene_expression.isna().sum(axis=1) / gene_expression.shape[1]
keep_probes = missing_pct < 0.2  # Less than 20% missing

gene_expression_qc = gene_expression[keep_probes].copy()
print(f"\nAfter removing high-missing probes (>20%):")
print(f"  Remaining features: {gene_expression_qc.shape[0]}")
print(f"  Removed: {sum(~keep_probes)}")

# Impute remaining missing values with median
for col in gene_expression_qc.columns:
    if gene_expression_qc[col].isna().any():
        median_val = gene_expression_qc[col].median()
        gene_expression_qc[col].fillna(median_val, inplace=True)

print(f"  Missing values after imputation: {gene_expression_qc.isna().sum().sum()}")

# Remove low-variance probes (bottom 10%)
variances = gene_expression_qc.var(axis=1)
variance_threshold = np.percentile(variances, 10)
high_var_probes = variances > variance_threshold

gene_expression_qc = gene_expression_qc[high_var_probes].copy()
print(f"\nAfter removing low-variance probes (bottom 10%):")
print(f"  Remaining features: {gene_expression_qc.shape[0]}")


# ============================================================================
# 7. Normalization
# ============================================================================

print(f"\n{'='*80}")
print("STEP 5: NORMALIZATION")
print(f"{'='*80}")

# Log2 transform if not already (check if values > 20)
if gene_expression_qc.max().max() > 20:
    print("  Applying log2 transformation...")
    gene_expression_qc = np.log2(gene_expression_qc + 1)
else:
    print("  Data appears to be already log-transformed")

# Z-score normalization per gene
print("  Applying z-score normalization per gene...")
gene_expression_norm = gene_expression_qc.T  # Transpose to samples × genes

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
gene_expression_scaled = scaler.fit_transform(gene_expression_norm)
gene_expression_scaled = pd.DataFrame(
    gene_expression_scaled,
    index=gene_expression_norm.index,
    columns=gene_expression_norm.columns
)

print(f"  Normalized matrix: {gene_expression_scaled.shape}")
print(f"  Mean: {gene_expression_scaled.mean().mean():.4f}")
print(f"  Std: {gene_expression_scaled.std().mean():.4f}")


# ============================================================================
# 8. Create Labels
# ============================================================================

print(f"\n{'='*80}")
print("STEP 6: CREATING LABELS")
print(f"{'='*80}")

# Create label DataFrame
labels_df = pd.DataFrame({
    'sample_id': sample_ids,
    'diabetes_status': sample_df_filtered['diabetes_status'].values,
    'label': (sample_df_filtered['diabetes_status'] == 'Diabetes').astype(int).values
})

print(f"  Total samples: {len(labels_df)}")
print(f"  Diabetes (1): {sum(labels_df['label'] == 1)}")
print(f"  Control (0): {sum(labels_df['label'] == 0)}")


# ============================================================================
# 9. Train/Val/Test Split
# ============================================================================

print(f"\n{'='*80}")
print("STEP 7: TRAIN/VAL/TEST SPLIT")
print(f"{'='*80}")

from sklearn.model_selection import train_test_split

# First split: 70% train, 30% temp
X = gene_expression_scaled.values
y = labels_df['label'].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Second split: 15% val, 15% test from temp
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"  Train: {X_train.shape[0]} samples ({sum(y_train==1)} diabetes, {sum(y_train==0)} control)")
print(f"  Val:   {X_val.shape[0]} samples ({sum(y_val==1)} diabetes, {sum(y_val==0)} control)")
print(f"  Test:  {X_test.shape[0]} samples ({sum(y_test==1)} diabetes, {sum(y_test==0)} control)")


# ============================================================================
# 10. Save in Pipeline-Compatible Format
# ============================================================================

print(f"\n{'='*80}")
print("STEP 8: SAVING PROCESSED DATA")
print(f"{'='*80}")

# Convert back to DataFrames
feature_names = gene_expression_scaled.columns

X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_val_df = pd.DataFrame(X_val, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

y_train_df = pd.DataFrame({'label': y_train})
y_val_df = pd.DataFrame({'label': y_val})
y_test_df = pd.DataFrame({'label': y_test})

# Save
X_train_df.to_csv(OUTPUT_DIR / 'X_train.csv')
X_val_df.to_csv(OUTPUT_DIR / 'X_val.csv')
X_test_df.to_csv(OUTPUT_DIR / 'X_test.csv')

y_train_df.to_csv(OUTPUT_DIR / 'y_train.csv', index=False)
y_val_df.to_csv(OUTPUT_DIR / 'y_val.csv', index=False)
y_test_df.to_csv(OUTPUT_DIR / 'y_test.csv', index=False)

# Save gene names
gene_names_df = pd.DataFrame({'gene': feature_names})
gene_names_df.to_csv(OUTPUT_DIR / 'gene_names.csv', index=False)

print(f"\n✓ Saved processed data to {OUTPUT_DIR}:")
print(f"  X_train.csv: {X_train_df.shape}")
print(f"  X_val.csv: {X_val_df.shape}")
print(f"  X_test.csv: {X_test_df.shape}")
print(f"  y_train.csv, y_val.csv, y_test.csv")
print(f"  gene_names.csv: {len(feature_names)} features")


# ============================================================================
# 11. Summary Report
# ============================================================================

print(f"\n{'='*80}")
print("PROCESSING SUMMARY")
print(f"{'='*80}")

summary = f"""
GEO DATA PROCESSING COMPLETED
{'='*80}

PRIMARY DATASET: {primary_id}
Platform: {primary_data['series_meta'].get('platform_id', 'Unknown')}

DATASETS PROCESSED: {len(datasets)}
{chr(10).join([f'  - {geo_id}: {len(data["sample_df"])} samples (priority {data["priority"]})'
               for geo_id, data in sorted(datasets.items(), key=lambda x: x[1]["priority"])])}

QUALITY CONTROL:
  Original probes: {len(expr_filtered)}
  After missing value filter: {len(gene_expression_qc)}
  After low-variance filter: {gene_expression_scaled.shape[1]}
  Final features: {gene_expression_scaled.shape[1]}

FINAL DATASET:
  Total samples: {len(labels_df)}
  Diabetes: {sum(labels_df['label'] == 1)}
  Control: {sum(labels_df['label'] == 0)}

SPLIT:
  Train: {len(y_train)} ({sum(y_train==1)}D, {sum(y_train==0)}C)
  Val:   {len(y_val)} ({sum(y_val==1)}D, {sum(y_val==0)}C)
  Test:  {len(y_test)} ({sum(y_test==1)}D, {sum(y_test==0)}C)

OUTPUT LOCATION:
  {OUTPUT_DIR}

FILES CREATED:
  ✓ X_train.csv, X_val.csv, X_test.csv
  ✓ y_train.csv, y_val.csv, y_test.csv
  ✓ gene_names.csv
  ✓ Raw data: {primary_id}_raw_expression.csv, {primary_id}_raw_metadata.csv

NEXT STEPS:
  1. Run feature selection: python scripts/05_feature_selection.py
     (modify to use OUTPUT_DIR instead of data/processed)

  2. Run model training: python scripts/06_ml_models.py

  3. Run clinical optimization: python scripts/10_clinical_grade_optimization.py

NOTE:
  - Currently using probe IDs as features
  - For better results, add GPL570 annotation to map probes to genes
  - Consider combining multiple datasets for larger sample size

CITATION:
  Dataset: {primary_id}
  PubMed ID: {primary_data['series_meta'].get('pubmed_id', 'N/A')}
"""

print(summary)

# Save summary
with open(OUTPUT_DIR / 'processing_summary.txt', 'w') as f:
    f.write(summary)

print(f"\n✓ Saved summary: {OUTPUT_DIR / 'processing_summary.txt'}")

print(f"\n{'='*80}")
print("✓ PROCESSING COMPLETE!")
print(f"{'='*80}")
print(f"\nYou can now run the ML pipeline with this real GEO data!")
print(f"Modify other scripts to use: {OUTPUT_DIR}")
