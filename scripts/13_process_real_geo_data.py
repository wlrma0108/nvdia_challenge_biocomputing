"""
Process Real GEO Series Matrix Files

This script processes actual GEO series matrix files (.txt.gz format)
and prepares them for the ML pipeline.

Target datasets:
- GSE164416: Pancreatic islet samples
- GSE76894: Blood/islet samples (206 samples)
- GSE25724: Validation cohort
- GSE81608, GSE86468, GSE86469: Additional cohorts

Author: Claude
Date: 2025-11-17
"""

import pandas as pd
import numpy as np
import gzip
from pathlib import Path
import re

# Set random seed
np.random.seed(42)

# Define paths
BASE_DIR = Path(__file__).parent.parent
GEO_DIR = BASE_DIR / 'data' / 'geo_datasets'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed_geo'
RAW_DIR = BASE_DIR / 'data' / 'raw_geo'

# Create directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PROCESSING REAL GEO SERIES MATRIX FILES")
print("="*80)

# ============================================================================
# 1. Parse GEO Series Matrix File
# ============================================================================

def parse_series_matrix(filepath):
    """
    Parse a GEO series matrix .txt.gz file

    Returns:
        metadata_dict: Dictionary of series/sample metadata
        expression_df: DataFrame of gene expression data
    """
    print(f"\nParsing {filepath.name}...")

    metadata = {}
    sample_metadata = {}
    expression_data = []
    in_data_section = False

    # Open gzipped file
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()

            # Series-level metadata
            if line.startswith('!Series_'):
                match = re.match(r'!Series_(\w+)\s+"(.+)"', line)
                if match:
                    key, value = match.groups()
                    metadata[f'Series_{key}'] = value

            # Sample-level metadata
            elif line.startswith('!Sample_'):
                match = re.match(r'!Sample_(\w+)\s+(.+)', line)
                if match:
                    key, values = match.groups()
                    # Split tab-separated values
                    sample_values = [v.strip('"') for v in values.split('\t')]
                    sample_metadata[f'Sample_{key}'] = sample_values

            # Expression data section
            elif line.startswith('!series_matrix_table_begin'):
                in_data_section = True
                continue
            elif line.startswith('!series_matrix_table_end'):
                break
            elif in_data_section:
                # Parse expression data
                expression_data.append(line.split('\t'))

    # Convert expression data to DataFrame
    if expression_data:
        # First row is header (ID_REF + sample IDs)
        header = [h.strip('"') for h in expression_data[0]]
        data_rows = [[val.strip('"') for val in row] for row in expression_data[1:]]

        expression_df = pd.DataFrame(data_rows, columns=header)
        expression_df = expression_df.set_index('ID_REF')

        # Convert to numeric
        for col in expression_df.columns:
            expression_df[col] = pd.to_numeric(expression_df[col], errors='coerce')

        print(f"  Expression matrix: {expression_df.shape[0]} probes × {expression_df.shape[1]} samples")
    else:
        expression_df = pd.DataFrame()
        print("  WARNING: No expression data found!")

    # Convert sample metadata to DataFrame
    if sample_metadata:
        # All arrays should have same length
        sample_df = pd.DataFrame(sample_metadata)
        print(f"  Sample metadata: {len(sample_df)} samples, {len(sample_df.columns)} fields")
    else:
        sample_df = pd.DataFrame()

    return metadata, sample_df, expression_df


# ============================================================================
# 2. Extract Diabetes Status from Metadata
# ============================================================================

def extract_diabetes_status(sample_df, geo_id):
    """
    Extract diabetes/control labels from sample metadata

    Different GEO datasets use different field names:
    - Some use 'Sample_characteristics_ch1'
    - Some use 'Sample_title'
    - Need to search for keywords: diabetes, diabetic, T2D, ND, control
    """
    print(f"\nExtracting diabetes status for {geo_id}...")

    # Common patterns
    diabetes_keywords = ['diabetes', 'diabetic', 't2d', 'type 2', 'type2']
    control_keywords = ['control', 'non-diabetic', 'nondiabetic', 'nd', 'normal', 'healthy']

    labels = []

    # Check each row
    for idx, row in sample_df.iterrows():
        # Convert all values to string and lowercase
        row_text = ' '.join([str(v).lower() for v in row.values if pd.notna(v)])

        # Check for diabetes
        is_diabetes = any(keyword in row_text for keyword in diabetes_keywords)
        is_control = any(keyword in row_text for keyword in control_keywords)

        if is_diabetes and not is_control:
            labels.append('Diabetes')
        elif is_control and not is_diabetes:
            labels.append('Control')
        else:
            # Ambiguous - mark as unknown
            labels.append('Unknown')

    sample_df['diabetes_status'] = labels

    # Print summary
    status_counts = sample_df['diabetes_status'].value_counts()
    print(f"  Status counts:")
    for status, count in status_counts.items():
        print(f"    {status}: {count}")

    return sample_df


# ============================================================================
# 3. Process All Available GEO Files
# ============================================================================

print(f"\n1. Scanning for GEO series matrix files in {GEO_DIR}...")

geo_files = list(GEO_DIR.glob('GSE*_series_matrix.txt.gz'))

if not geo_files:
    print(f"\nERROR: No GEO series matrix files found in {GEO_DIR}")
    print(f"Expected files like: GSE164416_series_matrix.txt.gz")
    print(f"\nPlease ensure files are placed in: {GEO_DIR}")
    exit(1)

print(f"Found {len(geo_files)} GEO series matrix files:")
for f in geo_files:
    print(f"  - {f.name}")

# Process each file
all_datasets = {}

for geo_file in geo_files:
    geo_id = geo_file.stem.split('_')[0]  # Extract GSE number

    try:
        # Parse file
        metadata, sample_df, expression_df = parse_series_matrix(geo_file)

        # Extract diabetes status
        sample_df = extract_diabetes_status(sample_df, geo_id)

        # Store
        all_datasets[geo_id] = {
            'metadata': metadata,
            'samples': sample_df,
            'expression': expression_df,
            'file': geo_file
        }

        # Save individual dataset
        expression_df.to_csv(RAW_DIR / f'{geo_id}_expression.csv')
        sample_df.to_csv(RAW_DIR / f'{geo_id}_samples.csv', index=False)

        print(f"✓ Saved {geo_id} to {RAW_DIR}")

    except Exception as e:
        print(f"✗ Error processing {geo_file.name}: {e}")
        continue

# ============================================================================
# 4. Select Primary Dataset for Analysis
# ============================================================================

print("\n" + "="*80)
print("DATASET SELECTION")
print("="*80)

# Priority order based on user recommendations
priority_order = ['GSE164416', 'GSE76894', 'GSE25724', 'GSE86469', 'GSE81608', 'GSE86468']

primary_dataset = None
for geo_id in priority_order:
    if geo_id in all_datasets:
        primary_dataset = geo_id
        break

if not primary_dataset:
    # Use first available
    primary_dataset = list(all_datasets.keys())[0]

print(f"\nPrimary dataset selected: {primary_dataset}")

dataset = all_datasets[primary_dataset]
expression_df = dataset['expression']
sample_df = dataset['samples']

# Filter to only Diabetes and Control samples
valid_samples = sample_df[sample_df['diabetes_status'].isin(['Diabetes', 'Control'])].copy()
print(f"\nFiltering to valid samples:")
print(f"  Original: {len(sample_df)} samples")
print(f"  Valid (Diabetes/Control): {len(valid_samples)} samples")
print(f"  Removed (Unknown): {len(sample_df) - len(valid_samples)} samples")

# Get corresponding sample IDs
sample_ids = valid_samples['Sample_geo_accession'].values
expression_filtered = expression_df[sample_ids]

print(f"\nFiltered expression matrix: {expression_filtered.shape}")

# ============================================================================
# 5. Gene ID Mapping (Probe ID to Gene Symbol)
# ============================================================================

print("\n" + "="*80)
print("GENE ID MAPPING")
print("="*80)

# For Affymetrix arrays, probe IDs need to be mapped to gene symbols
# This typically requires annotation files from Bioconductor
# For now, we'll use probe IDs directly

print(f"\nProbe IDs: {len(expression_filtered)}")
print(f"Note: For full analysis, probe IDs should be mapped to gene symbols")
print(f"      using platform annotation (GPL570 for Affymetrix U133 Plus 2.0)")

# ============================================================================
# 6. Save Processed Data
# ============================================================================

print("\n" + "="*80)
print("SAVING PROCESSED DATA")
print("="*80)

# Save expression data
expression_filtered.to_csv(PROCESSED_DIR / 'expression_matrix.csv')
print(f"✓ Saved expression matrix: {PROCESSED_DIR / 'expression_matrix.csv'}")

# Save metadata
valid_samples.to_csv(PROCESSED_DIR / 'sample_metadata.csv', index=False)
print(f"✓ Saved sample metadata: {PROCESSED_DIR / 'sample_metadata.csv'}")

# Create labels file
labels_df = pd.DataFrame({
    'sample_id': valid_samples['Sample_geo_accession'].values,
    'diabetes_status': valid_samples['diabetes_status'].values,
    'label': (valid_samples['diabetes_status'] == 'Diabetes').astype(int).values
})
labels_df.to_csv(PROCESSED_DIR / 'labels.csv', index=False)
print(f"✓ Saved labels: {PROCESSED_DIR / 'labels.csv'}")

# ============================================================================
# 7. Summary Statistics
# ============================================================================

print("\n" + "="*80)
print("PROCESSING SUMMARY")
print("="*80)

summary = f"""
PRIMARY DATASET: {primary_dataset}
{'='*60}

Total Datasets Processed: {len(all_datasets)}
{chr(10).join([f'  - {geo_id}: {len(data["samples"])} samples' for geo_id, data in all_datasets.items()])}

SELECTED DATASET STATISTICS:
  Total Probes: {expression_filtered.shape[0]:,}
  Total Samples: {expression_filtered.shape[1]}

  Class Distribution:
    Diabetes: {sum(labels_df['label'] == 1)}
    Control:  {sum(labels_df['label'] == 0)}

  Platform: {dataset['metadata'].get('Series_platform_id', 'Unknown')}

OUTPUT FILES:
  {PROCESSED_DIR / 'expression_matrix.csv'}
  {PROCESSED_DIR / 'sample_metadata.csv'}
  {PROCESSED_DIR / 'labels.csv'}

NEXT STEPS:
  1. Run preprocessing pipeline (scripts/02_preprocessing.py)
  2. Modify to use PROCESSED_DIR instead of raw data
  3. Continue with feature selection and modeling

NOTE: This data uses probe IDs instead of gene symbols.
      For optimal results, map probe IDs to gene symbols using
      platform annotation files (GPL570).
"""

print(summary)

# Save summary
with open(PROCESSED_DIR / 'processing_summary.txt', 'w') as f:
    f.write(summary)

print(f"\n✓ Saved summary: {PROCESSED_DIR / 'processing_summary.txt'}")
print("="*80)
print("Processing complete!")
print("="*80)
