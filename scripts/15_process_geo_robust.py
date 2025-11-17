"""
Production-Grade GEO Data Processing Pipeline with Rich Visualizations

Handles all edge cases and provides comprehensive visual diagnostics.

Author: Claude
Date: 2025-11-17
Version: 2.0 (Complete Rewrite)
"""

import pandas as pd
import numpy as np
import gzip
import re
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_style('whitegrid')

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
GEO_DIR = BASE_DIR / 'data' / 'geo_datasets'
OUTPUT_DIR = BASE_DIR / 'data' / 'real_geo_processed'
FIG_DIR = OUTPUT_DIR / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PRIORITY = {
    'GSE164416': 1,
    'GSE76894': 2,
    'GSE25724': 3,
    'GSE86469': 4,
    'GSE81608': 5,
    'GSE86468': 6
}

print("="*80)
print("🔬 PRODUCTION-GRADE GEO DATA PROCESSING PIPELINE v2.0")
print("="*80)

# ============================================================================
# IMPROVED PARSER with Better Error Handling
# ============================================================================

def parse_series_matrix_robust(filepath):
    """
    Robust parser with detailed diagnostics
    """
    print(f"\n📄 Parsing: {filepath.name}")

    series_meta = {}
    sample_meta_raw = defaultdict(list)
    expression_lines = []
    in_table = False
    line_count = 0

    # Open file
    try:
        if filepath.suffix == '.gz':
            f = gzip.open(filepath, 'rt', encoding='utf-8', errors='replace')
        else:
            f = open(filepath, 'r', encoding='utf-8', errors='replace')
    except Exception as e:
        print(f"  ❌ Cannot open file: {e}")
        return None, None, None

    try:
        for line in f:
            line_count += 1
            line = line.strip()

            if not line or line.startswith('"'):
                continue

            # Series metadata
            if line.startswith('!Series_'):
                try:
                    match = re.match(r'!Series_(\w+)\s+"?(.+?)"?\s*$', line)
                    if match:
                        key, value = match.groups()
                        series_meta[key] = value.strip('"')
                except:
                    pass

            # Sample metadata - IMPROVED
            elif line.startswith('!Sample_'):
                try:
                    # More flexible parsing
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        header = parts[0].replace('!Sample_', '')
                        values_str = parts[1]
                        # Split by tab and clean quotes
                        values = [v.strip(' "') for v in values_str.split('\t')]
                        sample_meta_raw[header] = values
                except Exception as e:
                    print(f"  ⚠️ Line {line_count}: Failed to parse sample metadata: {e}")
                    continue

            # Expression table
            elif line == '!series_matrix_table_begin':
                in_table = True
                continue
            elif line == '!series_matrix_table_end':
                in_table = False
                break
            elif in_table:
                expression_lines.append(line)

    finally:
        f.close()

    # Build sample DataFrame
    if sample_meta_raw:
        try:
            sample_df = pd.DataFrame(sample_meta_raw)
            print(f"  ✓ Metadata: {len(sample_df)} samples × {len(sample_df.columns)} fields")
        except Exception as e:
            print(f"  ❌ Failed to create sample DataFrame: {e}")
            sample_df = pd.DataFrame()
    else:
        print(f"  ⚠️ No sample metadata found")
        sample_df = pd.DataFrame()

    # Build expression DataFrame - IMPROVED
    if expression_lines:
        try:
            # Parse header
            header_line = expression_lines[0]
            header = [h.strip(' "') for h in header_line.split('\t')]

            if len(header) < 2:
                print(f"  ⚠️ Invalid header: {header}")
                return series_meta, sample_df, pd.DataFrame()

            # Parse data
            data_rows = []
            for i, line in enumerate(expression_lines[1:], start=1):
                try:
                    values = [v.strip(' "') for v in line.split('\t')]
                    if len(values) == len(header):
                        data_rows.append(values)
                    elif i < 10:  # Only warn for first few rows
                        print(f"  ⚠️ Row {i}: Expected {len(header)} values, got {len(values)}")
                except:
                    pass

            if data_rows:
                expr_df = pd.DataFrame(data_rows, columns=header)

                # Set index
                if 'ID_REF' in expr_df.columns:
                    expr_df = expr_df.set_index('ID_REF')
                else:
                    # Use first column as index
                    expr_df = expr_df.set_index(expr_df.columns[0])

                # Convert to numeric
                for col in expr_df.columns:
                    expr_df[col] = pd.to_numeric(expr_df[col], errors='coerce')

                # Check if we actually have data
                non_null_counts = expr_df.notna().sum().sum()
                if non_null_counts == 0:
                    print(f"  ❌ All expression values are null!")
                    expr_df = pd.DataFrame()
                else:
                    print(f"  ✓ Expression: {expr_df.shape[0]} probes × {expr_df.shape[1]} samples")
                    print(f"    Non-null values: {non_null_counts:,} ({non_null_counts/(expr_df.shape[0]*expr_df.shape[1])*100:.1f}%)")
            else:
                print(f"  ❌ No valid expression data rows")
                expr_df = pd.DataFrame()

        except Exception as e:
            print(f"  ❌ Failed to parse expression data: {e}")
            import traceback
            traceback.print_exc()
            expr_df = pd.DataFrame()
    else:
        print(f"  ⚠️ No expression table found (may use supplementary files)")
        expr_df = pd.DataFrame()

    return series_meta, sample_df, expr_df


# ============================================================================
# IMPROVED LABEL EXTRACTION
# ============================================================================

def extract_labels_improved(sample_df, geo_id):
    """
    Enhanced label extraction with better pattern matching
    """
    print(f"  🏷️ Extracting labels...")

    if len(sample_df) == 0:
        print(f"    ❌ No samples to label")
        return sample_df

    # Check available columns
    print(f"    Available columns: {list(sample_df.columns[:5])}...")

    # Diabetes patterns - EXPANDED
    diabetes_patterns = [
        r'\bt2d\b', r'\bt2dm\b', r'\bdiabetes\b', r'\bdiabetic\b',
        r'type\s*2', r'type\s*ii', r'type2', r'typeii',
        r'diabetes mellitus', r'dm\b', r'diabete'
    ]

    # Control patterns - EXPANDED
    control_patterns = [
        r'\bcontrol\b', r'\bnon-diabetic\b', r'\bnondiabetic\b',
        r'\bnd\b', r'\bnormal\b', r'\bhealthy\b',
        r'non\s*diabetic', r'non-dm', r'no diabetes',
        r'ctrl\b', r'normal glucose'
    ]

    labels = []
    debug_info = []

    for idx, row in sample_df.iterrows():
        # Collect all text from sample
        text_parts = []

        # Try different column name patterns
        for col in sample_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['characteristics', 'title', 'description', 'source', 'type']):
                val = row[col]
                if pd.notna(val):
                    text_parts.append(str(val))

        combined = ' '.join(text_parts).lower()

        # Check patterns
        has_diabetes = any(re.search(p, combined, re.IGNORECASE) for p in diabetes_patterns)
        has_control = any(re.search(p, combined, re.IGNORECASE) for p in control_patterns)

        # Decision logic
        if has_diabetes and not has_control:
            label = 'Diabetes'
        elif has_control and not has_diabetes:
            label = 'Control'
        elif 'igt' in combined or 'impaired glucose' in combined:
            label = 'IGT'
        elif 't3c' in combined or 't3d' in combined:
            label = 'T3cD'
        else:
            label = 'Unknown'

        labels.append(label)

        # Store debug info for first 5
        if idx < 5:
            debug_info.append({
                'index': idx,
                'text_snippet': combined[:100],
                'label': label
            })

    sample_df['diabetes_status'] = labels

    # Print label distribution
    counts = sample_df['diabetes_status'].value_counts()
    print(f"    Label distribution:")
    for status, count in counts.items():
        pct = count / len(sample_df) * 100
        print(f"      {status}: {count} ({pct:.1f}%)")

    # Show debug info
    if debug_info and all(d['label'] == 'Unknown' for d in debug_info):
        print(f"    ⚠️ All samples labeled as Unknown. Sample text:")
        for d in debug_info[:3]:
            print(f"      #{d['index']}: {d['text_snippet'][:80]}...")

    return sample_df


# ============================================================================
# STEP 1: Load and Parse All Datasets
# ============================================================================

print(f"\n{'='*80}")
print("📥 STEP 1: LOADING GEO DATASETS")
print(f"{'='*80}")

geo_files = sorted(GEO_DIR.glob('GSE*_series_matrix.txt*'))

if not geo_files:
    print(f"\n❌ ERROR: No GEO files found in {GEO_DIR}")
    print(f"Expected: GSE164416_series_matrix.txt or .txt.gz")
    exit(1)

print(f"\nFound {len(geo_files)} files:")
for f in geo_files:
    geo_id = f.stem.split('_')[0]
    priority = DATASET_PRIORITY.get(geo_id, 99)
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"  [{priority}] {f.name} ({size_mb:.1f} MB)")

# Parse all
datasets = {}
parsing_summary = []

for filepath in geo_files:
    geo_id = filepath.stem.split('_')[0]

    series_meta, sample_df, expr_df = parse_series_matrix_robust(filepath)

    # Skip if completely failed
    if sample_df is None:
        parsing_summary.append({
            'geo_id': geo_id,
            'status': '❌ Failed',
            'samples': 0,
            'probes': 0,
            'valid_samples': 0
        })
        continue

    # Extract labels
    if len(sample_df) > 0:
        sample_df = extract_labels_improved(sample_df, geo_id)

    # Calculate valid samples
    if 'diabetes_status' in sample_df.columns:
        valid_count = sample_df['diabetes_status'].isin(['Diabetes', 'Control']).sum()
    else:
        valid_count = 0

    datasets[geo_id] = {
        'series_meta': series_meta,
        'sample_df': sample_df,
        'expr_df': expr_df,
        'priority': DATASET_PRIORITY.get(geo_id, 99),
        'valid_samples': valid_count
    }

    parsing_summary.append({
        'geo_id': geo_id,
        'status': '✓ OK' if valid_count > 0 else '⚠️ No valid',
        'samples': len(sample_df),
        'probes': len(expr_df) if not expr_df.empty else 0,
        'valid_samples': valid_count
    })

    # Save raw
    if not expr_df.empty:
        expr_df.to_csv(OUTPUT_DIR / f'{geo_id}_raw_expression.csv')
    if len(sample_df) > 0:
        sample_df.to_csv(OUTPUT_DIR / f'{geo_id}_raw_metadata.csv', index=False)

# Create parsing summary visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

summary_df = pd.DataFrame(parsing_summary)
print(f"\n{'='*80}")
print("📊 PARSING SUMMARY")
print(f"{'='*80}")
print(summary_df.to_string(index=False))

# Plot 1: Valid samples by dataset
ax1.barh(summary_df['geo_id'], summary_df['valid_samples'],
         color=['green' if v > 0 else 'red' for v in summary_df['valid_samples']])
ax1.set_xlabel('Valid Samples (Diabetes + Control)')
ax1.set_title('Valid Samples by Dataset', fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

for i, (geo, count) in enumerate(zip(summary_df['geo_id'], summary_df['valid_samples'])):
    ax1.text(count + 1, i, str(count), va='center', fontweight='bold')

# Plot 2: Probes by dataset
ax2.barh(summary_df['geo_id'], summary_df['probes'],
         color=['blue' if p > 0 else 'gray' for p in summary_df['probes']])
ax2.set_xlabel('Number of Probes')
ax2.set_title('Expression Probes by Dataset', fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()
plt.savefig(FIG_DIR / '01_parsing_summary.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {FIG_DIR / '01_parsing_summary.png'}")
plt.close()

# Check if we have any valid data
total_valid = sum(d['valid_samples'] for d in datasets.values())
if total_valid == 0:
    print(f"\n❌ CRITICAL: No valid diabetes/control samples found in any dataset!")
    print(f"\n💡 TROUBLESHOOTING:")
    print(f"   1. Check sample metadata in *_raw_metadata.csv files")
    print(f"   2. Look for 'characteristics' or 'title' columns")
    print(f"   3. Verify diabetes/control keywords are present")
    print(f"   4. May need to manually annotate samples")
    exit(1)

print(f"\n✓ Total valid samples across all datasets: {total_valid}")

# ============================================================================
# STEP 2: Select Best Dataset
# ============================================================================

print(f"\n{'='*80}")
print("🎯 STEP 2: SELECTING BEST DATASET")
print(f"{'='*80}")

# Filter to datasets with valid samples and expression data
valid_datasets = {
    geo_id: data for geo_id, data in datasets.items()
    if data['valid_samples'] > 0 and not data['expr_df'].empty
}

if not valid_datasets:
    print(f"\n❌ No datasets have both valid samples AND expression data!")
    print(f"\nDatasets with valid samples but no expression:")
    for geo_id, data in datasets.items():
        if data['valid_samples'] > 0:
            print(f"  - {geo_id}: {data['valid_samples']} valid samples (expression data missing)")
    exit(1)

# Sort by priority and valid sample count
sorted_datasets = sorted(
    valid_datasets.items(),
    key=lambda x: (x[1]['priority'], -x[1]['valid_samples'])
)

primary_id, primary_data = sorted_datasets[0]

print(f"\n🏆 Selected: {primary_id}")
print(f"   Priority: {primary_data['priority']}")
print(f"   Valid samples: {primary_data['valid_samples']}")
print(f"   Expression probes: {len(primary_data['expr_df'])}")
print(f"   Platform: {primary_data['series_meta'].get('platform_id', 'Unknown')}")

# Filter to valid samples
sample_df = primary_data['sample_df']
expr_df = primary_data['expr_df']

valid_mask = sample_df['diabetes_status'].isin(['Diabetes', 'Control'])
sample_df_filtered = sample_df[valid_mask].copy()

diabetes_count = (sample_df_filtered['diabetes_status'] == 'Diabetes').sum()
control_count = (sample_df_filtered['diabetes_status'] == 'Control').sum()

print(f"\n📊 Class Distribution:")
print(f"   Diabetes: {diabetes_count}")
print(f"   Control: {control_count}")
print(f"   Total: {len(sample_df_filtered)}")
print(f"   Balance: {min(diabetes_count, control_count) / max(diabetes_count, control_count):.2f}")

# Get sample IDs
if 'geo_accession' in sample_df_filtered.columns:
    sample_ids = sample_df_filtered['geo_accession'].values
else:
    # Use first column
    sample_ids = sample_df_filtered.iloc[:, 0].values

# Filter expression
expr_filtered = expr_df[sample_ids]

print(f"\n✓ Filtered expression matrix: {expr_filtered.shape}")

# ============================================================================
# VISUALIZATION: Class Distribution
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Class distribution pie chart
ax = axes[0, 0]
sizes = [diabetes_count, control_count]
colors = ['#e74c3c', '#3498db']
explode = (0.05, 0.05)

ax.pie(sizes, explode=explode, labels=['Diabetes', 'Control'],
       colors=colors, autopct='%1.1f%%', startangle=90)
ax.set_title(f'Class Distribution\n{primary_id}', fontweight='bold')

# Plot 2: All datasets comparison
ax = axes[0, 1]
dataset_names = []
diabetes_counts = []
control_counts = []

for geo_id, data in datasets.items():
    if data['valid_samples'] > 0:
        sdf = data['sample_df']
        diabetes_counts.append((sdf['diabetes_status'] == 'Diabetes').sum())
        control_counts.append((sdf['diabetes_status'] == 'Control').sum())
        dataset_names.append(geo_id)

x = np.arange(len(dataset_names))
width = 0.35

ax.bar(x - width/2, diabetes_counts, width, label='Diabetes', color='#e74c3c', alpha=0.8)
ax.bar(x + width/2, control_counts, width, label='Control', color='#3498db', alpha=0.8)
ax.set_xlabel('Dataset')
ax.set_ylabel('Sample Count')
ax.set_title('Sample Distribution Across Datasets', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(dataset_names, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 3: Expression data heatmap (sample)
ax = axes[1, 0]
# Take first 50 probes, all samples
sample_expr = expr_filtered.iloc[:50, :].values
im = ax.imshow(sample_expr, aspect='auto', cmap='RdBu_r', interpolation='nearest')
ax.set_xlabel('Samples')
ax.set_ylabel('Probes (first 50)')
ax.set_title('Raw Expression Heatmap', fontweight='bold')
plt.colorbar(im, ax=ax, label='Expression')

# Plot 4: Missing data pattern
ax = axes[1, 1]
missing_pct = expr_filtered.isna().sum(axis=1) / expr_filtered.shape[1] * 100
ax.hist(missing_pct, bins=50, color='orange', alpha=0.7, edgecolor='black')
ax.set_xlabel('Missing Data (%)')
ax.set_ylabel('Number of Probes')
ax.set_title(f'Missing Data Distribution\n{(missing_pct > 0).sum()} probes with missing values',
             fontweight='bold')
ax.axvline(20, color='red', linestyle='--', linewidth=2, label='20% threshold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / '02_data_overview.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {FIG_DIR / '02_data_overview.png'}")
plt.close()

# ============================================================================
# STEP 3: Quality Control with Visualization
# ============================================================================

print(f"\n{'='*80}")
print("🔍 STEP 3: QUALITY CONTROL")
print(f"{'='*80}")

gene_expr_qc = expr_filtered.copy()

print(f"\nBefore QC:")
print(f"  Probes: {gene_expr_qc.shape[0]}")
print(f"  Samples: {gene_expr_qc.shape[1]}")

# QC Step 1: Remove high-missing probes
print(f"\n1️⃣ Removing high-missing probes (>20%)...")
missing_pct = gene_expr_qc.isna().sum(axis=1) / gene_expr_qc.shape[1]
keep_probes = missing_pct < 0.2

removed_count = (~keep_probes).sum()
gene_expr_qc = gene_expr_qc[keep_probes].copy()
print(f"   Removed: {removed_count} probes")
print(f"   Remaining: {gene_expr_qc.shape[0]} probes")

# QC Step 2: Impute remaining missing
print(f"\n2️⃣ Imputing remaining missing values...")
before_impute = gene_expr_qc.isna().sum().sum()
for col in gene_expr_qc.columns:
    if gene_expr_qc[col].isna().any():
        median_val = gene_expr_qc[col].median()
        gene_expr_qc[col].fillna(median_val, inplace=True)
after_impute = gene_expr_qc.isna().sum().sum()
print(f"   Imputed: {before_impute} missing values")
print(f"   Remaining: {after_impute}")

# QC Step 3: Remove low-variance probes
print(f"\n3️⃣ Removing low-variance probes (bottom 10%)...")
variances = gene_expr_qc.var(axis=1)

if len(variances) > 0:
    var_threshold = np.percentile(variances, 10)
    high_var = variances > var_threshold

    removed_count = (~high_var).sum()
    gene_expr_qc = gene_expr_qc[high_var].copy()
    print(f"   Variance threshold: {var_threshold:.4f}")
    print(f"   Removed: {removed_count} probes")
    print(f"   Remaining: {gene_expr_qc.shape[0]} probes")
else:
    print(f"   ⚠️ No probes to filter")

print(f"\nAfter QC:")
print(f"  Final probes: {gene_expr_qc.shape[0]}")
print(f"  Reduction: {(1 - gene_expr_qc.shape[0]/expr_filtered.shape[0])*100:.1f}%")

# QC Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Missing data before/after
ax = axes[0, 0]
missing_before = expr_filtered.isna().sum(axis=1) / expr_filtered.shape[1] * 100
missing_after = gene_expr_qc.isna().sum(axis=1) / gene_expr_qc.shape[1] * 100

ax.hist(missing_before, bins=50, alpha=0.5, label=f'Before ({len(missing_before)} probes)',
        color='red', edgecolor='black')
ax.hist(missing_after, bins=50, alpha=0.5, label=f'After ({len(missing_after)} probes)',
        color='green', edgecolor='black')
ax.axvline(20, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Missing Data (%)')
ax.set_ylabel('Frequency')
ax.set_title('Missing Data: Before vs After QC', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: Variance distribution
ax = axes[0, 1]
all_var = expr_filtered.var(axis=1)
qc_var = gene_expr_qc.var(axis=1)

ax.hist(np.log10(all_var + 1e-10), bins=50, alpha=0.5, label='Before QC',
        color='red', edgecolor='black')
ax.hist(np.log10(qc_var + 1e-10), bins=50, alpha=0.5, label='After QC',
        color='green', edgecolor='black')
ax.set_xlabel('log10(Variance)')
ax.set_ylabel('Frequency')
ax.set_title('Variance Distribution', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 3: Sample correlation heatmap (before)
ax = axes[0, 2]
sample_corr_before = expr_filtered.corr()
im = ax.imshow(sample_corr_before, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
ax.set_title('Sample Correlation (Before QC)', fontweight='bold')
plt.colorbar(im, ax=ax, label='Correlation')

# Plot 4: Sample correlation heatmap (after)
ax = axes[1, 0]
sample_corr_after = gene_expr_qc.corr()
im = ax.imshow(sample_corr_after, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
ax.set_title('Sample Correlation (After QC)', fontweight='bold')
plt.colorbar(im, ax=ax, label='Correlation')

# Plot 5: Expression distribution (before/after)
ax = axes[1, 1]
ax.boxplot([expr_filtered.values.flatten()[:10000], gene_expr_qc.values.flatten()[:10000]],
           labels=['Before', 'After'], patch_artist=True,
           boxprops=dict(facecolor='lightblue', alpha=0.7))
ax.set_ylabel('Expression Value')
ax.set_title('Expression Distribution\n(Sample of 10K values)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Plot 6: QC summary stats
ax = axes[1, 2]
ax.axis('off')

qc_summary = f"""
QUALITY CONTROL SUMMARY
{'='*40}

Initial Probes: {expr_filtered.shape[0]:,}
Final Probes: {gene_expr_qc.shape[0]:,}
Removed: {expr_filtered.shape[0] - gene_expr_qc.shape[0]:,} ({(1-gene_expr_qc.shape[0]/expr_filtered.shape[0])*100:.1f}%)

QC STEPS:
1. High Missing (>20%): {removed_count} probes
2. Imputation: {before_impute} values
3. Low Variance (bottom 10%): {removed_count} probes

FINAL DATA QUALITY:
• Missing values: {gene_expr_qc.isna().sum().sum()}
• Mean variance: {gene_expr_qc.var(axis=1).mean():.4f}
• Samples: {gene_expr_qc.shape[1]}
• Features: {gene_expr_qc.shape[0]:,}
"""

ax.text(0.1, 0.95, qc_summary, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(FIG_DIR / '03_quality_control.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {FIG_DIR / '03_quality_control.png'}")
plt.close()

# ============================================================================
# STEP 4: Normalization with Visualization
# ============================================================================

print(f"\n{'='*80}")
print("📊 STEP 4: NORMALIZATION")
print(f"{'='*80}")

gene_expr_norm = gene_expr_qc.copy()

# Check if log transform needed
max_val = gene_expr_norm.max().max()
print(f"\nMax expression value: {max_val:.2f}")

if max_val > 20:
    print(f"  Applying log2 transformation...")
    gene_expr_norm = np.log2(gene_expr_norm + 1)
    print(f"  ✓ Transformed. New max: {gene_expr_norm.max().max():.2f}")
else:
    print(f"  Data appears log-transformed already")

# Z-score normalization
print(f"\nApplying z-score normalization...")
scaler = StandardScaler()
gene_expr_scaled = scaler.fit_transform(gene_expr_norm.T).T  # Normalize per gene

gene_expr_scaled = pd.DataFrame(
    gene_expr_scaled,
    index=gene_expr_norm.index,
    columns=gene_expr_norm.columns
)

print(f"  Mean: {gene_expr_scaled.mean().mean():.6f}")
print(f"  Std: {gene_expr_scaled.std().mean():.6f}")

# Normalization visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Before normalization - histogram
ax = axes[0, 0]
ax.hist(gene_expr_norm.values.flatten()[:50000], bins=100, alpha=0.7, color='blue', edgecolor='black')
ax.set_xlabel('Expression Value')
ax.set_ylabel('Frequency')
ax.set_title('Distribution Before Normalization', fontweight='bold')
ax.axvline(gene_expr_norm.values.flatten().mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: After normalization - histogram
ax = axes[0, 1]
ax.hist(gene_expr_scaled.values.flatten()[:50000], bins=100, alpha=0.7, color='green', edgecolor='black')
ax.set_xlabel('Normalized Expression')
ax.set_ylabel('Frequency')
ax.set_title('Distribution After Z-score Normalization', fontweight='bold')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Mean=0')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 3: QQ plot
ax = axes[0, 2]
from scipy import stats
sample_data = gene_expr_scaled.values.flatten()[:10000]
stats.probplot(sample_data, dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
ax.grid(alpha=0.3)

# Plot 4: Boxplot by sample (before)
ax = axes[1, 0]
sample_subset = gene_expr_norm.iloc[:, ::max(1, gene_expr_norm.shape[1]//10)]  # Max 10 samples
ax.boxplot(sample_subset.values, patch_artist=True,
           boxprops=dict(facecolor='lightblue', alpha=0.7))
ax.set_xlabel('Sample (subset)')
ax.set_ylabel('Expression')
ax.set_title('Per-Sample Distribution (Before)', fontweight='bold')
ax.set_xticklabels([])
ax.grid(axis='y', alpha=0.3)

# Plot 5: Boxplot by sample (after)
ax = axes[1, 1]
sample_subset_norm = gene_expr_scaled.iloc[:, ::max(1, gene_expr_scaled.shape[1]//10)]
ax.boxplot(sample_subset_norm.values, patch_artist=True,
           boxprops=dict(facecolor='lightgreen', alpha=0.7))
ax.set_xlabel('Sample (subset)')
ax.set_ylabel('Normalized Expression')
ax.set_title('Per-Sample Distribution (After)', fontweight='bold')
ax.set_xticklabels([])
ax.grid(axis='y', alpha=0.3)

# Plot 6: PCA before vs after
ax = axes[1, 2]
pca_before = PCA(n_components=2).fit_transform(gene_expr_norm.T)
pca_after = PCA(n_components=2).fit_transform(gene_expr_scaled.T)

labels = (sample_df_filtered['diabetes_status'] == 'Diabetes').astype(int).values
colors = ['red' if l == 1 else 'blue' for l in labels]

ax.scatter(pca_before[:, 0], pca_before[:, 1], c=colors, alpha=0.3, s=100, label='Before', marker='o')
ax.scatter(pca_after[:, 0], pca_after[:, 1], c=colors, alpha=0.7, s=100, label='After', marker='s')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA: Before vs After Normalization', fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.5, label='Diabetes'),
    Patch(facecolor='blue', alpha=0.5, label='Control'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Before'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='After')
]
ax.legend(handles=legend_elements, loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / '04_normalization.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {FIG_DIR / '04_normalization.png'}")
plt.close()

# ============================================================================
# STEP 5: Create Labels and Split
# ============================================================================

print(f"\n{'='*80}")
print("✂️ STEP 5: TRAIN/VAL/TEST SPLIT")
print(f"{'='*80}")

# Transpose to samples × features
X = gene_expr_scaled.T.values
y = (sample_df_filtered['diabetes_status'] == 'Diabetes').astype(int).values
feature_names = gene_expr_scaled.index.values

print(f"\nFinal dataset:")
print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Diabetes: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
print(f"  Control: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")

# Split: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nSplit distribution:")
print(f"  Train: {len(y_train)} samples ({sum(y_train)} diabetes, {len(y_train)-sum(y_train)} control)")
print(f"  Val:   {len(y_val)} samples ({sum(y_val)} diabetes, {len(y_val)-sum(y_val)} control)")
print(f"  Test:  {len(y_test)} samples ({sum(y_test)} diabetes, {len(y_test)-sum(y_test)} control)")

# Split visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (split_name, y_split) in zip(axes, [('Train', y_train), ('Val', y_val), ('Test', y_test)]):
    counts = [sum(y_split == 0), sum(y_split == 1)]
    colors = ['#3498db', '#e74c3c']
    ax.pie(counts, labels=['Control', 'Diabetes'], colors=colors,
           autopct='%1.1f%%', startangle=90)
    ax.set_title(f'{split_name} Set\n({len(y_split)} samples)', fontweight='bold')

plt.tight_layout()
plt.savefig(FIG_DIR / '05_data_split.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {FIG_DIR / '05_data_split.png'}")
plt.close()

# ============================================================================
# STEP 6: Save Processed Data
# ============================================================================

print(f"\n{'='*80}")
print("💾 STEP 6: SAVING PROCESSED DATA")
print(f"{'='*80}")

# Convert to DataFrames
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_val_df = pd.DataFrame(X_val, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

y_train_df = pd.DataFrame({'label': y_train})
y_val_df = pd.DataFrame({'label': y_val})
y_test_df = pd.DataFrame({'label': y_test})

gene_names_df = pd.DataFrame({'gene': feature_names})

# Save
X_train_df.to_csv(OUTPUT_DIR / 'X_train.csv')
X_val_df.to_csv(OUTPUT_DIR / 'X_val.csv')
X_test_df.to_csv(OUTPUT_DIR / 'X_test.csv')

y_train_df.to_csv(OUTPUT_DIR / 'y_train.csv', index=False)
y_val_df.to_csv(OUTPUT_DIR / 'y_val.csv', index=False)
y_test_df.to_csv(OUTPUT_DIR / 'y_test.csv', index=False)

gene_names_df.to_csv(OUTPUT_DIR / 'gene_names.csv', index=False)

print(f"\n✓ Saved to {OUTPUT_DIR}:")
print(f"  X_train.csv: {X_train_df.shape}")
print(f"  X_val.csv: {X_val_df.shape}")
print(f"  X_test.csv: {X_test_df.shape}")
print(f"  y_*.csv files")
print(f"  gene_names.csv: {len(feature_names)} features")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("✅ PROCESSING COMPLETE!")
print(f"{'='*80}")

summary_text = f"""
GEO DATA PROCESSING SUMMARY
{'='*80}

PRIMARY DATASET: {primary_id}
Platform: {primary_data['series_meta'].get('platform_id', 'Unknown')}

SAMPLE STATISTICS:
  Total samples: {len(sample_df)}
  Valid samples: {len(sample_df_filtered)}
  Diabetes: {diabetes_count} ({diabetes_count/len(sample_df_filtered)*100:.1f}%)
  Control: {control_count} ({control_count/len(sample_df_filtered)*100:.1f}%)

FEATURE STATISTICS:
  Original probes: {len(expr_filtered):,}
  After QC: {len(gene_expr_qc):,} ({len(gene_expr_qc)/len(expr_filtered)*100:.1f}%)
  Final features: {X.shape[1]:,}

DATA SPLIT:
  Train: {len(y_train)} ({sum(y_train)}D, {len(y_train)-sum(y_train)}C)
  Val:   {len(y_val)} ({sum(y_val)}D, {len(y_val)-sum(y_val)}C)
  Test:  {len(y_test)} ({sum(y_test)}D, {len(y_test)-sum(y_test)}C)

OUTPUT FILES:
  {OUTPUT_DIR}/X_train.csv, X_val.csv, X_test.csv
  {OUTPUT_DIR}/y_train.csv, y_val.csv, y_test.csv
  {OUTPUT_DIR}/gene_names.csv

VISUALIZATIONS:
  {FIG_DIR}/01_parsing_summary.png
  {FIG_DIR}/02_data_overview.png
  {FIG_DIR}/03_quality_control.png
  {FIG_DIR}/04_normalization.png
  {FIG_DIR}/05_data_split.png

NEXT STEPS:
  1. Feature selection: python scripts/05_feature_selection.py
     (Update data path to {OUTPUT_DIR})

  2. Model training: python scripts/06_ml_models.py

  3. Clinical optimization: python scripts/10_clinical_grade_optimization.py

IMPORTANT NOTES:
  • Using probe IDs as features
  • For better results, map to gene symbols using platform annotation
  • Consider combining multiple datasets for larger sample size
  • Validate results on independent test set
"""

print(summary_text)

with open(OUTPUT_DIR / 'PROCESSING_SUMMARY.txt', 'w') as f:
    f.write(summary_text)

print(f"\n✓ Summary saved: {OUTPUT_DIR / 'PROCESSING_SUMMARY.txt'}")
print(f"\n🎉 All done! Ready for ML pipeline.")
