"""
Step 1: Data Collection from NCBI GEO (Alternative Method)
Download diabetes-related RNA expression datasets using direct HTTPS access
"""

import pandas as pd
import numpy as np
import requests
import gzip
import io
import os
from pathlib import Path
import time

# Create data directory
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_geo_matrix(geo_id):
    """
    Download GEO series matrix file (processed data) via HTTPS
    This is more reliable than SOFT files
    """
    print(f"\n{'='*80}")
    print(f"Downloading {geo_id} series matrix")
    print(f"{'='*80}")

    try:
        # Construct URL for series matrix file
        series_stub = geo_id[:-3] + "nnn"  # e.g., GSE38642 -> GSE38nnn
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{series_stub}/{geo_id}/matrix/{geo_id}_series_matrix.txt.gz"

        print(f"📥 Downloading from: {url}")

        # Download with timeout and retry
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=60)
                if response.status_code == 200:
                    break
                elif attempt < 2:
                    print(f"   Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)
            except requests.exceptions.RequestException as e:
                if attempt < 2:
                    print(f"   Connection error, retrying... ({e})")
                    time.sleep(2)
                else:
                    raise

        if response.status_code != 200:
            print(f"❌ Failed to download: HTTP {response.status_code}")
            return None

        # Decompress the gzipped content
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
            content = gz.read().decode('utf-8')

        # Save raw file
        raw_file = DATA_DIR / f"{geo_id}_series_matrix.txt"
        with open(raw_file, 'w') as f:
            f.write(content)

        print(f"✅ Downloaded and saved to {raw_file.name}")

        # Parse the file
        lines = content.split('\n')

        # Extract metadata
        metadata = {}
        sample_info = {}
        data_start_idx = None

        for idx, line in enumerate(lines):
            if line.startswith('!Series_'):
                key = line.split('\t')[0].replace('!Series_', '')
                value = '\t'.join(line.split('\t')[1:]).strip('"')
                metadata[key] = value
            elif line.startswith('!Sample_'):
                parts = line.split('\t')
                key = parts[0].replace('!Sample_', '')
                values = [v.strip('"') for v in parts[1:]]
                if key not in sample_info:
                    sample_info[key] = []
                sample_info[key] = values
            elif line.startswith('!series_matrix_table_begin'):
                data_start_idx = idx + 1
                break

        # Print metadata
        print(f"\n📊 Dataset Information:")
        print(f"Title: {metadata.get('title', 'N/A')}")
        print(f"Summary: {metadata.get('summary', 'N/A')[:200]}...")
        print(f"Organism: {metadata.get('organism', 'N/A')}")
        print(f"Platform: {metadata.get('platform_id', 'N/A')}")

        # Extract sample metadata
        if sample_info:
            sample_df = pd.DataFrame(sample_info)
            print(f"\n📋 Number of samples: {len(sample_df.columns) if not sample_df.empty else 0}")

            # Transpose so samples are rows
            if not sample_df.empty:
                sample_df = sample_df.T
                sample_df.to_csv(DATA_DIR / f"{geo_id}_sample_info.csv")
                print(f"\n🔍 Sample characteristics preview:")
                print(sample_df.head())

        # Extract expression data
        if data_start_idx:
            expr_lines = []
            for line in lines[data_start_idx:]:
                if line.startswith('!series_matrix_table_end'):
                    break
                if line.strip():
                    expr_lines.append(line)

            if expr_lines:
                # Parse expression data
                from io import StringIO
                expr_data = pd.read_csv(StringIO('\n'.join(expr_lines)), sep='\t', index_col=0)

                print(f"\n🧬 Expression Matrix:")
                print(f"Shape: {expr_data.shape}")
                print(f"Features (genes/probes): {expr_data.shape[0]}")
                print(f"Samples: {expr_data.shape[1]}")

                # Convert to numeric
                expr_data = expr_data.apply(pd.to_numeric, errors='coerce')

                # Basic statistics
                print(f"\n📈 Expression Statistics:")
                print(f"Min value: {expr_data.min().min():.2f}")
                print(f"Max value: {expr_data.max().max():.2f}")
                print(f"Mean value: {expr_data.mean().mean():.2f}")
                print(f"Missing values: {expr_data.isna().sum().sum()}")

                # Save expression matrix
                expr_data.to_csv(DATA_DIR / f"{geo_id}_expression.csv")
                print(f"✅ Expression matrix saved")

                return {
                    'geo_id': geo_id,
                    'title': metadata.get('title', 'N/A'),
                    'platform': metadata.get('platform_id', 'N/A'),
                    'n_samples': expr_data.shape[1],
                    'n_features': expr_data.shape[0],
                    'expr_min': float(expr_data.min().min()),
                    'expr_max': float(expr_data.max().max()),
                    'expr_mean': float(expr_data.mean().mean()),
                    'missing_values': int(expr_data.isna().sum().sum()),
                    'success': True
                }

        return {
            'geo_id': geo_id,
            'success': False,
            'error': 'Could not parse expression data'
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'geo_id': geo_id,
            'success': False,
            'error': str(e)
        }

def search_alternative_diabetes_datasets():
    """
    Suggest alternative diabetes datasets that might be more accessible
    """
    print("\n\n🔍 Alternative Diabetes RNA-seq/Expression Datasets:")
    print("="*80)

    alternatives = [
        {
            'id': 'GSE76894',
            'description': 'Type 2 Diabetes blood samples, large cohort',
            'tissue': 'Whole blood',
            'samples': '199 samples'
        },
        {
            'id': 'GSE20966',
            'description': 'Type 2 Diabetes skeletal muscle',
            'tissue': 'Muscle',
            'samples': '40 samples'
        },
        {
            'id': 'GSE15932',
            'description': 'Type 1 Diabetes pancreatic islets',
            'tissue': 'Pancreatic islets',
            'samples': '20 samples'
        },
        {
            'id': 'GSE50397',
            'description': 'Prediabetes vs T2D blood samples',
            'tissue': 'Blood',
            'samples': '80 samples'
        },
    ]

    for ds in alternatives:
        print(f"\n📌 {ds['id']}")
        print(f"   Description: {ds['description']}")
        print(f"   Tissue: {ds['tissue']}")
        print(f"   Samples: {ds['samples']}")

    return alternatives

def main():
    """
    Main function to download datasets
    """
    print("🔬 RNA-based Diabetes Detection - Data Collection v2")
    print("Using direct HTTPS access to GEO")
    print("="*80)

    # Try original suggested datasets
    original_datasets = ['GSE38642', 'GSE25724', 'GSE9006']

    # Also try some alternatives that might be more accessible
    all_datasets = original_datasets + ['GSE76894', 'GSE20966']

    results = []

    for geo_id in all_datasets:
        result = download_geo_matrix(geo_id)
        if result:
            results.append(result)
        time.sleep(1)  # Be polite to the server

    # Show alternatives
    search_alternative_diabetes_datasets()

    # Summary
    print(f"\n\n{'='*80}")
    print("📊 DATA COLLECTION SUMMARY")
    print(f"{'='*80}")

    if results:
        summary_df = pd.DataFrame(results)
        print(summary_df.to_string())

        # Save summary
        summary_df.to_csv(DATA_DIR / "download_summary.csv", index=False)

        successful = sum(1 for r in results if r.get('success', False))
        print(f"\n✅ Successfully downloaded: {successful}/{len(all_datasets)} datasets")

        if successful > 0:
            print("\n🎯 Successfully downloaded datasets:")
            for r in results:
                if r.get('success'):
                    print(f"   • {r['geo_id']}: {r['n_samples']} samples, {r['n_features']} features")

            print("\n📝 Next steps:")
            print("1. Review sample_info.csv files to identify diabetes/control groups")
            print("2. Check expression data quality and distribution")
            print("3. Proceed with preprocessing and normalization")
    else:
        print("⚠️  No results to display")

if __name__ == "__main__":
    main()
