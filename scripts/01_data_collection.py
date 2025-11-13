"""
Step 1: Data Collection from NCBI GEO
Download and explore diabetes-related RNA expression datasets
"""

import GEOparse
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# Create data directory
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_and_explore_geo(geo_id):
    """
    Download GEO dataset and extract key information
    """
    print(f"\n{'='*80}")
    print(f"Downloading and exploring {geo_id}")
    print(f"{'='*80}")

    try:
        # Download the dataset
        gse = GEOparse.get_GEO(geo=geo_id, destdir=str(DATA_DIR))

        # Save the GEO object
        with open(DATA_DIR / f"{geo_id}.pkl", 'wb') as f:
            pickle.dump(gse, f)

        # Extract metadata
        metadata = gse.metadata

        print(f"\n📊 Dataset: {geo_id}")
        print(f"Title: {metadata.get('title', ['N/A'])[0]}")
        print(f"Summary: {metadata.get('summary', ['N/A'])[0][:200]}...")
        print(f"Organism: {metadata.get('organism', ['N/A'])[0]}")
        print(f"Platform(s): {list(gse.gpls.keys())}")
        print(f"Number of samples: {len(gse.gsms)}")

        # Extract sample information
        sample_info = []
        for gsm_name, gsm in gse.gsms.items():
            sample_data = {
                'sample_id': gsm_name,
                'title': gsm.metadata.get('title', [''])[0],
                'source': gsm.metadata.get('source_name_ch1', [''])[0],
                'characteristics': str(gsm.metadata.get('characteristics_ch1', [])),
            }
            sample_info.append(sample_data)

        sample_df = pd.DataFrame(sample_info)
        print(f"\n📋 Sample characteristics preview:")
        print(sample_df.head())

        # Save sample metadata
        sample_df.to_csv(DATA_DIR / f"{geo_id}_samples.csv", index=False)

        # Try to extract expression data
        print(f"\n🧬 Extracting expression data...")

        # Get the first platform (most datasets have one platform)
        gpl_name = list(gse.gpls.keys())[0]

        # Extract expression matrix
        expression_data = []
        for gsm_name, gsm in gse.gsms.items():
            if hasattr(gsm.table, 'VALUE'):
                expression_data.append(gsm.table['VALUE'])
            elif hasattr(gsm.table, 'value'):
                expression_data.append(gsm.table['value'])
            elif 'VALUE' in gsm.table.columns:
                expression_data.append(gsm.table['VALUE'])
            else:
                # Try to find expression column
                numeric_cols = gsm.table.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    expression_data.append(gsm.table[numeric_cols[-1]])

        if expression_data:
            # Get gene identifiers from first sample
            first_gsm = list(gse.gsms.values())[0]
            if 'ID_REF' in first_gsm.table.columns:
                gene_ids = first_gsm.table['ID_REF']
            elif 'IDENTIFIER' in first_gsm.table.columns:
                gene_ids = first_gsm.table['IDENTIFIER']
            else:
                gene_ids = first_gsm.table.iloc[:, 0]

            # Create expression matrix
            expr_matrix = pd.DataFrame(expression_data).T
            expr_matrix.columns = list(gse.gsms.keys())
            expr_matrix.index = gene_ids

            print(f"Expression matrix shape: {expr_matrix.shape}")
            print(f"Features (genes/probes): {expr_matrix.shape[0]}")
            print(f"Samples: {expr_matrix.shape[1]}")

            # Save expression matrix
            expr_matrix.to_csv(DATA_DIR / f"{geo_id}_expression.csv")
            print(f"✅ Expression data saved to {geo_id}_expression.csv")

            # Basic statistics
            print(f"\n📈 Expression data statistics:")
            print(f"Min value: {expr_matrix.min().min():.2f}")
            print(f"Max value: {expr_matrix.max().max():.2f}")
            print(f"Mean value: {expr_matrix.mean().mean():.2f}")
            print(f"Median value: {expr_matrix.median().median():.2f}")

            return {
                'geo_id': geo_id,
                'n_samples': len(gse.gsms),
                'n_features': expr_matrix.shape[0],
                'platform': gpl_name,
                'success': True,
                'expression_range': (expr_matrix.min().min(), expr_matrix.max().max())
            }
        else:
            print(f"⚠️  Could not extract expression data automatically")
            return {
                'geo_id': geo_id,
                'n_samples': len(gse.gsms),
                'success': False,
                'platform': gpl_name
            }

    except Exception as e:
        print(f"❌ Error downloading {geo_id}: {str(e)}")
        return {
            'geo_id': geo_id,
            'success': False,
            'error': str(e)
        }

def main():
    """
    Main function to download all suggested datasets
    """
    print("🔬 RNA-based Diabetes Detection - Data Collection")
    print("=" * 80)

    # Suggested datasets
    datasets = [
        'GSE38642',  # T2D blood samples
        'GSE25724',  # T2D pancreatic islets
        'GSE9006',   # T1D blood samples
    ]

    results = []
    for geo_id in datasets:
        result = download_and_explore_geo(geo_id)
        results.append(result)

    # Summary
    print(f"\n\n{'='*80}")
    print("📊 DATA COLLECTION SUMMARY")
    print(f"{'='*80}")

    summary_df = pd.DataFrame(results)
    print(summary_df)

    # Save summary
    summary_df.to_csv(DATA_DIR / "download_summary.csv", index=False)

    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n✅ Successfully downloaded: {successful}/{len(datasets)} datasets")

    if successful > 0:
        print("\n🎯 Next steps:")
        print("1. Review the sample metadata files (*_samples.csv)")
        print("2. Identify diabetes vs control groups in each dataset")
        print("3. Proceed with data preprocessing")
    else:
        print("\n⚠️  No datasets were successfully downloaded.")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()
