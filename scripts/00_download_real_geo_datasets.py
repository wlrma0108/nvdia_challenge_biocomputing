"""
Download Real GEO Datasets for RNA-based Diabetes Detection

Download the optimal combination of datasets:
Core Datasets (Primary):
- GSE164416: Islet samples (Primary training data)
- GSE76894: Blood samples (Blood-based biomarker)
- GSE86469/GSE81608: Cell-type resolution data (Mechanism understanding)

Validation Datasets:
- GSE25724: External validation
- GSE86468: Additional validation
"""

import GEOparse
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import time

# Create data directory
DATA_DIR = Path("data/geo_datasets")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_geo_dataset(geo_id, max_retries=3):
    """
    Download GEO dataset with retry logic
    """
    print(f"\n{'='*80}")
    print(f"📥 Downloading {geo_id}")
    print(f"{'='*80}")

    for attempt in range(max_retries):
        try:
            print(f"   Attempt {attempt + 1}/{max_retries}...")

            # Download the dataset
            gse = GEOparse.get_GEO(geo=geo_id, destdir=str(DATA_DIR), silent=False)

            # Save the GEO object
            with open(DATA_DIR / f"{geo_id}.pkl", 'wb') as f:
                pickle.dump(gse, f)

            # Extract metadata
            metadata = gse.metadata

            print(f"\n✅ Successfully downloaded {geo_id}")
            print(f"   Title: {metadata.get('title', ['N/A'])[0]}")
            print(f"   Organism: {metadata.get('organism', ['N/A'])[0]}")
            print(f"   Platform(s): {list(gse.gpls.keys())}")
            print(f"   Number of samples: {len(gse.gsms)}")

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
            sample_df.to_csv(DATA_DIR / f"{geo_id}_samples.csv", index=False)

            print(f"\n📋 Sample information preview:")
            print(sample_df[['sample_id', 'title']].head(3))

            # Try to extract expression data
            try:
                print(f"\n🧬 Extracting expression data...")

                # Get expression matrix
                expression_data = []
                gene_ids = None

                for gsm_name, gsm in gse.gsms.items():
                    if gene_ids is None:
                        # Get gene identifiers from first sample
                        if 'ID_REF' in gsm.table.columns:
                            gene_ids = gsm.table['ID_REF'].values
                        elif 'IDENTIFIER' in gsm.table.columns:
                            gene_ids = gsm.table['IDENTIFIER'].values
                        else:
                            gene_ids = gsm.table.iloc[:, 0].values

                    # Get expression values
                    if 'VALUE' in gsm.table.columns:
                        expression_data.append(gsm.table['VALUE'].values)
                    elif hasattr(gsm.table, 'VALUE'):
                        expression_data.append(gsm.table.VALUE.values)
                    else:
                        # Find numeric column
                        numeric_cols = gsm.table.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            expression_data.append(gsm.table[numeric_cols[-1]].values)

                if expression_data and gene_ids is not None:
                    # Create expression matrix
                    expr_matrix = pd.DataFrame(
                        np.array(expression_data).T,
                        index=gene_ids,
                        columns=list(gse.gsms.keys())
                    )

                    print(f"   Expression matrix shape: {expr_matrix.shape}")
                    print(f"   Expression range: {expr_matrix.min().min():.2f} to {expr_matrix.max().max():.2f}")

                    # Save expression matrix
                    expr_matrix.to_csv(DATA_DIR / f"{geo_id}_expression.csv")
                    print(f"   ✅ Expression data saved")

                    return {
                        'geo_id': geo_id,
                        'success': True,
                        'n_samples': len(gse.gsms),
                        'n_features': expr_matrix.shape[0],
                        'has_expression': True
                    }
                else:
                    print(f"   ⚠️  Could not extract expression matrix automatically")
                    return {
                        'geo_id': geo_id,
                        'success': True,
                        'n_samples': len(gse.gsms),
                        'has_expression': False
                    }

            except Exception as e:
                print(f"   ⚠️  Expression extraction error: {str(e)}")
                return {
                    'geo_id': geo_id,
                    'success': True,
                    'n_samples': len(gse.gsms),
                    'has_expression': False,
                    'error': str(e)
                }

        except Exception as e:
            print(f"   ❌ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"   Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"   ❌ All attempts failed for {geo_id}")
                return {
                    'geo_id': geo_id,
                    'success': False,
                    'error': str(e)
                }

def main():
    """
    Download all recommended datasets
    """
    print("="*80)
    print("🔬 RNA-based Diabetes Detection - Real GEO Dataset Download")
    print("="*80)
    print("\n📋 Dataset Selection Strategy:")
    print("   Core Datasets (Primary):")
    print("   • GSE164416: Pancreatic islet samples (Primary training)")
    print("   • GSE76894: Blood samples (Biomarker translation)")
    print("   • GSE86469: Cell-type resolution (Mechanism)")
    print("   • GSE81608: Cell-type resolution (Mechanism)")
    print("\n   Validation Datasets:")
    print("   • GSE25724: External validation")
    print("   • GSE86468: Additional validation")

    # Core datasets
    core_datasets = [
        'GSE164416',  # Islet - Primary training
        'GSE76894',   # Blood - Biomarker
        'GSE86469',   # Cell-type resolution
        'GSE81608',   # Cell-type resolution
    ]

    # Validation datasets
    validation_datasets = [
        'GSE25724',   # External validation
        'GSE86468',   # Additional validation
    ]

    all_datasets = core_datasets + validation_datasets

    results = []

    print(f"\n{'='*80}")
    print(f"Starting Downloads ({len(all_datasets)} datasets)")
    print(f"{'='*80}")

    for geo_id in all_datasets:
        result = download_geo_dataset(geo_id)
        if result:
            results.append(result)
        # Be polite to the server
        time.sleep(3)

    # Summary
    print(f"\n\n{'='*80}")
    print("📊 DOWNLOAD SUMMARY")
    print(f"{'='*80}")

    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_df.to_csv(DATA_DIR / "download_summary.csv", index=False)

    successful = sum(1 for r in results if r.get('success', False))
    with_expression = sum(1 for r in results if r.get('has_expression', False))

    print(f"\n✅ Successfully downloaded: {successful}/{len(all_datasets)} datasets")
    print(f"📊 With expression data: {with_expression}/{successful} datasets")

    if successful > 0:
        print("\n🎯 Next Steps:")
        print("1. Review downloaded datasets in data/geo_datasets/")
        print("2. Examine sample metadata (*_samples.csv)")
        print("3. Identify control vs diabetes samples")
        print("4. Combine datasets for training")
        print("5. Proceed with analysis pipeline")
    else:
        print("\n⚠️  No datasets downloaded successfully")
        print("This may be due to network restrictions or server access issues")
        print("\n💡 Alternative approaches:")
        print("1. Use the simulated dataset we already generated")
        print("2. Download datasets manually from GEO website")
        print("3. Use GEO2R online tool for preliminary analysis")

if __name__ == "__main__":
    main()
