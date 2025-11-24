import pandas as pd
import numpy as np

def load_geo_series_matrix(filepath):

    
    metadata = {}
    data_lines = []
    reading_data = False
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Metadata
            if line.startswith('!'):
                parts = line.split('\t')
                key = parts[0].replace('!', '').strip()
                values = [v.strip('"').strip() for v in parts[1:]]
                if values:
                    metadata[key] = values
            
            # Data matrix
            elif line.startswith('"ID_REF"') or line.startswith('ID_REF'):
                reading_data = True
                headers = [h.strip('"').strip() for h in line.split('\t')]
                data_lines.append(headers)
            
            elif reading_data:
                data_lines.append(line.split('\t'))
    
    expr_df = pd.DataFrame(data_lines[1:], columns=data_lines[0])
    expr_df.set_index('ID_REF', inplace=True)
    expr_df = expr_df.apply(pd.to_numeric, errors='coerce')
    sample_ids = metadata.get('Sample_geo_accession', [])
    sample_meta = pd.DataFrame({'sample_id': sample_ids})
    
    for key, values in metadata.items():
        if key.startswith('Sample_') and len(values) == len(sample_ids):
            sample_meta[key] = values
    
    print(f"Loaded: {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")
    
    return expr_df, sample_meta

expr_164416, samples_164416 = load_geo_series_matrix(
    '/mnt/user-data/uploads/GSE164416_series_matrix.txt'
)


def extract_diabetes_labels_gse164416(sample_df):
    """Extract ND/IGT/T3cD/T2D labels from GSE164416"""
    
    labels = []
    
    for idx, row in sample_df.iterrows():
        found = False
        for col in row:
            val = str(col).lower()
            
            if 'igt' in val or 'impaired glucose tolerance' in val:
                labels.append('IGT')
                found = True
                break
            elif 't3cd' in val or 'type 3c' in val or 'type3c' in val:
                labels.append('T3cD')
                found = True
                break
            elif 't2d' in val or 'type 2' in val or 'type2' in val:
                labels.append('T2DM')
                found = True
                break
            elif any(kw in val for kw in ['nd', 'non-diabetic', 'non diabetic', 'normal']):
                labels.append('ND')
                found = True
                break
        
        if not found:
            labels.append('Unknown')
    
    return labels

samples_164416['diabetes_label'] = extract_diabetes_labels_gse164416(samples_164416)

print("\nGroup distribution:")
print(samples_164416['diabetes_label'].value_counts())