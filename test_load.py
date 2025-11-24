import pandas as pd
import gzip

# GSE86468 테스트
print("Testing GSE86468...")
try:
    with gzip.open('suppl_data/GSE86468/GSE86468_GEO.bulk.islet.processed.data.RSEM.raw.expected.counts.csv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"Line {i}: {line[:100]}")  
            else:
                break
    
    df = pd.read_csv('suppl_data/GSE86468/GSE86468_GEO.bulk.islet.processed.data.RSEM.raw.expected.counts.csv.gz',
                     compression='gzip', index_col=0)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:5]}") 
    print(f"Index: {df.index.tolist()[:5]}")  
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*50 + "\n")

# GSE86469 테스트
print("Testing GSE86469...")
try:
    with gzip.open('suppl_data/GSE86469/GSE86469_GEO.islet.single.cell.processed.data.RSEM.raw.expected.counts.csv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"Line {i}: {line[:100]}")
            else:
                break
    
    df = pd.read_csv('suppl_data/GSE86469/GSE86469_GEO.islet.single.cell.processed.data.RSEM.raw.expected.counts.csv.gz',
                     compression='gzip', index_col=0)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:5]}")
    print(f"Index: {df.index.tolist()[:5]}")
except Exception as e:
    print(f"Error: {e}")