import pandas as pd
import gzip

with gzip.open('./suppl_data/GSE164416/GSE164416_DP_htseq_counts.txt.gz', 'rt') as f:
    expr = pd.read_csv(f, sep='\t', index_col=0, nrows=5)

print("Expression 샘플 ID (처음 5개):")
print(expr.columns.tolist()[:5])

meta = pd.read_csv('./outputdata/GSE164416_metadata.csv', index_col=0)
print("\nMetadata 샘플 ID (처음 5개):")
print(meta.index.tolist()[:5])