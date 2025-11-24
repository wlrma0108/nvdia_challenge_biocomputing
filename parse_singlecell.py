import pandas as pd
import gzip
import os
from pathlib import Path

def safe_read_gzip(filepath):
    try:
        with gzip.open(filepath, 'rt') as f:
            df = pd.read_csv(f, sep='\t', index_col=0, nrows=5)
        
        with gzip.open(filepath, 'rt') as f:
            df = pd.read_csv(f, sep='\t', index_col=0)
        
        return df
    except:
        try:
            # CSV 시도
            with gzip.open(filepath, 'rt') as f:
                df = pd.read_csv(f, index_col=0)
            return df
        except:
            return None

def load_singlecell_data_v2(geo_id, suppl_dir='./suppl_data'):
    
    geo_dir = Path(suppl_dir) / geo_id
    
    if not geo_dir.exists():
        print(f"❌ {geo_dir} not found")
        return None, None
    
    print(f"🔍 Loading {geo_id}...")
    
    all_files = list(geo_dir.glob('*'))
    
    if not all_files:
        print(f"   ❌ No files found in {geo_dir}")
        return None, None
    
    print(f"   📁 Found {len(all_files)} file(s):")
    for f in all_files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"      • {f.name} ({size_mb:.2f} MB)")
    
    # Expression 파일 찾기
    expr_keywords = ['count', 'matrix', 'rpkm', 'tpm', 'fpkm', 'expr', 'processed.data']
    expr_file = None
    
    for keyword in expr_keywords:
        for f in all_files:
            if keyword in f.name.lower():
                expr_file = f
                break
        if expr_file:
            break
    
    # Metadata 파일 찾기
    meta_keywords = ['meta', 'cell', 'sample', 'annotation']
    meta_file = None
    
    for keyword in meta_keywords:
        for f in all_files:
            if keyword in f.name.lower() and f != expr_file:
                meta_file = f
                break
        if meta_file:
            break
    
    if not expr_file:
        print(f"   ⚠️  No expression file found")
        return None, None
    
    # Expression data 로드
    print(f"   📊 Loading expression: {expr_file.name}")
    
    expr = None
    
    try:
        if expr_file.suffix == '.gz':
            # Gzip 파일 처리
            if '.csv' in expr_file.name:
                # CSV.gz
                with gzip.open(expr_file, 'rt') as f:
                    expr = pd.read_csv(f, index_col=0)
            else:
                # TSV.gz
                with gzip.open(expr_file, 'rt') as f:
                    expr = pd.read_csv(f, sep='\t', index_col=0)
        else:
            # 압축 안된 파일
            if expr_file.suffix == '.csv':
                expr = pd.read_csv(expr_file, index_col=0)
            else:
                expr = pd.read_csv(expr_file, sep='\t', index_col=0)
        
        if expr is not None:
            print(f"   ✅ Loaded: {expr.shape[0]} genes × {expr.shape[1]} samples")
        else:
            print(f"   ❌ Failed to load expression data")
            return None, None
            
    except Exception as e:
        print(f"   ❌ Error loading: {e}")
        return None, None
    
    # Metadata 로드
    metadata = None
    if meta_file:
        print(f"   📋 Loading metadata: {meta_file.name}")
        try:
            if meta_file.suffix == '.gz':
                with gzip.open(meta_file, 'rt') as f:
                    metadata = pd.read_csv(f, sep=None, engine='python', index_col=0)
            else:
                metadata = pd.read_csv(meta_file, sep=None, engine='python', index_col=0)
            
            if metadata is not None:
                print(f"   ✅ Loaded metadata: {metadata.shape}")
        except Exception as e:
            print(f"   ⚠️  Could not load metadata: {e}")
    else:
        print(f"   ℹ️  No metadata file found")
    
    return expr, metadata

if __name__ == '__main__':
    geo_ids = ['GSE81608', 'GSE164416', 'GSE86468', 'GSE86469']
    
    print("="*80)
    print("🧬 Single-cell Data Parser v3")
    print("="*80)
    print()
    
    results = {}
    
    for geo_id in geo_ids:
        expr, meta = load_singlecell_data_v2(geo_id)
        results[geo_id] = {
            'success': expr is not None,
            'shape': expr.shape if expr is not None else None
        }
        print()
    
    print("="*80)
    print("📊 Parsing Summary")
    print("="*80)
    
    for geo_id, result in results.items():
        if result['success']:
            print(f"   ✅ {geo_id}: {result['shape'][0]} genes × {result['shape'][1]} samples")
        else:
            print(f"   ❌ {geo_id}: Failed")
    
    print("="*80)