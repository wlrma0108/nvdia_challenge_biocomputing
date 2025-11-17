"""
Quick Verification Script for GEO Data Files

Checks if .txt files are valid and provides quick statistics.

Author: Claude
Date: 2025-11-17
"""

from pathlib import Path
import gzip

BASE_DIR = Path(__file__).parent.parent
GEO_DIR = BASE_DIR / 'data' / 'geo_datasets'

print("="*80)
print("🔍 GEO DATA FILES VERIFICATION")
print("="*80)

# Find all GEO files
txt_files = list(GEO_DIR.glob('GSE*_series_matrix.txt'))
gz_files = list(GEO_DIR.glob('GSE*_series_matrix.txt.gz'))

print(f"\n📁 Scanning: {GEO_DIR}")
print(f"\nFound files:")
print(f"  .txt files: {len(txt_files)}")
print(f"  .txt.gz files: {len(gz_files)}")

all_files = txt_files + gz_files

if not all_files:
    print(f"\n❌ No GEO files found!")
    print(f"\nExpected file names:")
    print(f"  GSE164416_series_matrix.txt (or .txt.gz)")
    print(f"  GSE76894_series_matrix.txt")
    print(f"  GSE25724_series_matrix.txt")
    print(f"  GSE81608_series_matrix.txt")
    print(f"  GSE86468_series_matrix.txt")
    print(f"  GSE86469_series_matrix.txt")
    exit(1)

print(f"\nTotal: {len(all_files)} files\n")

# Quick check each file
results = []

for filepath in sorted(all_files):
    geo_id = filepath.stem.split('_')[0]
    size_mb = filepath.stat().st_size / 1024 / 1024

    print(f"📄 {filepath.name}")
    print(f"   Size: {size_mb:.2f} MB")

    # Try to read first few lines
    try:
        if filepath.suffix == '.gz':
            f = gzip.open(filepath, 'rt', encoding='utf-8', errors='replace')
        else:
            f = open(filepath, 'r', encoding='utf-8', errors='replace')

        lines_read = 0
        series_info = {}
        sample_count = 0
        has_table = False

        for i, line in enumerate(f):
            if i > 500:  # Only read first 500 lines for quick check
                break

            line = line.strip()

            if line.startswith('!Series_title'):
                series_info['title'] = line.split('"')[1][:50] if '"' in line else "Unknown"
            elif line.startswith('!Series_sample_id'):
                # Count samples
                sample_count = len(line.split('\t')) - 1
            elif line == '!series_matrix_table_begin':
                has_table = True

            lines_read += 1

        f.close()

        print(f"   Lines read: {lines_read}")
        print(f"   Title: {series_info.get('title', 'Not found')}")
        print(f"   Samples: ~{sample_count}")
        print(f"   Has table: {'✓ Yes' if has_table else '⚠️ Not found in first 500 lines'}")

        results.append({
            'geo_id': geo_id,
            'file': filepath.name,
            'size_mb': size_mb,
            'samples': sample_count,
            'status': '✓ OK' if has_table else '⚠️ Check'
        })

    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append({
            'geo_id': geo_id,
            'file': filepath.name,
            'size_mb': size_mb,
            'samples': 0,
            'status': '❌ Error'
        })

    print()

# Summary table
print("="*80)
print("📊 SUMMARY")
print("="*80)
print(f"\n{'GEO ID':<12} {'Filename':<40} {'Size (MB)':<10} {'Samples':<8} {'Status'}")
print("-"*80)

for r in results:
    print(f"{r['geo_id']:<12} {r['file']:<40} {r['size_mb']:<10.2f} {r['samples']:<8} {r['status']}")

# Recommendations
print(f"\n{'='*80}")
print("✅ NEXT STEPS")
print("="*80)

ok_count = sum(1 for r in results if r['status'] == '✓ OK')
total = len(results)

if ok_count == total:
    print(f"\n🎉 All {total} files look good!")
    print(f"\nReady to run:")
    print(f"  python scripts/15_process_geo_robust.py")
elif ok_count > 0:
    print(f"\n⚠️ {ok_count}/{total} files verified successfully")
    print(f"\nProblematic files:")
    for r in results:
        if r['status'] != '✓ OK':
            print(f"  - {r['file']}: {r['status']}")
    print(f"\nYou can still proceed with working files:")
    print(f"  python scripts/15_process_geo_robust.py")
else:
    print(f"\n❌ No valid files found!")
    print(f"\nPlease check:")
    print(f"  1. Files are in correct location: {GEO_DIR}")
    print(f"  2. File names match pattern: GSE*_series_matrix.txt")
    print(f"  3. Files are not corrupted")

print("\n" + "="*80)
