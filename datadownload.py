import GEOparse

gse = GEOparse.get_GEO("GSE81608", destdir="./data")

print("=== GSE Level Supplementary Files ===")
if hasattr(gse, 'download_supplementary_files'):
    supp_files = gse.download_supplementary_files(directory="./data/GSE81608_suppl")
    print(supp_files)
else:
    print(gse.metadata.get('supplementary_file', 'No files'))