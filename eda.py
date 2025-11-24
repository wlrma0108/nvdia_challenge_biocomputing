import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DiabetesDataEDA:
    def __init__(self, expression_data, metadata, dataset_id):
        self.expr = expression_data
        self.metadata = metadata
        self.dataset_id = dataset_id
        self.output_dir = None
        
        print(f"\n{'='*80}")
        print(f"📊 EDA for {dataset_id}")
        print('='*80)
        print(f"Expression: {self.expr.shape[0]:,} genes × {self.expr.shape[1]} samples")
        
        if self.metadata is not None and 'diabetes_status' in self.metadata.columns:
            labels = self.metadata['diabetes_status'].value_counts().to_dict()
            print(f"Labels: {labels}")
        else:
            print("Labels: None available")
    
    def generate_report(self, output_dir):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("📊 GENERATING EDA REPORT")
        print('='*80)
        
        self.basic_statistics()
        self.plot_distributions()
        self.perform_pca()
        self.plot_pca()
        self.plot_correlation_heatmap()
        self.find_variable_genes()
        
        if self.metadata is not None and 'diabetes_status' in self.metadata.columns:
            unique = self.metadata['diabetes_status'].unique()
            if len(unique) == 2:
                try:
                    self.differential_expression(unique[0], unique[1])
                except Exception as e:
                    print(f"   ⚠️  Skipping differential expression: {e}")
        
        print(f"\n✅ Report complete: {self.output_dir}")
    
    def basic_statistics(self):
        print(f"\n📈 BASIC STATISTICS")
        
        stats_dict = {
            'Total genes': self.expr.shape[0],
            'Total samples': self.expr.shape[1],
            'Mean expression': f"{self.expr.values.mean():.3f}",
            'Median': f"{np.median(self.expr.values):.3f}",
            'Std': f"{self.expr.values.std():.3f}",
            'Min': f"{self.expr.values.min():.3f}",
            'Max': f"{self.expr.values.max():.3f}",
            'Missing %': f"{(self.expr.isna().sum().sum() / self.expr.size * 100):.2f}%",
            'Zero %': f"{(self.expr == 0).sum().sum() / self.expr.size * 100:.2f}%"
        }
        
        for key, val in stats_dict.items():
            print(f"  {key}: {val}")
        
        pd.DataFrame([stats_dict]).T.to_csv(f'{self.output_dir}/basic_stats.csv', header=['Value'])
    
    def plot_distributions(self):
        print(f"\n📊 Plotting distributions...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        sample_means = self.expr.mean(axis=0)
        axes[0, 0].hist(sample_means, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Mean Expression')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Sample Mean Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        gene_means = self.expr.mean(axis=1)
        axes[0, 1].hist(gene_means, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Mean Expression')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Gene Mean Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        gene_vars = self.expr.var(axis=1)
        axes[1, 0].hist(np.log10(gene_vars + 1), bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Log10(Variance + 1)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Gene Variance Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        flat_values = self.expr.values.flatten()
        sample_indices = np.random.choice(len(flat_values), size=min(10000, len(flat_values)), replace=False)
        axes[1, 1].hist(flat_values[sample_indices], bins=50, edgecolor='black', alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Expression Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Overall Expression Distribution (sampled)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{self.dataset_id}_dist.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved: {self.output_dir}/{self.dataset_id}_dist.png")
    
    def perform_pca(self):
        print(f"\n🔄 Performing PCA...")
        
        expr_t = self.expr.T
        expr_clean = expr_t.fillna(expr_t.mean())
        expr_clean = expr_clean.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        scaler = StandardScaler()
        expr_scaled = scaler.fit_transform(expr_clean)
        
        n_components = min(10, expr_scaled.shape[0], expr_scaled.shape[1])
        pca = PCA(n_components=n_components)
        self.pca_result = pca.fit_transform(expr_scaled)
        self.pca_variance = pca.explained_variance_ratio_
        
        print(f"  PC1: {self.pca_variance[0]*100:.2f}%")
        print(f"  PC2: {self.pca_variance[1]*100:.2f}%")
        
        pca_df = pd.DataFrame(
            self.pca_result[:, :5],
            index=expr_clean.index,
            columns=[f'PC{i+1}' for i in range(min(5, n_components))]
        )
        pca_df.to_csv(f'{self.output_dir}/pca_coordinates.csv')
        
        variance_df = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(len(self.pca_variance))],
            'Variance_Explained': self.pca_variance
        })
        variance_df.to_csv(f'{self.output_dir}/pca_variance.csv', index=False)
    
    def plot_pca(self):
        print(f"\n📊 Plotting PCA...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if self.metadata is not None and 'diabetes_status' in self.metadata.columns:
            pca_df = pd.DataFrame(
                self.pca_result[:, :2],
                index=self.expr.columns,
                columns=['PC1', 'PC2']
            )
            pca_df['diabetes_status'] = self.metadata['diabetes_status']
            
            for label in pca_df['diabetes_status'].unique():
                mask = pca_df['diabetes_status'] == label
                axes[0].scatter(
                    pca_df.loc[mask, 'PC1'],
                    pca_df.loc[mask, 'PC2'],
                    label=label,
                    alpha=0.6,
                    s=50
                )
            axes[0].legend()
        else:
            axes[0].scatter(self.pca_result[:, 0], self.pca_result[:, 1], alpha=0.6, s=50)
        
        axes[0].set_xlabel(f'PC1 ({self.pca_variance[0]*100:.1f}%)')
        axes[0].set_ylabel(f'PC2 ({self.pca_variance[1]*100:.1f}%)')
        axes[0].set_title(f'{self.dataset_id} - PCA')
        axes[0].grid(True, alpha=0.3)
        
        n_components = len(self.pca_variance)
        axes[1].bar(range(1, n_components + 1), self.pca_variance * 100, alpha=0.7)
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('Variance Explained (%)')
        axes[1].set_title('Scree Plot')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{self.dataset_id}_pca.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved: {self.output_dir}/{self.dataset_id}_pca.png")
    
    def plot_correlation_heatmap(self):
        print(f"\n📊 Plotting correlation heatmap...")
        
        if self.metadata is None:
            print("   ⚠️  No metadata available")
            return
        
        try:
            numeric_data = pd.DataFrame()
            
            for col in self.metadata.columns:
                try:
                    converted = pd.to_numeric(self.metadata[col], errors='coerce')
                    if converted.notna().sum() > len(self.metadata) * 0.5:
                        numeric_data[col] = converted
                except:
                    continue
            
            if numeric_data.shape[1] < 2:
                print("   ⚠️  Not enough numeric columns for correlation")
                return
            
            numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
            numeric_data = numeric_data.loc[:, numeric_data.std() > 0]
            
            if numeric_data.shape[1] < 2:
                print("   ⚠️  Not enough variable columns for correlation")
                return
            
            corr = numeric_data.corr()
            
            plt.figure(figsize=(max(8, len(corr.columns) * 0.8), max(6, len(corr.columns) * 0.8)))
            
            sns.heatmap(
                corr,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={'shrink': 0.8}
            )
            
            plt.title(f'{self.dataset_id} - Metadata Correlation Heatmap', fontsize=14, pad=20)
            plt.tight_layout()
            
            plt.savefig(f'{self.output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved correlation heatmap ({numeric_data.shape[1]} variables)")
            
        except Exception as e:
            print(f"   ⚠️  Skipped correlation heatmap: {str(e)}")
    
    def find_variable_genes(self, top_n=100):
        print(f"\n🧬 Finding top variable genes...")
        
        gene_vars = self.expr.var(axis=1).sort_values(ascending=False)
        top_genes = gene_vars.head(top_n)
        
        top_genes.to_csv(f'{self.output_dir}/top_variable_genes.csv', header=['Variance'])
        
        print(f"   💾 Saved top {top_n} variable genes")
        print(f"   📊 Top 5 genes:")
        for gene, var in top_genes.head(5).items():
            print(f"      {gene}: {var:.3f}")
    
    def differential_expression(self, group1, group2, top_n=50):
        print(f"\n🧬 Differential Expression: {group1} vs {group2}")
        
        try:
            if 'diabetes_status' not in self.metadata.columns:
                print("   ⚠️  No diabetes_status column found")
                return None
            
            mask1 = self.metadata['diabetes_status'] == group1
            mask2 = self.metadata['diabetes_status'] == group2
            
            samples1 = self.metadata[mask1].index.tolist()
            samples2 = self.metadata[mask2].index.tolist()
            
            expr1_cols = [col for col in samples1 if col in self.expr.columns]
            expr2_cols = [col for col in samples2 if col in self.expr.columns]
            
            if len(expr1_cols) < 2 or len(expr2_cols) < 2:
                print(f"   ⚠️  Not enough samples ({len(expr1_cols)} vs {len(expr2_cols)})")
                return None
            
            expr1 = self.expr[expr1_cols]
            expr2 = self.expr[expr2_cols]
            
            results = []
            
            for gene in self.expr.index:
                try:
                    vals1 = expr1.loc[gene].values
                    vals2 = expr2.loc[gene].values
                    
                    if vals1.std() == 0 and vals2.std() == 0:
                        continue
                    
                    t_stat, p_val = stats.ttest_ind(vals1, vals2)
                    
                    mean1 = vals1.mean()
                    mean2 = vals2.mean()
                    
                    if mean2 == 0:
                        log2fc = np.nan
                    else:
                        log2fc = np.log2((mean1 + 1) / (mean2 + 1))
                    
                    results.append({
                        'gene': gene,
                        'log2fc': log2fc,
                        'pvalue': p_val,
                        'mean_group1': mean1,
                        'mean_group2': mean2
                    })
                except:
                    continue
            
            if not results:
                print("   ⚠️  No results from differential expression")
                return None
            
            de_df = pd.DataFrame(results)
            de_df['padj'] = de_df['pvalue'] * len(de_df)
            de_df['padj'] = de_df['padj'].clip(upper=1.0)
            de_df = de_df.sort_values('pvalue')
            
            print(f"   ✅ Found {len(de_df)} genes")
            print(f"   📊 Top {min(5, len(de_df))} significant genes:")
            
            for idx, row in de_df.head(5).iterrows():
                print(f"      {row['gene']}: log2FC={row['log2fc']:.2f}, p={row['pvalue']:.2e}")
            
            de_df.head(top_n).to_csv(f'{self.output_dir}/top_de_genes.csv', index=False)
            
            self.plot_volcano(de_df)
            
            return de_df
            
        except Exception as e:
            print(f"   ❌ Error in differential expression: {e}")
            return None
    
    def plot_volcano(self, de_df):
        print(f"   📊 Plotting volcano plot...")
        
        try:
            de_clean = de_df.dropna(subset=['log2fc', 'pvalue'])
            de_clean['-log10(p)'] = -np.log10(de_clean['pvalue'])
            
            plt.figure(figsize=(10, 8))
            
            sig_mask = (de_clean['padj'] < 0.05) & (np.abs(de_clean['log2fc']) > 1)
            
            plt.scatter(
                de_clean.loc[~sig_mask, 'log2fc'],
                de_clean.loc[~sig_mask, '-log10(p)'],
                c='gray',
                alpha=0.3,
                s=10,
                label='Not significant'
            )
            
            plt.scatter(
                de_clean.loc[sig_mask, 'log2fc'],
                de_clean.loc[sig_mask, '-log10(p)'],
                c='red',
                alpha=0.6,
                s=20,
                label='Significant'
            )
            
            plt.axhline(y=-np.log10(0.05), color='blue', linestyle='--', alpha=0.5, linewidth=1)
            plt.axvline(x=-1, color='blue', linestyle='--', alpha=0.5, linewidth=1)
            plt.axvline(x=1, color='blue', linestyle='--', alpha=0.5, linewidth=1)
            
            plt.xlabel('Log2 Fold Change', fontsize=12)
            plt.ylabel('-Log10(p-value)', fontsize=12)
            plt.title(f'{self.dataset_id} - Volcano Plot', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/volcano_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   💾 Saved volcano plot")
            
        except Exception as e:
            print(f"   ⚠️  Could not create volcano plot: {e}")


if __name__ == '__main__':
    from geo_parser import parse_all_geo_files
    from parse_singlecell import load_singlecell_data_v2
    
    print("="*80)
    print("🧬 COMPREHENSIVE DIABETES TRANSCRIPTOMICS EDA")
    print("="*80)
    
    print("\n📊 Loading Microarray Data (Series Matrix)...")
    print("-"*80)
    parsers = parse_all_geo_files('./data', './outputdata')
    
    print("\n🧬 Loading RNA-seq/Single-cell Data (Supplementary)...")
    print("-"*80)
    
    sc_geo_ids = ['GSE81608', 'GSE164416', 'GSE86468', 'GSE86469']
    sc_data = {}
    
    for geo_id in sc_geo_ids:
        expr, _ = load_singlecell_data_v2(geo_id)
        
        if expr is not None and not expr.empty and expr.shape[1] > 0:
            meta_path = f'./outputdata/{geo_id}_metadata.csv'
            if os.path.exists(meta_path):
                metadata = pd.read_csv(meta_path, index_col=0)
            else:
                metadata = None
            
            sc_data[geo_id] = (expr, metadata)
    
    print("\n📦 Consolidating All Datasets...")
    print("-"*80)
    
    all_datasets = {}
    
    for geo_id, parser in parsers.items():
        expr = parser.get_expression_matrix()
        if expr is not None and not expr.empty:
            metadata = parser.get_sample_metadata()
            all
# eda.py 수정
if __name__ == '__main__':
    from geo_parser import parse_all_geo_files
    from parse_singlecell import load_singlecell_data_v2
    import pandas as pd
    import os
    
    print("="*80)
    print("🧬 COMPREHENSIVE DIABETES TRANSCRIPTOMICS EDA")
    print("="*80)
    
    # 1️⃣ Series Matrix 데이터 파싱
    print("\n📊 Loading Microarray Data (Series Matrix)...")
    print("-"*80)
    parsers = parse_all_geo_files('./data', './outputdata')
    
    # 2️⃣ Single-cell Supplementary 데이터 로드
    print("\n🧬 Loading RNA-seq/Single-cell Data (Supplementary)...")
    print("-"*80)
    
    sc_geo_ids = ['GSE81608', 'GSE164416', 'GSE86468', 'GSE86469']
    sc_data = {}
    
    for geo_id in sc_geo_ids:
        expr, _ = load_singlecell_data_v2(geo_id)
        
        if expr is not None and not expr.empty and expr.shape[1] > 0:
            # Series matrix metadata 로드
            meta_path = f'./outputdata/{geo_id}_metadata.csv'
            if os.path.exists(meta_path):
                metadata = pd.read_csv(meta_path, index_col=0)
            else:
                metadata = None
            
            sc_data[geo_id] = (expr, metadata)
    
    # 3️⃣ 모든 데이터셋 통합
    print("\n📦 Consolidating All Datasets...")
    print("-"*80)
    
    all_datasets = {}
    
    # Microarray 데이터
    for geo_id, parser in parsers.items():
        expr = parser.get_expression_matrix()
        if expr is not None and not expr.empty:
            metadata = parser.get_sample_metadata()
            all_datasets[geo_id] = (expr, metadata)
            
            # 라벨 분포 확인
            if metadata is not None and 'diabetes_status' in metadata.columns:
                labels = metadata['diabetes_status'].value_counts().to_dict()
                label_str = ', '.join([f"{k}: {v}" for k, v in labels.items()])
            else:
                label_str = "No labels"
            
            print(f"✅ {geo_id}: {expr.shape[0]:,} genes × {expr.shape[1]} samples ({label_str})")
    
    # Single-cell 데이터
    for geo_id, (expr, metadata) in sc_data.items():
        all_datasets[geo_id] = (expr, metadata)
        
        if metadata is not None and 'diabetes_status' in metadata.columns:
            labels = metadata['diabetes_status'].value_counts().to_dict()
            label_str = ', '.join([f"{k}: {v}" for k, v in labels.items()])
        else:
            label_str = "No labels"
        
        print(f"✅ {geo_id}: {expr.shape[0]:,} genes × {expr.shape[1]} samples ({label_str})")
    
    total_samples = sum([expr.shape[1] for expr, _ in all_datasets.values()])
    print(f"\n📊 Total: {len(all_datasets)} datasets, {total_samples:,} samples")
    
    # 4️⃣ 각 데이터셋 EDA 실행
    print("\n" + "="*80)
    print("🔬 RUNNING COMPREHENSIVE EDA")
    print("="*80)
    
    success_count = 0
    failed_datasets = []
    
    for i, (geo_id, (expr, metadata)) in enumerate(all_datasets.items(), 1):
        print(f"\n[{i}/{len(all_datasets)}] Analyzing {geo_id}...")
        print("-"*80)
        
        try:
            eda = DiabetesDataEDA(expr, metadata, geo_id)
            eda.generate_report(f'./eda_results/{geo_id}')
            print(f"✅ {geo_id} complete!")
            success_count += 1
        except Exception as e:
            print(f"❌ Error analyzing {geo_id}: {e}")
            failed_datasets.append(geo_id)
            import traceback
            traceback.print_exc()
    
    # 5️⃣ 최종 요약
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    print(f"✅ Successfully analyzed: {success_count}/{len(all_datasets)} datasets")
    
    if failed_datasets:
        print(f"❌ Failed datasets: {', '.join(failed_datasets)}")
    else:
        print("🎉 All datasets analyzed successfully!")
    
    print(f"\n📁 Results saved in: ./eda_results/")
    print(f"💾 Raw data saved in: ./outputdata/")
    print("="*80)