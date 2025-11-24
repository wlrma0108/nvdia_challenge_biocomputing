import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import gzip
import os
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class DifferentialExpressionAnalysis:
    def __init__(self, expression_data, metadata, output_dir='./de_results'):
        self.expr = expression_data
        self.metadata = metadata
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("🧬 DIFFERENTIAL EXPRESSION ANALYSIS")
        print("="*80)
        print(f"\n📊 Data:")
        print(f"   Expression: {self.expr.shape[0]} genes × {self.expr.shape[1]} samples")
        print(f"   Metadata: {self.metadata.shape[0]} samples")
    
    def prepare_data(self, group_column='diabetes_label', exclude_classes=['T3cD']):
        print(f"\n🔧 PREPARING DATA")
        print("-"*80)
        
        if group_column not in self.metadata.columns:
            raise ValueError(f"Column '{group_column}' not found in metadata")
        
        print(f"Original groups:")
        print(self.metadata[group_column].value_counts())
        
        print(f"\n🔄 Matching sample IDs...")
        
        if 'title' in self.metadata.columns:
            import re
            
            def extract_sample_id(title):
                match = re.search(r'(DP\d+)', str(title))
                return match.group(1) if match else None
            
            self.metadata['sample_id'] = self.metadata['title'].apply(extract_sample_id)
            valid_mask = self.metadata['sample_id'].notna()
            self.metadata = self.metadata[valid_mask]
            self.metadata = self.metadata.set_index('sample_id', drop=False)
        
        common_samples = list(set(self.expr.columns) & set(self.metadata.index))
        
        if len(common_samples) == 0:
            raise ValueError("No matching samples between expression and metadata")
        
        self.expr = self.expr[common_samples]
        self.metadata = self.metadata.loc[common_samples]
        
        print(f"   Matched {len(common_samples)} samples")
        
        if exclude_classes:
            mask = ~self.metadata[group_column].isin(exclude_classes)
            self.metadata = self.metadata[mask]
            filtered_samples = self.metadata.index.tolist()
            self.expr = self.expr[filtered_samples]
            
            print(f"\n✅ After filtering (excluded: {exclude_classes}):")
            print(self.metadata[group_column].value_counts())
        
        self.group_column = group_column
        
        print(f"\n✅ Final: {self.expr.shape[0]} genes × {self.expr.shape[1]} samples")
    
    def differential_expression(self, group1, group2):
        print(f"\n🔬 DIFFERENTIAL EXPRESSION: {group1} vs {group2}")
        print("-"*80)
        
        samples1 = self.metadata[self.metadata[self.group_column] == group1].index.tolist()
        samples2 = self.metadata[self.metadata[self.group_column] == group2].index.tolist()
        
        samples1 = [s for s in samples1 if s in self.expr.columns]
        samples2 = [s for s in samples2 if s in self.expr.columns]
        
        print(f"   {group1}: {len(samples1)} samples")
        print(f"   {group2}: {len(samples2)} samples")
        
        if len(samples1) < 2 or len(samples2) < 2:
            print(f"   ❌ Not enough samples for comparison")
            return None
        
        de_results = []
        
        print(f"\n   Analyzing {len(self.expr.index)} genes...")
        
        for idx, gene in enumerate(self.expr.index):
            if idx % 5000 == 0 and idx > 0:
                print(f"      Processed {idx}/{len(self.expr.index)} genes...")
            
            try:
                vals1 = self.expr.loc[gene, samples1].values.astype(float)
                vals2 = self.expr.loc[gene, samples2].values.astype(float)
                
                if vals1.std() == 0 and vals2.std() == 0:
                    continue
                
                t_stat, p_val = stats.ttest_ind(vals1, vals2, equal_var=False)
                
                mean1 = vals1.mean()
                mean2 = vals2.mean()
                
                if mean1 == 0 and mean2 == 0:
                    log2fc = 0
                elif mean2 == 0:
                    log2fc = np.inf
                else:
                    log2fc = np.log2((mean1 + 1) / (mean2 + 1))
                
                de_results.append({
                    'gene': gene,
                    'mean_group1': mean1,
                    'mean_group2': mean2,
                    'log2fc': log2fc,
                    'pvalue': p_val,
                    't_statistic': t_stat
                })
            except Exception as e:
                continue
        
        if not de_results:
            print(f"   ❌ No valid results")
            return None
        
        de_df = pd.DataFrame(de_results)
        
        de_df['abs_log2fc'] = de_df['log2fc'].abs()
        
        de_df = de_df.replace([np.inf, -np.inf], np.nan)
        de_df = de_df.dropna(subset=['log2fc', 'pvalue'])
        
        de_df['padj'] = de_df['pvalue'] * len(de_df)
        de_df['padj'] = de_df['padj'].clip(upper=1.0)
        
        de_df['-log10_pval'] = -np.log10(de_df['pvalue'])
        
        de_df = de_df.sort_values('pvalue')
        
        print(f"\n   ✅ Analyzed {len(de_df)} genes")
        
        sig_genes = de_df[(de_df['padj'] < 0.05) & (de_df['abs_log2fc'] > 1)]
        print(f"   📊 Significant genes (padj<0.05, |log2FC|>1): {len(sig_genes)}")
        
        upregulated = sig_genes[sig_genes['log2fc'] > 0]
        downregulated = sig_genes[sig_genes['log2fc'] < 0]
        
        print(f"      Upregulated in {group1}: {len(upregulated)}")
        print(f"      Downregulated in {group1}: {len(downregulated)}")
        
        return de_df
    
    def run_all_comparisons(self):
        print(f"\n🔬 RUNNING ALL PAIRWISE COMPARISONS")
        print("="*80)
        
        unique_groups = self.metadata[self.group_column].unique()
        print(f"Groups: {unique_groups}")
        
        all_results = {}
        
        for group1, group2 in combinations(unique_groups, 2):
            comparison_name = f"{group1}_vs_{group2}"
            print(f"\n{'='*80}")
            
            de_df = self.differential_expression(group1, group2)
            
            if de_df is not None:
                all_results[comparison_name] = de_df
                
                output_file = f"{self.output_dir}/DE_{comparison_name}.csv"
                de_df.to_csv(output_file, index=False)
                print(f"\n   💾 Saved: {output_file}")
                
                self.plot_volcano(de_df, group1, group2, comparison_name)
                
                top_genes = de_df.head(20)
                print(f"\n   📊 Top 20 genes by p-value:")
                for idx, row in top_genes.iterrows():
                    print(f"      {row['gene']}: log2FC={row['log2fc']:.2f}, p={row['pvalue']:.2e}")
        
        return all_results
    
    def plot_volcano(self, de_df, group1, group2, comparison_name):
        print(f"\n   🌋 Creating volcano plot...")
        
        try:
            de_clean = de_df.copy()
            
            sig_mask = (de_clean['padj'] < 0.05) & (de_clean['abs_log2fc'] > 1)
            
            plt.figure(figsize=(10, 8))
            
            plt.scatter(
                de_clean.loc[~sig_mask, 'log2fc'],
                de_clean.loc[~sig_mask, '-log10_pval'],
                c='gray',
                alpha=0.3,
                s=10,
                label='Not significant'
            )
            
            sig_up = de_clean[sig_mask & (de_clean['log2fc'] > 0)]
            plt.scatter(
                sig_up['log2fc'],
                sig_up['-log10_pval'],
                c='red',
                alpha=0.6,
                s=20,
                label=f'Upregulated in {group1}'
            )
            
            sig_down = de_clean[sig_mask & (de_clean['log2fc'] < 0)]
            plt.scatter(
                sig_down['log2fc'],
                sig_down['-log10_pval'],
                c='blue',
                alpha=0.6,
                s=20,
                label=f'Downregulated in {group1}'
            )
            
            plt.axhline(y=-np.log10(0.05), color='green', linestyle='--', alpha=0.5, linewidth=1, label='p=0.05')
            plt.axvline(x=-1, color='purple', linestyle='--', alpha=0.5, linewidth=1)
            plt.axvline(x=1, color='purple', linestyle='--', alpha=0.5, linewidth=1, label='|log2FC|=1')
            
            top_genes = de_clean.nsmallest(10, 'pvalue')
            for idx, row in top_genes.iterrows():
                if abs(row['log2fc']) > 1:
                    plt.annotate(
                        row['gene'],
                        xy=(row['log2fc'], row['-log10_pval']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7
                    )
            
            plt.xlabel('Log2 Fold Change', fontsize=12)
            plt.ylabel('-Log10(p-value)', fontsize=12)
            plt.title(f'Volcano Plot: {group1} vs {group2}', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/volcano_{comparison_name}.png', dpi=300)
            plt.close()
            
            print(f"      💾 Saved volcano plot")
            
        except Exception as e:
            print(f"      ⚠️  Could not create volcano plot: {e}")
    
    def plot_heatmap(self, de_results, comparison_name, top_n=50):
        print(f"\n   🔥 Creating heatmap for top {top_n} genes...")
        
        try:
            top_genes = de_results.nsmallest(top_n, 'pvalue')['gene'].tolist()
            
            expr_subset = self.expr.loc[top_genes]
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            expr_scaled = scaler.fit_transform(expr_subset.T).T
            expr_scaled_df = pd.DataFrame(
                expr_scaled,
                index=expr_subset.index,
                columns=expr_subset.columns
            )
            
            group_colors = self.metadata[self.group_column].map({
                'Control': 'green',
                'IGT': 'orange',
                'T2DM': 'red'
            })
            
            plt.figure(figsize=(12, max(8, top_n * 0.2)))
            
            sns.clustermap(
                expr_scaled_df,
                cmap='RdBu_r',
                center=0,
                col_colors=group_colors,
                figsize=(12, max(8, top_n * 0.2)),
                yticklabels=True,
                xticklabels=False,
                cbar_kws={'label': 'Z-score'}
            )
            
            plt.savefig(f'{self.output_dir}/heatmap_{comparison_name}_top{top_n}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"      💾 Saved heatmap")
            
        except Exception as e:
            print(f"      ⚠️  Could not create heatmap: {e}")
    
    def get_top_de_genes(self, all_results, top_n=100):
        print(f"\n📊 EXTRACTING TOP {top_n} DE GENES ACROSS ALL COMPARISONS")
        print("-"*80)
        
        all_genes_scores = {}
        
        for comparison, de_df in all_results.items():
            for idx, row in de_df.iterrows():
                gene = row['gene']
                score = row['abs_log2fc'] * row['-log10_pval']
                
                if gene not in all_genes_scores:
                    all_genes_scores[gene] = []
                
                all_genes_scores[gene].append(score)
        
        gene_max_scores = {gene: max(scores) for gene, scores in all_genes_scores.items()}
        
        sorted_genes = sorted(gene_max_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_genes = [gene for gene, score in sorted_genes[:top_n]]
        
        print(f"\n✅ Top {len(top_genes)} DE genes selected")
        print(f"\nTop 20:")
        for i, (gene, score) in enumerate(sorted_genes[:20], 1):
            print(f"   {i}. {gene}: {score:.3f}")
        
        top_genes_df = pd.DataFrame(sorted_genes[:top_n], columns=['gene', 'max_de_score'])
        top_genes_df.to_csv(f'{self.output_dir}/top_{top_n}_de_genes.csv', index=False)
        
        print(f"\n💾 Saved: {self.output_dir}/top_{top_n}_de_genes.csv")
        
        return top_genes
    
    def generate_summary_report(self, all_results):
        print(f"\n📋 GENERATING SUMMARY REPORT")
        print("-"*80)
        
        summary_data = []
        
        for comparison, de_df in all_results.items():
            sig_genes = de_df[(de_df['padj'] < 0.05) & (de_df['abs_log2fc'] > 1)]
            
            summary_data.append({
                'Comparison': comparison,
                'Total_Genes': len(de_df),
                'Significant_Genes': len(sig_genes),
                'Upregulated': len(sig_genes[sig_genes['log2fc'] > 0]),
                'Downregulated': len(sig_genes[sig_genes['log2fc'] < 0]),
                'Max_Log2FC': de_df['log2fc'].abs().max(),
                'Min_PValue': de_df['pvalue'].min()
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        print(summary_df.to_string(index=False))
        
        summary_df.to_csv(f'{self.output_dir}/summary_report.csv', index=False)
        
        print(f"\n💾 Saved: {self.output_dir}/summary_report.csv")
        
        return summary_df


def load_gse164416_data():
    print("="*80)
    print("📥 LOADING GSE164416 DATA")
    print("="*80)
    
    print("\n🧬 Loading expression data...")
    expr_path_gz = './suppl_data/GSE164416/GSE164416_DP_htseq_counts.txt.gz'
    
    if not os.path.exists(expr_path_gz):
        raise FileNotFoundError(f"Expression file not found: {expr_path_gz}")
    
    with gzip.open(expr_path_gz, 'rt') as f:
        expression_data = pd.read_csv(f, sep='\t', index_col=0)
    
    print(f"✅ Loaded: {expression_data.shape[0]} genes × {expression_data.shape[1]} samples")
    
    print("\n📋 Loading metadata...")
    metadata_path = './outputdata/GSE164416_metadata.csv'
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata = pd.read_csv(metadata_path, index_col=0)
    print(f"✅ Loaded: {metadata.shape[0]} samples")
    
    return expression_data, metadata


if __name__ == '__main__':
    
    expression_data, metadata = load_gse164416_data()
    
    de_analysis = DifferentialExpressionAnalysis(expression_data, metadata)
    
    de_analysis.prepare_data(group_column='diabetes_label', exclude_classes=['T3cD'])
    
    all_results = de_analysis.run_all_comparisons()
    
    if all_results:
        top_genes = de_analysis.get_top_de_genes(all_results, top_n=1000)
        
        summary = de_analysis.generate_summary_report(all_results)
        
        print("\n" + "="*80)
        print("✅ DIFFERENTIAL EXPRESSION ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\n📁 Results saved in: ./de_results/")
        print("\nGenerated files:")
        print("   • DE_[comparison].csv - Full results for each comparison")
        print("   • volcano_[comparison].png - Volcano plots")
        print("   • top_1000_de_genes.csv - Top DE genes across all comparisons")
        print("   • summary_report.csv - Summary statistics")
        print("="*80)
    else:
        print("\n❌ No DE results generated")