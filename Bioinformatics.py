import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import gzip
import os
import warnings
warnings.filterwarnings('ignore')

class BioinformaticsInsights:
    def __init__(self, expression_data, metadata, de_results_dir='./de_results', output_dir='./bioinfo_insights'):
        self.expr = expression_data
        self.metadata = metadata
        self.de_results_dir = de_results_dir
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("🧬 BIOINFORMATICS INSIGHTS ANALYSIS")
        print("="*80)
        print(f"\n📊 Data:")
        print(f"   Expression: {self.expr.shape[0]} genes × {self.expr.shape[1]} samples")
        print(f"   Metadata: {self.metadata.shape[0]} samples")
    
    def prepare_data(self, group_column='diabetes_label', exclude_classes=['T3cD']):
        print(f"\n🔧 PREPARING DATA")
        print("-"*80)
        
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
        self.expr = self.expr[common_samples]
        self.metadata = self.metadata.loc[common_samples]
        
        if exclude_classes:
            mask = ~self.metadata[group_column].isin(exclude_classes)
            self.metadata = self.metadata[mask]
            filtered_samples = self.metadata.index.tolist()
            self.expr = self.expr[filtered_samples]
        
        self.group_column = group_column
        self.metadata['group'] = self.metadata[group_column]
        
        print(f"✅ Prepared {self.expr.shape[1]} samples")
        print(self.metadata['group'].value_counts())
    
    def progressive_gene_changes(self):
        print(f"\n📈 ANALYZING PROGRESSIVE GENE CHANGES")
        print("="*80)
        print("Looking for genes that change progressively: Control → IGT → T2DM")
        
        control_samples = self.metadata[self.metadata['group'] == 'Control'].index
        igt_samples = self.metadata[self.metadata['group'] == 'IGT'].index
        t2dm_samples = self.metadata[self.metadata['group'] == 'T2DM'].index
        
        progressive_genes = []
        
        print(f"\nAnalyzing {len(self.expr)} genes...")
        
        for gene in self.expr.index:
            try:
                control_mean = self.expr.loc[gene, control_samples].mean()
                igt_mean = self.expr.loc[gene, igt_samples].mean()
                t2dm_mean = self.expr.loc[gene, t2dm_samples].mean()
                
                if control_mean == 0 and igt_mean == 0 and t2dm_mean == 0:
                    continue
                
                trend_up = (control_mean < igt_mean < t2dm_mean)
                trend_down = (control_mean > igt_mean > t2dm_mean)
                
                if trend_up or trend_down:
                    control_vals = self.expr.loc[gene, control_samples].values
                    igt_vals = self.expr.loc[gene, igt_samples].values
                    t2dm_vals = self.expr.loc[gene, t2dm_samples].values
                    
                    _, p_control_igt = stats.ttest_ind(control_vals, igt_vals)
                    _, p_igt_t2dm = stats.ttest_ind(igt_vals, t2dm_vals)
                    _, p_control_t2dm = stats.ttest_ind(control_vals, t2dm_vals)
                    
                    if p_control_t2dm < 0.05:
                        fc_control_igt = np.log2((igt_mean + 1) / (control_mean + 1))
                        fc_igt_t2dm = np.log2((t2dm_mean + 1) / (igt_mean + 1))
                        fc_control_t2dm = np.log2((t2dm_mean + 1) / (control_mean + 1))
                        
                        progressive_genes.append({
                            'gene': gene,
                            'control_mean': control_mean,
                            'igt_mean': igt_mean,
                            't2dm_mean': t2dm_mean,
                            'fc_control_igt': fc_control_igt,
                            'fc_igt_t2dm': fc_igt_t2dm,
                            'fc_control_t2dm': fc_control_t2dm,
                            'p_control_igt': p_control_igt,
                            'p_igt_t2dm': p_igt_t2dm,
                            'p_control_t2dm': p_control_t2dm,
                            'trend': 'increasing' if trend_up else 'decreasing'
                        })
            except:
                continue
        
        if not progressive_genes:
            print("❌ No progressive genes found")
            return None
        
        prog_df = pd.DataFrame(progressive_genes)
        prog_df = prog_df.sort_values('p_control_t2dm')
        
        print(f"\n✅ Found {len(prog_df)} progressive genes")
        print(f"   Increasing: {len(prog_df[prog_df['trend']=='increasing'])}")
        print(f"   Decreasing: {len(prog_df[prog_df['trend']=='decreasing'])}")
        
        prog_df.to_csv(f'{self.output_dir}/progressive_genes.csv', index=False)
        
        self.plot_progressive_heatmap(prog_df, top_n=50)
        self.plot_progressive_trajectories(prog_df, top_n=20)
        
        return prog_df
    
    def plot_progressive_heatmap(self, prog_df, top_n=50):
        print(f"\n🔥 Creating progressive genes heatmap...")
        
        top_genes = prog_df.head(top_n)['gene'].tolist()
        
        expr_subset = self.expr.loc[top_genes]
        
        scaler = StandardScaler()
        expr_scaled = scaler.fit_transform(expr_subset.T).T
        
        expr_scaled_df = pd.DataFrame(
            expr_scaled,
            index=expr_subset.index,
            columns=expr_subset.columns
        )
        
        group_order = self.metadata.sort_values('group')
        expr_ordered = expr_scaled_df[group_order.index]
        
        group_colors = group_order['group'].map({
            'Control': '#2ecc71',
            'IGT': '#f39c12', 
            'T2DM': '#e74c3c'
        })
        
        fig, ax = plt.subplots(figsize=(14, max(10, top_n * 0.3)))
        
        sns.heatmap(
            expr_ordered,
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Z-score'},
            yticklabels=True,
            xticklabels=False,
            ax=ax
        )
        
        for i, (idx, row) in enumerate(group_order.iterrows()):
            ax.add_patch(plt.Rectangle((i, 0), 1, len(top_genes), 
                                      facecolor=group_colors[idx], 
                                      edgecolor='none', alpha=0.3))
        
        plt.title(f'Progressive Gene Expression Changes\n(Control → IGT → T2DM)', 
                 fontsize=14, pad=20)
        plt.xlabel('Samples (ordered by group)', fontsize=12)
        plt.ylabel('Genes', fontsize=12)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Control'),
            Patch(facecolor='#f39c12', label='IGT'),
            Patch(facecolor='#e74c3c', label='T2DM')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/progressive_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved progressive heatmap")
    
    def plot_progressive_trajectories(self, prog_df, top_n=20):
        print(f"\n📊 Creating trajectory plots...")
        
        top_increasing = prog_df[prog_df['trend']=='increasing'].head(top_n//2)
        top_decreasing = prog_df[prog_df['trend']=='decreasing'].head(top_n//2)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        for ax, subset, title in zip(axes, 
                                     [top_increasing, top_decreasing],
                                     ['Top Increasing Genes', 'Top Decreasing Genes']):
            
            for _, row in subset.iterrows():
                gene = row['gene']
                means = [row['control_mean'], row['igt_mean'], row['t2dm_mean']]
                
                ax.plot([0, 1, 2], means, marker='o', alpha=0.6, linewidth=1.5)
            
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['Control', 'IGT', 'T2DM'], fontsize=11)
            ax.set_ylabel('Mean Expression', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/progressive_trajectories.png', dpi=300)
        plt.close()
        
        print(f"   💾 Saved trajectory plots")
    
    def early_vs_late_markers(self):
        print(f"\n🎯 IDENTIFYING EARLY vs LATE MARKERS")
        print("="*80)
        
        control_samples = self.metadata[self.metadata['group'] == 'Control'].index
        igt_samples = self.metadata[self.metadata['group'] == 'IGT'].index
        t2dm_samples = self.metadata[self.metadata['group'] == 'T2DM'].index
        
        early_markers = []
        late_markers = []
        
        print("Analyzing genes...")
        
        for gene in self.expr.index:
            try:
                control_vals = self.expr.loc[gene, control_samples].values
                igt_vals = self.expr.loc[gene, igt_samples].values
                t2dm_vals = self.expr.loc[gene, t2dm_samples].values
                
                control_mean = control_vals.mean()
                igt_mean = igt_vals.mean()
                t2dm_mean = t2dm_vals.mean()
                
                _, p_control_igt = stats.ttest_ind(control_vals, igt_vals)
                _, p_igt_t2dm = stats.ttest_ind(igt_vals, t2dm_vals)
                _, p_control_t2dm = stats.ttest_ind(control_vals, t2dm_vals)
                
                fc_control_igt = np.log2((igt_mean + 1) / (control_mean + 1))
                fc_igt_t2dm = np.log2((t2dm_mean + 1) / (igt_mean + 1))
                
                if p_control_igt < 0.05 and abs(fc_control_igt) > 0.5:
                    if p_igt_t2dm > 0.1:
                        early_markers.append({
                            'gene': gene,
                            'fc_control_igt': fc_control_igt,
                            'fc_igt_t2dm': fc_igt_t2dm,
                            'p_control_igt': p_control_igt,
                            'p_igt_t2dm': p_igt_t2dm,
                            'control_mean': control_mean,
                            'igt_mean': igt_mean,
                            't2dm_mean': t2dm_mean
                        })
                
                if p_control_igt > 0.1 and p_igt_t2dm < 0.05 and abs(fc_igt_t2dm) > 0.5:
                    late_markers.append({
                        'gene': gene,
                        'fc_control_igt': fc_control_igt,
                        'fc_igt_t2dm': fc_igt_t2dm,
                        'p_control_igt': p_control_igt,
                        'p_igt_t2dm': p_igt_t2dm,
                        'control_mean': control_mean,
                        'igt_mean': igt_mean,
                        't2dm_mean': t2dm_mean
                    })
            except:
                continue
        
        early_df = pd.DataFrame(early_markers)
        late_df = pd.DataFrame(late_markers)
        
        if not early_df.empty:
            early_df = early_df.sort_values('p_control_igt')
            early_df.to_csv(f'{self.output_dir}/early_markers.csv', index=False)
            print(f"\n✅ Early markers (change in Control→IGT): {len(early_df)}")
            print("   Top 10:")
            for i, row in early_df.head(10).iterrows():
                print(f"      {row['gene']}: FC={row['fc_control_igt']:.2f}, p={row['p_control_igt']:.2e}")
        
        if not late_df.empty:
            late_df = late_df.sort_values('p_igt_t2dm')
            late_df.to_csv(f'{self.output_dir}/late_markers.csv', index=False)
            print(f"\n✅ Late markers (change in IGT→T2DM): {len(late_df)}")
            print("   Top 10:")
            for i, row in late_df.head(10).iterrows():
                print(f"      {row['gene']}: FC={row['fc_igt_t2dm']:.2f}, p={row['p_igt_t2dm']:.2e}")
        
        self.plot_early_late_comparison(early_df, late_df)
        
        return early_df, late_df
    
    def plot_early_late_comparison(self, early_df, late_df):
        print(f"\n📊 Creating early vs late markers plot...")
        
        if early_df.empty and late_df.empty:
            print("   ⚠️  No markers to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        if not early_df.empty:
            top_early = early_df.head(10)
            
            for _, row in top_early.iterrows():
                gene = row['gene']
                means = [row['control_mean'], row['igt_mean'], row['t2dm_mean']]
                axes[0].plot([0, 1, 2], means, marker='o', linewidth=2, label=gene[:15])
            
            axes[0].set_xticks([0, 1, 2])
            axes[0].set_xticklabels(['Control', 'IGT', 'T2DM'])
            axes[0].set_ylabel('Mean Expression')
            axes[0].set_title('Early Markers\n(Change in Control→IGT)', fontweight='bold')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            axes[0].grid(True, alpha=0.3)
        
        if not late_df.empty:
            top_late = late_df.head(10)
            
            for _, row in top_late.iterrows():
                gene = row['gene']
                means = [row['control_mean'], row['igt_mean'], row['t2dm_mean']]
                axes[1].plot([0, 1, 2], means, marker='o', linewidth=2, label=gene[:15])
            
            axes[1].set_xticks([0, 1, 2])
            axes[1].set_xticklabels(['Control', 'IGT', 'T2DM'])
            axes[1].set_ylabel('Mean Expression')
            axes[1].set_title('Late Markers\n(Change in IGT→T2DM)', fontweight='bold')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/early_late_markers.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved early vs late markers plot")
    
    def disease_progression_pca(self):
        print(f"\n🔄 DISEASE PROGRESSION PCA ANALYSIS")
        print("="*80)
        
        expr_t = self.expr.T
        expr_clean = expr_t.fillna(expr_t.mean())
        
        scaler = StandardScaler()
        expr_scaled = scaler.fit_transform(expr_clean)
        
        # ✅ 최대 10개 또는 샘플 수만큼만
        n_components = min(10, expr_scaled.shape[0], expr_scaled.shape[1])
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(expr_scaled)
        
        pca_df = pd.DataFrame(
            pca_result[:, :3],  # 처음 3개만 사용
            index=expr_clean.index,
            columns=['PC1', 'PC2', 'PC3']
        )
        pca_df['group'] = self.metadata.loc[pca_df.index, 'group']
        
        var_explained = pca.explained_variance_ratio_
        
        print(f"\n✅ PCA complete")
        print(f"   PC1: {var_explained[0]*100:.2f}%")
        print(f"   PC2: {var_explained[1]*100:.2f}%")
        print(f"   PC3: {var_explained[2]*100:.2f}%")
        
        fig = plt.figure(figsize=(16, 5))
        
        ax1 = fig.add_subplot(131)
        colors = {'Control': '#2ecc71', 'IGT': '#f39c12', 'T2DM': '#e74c3c'}
        
        for group in ['Control', 'IGT', 'T2DM']:
            mask = pca_df['group'] == group
            ax1.scatter(
                pca_df.loc[mask, 'PC1'],
                pca_df.loc[mask, 'PC2'],
                c=colors[group],
                label=group,
                s=100,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax1.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
        ax1.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
        ax1.set_title('Disease Progression in PCA Space', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(132, projection='3d')
        
        for group in ['Control', 'IGT', 'T2DM']:
            mask = pca_df['group'] == group
            ax2.scatter(
                pca_df.loc[mask, 'PC1'],
                pca_df.loc[mask, 'PC2'],
                pca_df.loc[mask, 'PC3'],
                c=colors[group],
                label=group,
                s=100,
                alpha=0.6
            )
        
        ax2.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
        ax2.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
        ax2.set_zlabel(f'PC3 ({var_explained[2]*100:.1f}%)')
        ax2.set_title('3D PCA View', fontweight='bold')
        ax2.legend()
        
        ax3 = fig.add_subplot(133)
        # ✅ 실제 컴포넌트 수만큼만 그리기
        n_bars = min(10, len(var_explained))
        ax3.bar(range(1, n_bars + 1), var_explained[:n_bars] * 100)
        ax3.set_xlabel('Principal Component')
        ax3.set_ylabel('Variance Explained (%)')
        ax3.set_title('Scree Plot', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/disease_progression_pca.png', dpi=300)
        plt.close()
        
        print(f"   💾 Saved PCA plots")
        
        pca_df.to_csv(f'{self.output_dir}/pca_coordinates.csv')
        
        return pca_df, pca
    
    def gene_expression_boxplots(self, top_genes, n_plots=12):
        print(f"\n📦 CREATING GENE EXPRESSION BOXPLOTS")
        print("-"*80)
        
        n_rows = (n_plots + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows*4))
        axes = axes.flatten() if n_plots > 3 else [axes]
        
        for idx, gene in enumerate(top_genes[:n_plots]):
            ax = axes[idx]
            
            data_to_plot = []
            labels = []
            
            for group in ['Control', 'IGT', 'T2DM']:
                samples = self.metadata[self.metadata['group'] == group].index
                values = self.expr.loc[gene, samples].values
                data_to_plot.append(values)
                labels.append(group)
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.set_ylabel('Expression Level')
            ax.set_title(gene, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        for idx in range(len(top_genes), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/gene_expression_boxplots.png', dpi=300)
        plt.close()
        
        print(f"   💾 Saved boxplots for {min(n_plots, len(top_genes))} genes")
    
    def hierarchical_clustering(self, top_genes=100):
        print(f"\n🌳 HIERARCHICAL CLUSTERING ANALYSIS")
        print("-"*80)
        
        expr_subset = self.expr.loc[top_genes]
        
        scaler = StandardScaler()
        expr_scaled = scaler.fit_transform(expr_subset.T).T
        
        group_colors = self.metadata['group'].map({
            'Control': '#2ecc71',
            'IGT': '#f39c12',
            'T2DM': '#e74c3c'
        })
        
        g = sns.clustermap(
            expr_scaled,
            cmap='RdBu_r',
            center=0,
            col_colors=group_colors,
            figsize=(14, 12),
            yticklabels=True,
            xticklabels=False,
            cbar_kws={'label': 'Z-score'},
            method='ward',
            metric='euclidean'
        )
        
        g.fig.suptitle('Hierarchical Clustering of Top Genes', 
                      fontsize=14, y=0.99, fontweight='bold')
        
        plt.savefig(f'{self.output_dir}/hierarchical_clustering.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved hierarchical clustering")
    
    def generate_biological_summary(self, prog_df, early_df, late_df):
        print(f"\n📋 GENERATING BIOLOGICAL SUMMARY")
        print("="*80)
        
        summary = []
        
        summary.append("="*80)
        summary.append("BIOLOGICAL INSIGHTS SUMMARY")
        summary.append("="*80)
        summary.append("")
        
        summary.append("1. PROGRESSIVE GENE CHANGES (Control → IGT → T2DM)")
        summary.append("-" * 80)
        if prog_df is not None and not prog_df.empty:
            summary.append(f"   Total progressive genes: {len(prog_df)}")
            increasing = prog_df[prog_df['trend']=='increasing']
            decreasing = prog_df[prog_df['trend']=='decreasing']
            summary.append(f"   - Increasing: {len(increasing)}")
            summary.append(f"   - Decreasing: {len(decreasing)}")
            summary.append("")
            summary.append("   Top 5 increasing genes:")
            for i, row in increasing.head(5).iterrows():
                summary.append(f"      {row['gene']}: {row['control_mean']:.1f} → {row['igt_mean']:.1f} → {row['t2dm_mean']:.1f}")
        else:
            summary.append("   No progressive genes found")
        summary.append("")
        
        summary.append("2. EARLY MARKERS (Control → IGT)")
        summary.append("-" * 80)
        if early_df is not None and not early_df.empty:
            summary.append(f"   Total early markers: {len(early_df)}")
            summary.append("   These genes change BEFORE diabetes develops")
            summary.append("   → Potential for early intervention")
            summary.append("")
            summary.append("   Top 5 early markers:")
            for i, row in early_df.head(5).iterrows():
                summary.append(f"      {row['gene']}: log2FC={row['fc_control_igt']:.2f}, p={row['p_control_igt']:.2e}")
        else:
            summary.append("   No early markers found")
        summary.append("")
        
        summary.append("3. LATE MARKERS (IGT → T2DM)")
        summary.append("-" * 80)
        if late_df is not None and not late_df.empty:
            summary.append(f"   Total late markers: {len(late_df)}")
            summary.append("   These genes change DURING disease progression")
            summary.append("   → Markers of disease severity")
            summary.append("")
            summary.append("   Top 5 late markers:")
            for i, row in late_df.head(5).iterrows():
                summary.append(f"      {row['gene']}: log2FC={row['fc_igt_t2dm']:.2f}, p={row['p_igt_t2dm']:.2e}")
        else:
            summary.append("   No late markers found")
        summary.append("")
        
        summary.append("4. CLINICAL IMPLICATIONS")
        summary.append("-" * 80)
        summary.append("   • Early markers → Screening/prevention strategies")
        summary.append("   • Progressive genes → Track disease progression")
        summary.append("   • Late markers → Therapeutic targets for established disease")
        summary.append("")
        
        summary.append("5. RECOMMENDED NEXT STEPS")
        summary.append("-" * 80)
        summary.append("   1. Pathway enrichment analysis (KEGG, GO)")
        summary.append("   2. Protein-protein interaction network analysis")
        summary.append("   3. Validation in independent cohorts")
        summary.append("   4. Functional studies on top candidates")
        summary.append("   5. Development of diagnostic panels")
        summary.append("")
        
        summary.append("="*80)
        
        summary_text = "\n".join(summary)
        print(summary_text)
        
        with open(f'{self.output_dir}/biological_summary.txt', 'w') as f:
            f.write(summary_text)
        
        print(f"\n💾 Saved biological summary")
    
    def run_full_analysis(self):
        print("\n" + "="*80)
        print("🚀 RUNNING FULL BIOINFORMATICS ANALYSIS")
        print("="*80)
        
        self.prepare_data()
        
        prog_df = self.progressive_gene_changes()
        
        early_df, late_df = self.early_vs_late_markers()
        
        pca_df, pca = self.disease_progression_pca()
        
        if prog_df is not None and not prog_df.empty:
            top_genes = prog_df.head(100)['gene'].tolist()
            self.hierarchical_clustering(top_genes)
            self.gene_expression_boxplots(top_genes[:12])
        
        self.generate_biological_summary(prog_df, early_df, late_df)
        
        print("\n" + "="*80)
        print("✅ BIOINFORMATICS ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\n📁 Results saved in: {self.output_dir}/")
        print("\nGenerated files:")
        print("   • progressive_genes.csv")
        print("   • early_markers.csv")
        print("   • late_markers.csv")
        print("   • progressive_heatmap.png")
        print("   • progressive_trajectories.png")
        print("   • early_late_markers.png")
        print("   • disease_progression_pca.png")
        print("   • hierarchical_clustering.png")
        print("   • gene_expression_boxplots.png")
        print("   • biological_summary.txt")
        print("="*80)


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
    
    bioinfo = BioinformaticsInsights(expression_data, metadata)
    
    bioinfo.run_full_analysis()
    
    print("\n" + "="*80)
    print("🎉 ALL DONE!")
    print("="*80)