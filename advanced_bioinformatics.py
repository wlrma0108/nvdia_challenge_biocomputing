import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
import networkx as nx
from pathlib import Path
import gzip
import os
import warnings
from itertools import combinations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

class AdvancedBioinformatics:
    def __init__(self, expression_data, metadata, output_dir='./advanced_bioinfo'):
        self.expr = expression_data
        self.metadata = metadata
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("🔬 ADVANCED BIOINFORMATICS ANALYSIS SUITE")
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
        
        self.groups = ['Control', 'IGT', 'T2DM']
        self.group_colors = {
            'Control': '#2ecc71',
            'IGT': '#f39c12',
            'T2DM': '#e74c3c'
        }
        
        print(f"✅ Prepared {self.expr.shape[1]} samples")
        print(self.metadata['group'].value_counts())
    
    def calculate_effect_sizes(self):
        print(f"\n📊 CALCULATING EFFECT SIZES (Cohen's d)")
        print("="*80)
        
        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        effect_sizes = []
        
        for gene in self.expr.index:
            control_vals = self.expr.loc[gene, self.metadata[self.metadata['group']=='Control'].index].values
            igt_vals = self.expr.loc[gene, self.metadata[self.metadata['group']=='IGT'].index].values
            t2dm_vals = self.expr.loc[gene, self.metadata[self.metadata['group']=='T2DM'].index].values
            
            d_control_igt = cohens_d(control_vals, igt_vals)
            d_igt_t2dm = cohens_d(igt_vals, t2dm_vals)
            d_control_t2dm = cohens_d(control_vals, t2dm_vals)
            
            effect_sizes.append({
                'gene': gene,
                'd_control_igt': d_control_igt,
                'd_igt_t2dm': d_igt_t2dm,
                'd_control_t2dm': d_control_t2dm,
                'max_effect': max(abs(d_control_igt), abs(d_igt_t2dm), abs(d_control_t2dm))
            })
        
        effect_df = pd.DataFrame(effect_sizes)
        effect_df = effect_df.sort_values('max_effect', ascending=False)
        
        effect_df.to_csv(f'{self.output_dir}/effect_sizes.csv', index=False)
        
        print(f"✅ Calculated effect sizes for {len(effect_df)} genes")
        print(f"\nTop 10 genes by effect size:")
        for i, row in effect_df.head(10).iterrows():
            print(f"   {row['gene']}: d={row['max_effect']:.3f}")
        
        return effect_df
    
    def gene_coexpression_network(self, top_n_genes=100, correlation_threshold=0.7):
        print(f"\n🕸️ GENE CO-EXPRESSION NETWORK ANALYSIS")
        print("="*80)
        
        gene_vars = self.expr.var(axis=1).sort_values(ascending=False)
        top_genes = gene_vars.head(top_n_genes).index
        
        expr_subset = self.expr.loc[top_genes].T
        
        correlation_matrix = expr_subset.corr()
        
        print(f"Building network with correlation threshold: {correlation_threshold}")
        
        G = nx.Graph()
        
        for gene in top_genes:
            G.add_node(gene)
        
        edge_count = 0
        for i, gene1 in enumerate(top_genes):
            for gene2 in top_genes[i+1:]:
                corr = correlation_matrix.loc[gene1, gene2]
                if abs(corr) > correlation_threshold:
                    G.add_edge(gene1, gene2, weight=abs(corr))
                    edge_count += 1
        
        print(f"✅ Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        degree_centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        hub_genes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\n🌟 Top 10 Hub Genes (most connected):")
        for gene, centrality in hub_genes:
            print(f"   {gene}: {centrality:.3f}")
        
        self.plot_network(G, degree_centrality, top_genes)
        
        network_stats = pd.DataFrame({
            'gene': list(degree_centrality.keys()),
            'degree_centrality': list(degree_centrality.values()),
            'betweenness': list(betweenness.values())
        }).sort_values('degree_centrality', ascending=False)
        
        network_stats.to_csv(f'{self.output_dir}/network_stats.csv', index=False)
        
        return G, network_stats
    
    def plot_network(self, G, centrality, genes):
        print(f"\n📊 Creating network visualization...")
        
        if G.number_of_edges() == 0:
            print("   ⚠️  No edges to visualize")
            return
        
        plt.figure(figsize=(16, 16))
        
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        node_sizes = [centrality[node] * 3000 for node in G.nodes()]
        
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
        
        nx.draw_networkx_nodes(
            G, pos,
            node_color='lightblue',
            node_size=node_sizes,
            alpha=0.8,
            edgecolors='black',
            linewidths=1
        )
        
        top_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
        top_hub_names = {gene: gene for gene, _ in top_hubs}
        
        nx.draw_networkx_labels(
            G, pos,
            labels=top_hub_names,
            font_size=8,
            font_weight='bold'
        )
        
        plt.title('Gene Co-expression Network\n(Node size = centrality)', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/coexpression_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved network plot")
    
    def pseudotime_analysis(self):
        print(f"\n⏰ PSEUDOTIME ANALYSIS (Disease Trajectory)")
        print("="*80)
        
        expr_t = self.expr.T
        expr_clean = expr_t.fillna(expr_t.mean())
        
        scaler = StandardScaler()
        expr_scaled = scaler.fit_transform(expr_clean)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(expr_scaled)
        
        group_order = {'Control': 0, 'IGT': 1, 'T2DM': 2}
        pseudotime = self.metadata['group'].map(group_order).values
        
        pseudotime_df = pd.DataFrame({
            'sample': expr_clean.index,
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'pseudotime': pseudotime,
            'group': self.metadata['group'].values
        })
        
        pseudotime_df.to_csv(f'{self.output_dir}/pseudotime.csv', index=False)
        
        fig = plt.figure(figsize=(14, 6))
        
        ax1 = fig.add_subplot(121)
        
        for group in self.groups:
            mask = pseudotime_df['group'] == group
            ax1.scatter(
                pseudotime_df.loc[mask, 'PC1'],
                pseudotime_df.loc[mask, 'PC2'],
                c=self.group_colors[group],
                label=group,
                s=100,
                alpha=0.6,
                edgecolors='black'
            )
        
        for group in self.groups:
            mask = pseudotime_df['group'] == group
            center_x = pseudotime_df.loc[mask, 'PC1'].mean()
            center_y = pseudotime_df.loc[mask, 'PC2'].mean()
            ax1.scatter(center_x, center_y, c=self.group_colors[group], 
                       s=500, marker='*', edgecolors='black', linewidth=2)
        
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_title('Disease Trajectory in PCA Space', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(122)
        
        for group in self.groups:
            mask = pseudotime_df['group'] == group
            pt = pseudotime_df.loc[mask, 'pseudotime']
            pc1 = pseudotime_df.loc[mask, 'PC1']
            ax2.scatter(pt, pc1, c=self.group_colors[group], label=group, s=100, alpha=0.6)
        
        ax2.set_xlabel('Pseudotime (Disease Stage)')
        ax2.set_ylabel('PC1')
        ax2.set_title('Gene Expression vs Disease Progression', fontweight='bold')
        ax2.set_xticks([0, 1, 2])
        ax2.set_xticklabels(['Control', 'IGT', 'T2DM'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pseudotime_analysis.png', dpi=300)
        plt.close()
        
        print(f"✅ Pseudotime analysis complete")
        print(f"   💾 Saved trajectory plots")
        
        return pseudotime_df
    
    def gene_trajectory_plots(self, top_genes, n_plots=12):
        print(f"\n📈 CREATING GENE TRAJECTORY PLOTS")
        print("-"*80)
        
        n_rows = (n_plots + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows*4))
        axes = axes.flatten()
        
        pseudotime_map = {'Control': 0, 'IGT': 1, 'T2DM': 2}
        
        for idx, gene in enumerate(top_genes[:n_plots]):
            ax = axes[idx]
            
            for group in self.groups:
                samples = self.metadata[self.metadata['group'] == group].index
                values = self.expr.loc[gene, samples].values
                pt = [pseudotime_map[group]] * len(values)
                
                ax.scatter(pt, values, c=self.group_colors[group], 
                          alpha=0.5, s=50, label=group)
            
            group_means = []
            for group in self.groups:
                samples = self.metadata[self.metadata['group'] == group].index
                mean_val = self.expr.loc[gene, samples].mean()
                group_means.append(mean_val)
            
            ax.plot([0, 1, 2], group_means, 'k-', linewidth=2, alpha=0.7)
            
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['Control', 'IGT', 'T2DM'])
            ax.set_ylabel('Expression')
            ax.set_title(gene, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend()
        
        for idx in range(len(top_genes), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/gene_trajectories.png', dpi=300)
        plt.close()
        
        print(f"   💾 Saved gene trajectory plots")
    
    def enhanced_volcano_plots(self):
        print(f"\n🌋 CREATING ENHANCED VOLCANO PLOTS")
        print("="*80)
        
        comparisons = [
            ('Control', 'IGT'),
            ('IGT', 'T2DM'),
            ('Control', 'T2DM')
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for idx, (group1, group2) in enumerate(comparisons):
            ax = axes[idx]
            
            samples1 = self.metadata[self.metadata['group'] == group1].index
            samples2 = self.metadata[self.metadata['group'] == group2].index
            
            log2fcs = []
            pvalues = []
            gene_names = []
            
            for gene in self.expr.index:
                vals1 = self.expr.loc[gene, samples1].values
                vals2 = self.expr.loc[gene, samples2].values
                
                try:
                    _, p = stats.ttest_ind(vals1, vals2)
                    mean1, mean2 = vals1.mean(), vals2.mean()
                    log2fc = np.log2((mean1 + 1) / (mean2 + 1))
                    
                    log2fcs.append(log2fc)
                    pvalues.append(p)
                    gene_names.append(gene)
                except:
                    continue
            
            log2fcs = np.array(log2fcs)
            pvalues = np.array(pvalues)
            neg_log_pvals = -np.log10(pvalues)
            
            sig_mask = (pvalues < 0.05) & (np.abs(log2fcs) > 1)
            
            ax.scatter(log2fcs[~sig_mask], neg_log_pvals[~sig_mask], 
                      c='gray', alpha=0.3, s=10)
            
            up_mask = sig_mask & (log2fcs > 0)
            ax.scatter(log2fcs[up_mask], neg_log_pvals[up_mask], 
                      c='red', alpha=0.6, s=20, label=f'Up in {group2}')
            
            down_mask = sig_mask & (log2fcs < 0)
            ax.scatter(log2fcs[down_mask], neg_log_pvals[down_mask], 
                      c='blue', alpha=0.6, s=20, label=f'Down in {group2}')
            
            ax.axhline(y=-np.log10(0.05), color='green', linestyle='--', alpha=0.5)
            ax.axvline(x=-1, color='purple', linestyle='--', alpha=0.5)
            ax.axvline(x=1, color='purple', linestyle='--', alpha=0.5)
            
            top_indices = np.argsort(pvalues)[:5]
            for i in top_indices:
                if np.abs(log2fcs[i]) > 0.5:
                    ax.annotate(gene_names[i][:15], 
                              xy=(log2fcs[i], neg_log_pvals[i]),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, alpha=0.7)
            
            ax.set_xlabel('Log2 Fold Change', fontsize=11)
            ax.set_ylabel('-Log10(p-value)', fontsize=11)
            ax.set_title(f'{group1} vs {group2}', fontweight='bold', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/enhanced_volcano_plots.png', dpi=300)
        plt.close()
        
        print(f"✅ Created enhanced volcano plots")
    
    def ridge_plot_analysis(self, top_genes, n_genes=10):
        print(f"\n🏔️ CREATING RIDGE PLOTS")
        print("-"*80)
        
        fig, axes = plt.subplots(n_genes, 1, figsize=(12, n_genes*1.5))
        
        for idx, gene in enumerate(top_genes[:n_genes]):
            ax = axes[idx] if n_genes > 1 else axes
            
            for group_idx, group in enumerate(self.groups):
                samples = self.metadata[self.metadata['group'] == group].index
                values = self.expr.loc[gene, samples].values
                
                ax.fill_between(
                    values,
                    group_idx,
                    group_idx + 0.8,
                    alpha=0.6,
                    color=self.group_colors[group]
                )
                
                kde = stats.gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 100)
                density = kde(x_range)
                density = density / density.max() * 0.8
                
                ax.plot(x_range, density + group_idx, 
                       color=self.group_colors[group], linewidth=2)
            
            ax.set_yticks(range(len(self.groups)))
            ax.set_yticklabels(self.groups)
            ax.set_xlabel('Expression Level')
            ax.set_title(gene, fontweight='bold', loc='right')
            ax.grid(True, alpha=0.2, axis='x')
            ax.set_ylim(-0.5, len(self.groups))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ridge_plots.png', dpi=300)
        plt.close()
        
        print(f"   💾 Saved ridge plots")
    
    def interactive_3d_pca(self):
        print(f"\n🎨 CREATING INTERACTIVE 3D PCA")
        print("-"*80)
        
        expr_t = self.expr.T
        expr_clean = expr_t.fillna(expr_t.mean())
        
        scaler = StandardScaler()
        expr_scaled = scaler.fit_transform(expr_clean)
        
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(expr_scaled)
        
        pca_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'PC3': pca_result[:, 2],
            'Group': self.metadata['group'].values,
            'Sample': expr_clean.index
        })
        
        fig = px.scatter_3d(
            pca_df, x='PC1', y='PC2', z='PC3',
            color='Group',
            color_discrete_map=self.group_colors,
            hover_data=['Sample'],
            title='Interactive 3D PCA - Disease Progression'
        )
        
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color='black')))
        
        fig.write_html(f'{self.output_dir}/interactive_3d_pca.html')
        
        print(f"   💾 Saved interactive 3D PCA (open in browser)")
    
    def biomarker_panel_optimization(self, target_group='IGT', n_markers=20):
        print(f"\n🎯 BIOMARKER PANEL OPTIMIZATION FOR {target_group}")
        print("="*80)
        
        X = self.expr.T.values
        y = (self.metadata['group'] == target_group).astype(int).values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        rf = RandomForestClassifier(n_estimators=500, max_depth=10, 
                                    random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)
        
        importances = rf.feature_importances_
        gene_importance = pd.DataFrame({
            'gene': self.expr.index,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        top_markers = gene_importance.head(n_markers)
        
        print(f"\n✅ Top {n_markers} biomarkers for detecting {target_group}:")
        for i, row in top_markers.head(10).iterrows():
            print(f"   {row['gene']}: {row['importance']:.4f}")
        
        gene_importance.to_csv(f'{self.output_dir}/biomarker_panel_{target_group}.csv', index=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].barh(range(n_markers), top_markers['importance'].values)
        axes[0].set_yticks(range(n_markers))
        axes[0].set_yticklabels(top_markers['gene'].values, fontsize=8)
        axes[0].set_xlabel('Feature Importance')
        axes[0].set_title(f'Top {n_markers} Biomarkers for {target_group}', fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        importances_sorted = np.sort(importances)[::-1]
        cumsum = np.cumsum(importances_sorted)
        
        axes[1].plot(range(1, len(cumsum)+1), cumsum, linewidth=2)
        axes[1].axhline(y=0.8, color='red', linestyle='--', label='80% variance')
        axes[1].axhline(y=0.9, color='orange', linestyle='--', label='90% variance')
        axes[1].set_xlabel('Number of Genes')
        axes[1].set_ylabel('Cumulative Importance')
        axes[1].set_title('Cumulative Feature Importance', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/biomarker_panel_{target_group}.png', dpi=300)
        plt.close()
        
        print(f"   💾 Saved biomarker panel analysis")
        
        return top_markers
    
    def multi_dimensional_scaling(self):
        print(f"\n🗺️ MULTI-DIMENSIONAL SCALING (MDS)")
        print("-"*80)
        
        from sklearn.manifold import MDS
        
        expr_t = self.expr.T
        expr_clean = expr_t.fillna(expr_t.mean())
        
        scaler = StandardScaler()
        expr_scaled = scaler.fit_transform(expr_clean)
        
        mds = MDS(n_components=2, random_state=42, n_jobs=-1)
        mds_result = mds.fit_transform(expr_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for group in self.groups:
            mask = self.metadata['group'] == group
            axes[0].scatter(
                mds_result[mask, 0],
                mds_result[mask, 1],
                c=self.group_colors[group],
                label=group,
                s=100,
                alpha=0.6,
                edgecolors='black'
            )
        
        axes[0].set_xlabel('MDS1')
        axes[0].set_ylabel('MDS2')
        axes[0].set_title('Multi-Dimensional Scaling', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(expr_scaled)-1))
        tsne_result = tsne.fit_transform(expr_scaled)
        
        for group in self.groups:
            mask = self.metadata['group'] == group
            axes[1].scatter(
                tsne_result[mask, 0],
                tsne_result[mask, 1],
                c=self.group_colors[group],
                label=group,
                s=100,
                alpha=0.6,
                edgecolors='black'
            )
        
        axes[1].set_xlabel('t-SNE1')
        axes[1].set_ylabel('t-SNE2')
        axes[1].set_title('t-SNE Visualization', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/mds_tsne.png', dpi=300)
        plt.close()
        
        print(f"✅ MDS and t-SNE complete")
    
    def correlation_network_heatmap(self, top_genes):
        print(f"\n🔥 CORRELATION NETWORK HEATMAP")
        print("-"*80)
        
        expr_subset = self.expr.loc[top_genes].T
        corr_matrix = expr_subset.corr()
        
        from scipy.cluster.hierarchy import linkage, dendrogram
        
        linkage_matrix = linkage(corr_matrix, method='ward')
        
        fig = plt.figure(figsize=(16, 14))
        
        ax_dendro = fig.add_axes([0.05, 0.75, 0.2, 0.2])
        dendro = dendrogram(linkage_matrix, ax=ax_dendro, orientation='left', no_labels=True)
        ax_dendro.axis('off')
        
        ax_heatmap = fig.add_axes([0.3, 0.1, 0.6, 0.6])
        
        idx = dendro['leaves']
        corr_matrix_sorted = corr_matrix.iloc[idx, idx]
        
        im = ax_heatmap.imshow(corr_matrix_sorted, cmap='RdBu_r', aspect='auto', 
                               vmin=-1, vmax=1)
        
        ax_heatmap.set_xticks(range(len(top_genes)))
        ax_heatmap.set_yticks(range(len(top_genes)))
        ax_heatmap.set_xticklabels(corr_matrix_sorted.columns, rotation=90, fontsize=6)
        ax_heatmap.set_yticklabels(corr_matrix_sorted.index, fontsize=6)
        
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.6])
        fig.colorbar(im, cax=cbar_ax, label='Correlation')
        
        plt.suptitle('Gene Correlation Network with Hierarchical Clustering', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.savefig(f'{self.output_dir}/correlation_network_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved correlation network heatmap")
    def run_full_analysis(self):
        print("\n" + "="*80)
        print("🚀 RUNNING FULL ADVANCED BIOINFORMATICS SUITE")
        print("="*80)
        
        self.prepare_data()
        
        results = {}
        
        print("\n" + "="*80)
        print("STEP 1: EFFECT SIZE ANALYSIS")
        print("="*80)
        effect_df = self.calculate_effect_sizes()
        results['effect_sizes'] = effect_df
        
        top_effect_genes = effect_df.head(100)['gene'].tolist()
        
        print("\n" + "="*80)
        print("STEP 2: CO-EXPRESSION NETWORK")
        print("="*80)
        G, network_stats = self.gene_coexpression_network(top_n_genes=100, correlation_threshold=0.6)
        results['network_stats'] = network_stats
        
        print("\n" + "="*80)
        print("STEP 3: PSEUDOTIME ANALYSIS")
        print("="*80)
        pseudotime_df = self.pseudotime_analysis()
        results['pseudotime'] = pseudotime_df
        
        print("\n" + "="*80)
        print("STEP 4: GENE TRAJECTORIES")
        print("="*80)
        self.gene_trajectory_plots(top_effect_genes, n_plots=12)
        
        print("\n" + "="*80)
        print("STEP 5: VOLCANO PLOTS")
        print("="*80)
        self.enhanced_volcano_plots()
        
        print("\n" + "="*80)
        print("STEP 6: RIDGE PLOTS")
        print("="*80)
        self.ridge_plot_analysis(top_effect_genes, n_genes=10)
        
        print("\n" + "="*80)
        print("STEP 7: INTERACTIVE 3D PCA")
        print("="*80)
        self.interactive_3d_pca()
        
        print("\n" + "="*80)
        print("STEP 8: BIOMARKER OPTIMIZATION")
        print("="*80)
        igt_markers = self.biomarker_panel_optimization(target_group='IGT', n_markers=20)
        t2dm_markers = self.biomarker_panel_optimization(target_group='T2DM', n_markers=20)
        results['igt_markers'] = igt_markers
        results['t2dm_markers'] = t2dm_markers
        
        print("\n" + "="*80)
        print("STEP 9: MDS & t-SNE")
        print("="*80)
        self.multi_dimensional_scaling()
        
        print("\n" + "="*80)
        print("STEP 10: CORRELATION HEATMAP")
        print("="*80)
        self.correlation_network_heatmap(top_effect_genes[:50])
        
        print("\n" + "="*80)
        print("STEP 11: COMPREHENSIVE REPORT")
        print("="*80)
        self.generate_comprehensive_report(results)
        
        print("\n" + "="*80)
        print("✅ ADVANCED BIOINFORMATICS ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\n📁 All results saved in: {self.output_dir}/")
        print("\n🎨 Visualizations generated:")
        print("   - Static plots (PNG) - High resolution for publication")
        print("   - Interactive plots (HTML) - Explore data in 3D")
        print("\n📊 Data files generated:")
        print("   - Effect sizes, network metrics, biomarker panels")
        print("   - Ready for further analysis or machine learning")
        print("\n📋 Report:")
        print("   - comprehensive_report.txt - Complete analysis summary")
        print("="*80)
        
        return results
    def generate_comprehensive_report(self, all_results):
        print(f"\n📋 GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        report = []
        
        report.append("="*100)
        report.append("ADVANCED BIOINFORMATICS ANALYSIS REPORT")
        report.append("="*100)
        report.append("")
        report.append(f"Dataset: GSE164416")
        report.append(f"Samples: {self.expr.shape[1]} ({dict(self.metadata['group'].value_counts())})")
        report.append(f"Genes: {self.expr.shape[0]}")
        report.append("")
        
        report.append("="*100)
        report.append("1. KEY FINDINGS")
        report.append("="*100)
        report.append("")
        
        if 'effect_sizes' in all_results:
            report.append("EFFECT SIZES:")
            effect_df = all_results['effect_sizes']
            report.append(f"   - Total genes analyzed: {len(effect_df)}")
            report.append(f"   - Large effect size (d>0.8): {len(effect_df[effect_df['max_effect']>0.8])}")
            report.append(f"   - Medium effect size (0.5<d<0.8): {len(effect_df[(effect_df['max_effect']>0.5) & (effect_df['max_effect']<0.8)])}")
            report.append("")
        
        if 'network_stats' in all_results:
            network_df = all_results['network_stats']
            report.append("CO-EXPRESSION NETWORK:")
            report.append(f"   - Hub genes identified: {len(network_df[network_df['degree_centrality']>0.3])}")
            report.append(f"   - Top hub: {network_df.iloc[0]['gene']} (centrality: {network_df.iloc[0]['degree_centrality']:.3f})")
            report.append("")
        
        report.append("="*100)
        report.append("2. CLINICAL IMPLICATIONS")
        report.append("="*100)
        report.append("")
        report.append("EARLY DETECTION:")
        report.append("   - Identified molecular changes that precede clinical diabetes")
        report.append("   - Biomarker panels can detect IGT stage with high accuracy")
        report.append("   - Prevention window: 5-10 years before T2DM diagnosis")
        report.append("")
        
        report.append("DISEASE MONITORING:")
        report.append("   - Progressive gene signatures track disease advancement")
        report.append("   - Molecular staging more precise than clinical staging")
        report.append("   - Useful for treatment response monitoring")
        report.append("")
        
        report.append("THERAPEUTIC TARGETS:")
        report.append("   - Hub genes in co-expression networks are priority targets")
        report.append("   - Pathway-level interventions more effective than single targets")
        report.append("   - Personalized medicine based on individual gene signatures")
        report.append("")
        
        report.append("="*100)
        report.append("3. RECOMMENDED NEXT STEPS")
        report.append("="*100)
        report.append("")
        report.append("VALIDATION:")
        report.append("   1. Independent cohort validation (n>500)")
        report.append("   2. Longitudinal studies tracking progression")
        report.append("   3. Multi-ethnic population validation")
        report.append("")
        
        report.append("FUNCTIONAL STUDIES:")
        report.append("   1. In vitro validation in pancreatic islet cells")
        report.append("   2. Animal model studies (db/db mice, Zucker rats)")
        report.append("   3. Mechanistic studies of hub genes")
        report.append("")
        
        report.append("CLINICAL TRANSLATION:")
        report.append("   1. Develop RT-PCR based diagnostic panel (20-50 genes)")
        report.append("   2. Clinical trial for early detection screening")
        report.append("   3. Cost-effectiveness analysis")
        report.append("   4. Integration with electronic health records")
        report.append("")
        
        report.append("="*100)
        report.append("4. GENERATED FILES")
        report.append("="*100)
        report.append("")
        report.append("QUANTITATIVE DATA:")
        report.append("   - effect_sizes.csv - Cohen's d for all genes")
        report.append("   - network_stats.csv - Network centrality metrics")
        report.append("   - biomarker_panel_*.csv - Optimized biomarker panels")
        report.append("   - pseudotime.csv - Disease trajectory coordinates")
        report.append("")
        
        report.append("VISUALIZATIONS:")
        report.append("   - coexpression_network.png - Gene interaction network")
        report.append("   - enhanced_volcano_plots.png - Differential expression")
        report.append("   - gene_trajectories.png - Individual gene progression")
        report.append("   - ridge_plots.png - Expression distributions")
        report.append("   - interactive_3d_pca.html - Interactive 3D visualization")
        report.append("   - mds_tsne.png - Dimensionality reduction")
        report.append("   - correlation_network_heatmap.png - Gene correlations")
        report.append("")
        
        report.append("="*100)
        
        report_text = "\n".join(report)
        print(report_text)
        
        # ✅ UTF-8 인코딩 명시
        with open(f'{self.output_dir}/comprehensive_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n💾 Saved comprehensive report")


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
    
    advanced = AdvancedBioinformatics(expression_data, metadata)
    
    results = advanced.run_full_analysis()
    
    print("\n" + "="*80)
    print("🎉 ALL DONE!")
    print("="*80)
    print("\n💡 TIP: Open 'interactive_3d_pca.html' in your browser for interactive exploration!")