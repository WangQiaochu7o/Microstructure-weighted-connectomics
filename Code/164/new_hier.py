import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import os
from collections import defaultdict, Counter
import nibabel as nib
from nilearn import plotting, datasets
from surfer import Brain
from PIL import Image
import csv
import seaborn as sns
from sklearn.cluster import SpectralClustering
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import bct
import networkx as nx
from pathlib import Path
import warnings
from scipy import stats
import pickle
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Fix for newer matplotlib versions
if not hasattr(mpl_cm, 'get_cmap'):
    def get_cmap(name):
        """Compatibility function for newer matplotlib versions"""
        try:
            return plt.get_cmap(name)
        except:
            return matplotlib.colormaps[name]
    
    mpl_cm.get_cmap = get_cmap

class AnalysisConfig:
    """Configuration for analysis parameters"""
    
    def __init__(self):
        # Data paths
        self.base_path = "/home/localadmin/Desktop/SP2"
        self.group_connectomes_dir = "/home/localadmin/Desktop/SP2/164output/pro_group_average"
        self.output_path = "/home/localadmin/Desktop/SP2/164output/group_averaged_output"
        self.freesurfer_home = '/home/localadmin/freesurfer'
        self.subjects_dir = os.path.join(self.freesurfer_home, 'subjects')
        
        # Metrics
        self.metrics = ['R1', 'RK_DKI', 'RD_DTI_reciprocal', 'NOS', 'F_SMI', 'FA_DTI', 'DePerp_SMI_reciprocal']
        
        # Hierarchical clustering parameters
        self.hierarchical_max_clusters = 20
        self.hierarchical_linkage_methods = ['ward'] # Only use 'ward' method

config = AnalysisConfig()

# Set environment variables
os.environ['FREESURFER_HOME'] = config.freesurfer_home
os.environ['SUBJECTS_DIR'] = config.subjects_dir

# Define colors for different weighted metrics
metric_colors = {
    'R1': '#1E90FF',           # Blue
    'DePerp_SMI_reciprocal': '#FF8C00',       # Orange  
    'FA_DTI': '#228B22',  # Green
    'RD_DTI_reciprocal': '#DC143C',          # Red
    'NOS': '#9932CC',        # Purple
    'RK_DKI': '#FF69B4',       # Pink
    'F_SMI': '#8B4513'  # Brown
}

def ensure_symmetric_matrix(W):
    """Ensure matrix is symmetric and remove diagonal"""
    W_sym = (W + W.T) / 2
    np.fill_diagonal(W_sym, 0)
    W_sym = np.round(W_sym, decimals=3)
    return W_sym

def get_corrected_a2009s_atlas_mapping():
    """
    CORRECTED mapping for MRtrix3 FreeSurfer a2009s parcellation (Destrieux atlas)
    Based on the fs_a2009s.txt lookup table
    """
    cortical_regions = [
        'G_and_S_frontomargin', 'G_and_S_occipital_inf', 'G_and_S_paracentral',
        'G_and_S_subcentral', 'G_and_S_transv_frontopol', 'G_and_S_cingul-Ant',
        'G_and_S_cingul-Mid-Ant', 'G_and_S_cingul-Mid-Post', 'G_cingul-Post-dorsal',
        'G_cingul-Post-ventral', 'G_cuneus', 'G_front_inf-Opercular',
        'G_front_inf-Orbital', 'G_front_inf-Triangul', 'G_front_middle',
        'G_front_sup', 'G_Ins_lg_and_S_cent_ins', 'G_insular_short',
        'G_occipital_middle', 'G_occipital_sup', 'G_oc-temp_lat-fusifor',
        'G_oc-temp_med-Lingual', 'G_oc-temp_med-Parahip', 'G_orbital',
        'G_pariet_inf-Angular', 'G_pariet_inf-Supramar', 'G_parietal_sup',
        'G_postcentral', 'G_precentral', 'G_precuneus', 'G_rectus',
        'G_subcallosal', 'G_temp_sup-G_T_transv', 'G_temp_sup-Lateral',
        'G_temp_sup-Plan_polar', 'G_temp_sup-Plan_tempo', 'G_temporal_inf',
        'G_temporal_middle', 'Lat_Fis-ant-Horizont', 'Lat_Fis-ant-Vertical',
        'Lat_Fis-post', 'Pole_occipital', 'Pole_temporal', 'S_calcarine',
        'S_central', 'S_cingul-Marginalis', 'S_circular_insula_ant',
        'S_circular_insula_inf', 'S_circular_insula_sup', 'S_collat_transv_ant',
        'S_collat_transv_post', 'S_front_inf', 'S_front_middle', 'S_front_sup',
        'S_interm_prim-Jensen', 'S_intrapariet_and_P_trans', 'S_oc_middle_and_Lunatus',
        'S_oc_sup_and_transversal', 'S_occipital_ant', 'S_oc-temp_lat',
        'S_oc-temp_med_and_Lingual', 'S_orbital_lateral', 'S_orbital_med-olfact',
        'S_orbital-H_Shaped', 'S_parieto_occipital', 'S_pericallosal',
        'S_postcentral', 'S_precentral-inf-part', 'S_precentral-sup-part',
        'S_suborbital', 'S_subparietal', 'S_temporal_inf', 'S_temporal_sup',
        'S_temporal_transverse'
    ]
    node_to_region = {}
    for i, region in enumerate(cortical_regions):
        node_to_region[i + 1] = f"lh.{region}"
    for i, region in enumerate(cortical_regions):
        node_to_region[90 + i] = f"rh.{region}"
    return node_to_region

def compute_uncentered_correlation_distance(W):
    """Compute distance matrix using uncentered (non-centered) Pearson correlation coefficient"""
    W_sym = ensure_symmetric_matrix(W)
    n_regions = W_sym.shape[0]
    correlation_matrix = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(i, n_regions):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                x, y = W_sym[i, :], W_sym[j, :]
                numerator = np.sum(x * y)
                denominator = np.sqrt(np.sum(x**2) * np.sum(y**2))
                corr = numerator / denominator if denominator > 1e-10 else 0.0
                correlation_matrix[i, j] = correlation_matrix[j, i] = corr
    distance_matrix = 1 - np.abs(correlation_matrix)
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix, correlation_matrix

def perform_hierarchical_clustering(distance_matrix, linkage_method='ward'):
    """Perform hierarchical clustering using specified linkage method"""
    distance_condensed = squareform(distance_matrix)
    linkage_matrix = linkage(distance_condensed, method=linkage_method)
    return linkage_matrix

def find_optimal_clusters_hierarchical(linkage_matrix, W_sym, max_clusters=20):
    """Find optimal number of clusters using silhouette analysis"""
    silhouette_scores = []
    cluster_range = range(2, min(max_clusters + 1, W_sym.shape[0]))
    for n_clusters in cluster_range:
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        if len(np.unique(cluster_labels)) > 1:
            try:
                score = silhouette_score(W_sym, cluster_labels, metric='euclidean')
                silhouette_scores.append(score)
            except:
                silhouette_scores.append(-1)
        else:
            silhouette_scores.append(-1)
    
    if len(silhouette_scores) > 0 and max(silhouette_scores) > -1:
        optimal_idx = np.argmax(silhouette_scores)
        optimal_n_clusters = cluster_range[optimal_idx]
        optimal_silhouette = silhouette_scores[optimal_idx]
    else:
        optimal_n_clusters = 2
        optimal_silhouette = -1
    
    return optimal_n_clusters, optimal_silhouette, list(cluster_range), silhouette_scores

def compute_hierarchical_clustering_analysis(W):
    """Compute hierarchical clustering analysis using only the 'ward' method"""
    print("    Starting hierarchical clustering computation...")
    W_sym = ensure_symmetric_matrix(W)
    
    print("    Computing uncentered correlation distance matrix...")
    distance_matrix, correlation_matrix = compute_uncentered_correlation_distance(W_sym)
    
    results = {}
    linkage_method = 'ward'
    
    try:
        print(f"      Running linkage method: {linkage_method}")
        linkage_matrix = perform_hierarchical_clustering(distance_matrix, linkage_method)
        
        print(f"        Finding optimal number of clusters (max: {config.hierarchical_max_clusters})...")
        optimal_n_clusters, optimal_silhouette, cluster_range, silhouette_scores = find_optimal_clusters_hierarchical(
            linkage_matrix, W_sym, config.hierarchical_max_clusters
        )
        
        cluster_assignments = fcluster(linkage_matrix, optimal_n_clusters, criterion='maxclust')
        
        method_result = {
            'linkage_method': linkage_method,
            'linkage_matrix': linkage_matrix,
            'distance_matrix': distance_matrix,
            'correlation_matrix': correlation_matrix,
            'optimal_n_clusters': optimal_n_clusters,
            'optimal_silhouette': optimal_silhouette,
            'cluster_assignments': cluster_assignments,
            'cluster_range': cluster_range,
            'silhouette_scores': silhouette_scores,
            'success': True
        }
        results[linkage_method] = method_result
        print(f"        Result: {optimal_n_clusters} clusters, silhouette: {optimal_silhouette:.3f}")
        
    except Exception as e:
        print(f"        Failed: {e}")
        results[linkage_method] = {'success': False, 'error': str(e)}
    
    return {
        'all_methods': results,
        'best_result': results.get('ward'),
        'distance_matrix': distance_matrix,
        'correlation_matrix': correlation_matrix
    }

def save_hierarchical_results(results, output_dir, metric):
    """Save hierarchical clustering analysis results"""
    best_result = results.get('best_result')
    if not best_result or not best_result['success']:
        print(f"    [ERROR] No valid hierarchical clustering results for {metric}")
        return

    print(f"    Saving results for {metric}...")
    
    # Save silhouette sweep data
    sweep_df = pd.DataFrame({
        'n_clusters': best_result['cluster_range'],
        'silhouette_score': best_result['silhouette_scores']
    })
    sweep_path = os.path.join(output_dir, f'hierarchical_silhouette_sweep_ward_{metric}.csv')
    sweep_df.to_csv(sweep_path, index=False)
    print(f"      Saved silhouette sweep data: {sweep_path}")

def generate_dynamic_colormap(n_communities):
    """Generate a dynamic colormap based on the number of communities"""
    if n_communities <= 10:
        base_colors = ['#FF0000', '#0000FF', '#00AA00', '#FF66CC', '#FF8800', '#8B00CC', '#996633', '#00CCCC', '#FFFF00', '#888888']
        return ListedColormap(base_colors[:n_communities])
    elif n_communities <= 20:
        return plt.get_cmap('tab20')
    else:
        colors = []
        cmaps = ['tab20', 'tab20b', 'tab20c']
        for i in range(n_communities):
            cmap_idx, color_idx = divmod(i, 20)
            if cmap_idx < len(cmaps):
                cmap = plt.get_cmap(cmaps[cmap_idx])
                colors.append(cmap(color_idx / 19.0))
            else:
                np.random.seed(i)
                colors.append(np.random.rand(3))
        return ListedColormap(colors)

def load_freesurfer_destrieux_atlas():
    """Load FreeSurfer Destrieux atlas (a2009s) for 164 regions"""
    try:
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
        lh_annot_path = f"{config.freesurfer_home}/subjects/fsaverage/label/lh.aparc.a2009s.annot"
        rh_annot_path = f"{config.freesurfer_home}/subjects/fsaverage/label/rh.aparc.a2009s.annot"
        
        lh_labels, _, lh_names = nib.freesurfer.read_annot(lh_annot_path)
        rh_labels, _, rh_names = nib.freesurfer.read_annot(rh_annot_path)
        
        lh_names = [name.decode('utf-8') for name in lh_names]
        rh_names = [name.decode('utf-8') for name in rh_names]
            
        return lh_labels, rh_labels, fsaverage, lh_names, rh_names
    except Exception as e:
        print(f"Error loading Destrieux atlas: {e}")
        return None, None, None, None, None

def map_data_to_surface(data_values, lh_labels, rh_labels, lh_names, rh_names):
    """Map data values to brain surface using Destrieux atlas"""
    lh_data = np.zeros(len(lh_labels), dtype=float)
    rh_data = np.zeros(len(rh_labels), dtype=float)
    
    node_to_region = get_corrected_a2009s_atlas_mapping()
    
    lh_name_to_label = {f"lh.{name}": i for i, name in enumerate(lh_names)}
    rh_name_to_label = {f"rh.{name}": i for i, name in enumerate(rh_names)}
    
    for connectome_idx, data_value in enumerate(data_values):
        fs_node_idx = connectome_idx + 1
        if fs_node_idx not in node_to_region or data_value == 0:
            continue
        
        region_name = node_to_region[fs_node_idx]
        
        if region_name.startswith('lh.'):
            label_idx = lh_name_to_label.get(region_name)
            if label_idx is not None:
                lh_data[lh_labels == label_idx] = float(data_value)
        else: # rh
            label_idx = rh_name_to_label.get(region_name)
            if label_idx is not None:
                rh_data[rh_labels == label_idx] = float(data_value)
                
    return np.nan_to_num(lh_data), np.nan_to_num(rh_data)

def create_ward_silhouette_sweep_plot(sweep_data, metric, output_dir):
    """Create a silhouette sweep plot for the 'ward' method for a given metric."""
    print(f"Creating silhouette sweep plot for {metric}...")
    if not isinstance(sweep_data, pd.DataFrame) or sweep_data.empty:
        print(f"  No sweep data for {metric}.")
        return

    color = metric_colors.get(metric, '#888888')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sweep_data['n_clusters'], sweep_data['silhouette_score'], 
            color=color, linewidth=3, marker='o', markersize=8, label=metric)
    
    if not sweep_data.empty:
        best_idx = sweep_data['silhouette_score'].idxmax()
        optimal_n = sweep_data.loc[best_idx, 'n_clusters']
        optimal_score = sweep_data.loc[best_idx, 'silhouette_score']
        
        ax.axvline(x=optimal_n, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.plot(optimal_n, optimal_score, 'D', color='red', markersize=10, 
                markeredgecolor='black', markeredgewidth=1.5,
                label=f'Optimal k={optimal_n} ({optimal_score:.3f})')
    
    ax.set_xlabel('Number of Clusters (k)', fontsize=14)
    ax.set_ylabel('Silhouette Score', fontsize=14)
    ax.set_title(f'Silhouette Score vs. Number of Clusters for {metric}\n(Ward Method)', fontsize=16, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=12)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"hierarchical_silhouette_sweep_ward_{metric}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {output_file}")

def create_brain_visualization_for_cluster_numbers(metric, linkage_matrix, output_dir):
    """Create brain visualizations showing how cluster assignments change with the number of clusters."""
    print(f"\nCreating brain visualization across cluster numbers for {metric}...")
    
    lh_labels, rh_labels, fsaverage, lh_names, rh_names = load_freesurfer_destrieux_atlas()
    if lh_labels is None:
        print("  Failed to load FreeSurfer atlas, skipping visualization.")
        return

    views = [('lateral', 'lh', 'Left Lateral'), ('medial', 'lh', 'Left Medial'), 
             ('lateral', 'rh', 'Right Lateral'), ('medial', 'rh', 'Right Medial')]
    
    cluster_numbers_to_plot = range(2, 21) # Plot for k=2, 3, 4, 5, 6
    
    fig, axes = plt.subplots(len(cluster_numbers_to_plot), len(views), 
                             figsize=(16, 4 * len(cluster_numbers_to_plot)),
                             gridspec_kw={'wspace': 0.01, 'hspace': 0.15})
    
    for row, n_clusters in enumerate(cluster_numbers_to_plot):
        print(f"  Processing for k={n_clusters} clusters...")
        cluster_assignments = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        unique_clusters = np.unique(cluster_assignments[cluster_assignments > 0])
        num_clusters_found = len(unique_clusters)
        
        if num_clusters_found == 0:
            print(f"    No clusters found for k={n_clusters}")
            # Fill row with placeholder
            for col in range(len(views)):
                axes[row, col].text(0.5, 0.5, 'No Clusters', ha='center', va='center')
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            continue

        custom_cmap = generate_dynamic_colormap(num_clusters_found)
        lh_data, rh_data = map_data_to_surface(cluster_assignments, lh_labels, rh_labels, lh_names, rh_names)
        
        for col, (view, hemi, view_title) in enumerate(views):
            ax = axes[row, col]
            try:
                brain = Brain('fsaverage', hemi, 'pial', cortex='low_contrast', background='white', size=(800, 600))
                surface_data = lh_data if hemi == 'lh' else rh_data
                
                brain.add_data(surface_data, min=0.5, max=np.max(unique_clusters), colormap=custom_cmap,
                               alpha=1.0, smoothing_steps=0, thresh=0.5, colorbar=False)
                brain.show_view(view)
                
                temp_file = f"{output_dir}/temp_{metric}_{n_clusters}_{hemi}_{view}.png"
                brain.save_image(temp_file, mode='rgb', antialiased=True)
                img = Image.open(temp_file)
                ax.imshow(np.array(img))
                brain.close()
                os.remove(temp_file)
            except Exception as e:
                print(f"    Error creating view {view_title} for k={n_clusters}: {e}")
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', color='red')
            
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(view_title, fontsize=14, fontweight='bold', pad=10)
            if col == 0:
                ax.set_ylabel(f'k = {n_clusters}', rotation=0, labelpad=30, fontsize=12, fontweight='bold', ha='right', va='center')

    fig.suptitle(f'Hierarchical Clustering Assignments for {metric} (Ward Method)', fontsize=16, fontweight='bold', y=0.98)
    
    output_file = f"{output_dir}/hierarchical_assignments_by_n_clusters_{metric}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved brain visualization grid: {output_file}")

def process_connectome_hierarchical_analysis(connectome_path, output_dir, metric):
    """Process a single connectome with hierarchical clustering analysis."""
    try:
        print(f"\nProcessing Hierarchical Analysis for Metric: {metric}")
        W = pd.read_csv(connectome_path, header=None, index_col=None).values
        if W.shape[0] != 164 or W.shape[1] != 164:
            print(f"  [ERROR] Connectome matrix is not 164x164. Shape is {W.shape}")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        
        hierarchical_results = compute_hierarchical_clustering_analysis(W)
        
        if hierarchical_results and hierarchical_results.get('best_result'):
            print("  Hierarchical clustering analysis completed successfully.")
            save_hierarchical_results(hierarchical_results, output_dir, metric)
            return hierarchical_results
        else:
            print("  [ERROR] Hierarchical clustering analysis failed.")
            return None
            
    except Exception as e:
        print(f"  [FATAL ERROR] processing metric {metric}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function for hierarchical clustering analysis."""
    print("="*80)
    print("HIERARCHICAL CLUSTERING ANALYSIS - WARD METHOD")
    print("="*80)
    
    os.makedirs(config.output_path, exist_ok=True)
    processed, failed = 0, 0
    
    for metric in config.metrics:
        connectome_path = os.path.join(config.group_connectomes_dir, f"group_{metric}_median.csv")
        
        if os.path.exists(connectome_path):
            results = process_connectome_hierarchical_analysis(connectome_path, config.output_path, metric)
            
            if results and results.get('best_result'):
                # Plot 1: Silhouette Sweep Plot
                sweep_data_path = os.path.join(config.output_path, f'hierarchical_silhouette_sweep_ward_{metric}.csv')
                if os.path.exists(sweep_data_path):
                    sweep_df = pd.read_csv(sweep_data_path)
                    create_ward_silhouette_sweep_plot(sweep_df, metric, config.output_path)
                
                # Plot 2: Brain Visualization Grid
                linkage_matrix = results['best_result']['linkage_matrix']
                create_brain_visualization_for_cluster_numbers(metric, linkage_matrix, config.output_path)
                
                processed += 1
            else:
                failed += 1
                print(f"[FAILED] {metric} analysis was unsuccessful.")
        else:
            print(f"[ERROR] Connectome file not found: {connectome_path}")
            failed += 1
            
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Successfully processed: {processed}/{len(config.metrics)} metrics")
    print(f"Failed: {failed}/{len(config.metrics)} metrics")
    print("="*80)

if __name__ == "__main__":
    main()