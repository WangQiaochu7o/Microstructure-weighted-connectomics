import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import os
from PIL import Image
import nibabel as nib
from nilearn import plotting, datasets
from surfer import Brain
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

# --- [1. Configuration Module] ---

# Fix for newer matplotlib versions
if not hasattr(mpl_cm, 'get_cmap'):
    def get_cmap(name):
        try: return plt.get_cmap(name)
        except: return plt.colormaps[name]
    mpl_cm.get_cmap = get_cmap

class AnalysisConfig:
    """Configuration for analysis parameters"""
    def __init__(self):
        self.base_path = "/home/localadmin/Desktop/SP2"
        self.group_connectomes_dir = "/home/localadmin/Desktop/SP2/164output/pro_group_average"
        self.output_path = "/home/localadmin/Desktop/SP2/164output/group_averaged_output/mixed"
        self.freesurfer_home = '/home/localadmin/freesurfer'
        self.subjects_dir = os.path.join(self.freesurfer_home, 'subjects')
        self.metrics = ['R1', 'RK_DKI', 'RD_DTI_reciprocal', 'NOS', 'F_SMI', 'FA_DTI', 'DePerp_SMI_reciprocal']
        self.hierarchical_max_clusters = 20
        # MODIFICATION: Removed hierarchical_linkage_methods as it's now determined dynamically.

config = AnalysisConfig()

# Set environment variables
os.environ['FREESURFER_HOME'] = config.freesurfer_home
os.environ['SUBJECTS_DIR'] = config.subjects_dir

# Define colors and LaTeX names for metrics
metric_colors = {
    'R1': '#1E90FF', 'DePerp_SMI_reciprocal': '#FF8C00', 'FA_DTI': '#228B22',
    'RD_DTI_reciprocal': '#DC143C', 'NOS': '#9932CC', 'RK_DKI': '#FF69B4', 'F_SMI': '#8B4513'
}
metric_name_map_latex = {
    'NOS': r'$nos$', 'RK_DKI': r'$rk$', 'FA_DTI': r'$fa$',
    'RD_DTI_reciprocal': r'$1/rd$', 'DePerp_SMI_reciprocal': r'$1/D_{\perp e}$',
    'F_SMI': r'$f$', 'R1': r'$R1$'
}

# --- [2. Core Analysis Functions] ---

def ensure_symmetric_matrix(W):
    W_sym = (W + W.T) / 2
    np.fill_diagonal(W_sym, 0)
    return np.round(W_sym, decimals=3)

def compute_uncentered_correlation_distance(W):
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
    return distance_matrix

def perform_hierarchical_clustering(distance_matrix, linkage_method='ward'):
    distance_condensed = squareform(distance_matrix)
    return linkage(distance_condensed, method=linkage_method)

def find_optimal_clusters_hierarchical(linkage_matrix, W_sym, max_clusters=20):
    silhouette_scores = []
    cluster_range = range(2, min(max_clusters + 1, W_sym.shape[0]))
    for n_clusters in cluster_range:
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        if len(np.unique(cluster_labels)) > 1:
            score = silhouette_score(W_sym, cluster_labels, metric='euclidean')
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1)
    
    optimal_n_clusters, optimal_silhouette = 2, -1
    if silhouette_scores and max(silhouette_scores) > -1:
        optimal_idx = np.argmax(silhouette_scores)
        optimal_n_clusters = cluster_range[optimal_idx]
        optimal_silhouette = silhouette_scores[optimal_idx]
    
    return optimal_n_clusters, optimal_silhouette, list(cluster_range), silhouette_scores

# MODIFICATION: The function now accepts 'metric' to determine the linkage method.
def compute_hierarchical_clustering_analysis(W, metric):
    """
    Compute hierarchical clustering.
    Uses 'single' method for 'NOS' metric, and 'ward' for all others.
    """
    print("    Starting hierarchical clustering...")
    W_sym = ensure_symmetric_matrix(W)
    print("    Computing distance matrix...")
    distance_matrix = compute_uncentered_correlation_distance(W_sym)
    
    # MODIFICATION: Determine the linkage method based on the metric.
    if metric == 'NOS':
        linkage_method = 'single'
    else:
        linkage_method = 'ward'

    try:
        # MODIFICATION: Use the dynamically chosen linkage_method.
        print(f"      Running {linkage_method.capitalize()} linkage for metric '{metric}'...")
        linkage_matrix = perform_hierarchical_clustering(distance_matrix, linkage_method)
        print(f"        Finding optimal number of clusters (max: {config.hierarchical_max_clusters})...")
        optimal_n, optimal_score, cluster_range, scores = find_optimal_clusters_hierarchical(
            linkage_matrix, W_sym, config.hierarchical_max_clusters
        )
        result = {
            'linkage_matrix': linkage_matrix, 'optimal_n_clusters': optimal_n,
            'optimal_silhouette': optimal_score, 'cluster_range': cluster_range,
            'silhouette_scores': scores, 'success': True
        }
        print(f"        Success: Found {optimal_n} optimal clusters with score {optimal_score:.3f}")
        return result
    except Exception as e:
        print(f"        Failed: {e}")
        return {'success': False, 'error': str(e)}

# --- [3. Visualization and Helper Functions for Destrieux Atlas] ---

def get_corrected_a2009s_atlas_mapping():
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
    for i, region in enumerate(cortical_regions): node_to_region[i + 1] = f"lh.{region}"
    for i, region in enumerate(cortical_regions): node_to_region[90 + i] = f"rh.{region}"
    return node_to_region

def generate_dynamic_colormap(n_communities):
    if n_communities <= 10: return plt.get_cmap('tab10', n_communities)
    if n_communities <= 20: return plt.get_cmap('tab20', n_communities)
    return plt.get_cmap('gist_ncar', n_communities)

def load_freesurfer_destrieux_atlas():
    try:
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
        lh_annot_path = os.path.join(config.subjects_dir, 'fsaverage', 'label', 'lh.aparc.a2009s.annot')
        rh_annot_path = os.path.join(config.subjects_dir, 'fsaverage', 'label', 'rh.aparc.a2009s.annot')
        lh_labels, _, lh_names = nib.freesurfer.read_annot(lh_annot_path)
        rh_labels, _, rh_names = nib.freesurfer.read_annot(rh_annot_path)
        return lh_labels, rh_labels, [n.decode('utf-8') for n in lh_names], [n.decode('utf-8') for n in rh_names]
    except Exception as e:
        print(f"Error loading Destrieux atlas: {e}")
        return None, None, None, None

def map_data_to_surface_destrieux(data_values, lh_labels, rh_labels, lh_names, rh_names):
    lh_data, rh_data = np.zeros(len(lh_labels)), np.zeros(len(rh_labels))
    node_to_region = get_corrected_a2009s_atlas_mapping()
    lh_name_to_label = {name: i for i, name in enumerate(lh_names)}
    rh_name_to_label = {name: i for i, name in enumerate(rh_names)}
    
    for connectome_idx, data_value in enumerate(data_values):
        mrtrix_idx = connectome_idx + 1
        if mrtrix_idx in node_to_region and data_value != 0:
            hemi_prefix, region_name = node_to_region[mrtrix_idx].split('.', 1)
            label_map, surface_data, labels = (lh_name_to_label, lh_data, lh_labels) if hemi_prefix == 'lh' else (rh_name_to_label, rh_data, rh_labels)
            label_val = label_map.get(region_name)
            if label_val is not None:
                surface_data[labels == label_val] = float(data_value)
    return np.nan_to_num(lh_data), np.nan_to_num(rh_data)

# --- [4. Plotting Functions] ---

def create_aggregated_silhouette_sweep_plot(all_sweep_data, all_optimal_data, output_dir):
    print("Creating aggregated silhouette sweep plot for all metrics...")
    if not all_sweep_data:
        print("  No sweep data available to plot.")
        return

    plt.rcParams['mathtext.fontset'] = 'cm'
    fig, ax = plt.subplots(figsize=(14, 8))

    for metric, sweep_data in all_sweep_data.items():
        color = metric_colors.get(metric, '#888888')
        label = metric_name_map_latex.get(metric, metric)
        ax.plot(sweep_data['n_clusters'], sweep_data['silhouette_score'],
                color=color, linewidth=2.5, marker='o', markersize=5, label=label)

    ax.set_xlabel('Number of Clusters (k)', fontsize=14)
    ax.set_ylabel('Silhouette Score', fontsize=14)
    # MODIFICATION: Title updated to be more general.
    ax.set_title('Silhouette Score vs. Number of Clusters (Destrieux Atlas)', fontsize=16, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=12)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=12)

    text_lines = ["Optimal k:"]
    for metric, optimal_data in all_optimal_data.items():
        latex_label = metric_name_map_latex.get(metric, metric).ljust(18)
        text_lines.append(f"{latex_label} k = {optimal_data['k']}, Score = {optimal_data['score']:.3f}")
    
    info_text = "\n".join(text_lines)
    fig.text(1.04, 0.5, info_text, transform=ax.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.7))

    fig.subplots_adjust(right=0.75)
    output_file = os.path.join(output_dir, "hierarchical_silhouette_sweep_aggregated_164regions.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Aggregated sweep plot saved to: {output_file}")


def create_brain_visualization_for_cluster_numbers(metric, linkage_matrix, output_dir):
    print(f"\nCreating brain visualization grid for {metric} (Destrieux-164 atlas)...")
    
    plt.rcParams['mathtext.fontset'] = 'cm'
    lh_labels, rh_labels, lh_names, rh_names = load_freesurfer_destrieux_atlas()
    if lh_labels is None:
        print("  Failed to load FreeSurfer Destrieux atlas, skipping visualization.")
        return

    views = [('lateral', 'lh', 'Left Lateral'), ('medial', 'lh', 'Left Medial'), 
             ('lateral', 'rh', 'Right Lateral'), ('medial', 'rh', 'Right Medial')]
    
    cluster_numbers_to_plot = range(2, 11)
    
    fig, axes = plt.subplots(len(cluster_numbers_to_plot), len(views), 
                             figsize=(16, 4 * len(cluster_numbers_to_plot)),
                             gridspec_kw={'wspace': 0.01, 'hspace': 0.15})
    
    for row, n_clusters_k in enumerate(cluster_numbers_to_plot):
        print(f"  Processing k={n_clusters_k}...")
        cluster_assignments = fcluster(linkage_matrix, n_clusters_k, criterion='maxclust')
        num_clusters_found = len(np.unique(cluster_assignments[cluster_assignments > 0]))

        custom_cmap = generate_dynamic_colormap(num_clusters_found)
        lh_data, rh_data = map_data_to_surface_destrieux(cluster_assignments, lh_labels, rh_labels, lh_names, rh_names)
        
        for col, (view, hemi, view_title) in enumerate(views):
            ax = axes[row, col]
            try:
                brain = Brain('fsaverage', hemi, 'pial', cortex='low_contrast', background='white', size=(800, 600))
                surface_data = lh_data if hemi == 'lh' else rh_data
                
                brain.add_data(surface_data, min=0.5, max=num_clusters_found, colormap=custom_cmap,
                               alpha=1.0, smoothing_steps=0, thresh=0.5, colorbar=False)
                brain.show_view(view)
                
                temp_file = f"{output_dir}/temp_{metric}_{n_clusters_k}_164regions.png"
                brain.save_image(temp_file, mode='rgb', antialiased=True)
                ax.imshow(np.array(Image.open(temp_file)))
                brain.close()
                os.remove(temp_file)
            except Exception as e:
                print(f"    Error creating view for k={n_clusters_k}: {e}")
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', color='red')
            
            ax.set_axis_off()

            if row == 0:
                ax.set_title(view_title, fontsize=14, fontweight='bold', pad=10)
            if col == 0:
                label_text = f"k = {n_clusters_k}\n({num_clusters_found} clusters)"
                ax.text(-0.15, 0.5, label_text,
            transform=ax.transAxes,
            ha='center',
            va='center',
            rotation=90,
            fontweight='bold',
            fontsize=14)     

    latex_metric_name = metric_name_map_latex.get(metric, metric)
    fig.suptitle(f'Hierarchical Clustering on Destrieux Atlas ({latex_metric_name})', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    output_file = f"{output_dir}/hierarchical_assignments_by_k_{metric}_164regions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Brain visualization grid saved: {output_file}")

# --- [5. Main Execution Function] ---

def main():
    """Main function for hierarchical clustering analysis (Destrieux-164)."""
    print("="*80)
    # MODIFICATION: Title updated to reflect dynamic methods.
    print("Hierarchical Clustering Analysis - Dynamic Method (164-region Destrieux)")
    print("="*80)
    
    os.makedirs(config.output_path, exist_ok=True)
    
    all_sweep_data = {}
    all_optimal_data = {}
    
    for metric in config.metrics:
        connectome_path = os.path.join(config.group_connectomes_dir, f"group_{metric}_median.csv")

        if not os.path.exists(connectome_path):
            print(f"[Error] Connectome file not found, skipping: {connectome_path}")
            continue

        print(f"\nProcessing metric: {metric}")
        W = pd.read_csv(connectome_path, header=None).values
        if W.shape[0] != 164:
            print(f"  [Error] Connectome matrix is not 164x164. Shape: {W.shape}. Skipping.")
            continue
        
        # MODIFICATION: Pass the current metric to the analysis function.
        results = compute_hierarchical_clustering_analysis(W, metric)
        
        if results and results['success']:
            # --- Plot 1: Brain Visualization Grid (for each metric) ---
            linkage_matrix = results['linkage_matrix']
            create_brain_visualization_for_cluster_numbers(metric, linkage_matrix, config.output_path)
            
            # --- Collect data for the aggregated sweep plot ---
            all_sweep_data[metric] = pd.DataFrame({
                'n_clusters': results['cluster_range'],
                'silhouette_score': results['silhouette_scores']
            })
            all_optimal_data[metric] = {
                'k': results['optimal_n_clusters'],
                'score': results['optimal_silhouette']
            }
        else:
            print(f"[Failure] Analysis failed for metric: {metric}.")
            
    # --- Plot 2: Aggregated Silhouette Sweep Plot (after loop) ---
    if all_sweep_data:
        create_aggregated_silhouette_sweep_plot(all_sweep_data, all_optimal_data, config.output_path)
    
    print("\n" + "="*80)
    print("Analysis Complete.")
    print(f"Successfully processed {len(all_sweep_data)} / {len(config.metrics)} metrics.")
    print("="*80)

if __name__ == "__main__":
    main()