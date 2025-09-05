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
        
        # Analysis parameters (kept for context, though not used in this specific version)
        self.hub_top_threshold = 80
        self.hub_bottom_threshold = 20

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

metric_name_map_latex = {
        'NOS': r'$nos$',
        'RK_DKI': r'$rk$',
        'FA_DTI': r'$fa$',
        'RD_DTI_reciprocal': r'$1/rd$',
        'DePerp_SMI_reciprocal': r'1/$D_{\perp e}$',
        'F_SMI': r'$f$',
        'R1': r'$R1$'
}

def ensure_symmetric_matrix(W):
    """Ensure matrix is symmetric and remove diagonal"""
    W_sym = (W + W.T) / 2
    np.fill_diagonal(W_sym, 0)
    W_sym = np.round(W_sym, decimals=3)
    return W_sym

def create_length_matrix(W):
    """Convert weights to connection-length matrix by taking reciprocal"""
    L = np.zeros_like(W)
    mask = W > 0
    L[mask] = 1.0 / W[mask]
    L[~mask] = np.inf
    np.fill_diagonal(L, 0)
    return L

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
        'S_interm_prim-Jensen', 'S_intrapariet_and_P_trans',
        'S_oc_middle_and_Lunatus', 'S_oc_sup_and_transversal', 'S_occipital_ant',
        'S_oc-temp_lat', 'S_oc-temp_med_and_Lingual', 'S_orbital_lateral',
        'S_orbital_med-olfact', 'S_orbital-H_Shaped', 'S_parieto_occipital',
        'S_pericallosal', 'S_postcentral', 'S_precentral-inf-part',
        'S_precentral-sup-part', 'S_suborbital', 'S_subparietal',
        'S_temporal_inf', 'S_temporal_sup', 'S_temporal_transverse'
    ]
    node_to_region = {}
    for i, region in enumerate(cortical_regions):
        node_idx = i + 1
        node_to_region[node_idx] = f"lh.{region}"
    for i, region in enumerate(cortical_regions):
        node_idx = 90 + i
        node_to_region[node_idx] = f"rh.{region}"
    return node_to_region

# ============================================================================
# HUB ANALYSIS (CENTRALITY MEASURES)
# ============================================================================

def compute_centrality_measures(W):
    """
    Compute all centrality measures for a weighted connectome
    """
    W_sym = ensure_symmetric_matrix(W)
    L = create_length_matrix(W_sym)
    
    results = {}
    
    results['degree'] = bct.degrees_und(W_sym)
    results['betweenness'] = bct.betweenness_wei(L)
    
    D, B = bct.distance_wei(L)
    G = nx.from_numpy_array(D)
    closeness_dict = nx.closeness_centrality(G, distance='weight')
    results['closeness'] = np.array([closeness_dict[i] for i in range(len(closeness_dict))])
    
    results['eigenvector'] = bct.eigenvector_centrality_und(W_sym)

    # Cpos, Cneg = bct.clustering_coef_wu_sign(W_sym)
    # results['clustering'] = Cpos
    
    return results

def compute_hub_scores(centrality_measures):
    """
    Compute hub scores based on centrality measures
    """
    n_nodes = len(centrality_measures['degree'])
    hub_scores = np.zeros(n_nodes)
    
    # +1 for top 20% in each measure (except clustering, which is bottom 20%)
    hub_scores += (centrality_measures['degree'] >= np.percentile(centrality_measures['degree'], config.hub_top_threshold)).astype(int)
    hub_scores += (centrality_measures['betweenness'] >= np.percentile(centrality_measures['betweenness'], config.hub_top_threshold)).astype(int)
    hub_scores += (centrality_measures['closeness'] >= np.percentile(centrality_measures['closeness'], config.hub_top_threshold)).astype(int)
    hub_scores += (centrality_measures['eigenvector'] >= np.percentile(centrality_measures['eigenvector'], config.hub_top_threshold)).astype(int)
    # hub_scores += (centrality_measures['clustering'] <= np.percentile(centrality_measures['clustering'], config.hub_bottom_threshold)).astype(int)
    
    return hub_scores


def save_centrality_results(centrality_measures, hub_scores, output_dir, metric):
    """Save centrality analysis results"""
    
    centrality_df = pd.DataFrame({
        'node_id': range(len(hub_scores)),
        'degree': centrality_measures['degree'],
        'betweenness': centrality_measures['betweenness'],
        'closeness': centrality_measures['closeness'],
        'eigenvector': centrality_measures['eigenvector'],
        # 'clustering': centrality_measures['clustering'],
        'hub_score': hub_scores
    })
    centrality_df = centrality_df.round(3)
    centrality_df.to_csv(os.path.join(output_dir, f'hub_scores_{metric}.csv'), index=False)

def load_freesurfer_destrieux_atlas():
    """
    Load FreeSurfer Destrieux atlas (a2009s) for 164 regions
    """
    try:
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
        lh_annot_path = f"{config.freesurfer_home}/subjects/fsaverage/label/lh.aparc.a2009s.annot"
        rh_annot_path = f"{config.freesurfer_home}/subjects/fsaverage/label/rh.aparc.a2009s.annot"
        
        if os.path.exists(lh_annot_path) and os.path.exists(rh_annot_path):
            lh_labels, lh_ctab, lh_names = nib.freesurfer.read_annot(lh_annot_path)
            rh_labels, rh_ctab, rh_names = nib.freesurfer.read_annot(rh_annot_path)
            
            if len(lh_names) > 0 and isinstance(lh_names[0], bytes):
                lh_names = [name.decode('utf-8') for name in lh_names]
            if len(rh_names) > 0 and isinstance(rh_names[0], bytes):
                rh_names = [name.decode('utf-8') for name in rh_names]
                
            return lh_labels, rh_labels, fsaverage, lh_names, rh_names
        else:
            from nilearn.datasets import fetch_atlas_surf_destrieux
            atlas = fetch_atlas_surf_destrieux()
            lh_labels = atlas['map_left']
            rh_labels = atlas['map_right']
            lh_names = [f"lh_region_{i}" for i in range(148)]
            rh_names = [f"rh_region_{i}" for i in range(148)]
            return lh_labels, rh_labels, fsaverage, lh_names, rh_names
    except Exception as e:
        print(f"Error loading Destrieux atlas: {e}")
        return None, None, None, None, None

def map_data_to_surface(data_values, lh_labels, rh_labels, lh_names, rh_names):
    """
    Map data values to brain surface using Destrieux atlas (164 regions)
    """
    n_vertices_lh = len(lh_labels)
    n_vertices_rh = len(rh_labels)
    
    lh_data = np.zeros(n_vertices_lh, dtype=np.float64)
    rh_data = np.zeros(n_vertices_rh, dtype=np.float64)
    
    node_to_region = get_corrected_a2009s_atlas_mapping()
    
    lh_name_to_label = {}
    for label_val in np.unique(lh_labels):
        if label_val < len(lh_names):
            region_name = lh_names[label_val]
            if not region_name.startswith('lh.'):
                region_name = f"lh.{region_name}"
            lh_name_to_label[region_name] = label_val
    
    rh_name_to_label = {}
    for label_val in np.unique(rh_labels):
        if label_val < len(rh_names):
            region_name = rh_names[label_val]
            if not region_name.startswith('rh.'):
                region_name = f"rh.{region_name}"
            rh_name_to_label[region_name] = label_val
    
    for connectome_idx, data_value in enumerate(data_values):
        fs_node_idx = connectome_idx + 1
        
        if fs_node_idx not in node_to_region or data_value == 0:
            continue
            
        region_name = node_to_region[fs_node_idx]
        
        if region_name.startswith('lh.'):
            if region_name in lh_name_to_label:
                label_val = lh_name_to_label[region_name]
                lh_data[lh_labels == label_val] = float(data_value)
        else:
            if region_name in rh_name_to_label:
                label_val = rh_name_to_label[region_name]
                rh_data[rh_labels == label_val] = float(data_value)
    
    lh_data = np.nan_to_num(lh_data, nan=0.0)
    rh_data = np.nan_to_num(rh_data, nan=0.0)
        
    return lh_data, rh_data

def collect_group_averaged_hub_scores():
    """
    Collect hub scores for group averaged connectomes (164 regions)
    """
    hub_data = {}
    
    print("Collecting hub scores for group averaged connectomes (164 regions)...")
    
    for weighted_metric in config.metrics:
        print(f"Processing hub scores for: {weighted_metric}")
        
        hub_file = f"{config.output_path}/hub_scores_{weighted_metric}.csv"
        
        if os.path.exists(hub_file):
            try:
                df = pd.read_csv(hub_file)
                if 'hub_score' in df.columns:
                    full_hub_scores = np.zeros(164)
                    for node_id, hub_score in zip(df['node_id'].values, df['hub_score'].values):
                        if 0 <= node_id < 164:
                            full_hub_scores[int(node_id)] = hub_score
                    
                    hub_data[weighted_metric] = full_hub_scores
                    print(f"  Found hub scores for {np.sum(full_hub_scores > 0)} nodes")
                else:
                    print(f"  Warning: 'hub_score' column not found in {hub_file}")
            except Exception as e:
                print(f"Error reading {hub_file}: {e}")
        else:
            print(f"  File not found: {hub_file}")
    
    return hub_data

def create_hub_scores_brain_visualization(hub_data, output_dir):
    """
    Create brain surface visualizations showing hub scores for group averaged data (164 regions)
    """
    plt.rcParams['mathtext.fontset'] = 'cm'
    print("="*60)
    print("CREATING HUB SCORES BRAIN VISUALIZATIONS (GROUP AVERAGED - 164 REGIONS)")
    print("="*60)
    
    lh_labels, rh_labels, fsaverage, lh_names, rh_names = load_freesurfer_destrieux_atlas()
    if lh_labels is None:
        print("Failed to load FreeSurfer Destrieux atlas")
        return None
    
    print(f"Loaded Destrieux atlas: LH={len(lh_labels)} vertices, RH={len(rh_labels)} vertices")
    
    views = [
        ('lateral', 'lh', 'Left Lateral'),
        ('medial', 'lh', 'Left Medial'), 
        ('lateral', 'rh', 'Right Lateral'),
        ('medial', 'rh', 'Right Medial')
    ]
    
    plot_images = {}
    
    for metric in config.metrics:
        if metric not in hub_data:
            print(f"Skipping {metric} - no hub data")
            continue
            
        print(f"Processing {metric}...")
        hub_scores = hub_data[metric]
        print(f"  Max hub score: {np.max(hub_scores):.3f}")
        
        lh_data, rh_data = map_data_to_surface(hub_scores, lh_labels, rh_labels, lh_names, rh_names)
        plot_images[metric] = {}
        
        for view, hemi, view_title in views:
            print(f"  Creating {view_title} view...")
            try:
                brain = Brain('fsaverage', hemi, 'pial', cortex='low_contrast', background='white', size=(800, 600))
                surface_data = lh_data if hemi == 'lh' else rh_data
                brain.add_data(surface_data, min=0.0, max=5.0, colormap='plasma', alpha=1.0, smoothing_steps=5, colorbar=False)
                brain.show_view(view)
                
                temp_file = f"{output_dir}/temp_hubs_{metric}_{view_title.replace(' ', '_')}.png"
                brain.save_image(temp_file, mode='rgb', antialiased=True)
                img = Image.open(temp_file)
                plot_images[metric][view_title] = np.array(img)
                brain.close()
                os.remove(temp_file)
                print(f"    Successfully created {view_title}")
            except Exception as e:
                print(f"    Error creating {view_title}: {e}")
                plot_images[metric][view_title] = np.ones((600, 800, 3), dtype=np.uint8) * 128
    
    fig, axes = plt.subplots(len(config.metrics), len(views), figsize=(24, 6 * len(config.metrics)))
    if len(config.metrics) == 1:
        axes = axes.reshape(1, -1)
    
    for row, metric in enumerate(config.metrics):
        if metric not in plot_images:
            continue
        for col, (view, hemi, view_title) in enumerate(views):
            ax = axes[row, col]
            if view_title in plot_images[metric]:
                ax.imshow(plot_images[metric][view_title], interpolation='bilinear')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
            if row == 0:
                ax.set_title(view_title, fontsize=24, fontweight='bold', pad=15)
            if col == 0:
                latex_label = metric_name_map_latex.get(metric, metric) # Get LaTeX label
                ax.text(-0.1, 0.5, latex_label, transform=ax.transAxes,
                       fontsize=24, fontweight='bold', # Fontsize adjusted
                       verticalalignment='center', horizontalalignment='center')
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=5))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Hub Score', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    
    fig.suptitle('Hub Scores on Destrieux Atlas', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(left=0.05, right=0.90, top=0.94, bottom=0.05, hspace=0.1, wspace=0.02)
    
    output_file = f"{output_dir}/hub_scores_group_averaged_164_regions.png"
    plt.savefig(output_file, dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"Saved hub scores visualization: {output_file}")
    return output_file

def process_connectome_hub_analysis(connectome_path, output_dir, metric):
    """
    Process a single connectome with hub analysis
    """
    try:
        print(f"\nProcessing Hub Analysis: Group Average Metric {metric}")
        
        W = pd.read_csv(connectome_path, header=None, index_col=None).values
        print(f"  Loaded connectome: {W.shape}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("  Computing centrality measures...")
        try:
            centrality_measures = compute_centrality_measures(W)
            hub_scores = compute_hub_scores(centrality_measures)
            
            save_centrality_results(centrality_measures, hub_scores, output_dir, metric)
            print("Hub analysis completed")
            
            return {'measures': centrality_measures, 'hub_scores': hub_scores}
        except Exception as e:
            print(f"Hub analysis failed: {e}")
            return None
        
    except Exception as e:
        print(f"Error processing Group Average Metric {metric}: {e}")
        return None

def main():
    """
    Main function for hub analysis
    """
    print("="*80)
    print("HUB ANALYSIS - GROUP AVERAGED CONNECTOMES (164 REGIONS)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Base directory: {config.group_connectomes_dir}")
    print(f"  Output directory: {config.output_path}")
    print(f"  Metrics: {config.metrics}")
    
    os.makedirs(config.output_path, exist_ok=True)
    
    processed = 0
    failed = 0
    
    print(f"\nStarting hub analysis for {len(config.metrics)} group averaged connectome files...")
    print("="*80)
    
    for metric in config.metrics:
        connectome_path = os.path.join(
            config.group_connectomes_dir,
            f"group_{metric}_median.csv"
        )
        
        if os.path.exists(connectome_path):
            results = process_connectome_hub_analysis(
                connectome_path, config.output_path, metric
            )
            if results is not None:
                processed += 1
            else:
                failed += 1
        else:
            print(f"Connectome not found: {connectome_path}")
            failed += 1
    
    print("="*80)
    print("HUB ANALYSIS COMPLETED!")
    print(f"Successfully processed: {processed}/{len(config.metrics)} files")
    
    print("\nCreating hub scores visualizations...")
    hub_data = collect_group_averaged_hub_scores()
    
    if hub_data:
        create_hub_scores_brain_visualization(hub_data, config.output_path)

if __name__ == "__main__":
    main()