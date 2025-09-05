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
        self.group_connectomes_dir = "/home/localadmin/Desktop/SP2/84output/pro_group_average"
        self.output_path = "/home/localadmin/Desktop/SP2/84output/group_averaged_output"
        self.freesurfer_home = '/home/localadmin/freesurfer'
        self.subjects_dir = os.path.join(self.freesurfer_home, 'subjects')
        
        # Metrics
        self.metrics = ['R1', 'RK_DKI', 'RD_DTI_reciprocal', 'NOS', 'F_SMI', 'FA_DTI', 'DePerp_SMI_reciprocal']
        
        # Analysis parameters
        self.hub_top_threshold = 80  # Top 20% (80th percentile)
        self.hub_bottom_threshold = 20  # Bottom 20% (20th percentile)

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

def get_corrected_dk_atlas_mapping():
    """
    CORRECTED mapping for MRtrix3 FreeSurfer parcellation (84 regions)
    Based on the fs_default.txt lookup table
    
    Structure according to fs_default.txt:
    - Node 0: Unknown
    - Nodes 1-34: Left hemisphere cortical (34 regions)
    - Node 35: Left cerebellum  
    - Nodes 36-42: Left subcortical (7 regions)
    - Nodes 43-49: Right subcortical (7 regions)
    - Nodes 50-83: Right hemisphere cortical (34 regions)
    - Node 84: Right cerebellum
    """
    
    # Standard DK atlas cortical region names (34 regions per hemisphere)
    cortical_regions = [
        'bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus',
        'entorhinal', 'fusiform', 'inferiorparietal', 'inferiortemporal', 
        'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', 'lingual',
        'medialorbitofrontal', 'middletemporal', 'parahippocampal', 'paracentral',
        'parsopercularis', 'parsorbitalis', 'parstriangularis', 'pericalcarine',
        'postcentral', 'posteriorcingulate', 'precentral', 'precuneus',
        'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal',
        'superiorparietal', 'superiortemporal', 'supramarginal', 'frontalpole',
        'temporalpole', 'transversetemporal', 'insula'
    ]
    
    node_to_region = {}
    
    # Nodes 1-34: Left hemisphere cortical regions
    for i, region in enumerate(cortical_regions):
        node_idx = i + 1  # Start from node 1
        node_to_region[node_idx] = f"lh.{region}"
    
    # Nodes 50-83: Right hemisphere cortical regions  
    for i, region in enumerate(cortical_regions):
        node_idx = 50 + i  # Start from node 50
        node_to_region[node_idx] = f"rh.{region}"
    
    # Note: Subcortical and cerebellar regions (nodes 0, 35-49, 84) are 
    # excluded from cortical surface mapping but are part of the 84-node connectome
    
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
    
    # Degree centrality
    results['degree'] = bct.degrees_und(W_sym)
    
    # Betweenness centrality
    results['betweenness'] = bct.betweenness_wei(L)
    
    # Closeness centrality
    D, B = bct.distance_wei(L)
    G = nx.from_numpy_array(D)
    closeness_dict = nx.closeness_centrality(G, distance='weight')
    results['closeness'] = np.array([closeness_dict[i] for i in range(len(closeness_dict))])
    
    # Eigenvector centrality
    results['eigenvector'] = bct.eigenvector_centrality_und(W_sym)

    # # Clustering coefficient
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
    
    # All centrality measures
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

def load_freesurfer_dk_atlas():
    """
    Load FreeSurfer DK atlas for 84 regions
    """
    try:
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
        lh_annot_path = f"{config.freesurfer_home}/subjects/fsaverage/label/lh.aparc.annot"
        rh_annot_path = f"{config.freesurfer_home}/subjects/fsaverage/label/rh.aparc.annot"
        
        if os.path.exists(lh_annot_path) and os.path.exists(rh_annot_path):
            lh_labels, lh_ctab, lh_names = nib.freesurfer.read_annot(lh_annot_path)
            rh_labels, rh_ctab, rh_names = nib.freesurfer.read_annot(rh_annot_path)
            
            # Decode names if they're bytes
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
            lh_names = [f"lh_region_{i}" for i in range(42)]
            rh_names = [f"rh_region_{i}" for i in range(42)]
            return lh_labels, rh_labels, fsaverage, lh_names, rh_names
    except Exception as e:
        print(f"Error loading atlas: {e}")
        return None, None, None, None, None

def map_data_to_surface(data_values, lh_labels, rh_labels, lh_names, rh_names):
    """
    Map data values to brain surface using DK atlas (84 regions)
    """
    n_vertices_lh = len(lh_labels)
    n_vertices_rh = len(rh_labels)
    
    lh_data = np.zeros(n_vertices_lh, dtype=np.float64)
    rh_data = np.zeros(n_vertices_rh, dtype=np.float64)
    
    node_to_region = get_corrected_dk_atlas_mapping()
    
    # Create name-to-label mappings
    lh_name_to_label = {}
    unique_lh_labels = np.unique(lh_labels)
    for label_val in unique_lh_labels:
        if label_val < len(lh_names):
            region_name = lh_names[label_val]
            if not region_name.startswith('lh.'):
                region_name = f"lh.{region_name}"
            lh_name_to_label[region_name] = label_val
    
    rh_name_to_label = {}
    unique_rh_labels = np.unique(rh_labels)
    for label_val in unique_rh_labels:
        if label_val < len(rh_names):
            region_name = rh_names[label_val]
            if not region_name.startswith('rh.'):
                region_name = f"rh.{region_name}"
            rh_name_to_label[region_name] = label_val
    
    # Map data to vertices
    # FIXED: Add 1 to connectome index to align with fs_default.txt indexing
    for connectome_idx, data_value in enumerate(data_values):
        fs_node_idx = connectome_idx + 1  # Convert 0-83 to 1-84
        
        if fs_node_idx not in node_to_region or data_value == 0:
            continue
            
        region_name = node_to_region[fs_node_idx]
        
        if region_name.startswith('lh.'):
            # Left hemisphere
            if region_name in lh_name_to_label:
                label_val = lh_name_to_label[region_name]
                region_vertices = lh_labels == label_val
                lh_data[region_vertices] = float(data_value)
        else:
            # Right hemisphere
            if region_name in rh_name_to_label:
                label_val = rh_name_to_label[region_name]
                region_vertices = rh_labels == label_val
                rh_data[region_vertices] = float(data_value)
    
    lh_data = np.nan_to_num(lh_data, nan=0.0)
    rh_data = np.nan_to_num(rh_data, nan=0.0)
        
    return lh_data, rh_data

def collect_group_averaged_hub_scores():
    """
    Collect hub scores for group averaged connectomes
    Using 84-region atlas structure
    
    Returns:
        Dictionary: {weighted_metric: hub_scores_array}
    """
    hub_data = {}
    
    print("Collecting hub scores for group averaged connectomes...")
    
    for weighted_metric in config.metrics:
        print(f"Processing hub scores for: {weighted_metric}")
        
        # Path to group averaged hub scores
        hub_file = f"{config.output_path}/hub_scores_{weighted_metric}.csv"
        
        if os.path.exists(hub_file):
            try:
                df = pd.read_csv(hub_file)
                
                # Check if we have node and hub score columns
                if 'hub_score' in df.columns:
                    node_ids = df['node_id'].values
                    hub_scores = df['hub_score'].values
                    
                    # Create full hub scores array for 84 regions
                    full_hub_scores = np.zeros(84)
                    for node_id, hub_score in zip(node_ids, hub_scores):
                        if 0 <= node_id < 84:
                            full_hub_scores[int(node_id)] = hub_score
                    
                    hub_data[weighted_metric] = full_hub_scores
                    
                    print(f"  Found hub scores for {np.sum(full_hub_scores > 0)} nodes")
                else:
                    print(f"  Warning: Insufficient columns in {hub_file}")
            
            except Exception as e:
                print(f"Error reading {hub_file}: {e}")
        else:
            print(f"  File not found: {hub_file}")
    
    return hub_data

def create_hub_scores_brain_visualization(hub_data, output_dir):
    """
    Create brain surface visualizations showing hub scores for group averaged data
    Using scalar colormaps
    """
    plt.rcParams['mathtext.fontset'] = 'cm'
    print("="*60)
    print("CREATING HUB SCORES BRAIN VISUALIZATIONS (GROUP AVERAGED - 84 REGIONS)")
    print("="*60)
    
    # Load atlas
    lh_labels, rh_labels, fsaverage, lh_names, rh_names = load_freesurfer_dk_atlas()
    if lh_labels is None:
        print("Failed to load FreeSurfer DK atlas")
        return None
    
    print(f"Loaded DK atlas: LH={len(lh_labels)} vertices, RH={len(rh_labels)} vertices")
    
    views = [
        ('lateral', 'lh', 'Left Lateral'),
        ('medial', 'lh', 'Left Medial'), 
        ('lateral', 'rh', 'Right Lateral'),
        ('medial', 'rh', 'Right Medial')
    ]
    
    plot_images = {}
    
    for metric_idx, metric in enumerate(config.metrics):
        if metric not in hub_data:
            print(f"Skipping {metric} - no hub data")
            continue
            
        print(f"Processing {metric}...")
        
        hub_scores = hub_data[metric]
        max_hub_score = np.max(hub_scores)
        
        print(f"  Max hub score: {max_hub_score:.3f}")
        
        # Map to surface
        lh_data, rh_data = map_data_to_surface(
            hub_scores, lh_labels, rh_labels, lh_names, rh_names
        )
        
        plot_images[metric] = {}
        
        # Process each view
        for view, hemi, view_title in views:
            print(f"  Creating {view_title} view...")
            
            try:
                brain = Brain('fsaverage', hemi, 'pial', 
                             cortex='low_contrast',
                             background='white',
                             size=(800, 600))
                
                surface_data = lh_data if hemi == 'lh' else rh_data
                
                # Use scalar colormap for hub scores
                brain.add_data(surface_data, 
                              min=0.0,
                              max=5.0,  # Fixed range for hub scores
                              colormap='plasma',
                              alpha=1.0,
                              smoothing_steps=5,
                              thresh=None,
                              colorbar=False)
                
                brain.show_view(view)
                
                temp_file = f"{output_dir}/temp_hubs_{metric}_{view_title.replace(' ', '_')}.png"
                brain.save_image(temp_file, mode='rgb', antialiased=True)
                
                img = Image.open(temp_file)
                plot_images[metric][view_title] = np.array(img)
                
                brain.close()
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
                print(f"    Successfully created {view_title}")
                
            except Exception as e:
                print(f"    Error creating {view_title}: {e}")
                placeholder = np.ones((600, 800, 3), dtype=np.uint8) * 128
                plot_images[metric][view_title] = placeholder
    
    # Assemble final image
    print("Assembling final hub scores visualization...")
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
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=5))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Hub Score', fontsize=24, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    
    # Title
    fig.suptitle('Hub Scores on Desikan-Killiany Atlas', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Layout
    plt.subplots_adjust(left=0.05, right=0.90, top=0.94, bottom=0.05, hspace=0.1, wspace=0.02)
    
    # Save
    output_file = f"{output_dir}/hub_scores_group_averaged_84_regions.png"
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
        
        # Load connectome
        W = pd.read_csv(connectome_path, header=None, index_col=None).values
        print(f"  Loaded connectome: {W.shape}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # CENTRALITY & HUB ANALYSIS
        print("  Computing centrality measures...")
        try:
            centrality_measures = compute_centrality_measures(W)
            hub_scores = compute_hub_scores(centrality_measures)
            
            # Save results
            save_centrality_results(centrality_measures, hub_scores, output_dir, metric)
            print("Hub analysis completed")
            
            return {
                'measures': centrality_measures,
                'hub_scores': hub_scores
            }
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
    print("HUB ANALYSIS - GROUP AVERAGED CONNECTOMES")
    print("="*80)
    print(f"Configuration:")
    print(f"  Base directory: {config.group_connectomes_dir}")
    print(f"  Output directory: {config.output_path}")
    print(f"  Metrics: {config.metrics}")
    
    # Create output directory
    os.makedirs(config.output_path, exist_ok=True)
    
    processed = 0
    failed = 0
    
    print(f"\nStarting hub analysis for {len(config.metrics)} group averaged connectome files...")
    print("="*80)

        
    # Process each metric
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
    print("="*80)
    print(f"Successfully processed: {processed}/{len(config.metrics)} files")
    print(f"Failed: {failed}/{len(config.metrics)} files")
    
    # Create visualization
    print("\nCreating hub scores visualizations...")
    hub_data = collect_group_averaged_hub_scores()
    
    if hub_data:
        create_hub_scores_brain_visualization(hub_data, config.output_path)

if __name__ == "__main__":
    main()