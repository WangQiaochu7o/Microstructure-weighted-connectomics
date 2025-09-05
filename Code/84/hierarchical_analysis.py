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
        self.group_connectomes_dir = "/home/localadmin/Desktop/SP2/84output/pro_group_average"
        self.output_path = "/home/localadmin/Desktop/SP2/84output/group_averaged_output"
        self.freesurfer_home = '/home/localadmin/freesurfer'
        self.subjects_dir = os.path.join(self.freesurfer_home, 'subjects')
        
        # Metrics
        self.metrics = ['R1', 'RK_DKI', 'RD_DTI_reciprocal', 'NOS', 'F_SMI', 'FA_DTI', 'DePerp_SMI_reciprocal']
        
        # Hierarchical clustering parameters
        self.hierarchical_max_clusters = 20
        self.hierarchical_linkage_methods = ['ward', 'complete', 'average', 'single']

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
# HIERARCHICAL CLUSTERING ANALYSIS
# ============================================================================

def compute_uncentered_correlation_distance(W):
    """
    Compute distance matrix using uncentered (non-centered) Pearson correlation coefficient
    
    Uncentered correlation: r = Σ(XY) / sqrt(Σ(X²) * Σ(Y²))
    This does not subtract the mean, unlike standard Pearson correlation
    """
    print("      Computing uncentered Pearson correlation...")
    W_sym = ensure_symmetric_matrix(W)
    n_regions = W_sym.shape[0]
    
    # Compute uncentered correlation matrix
    correlation_matrix = np.zeros((n_regions, n_regions))
    
    for i in range(n_regions):
        if i % 20 == 0:  # Progress indicator
            print(f"        Processing region {i}/{n_regions}")
            
        for j in range(n_regions):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                x = W_sym[i, :]
                y = W_sym[j, :]
                
                # Uncentered correlation coefficient
                numerator = np.sum(x * y)
                denominator = np.sqrt(np.sum(x**2) * np.sum(y**2))
                
                if denominator > 1e-10:  # Avoid division by zero
                    correlation_matrix[i, j] = numerator / denominator
                else:
                    correlation_matrix[i, j] = 0.0
    
    print("      Computing distance matrix from correlations...")
    # Convert correlation to distance (1 - |correlation|)
    distance_matrix = 1 - np.abs(correlation_matrix)
    
    # Ensure diagonal is 0
    np.fill_diagonal(distance_matrix, 0)
    
    # Validate results
    print(f"        Correlation range: {np.min(correlation_matrix):.3f} to {np.max(correlation_matrix):.3f}")
    print(f"        Distance range: {np.min(distance_matrix):.3f} to {np.max(distance_matrix):.3f}")
    
    # Check for invalid values
    if np.any(np.isnan(correlation_matrix)) or np.any(np.isnan(distance_matrix)):
        print("        [WARNING] NaN values detected in correlation/distance matrices")
    
    return distance_matrix, correlation_matrix

def perform_hierarchical_clustering(distance_matrix, linkage_method='ward'):
    """
    Perform hierarchical clustering using specified linkage method
    """
    distance_condensed = squareform(distance_matrix)
    linkage_matrix = linkage(distance_condensed, method=linkage_method)
    
    return linkage_matrix

def find_optimal_clusters_hierarchical(linkage_matrix, W_sym, max_clusters=20):
    """
    Find optimal number of clusters using silhouette analysis and elbow method
    """
    silhouette_scores = []
    cluster_range = range(2, min(max_clusters + 1, W_sym.shape[0]))
    
    for n_clusters in cluster_range:
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Ensure we have more than 1 cluster
        unique_clusters = len(np.unique(cluster_labels))
        if unique_clusters > 1:
            try:
                score = silhouette_score(W_sym, cluster_labels, metric='euclidean')
                silhouette_scores.append(score)
            except:
                silhouette_scores.append(-1)
        else:
            silhouette_scores.append(-1)
    
    # Find optimal number of clusters
    if len(silhouette_scores) > 0 and max(silhouette_scores) > -1:
        optimal_idx = np.argmax(silhouette_scores)
        optimal_n_clusters = cluster_range[optimal_idx]
        optimal_silhouette = silhouette_scores[optimal_idx]
    else:
        optimal_n_clusters = 2
        optimal_silhouette = -1
    
    return optimal_n_clusters, optimal_silhouette, list(cluster_range), silhouette_scores

def compute_hierarchical_clustering_analysis(W):
    """
    Compute hierarchical clustering analysis with multiple linkage methods
    """
    print("    Starting hierarchical clustering computation...")
    W_sym = ensure_symmetric_matrix(W)
    print(f"    Processed connectome shape: {W_sym.shape}")
    print(f"    Non-zero connections: {np.sum(W_sym > 0)}")
    
    # Compute uncentered correlation distance matrix
    print("    Computing uncentered correlation distance matrix...")
    distance_matrix, correlation_matrix = compute_uncentered_correlation_distance(W_sym)
    print(f"    Distance matrix computed, shape: {distance_matrix.shape}")
    print(f"    Distance range: {np.min(distance_matrix):.3f} - {np.max(distance_matrix):.3f}")
    
    results = {}
    best_result = None
    best_silhouette = -1
    
    # Try different linkage methods
    print(f"    Testing {len(config.hierarchical_linkage_methods)} linkage methods...")
    for linkage_method in config.hierarchical_linkage_methods:
        try:
            print(f"      Testing linkage method: {linkage_method}")
            
            # Perform hierarchical clustering
            linkage_matrix = perform_hierarchical_clustering(distance_matrix, linkage_method)
            print(f"        Linkage matrix shape: {linkage_matrix.shape}")
            
            # Find optimal number of clusters
            print(f"        Finding optimal number of clusters (max: {config.hierarchical_max_clusters})...")
            optimal_n_clusters, optimal_silhouette, cluster_range, silhouette_scores = find_optimal_clusters_hierarchical(
                linkage_matrix, W_sym, config.hierarchical_max_clusters
            )
            
            # Get cluster assignments for optimal number
            cluster_assignments = fcluster(linkage_matrix, optimal_n_clusters, criterion='maxclust')
            unique_clusters = len(np.unique(cluster_assignments))
            
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
            
            # Track best method based on silhouette score
            if optimal_silhouette > best_silhouette:
                best_silhouette = optimal_silhouette
                best_result = method_result.copy()
                best_result['best_linkage_method'] = linkage_method
            
            print(f"        Result: {optimal_n_clusters} clusters, silhouette: {optimal_silhouette:.3f}")
            
        except Exception as e:
            print(f"        Failed: {e}")
            results[linkage_method] = {
                'linkage_method': linkage_method,
                'success': False,
                'error': str(e)
            }
    
    print(f"    Best result: {best_result['best_linkage_method'] if best_result else 'None'}")
    if best_result:
        print(f"      Best silhouette: {best_result['optimal_silhouette']:.3f}")
        print(f"      Best n_clusters: {best_result['optimal_n_clusters']}")
    
    return {
        'all_methods': results,
        'best_result': best_result,
        'distance_matrix': distance_matrix,
        'correlation_matrix': correlation_matrix
    }

def save_hierarchical_results(results, output_dir, metric):
    """Save hierarchical clustering analysis results in modularity-compatible format"""
    
    if results['best_result'] is None:
        print(f"    [ERROR] No valid hierarchical clustering results for {metric}")
        return
    
    best_result = results['best_result']
    print(f"    Saving results for {metric}...")
    print(f"      Best method: {best_result['best_linkage_method']}")
    print(f"      Optimal clusters: {best_result['optimal_n_clusters']}")
    print(f"      Silhouette score: {best_result['optimal_silhouette']:.3f}")
    
    # Save best method results (similar to modularity best gamma format)
    comprehensive_summary = {
        'metric': metric,
        'best_linkage_method': best_result['best_linkage_method'],
        'best_num_clusters': best_result['optimal_n_clusters'],
        'best_silhouette_score': best_result['optimal_silhouette'],
        'similarity_measure': 'uncentered_pearson_correlation'
    }
    
    comprehensive_df = pd.DataFrame([comprehensive_summary])
    comprehensive_df = comprehensive_df.round(3)
    comprehensive_path = os.path.join(output_dir, f'hierarchical_best_method_{metric}.csv')
    comprehensive_df.to_csv(comprehensive_path, index=False)
    print(f"      Saved: {comprehensive_path}")
    
    # Save cluster assignments (compatible with modularity community assignments format)
    community_df = pd.DataFrame({
        'node_id': range(len(best_result['cluster_assignments'])),
        'community_assignment': best_result['cluster_assignments']
    })
    community_path = os.path.join(output_dir, f'hierarchical_assignments_{metric}.csv')
    community_df.to_csv(community_path, index=False)
    print(f"      Saved: {community_path}")
    
    # Save detailed results for all methods
    method_summary = []
    for method_name, method_result in results['all_methods'].items():
        if method_result['success']:
            method_summary.append({
                'linkage_method': method_name,
                'optimal_n_clusters': method_result['optimal_n_clusters'],
                'optimal_silhouette': method_result['optimal_silhouette'],
                'max_silhouette': max(method_result['silhouette_scores']) if method_result['silhouette_scores'] else -1
            })
        else:
            method_summary.append({
                'linkage_method': method_name,
                'optimal_n_clusters': np.nan,
                'optimal_silhouette': np.nan,
                'max_silhouette': np.nan
            })
    
    method_df = pd.DataFrame(method_summary)
    method_df = method_df.round(3)
    method_path = os.path.join(output_dir, f'hierarchical_all_methods_{metric}.csv')
    method_df.to_csv(method_path, index=False)
    print(f"      Saved: {method_path}")
    
    # **[NEW]** Save silhouette sweep data for the best method
    if 'cluster_range' in best_result and 'silhouette_scores' in best_result:
        sweep_df = pd.DataFrame({
            'n_clusters': best_result['cluster_range'],
            'silhouette_score': best_result['silhouette_scores']
        })
        sweep_path = os.path.join(output_dir, f'hierarchical_silhouette_sweep_{metric}.csv')
        sweep_df.to_csv(sweep_path, index=False)
        print(f"      Saved: {sweep_path}")

    # Save distance and correlation matrices
    distance_df = pd.DataFrame(results['distance_matrix'])
    distance_path = os.path.join(output_dir, f'hierarchical_distance_matrix_{metric}.csv')
    distance_df.to_csv(distance_path, index=False)
    print(f"      Saved: {distance_path}")
    
    correlation_df = pd.DataFrame(results['correlation_matrix'])
    correlation_path = os.path.join(output_dir, f'hierarchical_correlation_matrix_{metric}.csv')
    correlation_df.to_csv(correlation_path, index=False)
    print(f"      Saved: {correlation_path}")
    
    print(f"    [SUCCESS] All results saved for {metric}")
    print(f"      Best method: {best_result['best_linkage_method']} "
          f"(clusters: {best_result['optimal_n_clusters']}, "
          f"silhouette: {best_result['optimal_silhouette']:.3f})")

def generate_dynamic_colormap(n_communities):
    """
    Generate a dynamic colormap based on the number of communities
    Uses distinguishable colors from different color spaces
    """
    if n_communities == 0:
        return ListedColormap(['#000000'])  # Black for no communities
    
    if n_communities <= 10:
        # Use predefined distinguishable colors for small numbers
        base_colors = [
            '#FF0000',  # Red
            '#0000FF',  # Blue  
            '#00AA00',  # Green
            '#FF66CC',  # Pink
            '#FF8800',  # Orange
            '#8800CC',  # Purple
            '#996633',  # Brown
            '#00CCCC',  # Cyan
            '#FFFF00',  # Yellow
            '#888888'   # Gray
        ]
        return ListedColormap(base_colors[:n_communities])
    else:
        # For larger numbers, generate colors using different methods
        if n_communities <= 20:
            # Use HSV color space for medium numbers
            colors = []
            for i in range(n_communities):
                hue = i / n_communities
                color = mcolors.hsv_to_rgb([hue, 0.8, 0.9])
                colors.append(color)
            return ListedColormap(colors)
        else:
            # Use tab20 colormap for very large numbers
            base_cmap = plt.get_cmap('tab20')
            if n_communities <= 20:
                return base_cmap
            else:
                # For more than 20, cycle through multiple colormaps
                colors = []
                cmaps = ['tab20', 'tab20b', 'tab20c']
                for i in range(n_communities):
                    cmap_idx = i // 20
                    color_idx = i % 20
                    if cmap_idx < len(cmaps):
                        cmap = plt.get_cmap(cmaps[cmap_idx])
                        colors.append(cmap(color_idx / 19))
                    else:
                        # Fallback to random colors
                        np.random.seed(i)
                        colors.append(np.random.rand(3))
                return ListedColormap(colors)

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

def collect_group_averaged_hierarchical_assignments():
    """
    Collect hierarchical cluster assignments for group averaged connectomes
    Using 84-region atlas structure
    
    Returns:
        Dictionary: {weighted_metric: {'assignments': [node_assignments], 'num_clusters': int}}
    """
    cluster_data = {}
    
    print("Collecting hierarchical cluster assignments for group averaged connectomes...")
    
    for weighted_metric in config.metrics:
        print(f"Processing hierarchical assignments for: {weighted_metric}")
        
        # Path to group averaged hierarchical assignments
        partition_file = f"{config.output_path}/hierarchical_assignments_{weighted_metric}.csv"
        
        if os.path.exists(partition_file):
            try:
                df = pd.read_csv(partition_file)
                
                # Check if we have node and cluster columns
                if len(df.columns) >= 2:
                    node_ids = df.iloc[:, 0].values
                    assignments = df.iloc[:, 1].values
                    
                    # Create full assignment array for 84 regions
                    full_assignments = np.zeros(84, dtype=int)
                    for node_id, assignment in zip(node_ids, assignments):
                        if 0 <= node_id < 84:
                            full_assignments[int(node_id)] = int(assignment)
                    
                    num_clusters = len(np.unique(assignments[assignments > 0]))
                    
                    cluster_data[weighted_metric] = {
                        'assignments': full_assignments,
                        'num_clusters': num_clusters
                    }
                    
                    print(f"  Found {num_clusters} clusters with {np.sum(full_assignments > 0)} assigned nodes")
                else:
                    print(f"  Warning: Insufficient columns in {partition_file}")
            
            except Exception as e:
                print(f"Error reading {partition_file}: {e}")
        else:
            print(f"  File not found: {partition_file}")
    
    return cluster_data

def create_hierarchical_brain_visualization(cluster_data, output_dir):
    """
    Create brain surface visualizations showing hierarchical cluster assignments for group averaged data
    Using dynamic colormaps based on the actual number of clusters found
    Updated for 84 regions
    """
    print("="*60)
    print("CREATING HIERARCHICAL CLUSTER BRAIN VISUALIZATIONS (GROUP AVERAGED - 84 REGIONS)")
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
    metric_cluster_info = {}
    
    for metric_idx, metric in enumerate(config.metrics):
        if metric not in cluster_data:
            print(f"Skipping {metric} - no cluster data")
            continue
            
        print(f"Processing {metric}...")
        
        cluster_assignments = cluster_data[metric]['assignments']
        unique_clusters = np.unique(cluster_assignments[cluster_assignments > 0])
        num_clusters = len(unique_clusters)
        
        if num_clusters == 0:
            print(f"No clusters found for {metric}")
            continue
        
        print(f"  Found {num_clusters} clusters: {unique_clusters}")
        metric_cluster_info[metric] = {'num_clusters': num_clusters, 'clusters': unique_clusters}
        
        # Generate dynamic colormap based on actual number of clusters
        custom_cmap = generate_dynamic_colormap(num_clusters)
        
        # Map to surface
        lh_data, rh_data = map_data_to_surface(
            cluster_assignments, lh_labels, rh_labels, lh_names, rh_names
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
                
                # Use dynamic colormap
                brain.add_data(surface_data, 
                              min=0.5,  # Start slightly above 0 to avoid showing unassigned regions
                              max=np.max(unique_clusters),
                              colormap=custom_cmap,
                              alpha=1.0,
                              smoothing_steps=0,  # No smoothing for discrete data
                              thresh=0.5,  # Only show assigned regions
                              colorbar=False)
                
                brain.show_view(view)
                
                temp_file = f"{output_dir}/temp_hierarchical_{metric}_{view_title.replace(' ', '_')}.png"
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
    print("Assembling final hierarchical clustering visualization...")
    fig, axes = plt.subplots(len(config.metrics), len(views), figsize=(20, 6 * len(config.metrics)))
    if len(config.metrics) == 1:
        axes = axes.reshape(1, -1)
    
    for row, metric in enumerate(config.metrics):
        if metric not in plot_images:
            continue
            
        for col, (view, hemi, view_title) in enumerate(views):
            ax = axes[row, col]
            if view_title in plot_images[metric]:
                ax.imshow(plot_images[metric][view_title], interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            
            if row == 0:
                ax.set_title(view_title, fontsize=14, fontweight='bold', pad=15)
            if col == 0:
                # Add metric name and cluster count
                if metric in metric_cluster_info:
                    num_clusters = metric_cluster_info[metric]['num_clusters']
                    label_text = f"{metric}\n({num_clusters} clusters)"
                else:
                    label_text = metric
                    
                ax.text(-0.15, 0.5, label_text, transform=ax.transAxes, 
                       fontsize=11, fontweight='bold', rotation=90, 
                       verticalalignment='center', horizontalalignment='center')
    
    # Add title
    fig.suptitle('Hierarchical Clustering: Group Averaged Connectomes (7 Weighted Metrics × 4 Brain Views)\n84-Region DK Atlas with Uncentered Pearson Correlation', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout to use full width
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.05, hspace=0.1, wspace=0.02)
    
    output_file = f"{output_dir}/hierarchical_clustering_group_averaged_84_regions.png"
    plt.savefig(output_file, dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"Saved hierarchical clustering visualization: {output_file}")
    
    # Print summary of clusters per metric
    print("\nHierarchical Clustering Summary:")
    for metric in config.metrics:
        if metric in metric_cluster_info:
            info = metric_cluster_info[metric]
            print(f"  {metric}: {info['num_clusters']} clusters {list(info['clusters'])}")
    
    return output_file

def save_hierarchical_assignments(cluster_data, output_dir):
    """
    Save detailed hierarchical cluster assignment data for 84 regions (group averaged)
    """
    print("Saving hierarchical cluster assignment data...")
    
    # Save cluster assignments
    assignment_df_data = []
    node_to_region = get_corrected_dk_atlas_mapping()
    
    for metric in config.metrics:
        if metric in cluster_data:
            assignments = cluster_data[metric]['assignments']
            for node_idx, cluster_id in enumerate(assignments):
                if node_idx < 84:  # Only process 84 regions
                    fs_node_idx = node_idx + 1
                    region_name = node_to_region.get(fs_node_idx, f"Unknown_Region_{node_idx}")
                    assignment_df_data.append({
                        'Metric': metric,
                        'Node_ID': node_idx,
                        'FS_Node_ID': fs_node_idx,  # Add the 1-based FreeSurfer index
                        'Region_Name': region_name,
                        'Cluster_Assignment': int(cluster_id)
                    })
    
    assignment_df = pd.DataFrame(assignment_df_data)
    assignment_file = f"{output_dir}/hierarchical_assignments_group_averaged_84_regions.csv"
    assignment_df.to_csv(assignment_file, index=False)
    print(f"Saved hierarchical assignments: {assignment_file}")
    
    # Save summary statistics
    summary_data = []
    for metric in config.metrics:
        if metric in cluster_data:
            assignments = cluster_data[metric]['assignments']
            unique_clusters = np.unique(assignments[assignments > 0])
            num_clusters = len(unique_clusters)
            num_assigned_nodes = np.sum(assignments > 0)
            
            summary_data.append({
                'Metric': metric,
                'Num_Clusters': num_clusters,
                'Assigned_Nodes': num_assigned_nodes,
                'Unassigned_Nodes': 84 - num_assigned_nodes,
                'Cluster_IDs': ','.join(map(str, unique_clusters))
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = f"{output_dir}/hierarchical_summary_group_averaged_84_regions.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary: {summary_file}")

# **[NEW]** Function to collect silhouette data
def collect_hierarchical_silhouette_data():
    """
    Collect silhouette sweep data for all metrics from hierarchical clustering.
    
    Returns:
        Dictionary: {metric: DataFrame with n_clusters, silhouette_score}
    """
    silhouette_data = {}
    print("\nCollecting silhouette sweep data for hierarchical clustering...")
    
    for metric in config.metrics:
        print(f"  Processing silhouette sweep data for: {metric}")
        
        file_path = f"{config.output_path}/hierarchical_silhouette_sweep_{metric}.csv"
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if not df.empty and 'n_clusters' in df.columns and 'silhouette_score' in df.columns:
                    silhouette_data[metric] = df.dropna()
                    print(f"    Found {len(df)} data points.")
                else:
                    print(f"    Warning: File is empty or missing required columns for {metric}")
            except Exception as e:
                print(f"    Error reading {file_path}: {e}")
        else:
            print(f"    File not found: {file_path}")
    
    return silhouette_data

# **[NEW]** Function to create silhouette plots
def create_silhouette_sweep_plots(silhouette_data, output_dir):
    """
    Create plots showing how the silhouette score changes with n_clusters for each metric.
    """
    print("Creating silhouette vs. number of clusters plots...")
    
    if not silhouette_data:
        print("No silhouette sweep data to plot.")
        return

    # Plot 1: Combined plot with all metrics
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for metric, df in silhouette_data.items():
        if df.empty:
            continue
        
        color = metric_colors.get(metric, '#888888')
        
        # Plot the curve
        ax.plot(df['n_clusters'], df['silhouette_score'], 
                color=color, linewidth=2.5, marker='o', markersize=6,
                label=metric, alpha=0.8)
        
        # Highlight the optimal point
        if not df['silhouette_score'].empty:
            best_idx = df['silhouette_score'].idxmax()
            optimal_n = df.loc[best_idx, 'n_clusters']
            optimal_score = df.loc[best_idx, 'silhouette_score']
            
            ax.plot(optimal_n, optimal_score, 'D', color=color, markersize=10, 
                    markeredgecolor='black', markeredgewidth=1.5)

    ax.set_xlabel('Number of Clusters (k)', fontsize=14)
    ax.set_ylabel('Silhouette Score', fontsize=14)
    ax.set_title('Silhouette Score vs. Number of Clusters', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    ax.tick_params(labelsize=12)
    
    # Set x-axis ticks to be integers
    max_n_cluster = max(df['n_clusters'].max() for df in silhouette_data.values() if not df.empty)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=1.5, right=max_n_cluster + 0.5)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    output_file = os.path.join(output_dir, "hierarchical_silhouette_sweep_plot_combined.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined silhouette sweep plot: {output_file}")

    # Plot 2: Individual plots for each metric
    create_individual_silhouette_plots(silhouette_data, output_dir)


# **[NEW]** Function to create individual silhouette plots
def create_individual_silhouette_plots(silhouette_data, output_dir):
    """
    Create an individual plot for each metric showing Silhouette Score vs. Number of Clusters.
    """
    print("Creating individual silhouette sweep plots...")
    
    for metric, df in silhouette_data.items():
        if df.empty:
            continue
        
        color = metric_colors.get(metric, '#888888')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df['n_clusters'], df['silhouette_score'], 
                color=color, linewidth=3, marker='o', markersize=8)
        
        # Highlight optimal point
        if not df['silhouette_score'].empty:
            best_idx = df['silhouette_score'].idxmax()
            optimal_n = df.loc[best_idx, 'n_clusters']
            optimal_score = df.loc[best_idx, 'silhouette_score']
            
            ax.axvline(x=optimal_n, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax.plot(optimal_n, optimal_score, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
            ax.text(optimal_n, optimal_score, f'  Best k={optimal_n}\n  Score={optimal_score:.3f}', 
                    verticalalignment='bottom', horizontalalignment='left', fontsize=12, fontweight='bold', color='red')

        ax.set_xlabel('Number of Clusters (k)', fontsize=14)
        ax.set_ylabel('Silhouette Score', fontsize=14)
        ax.set_title(f'Silhouette Score vs. Number of Clusters\n({metric})', fontsize=16, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=12)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f"hierarchical_silhouette_sweep_{metric}_individual.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved individual plot: {output_file}")


def process_connectome_hierarchical_analysis(connectome_path, output_dir, metric):
    """
    Process a single connectome with hierarchical clustering analysis
    """
    try:
        print(f"Processing Hierarchical Analysis: Group Average Metric {metric}")
        
        # Load connectome
        print(f"  Loading connectome from: {connectome_path}")
        W = pd.read_csv(connectome_path, header=None, index_col=None).values
        print(f"  Loaded connectome shape: {W.shape}")
        
        # Validate connectome
        if W.shape[0] != W.shape[1]:
            print(f"  [ERROR] Connectome is not square: {W.shape}")
            return None
            
        if W.shape[0] != 84:
            print(f"  [WARNING] Expected 84x84 matrix, got {W.shape}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # HIERARCHICAL CLUSTERING ANALYSIS
        print("  Computing hierarchical clustering analysis...")
        print(f"    Using uncentered Pearson correlation as similarity measure")
        print(f"    Testing linkage methods: {config.hierarchical_linkage_methods}")
        
        try:
            hierarchical_results = compute_hierarchical_clustering_analysis(W)
            
            if hierarchical_results is not None and hierarchical_results['best_result'] is not None:
                # Save results
                print("  Saving hierarchical clustering results...")
                save_hierarchical_results(hierarchical_results, output_dir, metric)
                print("  Hierarchical clustering analysis completed successfully")
                return hierarchical_results
            else:
                print("  [ERROR] Hierarchical clustering analysis failed - no valid results")
                return None
        except Exception as e:
            print(f"  [ERROR] Hierarchical clustering analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    except Exception as e:
        print(f"[ERROR] Error processing Group Average Metric {metric}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Main function for hierarchical clustering analysis
    """
    print("="*80)
    print("HIERARCHICAL CLUSTERING ANALYSIS - GROUP AVERAGED CONNECTOMES")
    print("="*80)
    print(f"Configuration:")
    print(f"  Base directory: {config.group_connectomes_dir}")
    print(f"  Output directory: {config.output_path}")
    print(f"  Metrics: {config.metrics}")
    print(f"  Similarity measure: Uncentered Pearson correlation")
    print(f"  Linkage methods: {config.hierarchical_linkage_methods}")
    print(f"  Max clusters: {config.hierarchical_max_clusters}")
    
    # Create output directory
    os.makedirs(config.output_path, exist_ok=True)
    
    processed = 0
    failed = 0
    
    print(f"\nStarting hierarchical clustering analysis for {len(config.metrics)} group averaged connectome files...")
    print("="*80)
    
    # STEP 1: Process each metric (ACTUAL ANALYSIS)
    for metric in config.metrics:
        connectome_path = os.path.join(
            config.group_connectomes_dir,
            f"group_{metric}_median.csv"
        )
        
        if os.path.exists(connectome_path):
            print(f"\n[INFO] Processing connectome: {connectome_path}")
            results = process_connectome_hierarchical_analysis(
                connectome_path, config.output_path, metric
            )
            
            if results is not None:
                processed += 1
                print(f"[SUCCESS] {metric} analysis completed")
            else:
                failed += 1
                print(f"[FAILED] {metric} analysis failed")
        else:
            print(f"[ERROR] Connectome not found: {connectome_path}")
            failed += 1
    
    print("="*80)
    print("HIERARCHICAL CLUSTERING ANALYSIS COMPLETED!")
    print("="*80)
    print(f"Successfully processed: {processed}/{len(config.metrics)} files")
    print(f"Failed: {failed}/{len(config.metrics)} files")
    
    # STEP 2: Create visualizations (only if we have successful results)
    if processed > 0:
        print("\nCreating hierarchical cluster visualizations...")
        cluster_data = collect_group_averaged_hierarchical_assignments()
        
        if cluster_data:
            save_hierarchical_assignments(cluster_data, config.output_path)
            create_hierarchical_brain_visualization(cluster_data, config.output_path)
        else:
            print("[WARNING] No cluster data found for brain visualization")
        
        # **[NEW]** STEP 3: Create silhouette sweep plot
        print("\nCreating silhouette sweep plots...")
        silhouette_data = collect_hierarchical_silhouette_data()
        if silhouette_data:
            create_silhouette_sweep_plots(silhouette_data, config.output_path)
        else:
            print("[WARNING] No silhouette sweep data found for plotting.")
    else:
        print("[WARNING] No successful analyses - skipping visualization and plotting")

if __name__ == "__main__":
    main()