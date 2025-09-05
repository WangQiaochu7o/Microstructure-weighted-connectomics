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
from netneurotools.modularity import consensus_modularity

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
        self.modularity_gamma_range = np.arange(0.5, 3.5, 0.1)
        self.modularity_n_runs = 1000

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
# MODULARITY ANALYSIS
# ============================================================================

def compute_modularity_consensus(W, gamma=1, n_runs=1000, seed_base=42):
    """
    Compute modularity using consensus modularity
    """
    W_sym = ensure_symmetric_matrix(W)
    
    try:
        consensus_community, Q_all, zrand_all = consensus_modularity(
            W_sym, 
            gamma=gamma, 
            B='modularity',
            repeats=n_runs,
            null_func=np.mean,
            seed=seed_base
        )
        
        return {
            'consensus_community': consensus_community,
            'modularity_values': Q_all,
            'zrand_scores': zrand_all,
            'num_communities': len(np.unique(consensus_community)),
            'gamma': gamma,
            'n_runs': n_runs,
            'success': True
        }
        
    except Exception as e:
        print(f"Warning: Failed for gamma={gamma}: {e}")
        return {
            'consensus_community': np.full(W_sym.shape[0], np.nan),
            'modularity_values': np.full(n_runs, np.nan),
            'zrand_scores': np.array([]),
            'num_communities': np.nan,
            'gamma': gamma,
            'n_runs': n_runs,
            'success': False
        }

def compute_modularity_analysis(W):
    """
    Compute modularity analysis across gamma range
    """
    all_results = {}
    summary_stats = []
    
    for gamma in config.modularity_gamma_range:
        results = compute_modularity_consensus(W, gamma=gamma, n_runs=config.modularity_n_runs)
        all_results[gamma] = results
        
        # Compute statistics
        if results['success']:
            modularity_values = np.array(results['modularity_values'])
            zrand_scores = np.array(results['zrand_scores'])
            
            valid_mod = modularity_values[~np.isnan(modularity_values)]
            valid_zrand = zrand_scores[~np.isnan(zrand_scores)] if len(zrand_scores) > 0 else np.array([])
            
            stats = {
                'gamma': gamma,
                'n_valid_runs': len(valid_mod),
                'modularity_mean': np.mean(valid_mod) if len(valid_mod) > 0 else np.nan,
                'modularity_std': np.std(valid_mod) if len(valid_mod) > 0 else np.nan,
                'modularity_min': np.min(valid_mod) if len(valid_mod) > 0 else np.nan,
                'modularity_max': np.max(valid_mod) if len(valid_mod) > 0 else np.nan,
                'num_communities': results['num_communities'],
                'zrand_mean': np.mean(valid_zrand) if len(valid_zrand) > 0 else np.nan,
                'zrand_std': np.std(valid_zrand) if len(valid_zrand) > 0 else np.nan,
                'zrand_min': np.min(valid_zrand) if len(valid_zrand) > 0 else np.nan,
                'zrand_max': np.max(valid_zrand) if len(valid_zrand) > 0 else np.nan
            }
        else:
            stats = {
                'gamma': gamma,
                'n_valid_runs': 0,
                'modularity_mean': np.nan,
                'modularity_std': np.nan,
                'modularity_min': np.nan,
                'modularity_max': np.nan,
                'num_communities': np.nan,
                'zrand_mean': np.nan,
                'zrand_std': np.nan,
                'zrand_min': np.nan,
                'zrand_max': np.nan
            }
        
        summary_stats.append(stats)
    
    # Find best gamma (highest zrand with >1 community)
    summary_df = pd.DataFrame(summary_stats)
    valid_rows = summary_df[
        (~summary_df['modularity_mean'].isna()) & 
        (summary_df['num_communities'] > 1)
    ]
        
    best_gamma_zrand_info = None
    if len(valid_rows) > 0:
        best_gamma_idx = valid_rows['zrand_mean'].idxmax()
        best_gamma = valid_rows.loc[best_gamma_idx, 'gamma']
        best_gamma_zrand_info = {
            'best_gamma': best_gamma,
            'best_modularity_mean': valid_rows.loc[best_gamma_idx, 'modularity_mean'],
            'best_modularity_std': valid_rows.loc[best_gamma_idx, 'modularity_std'],
            'best_num_communities': valid_rows.loc[best_gamma_idx, 'num_communities'],
            'best_zrand_mean': valid_rows.loc[best_gamma_idx, 'zrand_mean'],
            'consensus_community': all_results[best_gamma]['consensus_community']
        }
    
    return {
        'all_results': all_results,
        'summary_stats': summary_df,
        'best_gamma_zrand_info': best_gamma_zrand_info
    }

def save_modularity_results(results, output_dir, metric):
    """Save modularity analysis results - best gamma AND all gamma sweep data"""
    
    # Save best gamma zrand results (existing functionality)
    if results['best_gamma_zrand_info'] is not None:
        comprehensive_summary = {
            'metric': metric,
            **{k: v for k, v in results['best_gamma_zrand_info'].items() if k != 'consensus_community'}
        }
        
        comprehensive_df = pd.DataFrame([comprehensive_summary])
        comprehensive_df = comprehensive_df.round(2)
        comprehensive_path = os.path.join(output_dir, f'modularity_best_gamma_zrand_{metric}.csv')
        comprehensive_df.to_csv(comprehensive_path, index=False)
        
        # Save community assignments
        community_df = pd.DataFrame({
            'node_id': range(len(results['best_gamma_zrand_info']['consensus_community'])),
            'community_assignment': results['best_gamma_zrand_info']['consensus_community']
        })
        community_path = os.path.join(output_dir, f'community_assignments_zrand_{metric}.csv')
        community_df.to_csv(community_path, index=False)
        
        print(f"    Best gamma zrand: {results['best_gamma_zrand_info']['best_gamma']:.1f} "
              f"(modularity: {results['best_gamma_zrand_info']['best_modularity_mean']:.3f}, "
              f"communities: {int(results['best_gamma_zrand_info']['best_num_communities'])})")
    
    if 'summary_stats' in results and results['summary_stats'] is not None:
        gamma_sweep_df = results['summary_stats'].round(3)
        gamma_sweep_path = os.path.join(output_dir, f'modularity_gamma_sweep_{metric}.csv')
        gamma_sweep_df.to_csv(gamma_sweep_path, index=False)
        print(f"    Saved gamma sweep data: {gamma_sweep_path}")

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

def collect_group_averaged_gamma_sweep_data():
    """
    Collect gamma sweep data for all metrics
    
    Returns:
        Dictionary: {metric: DataFrame with gamma, modularity_mean, zrand_mean, etc.}
    """
    gamma_sweep_data = {}
    
    print("Collecting gamma sweep data for group averaged connectomes...")
    
    for metric in config.metrics:
        print(f"Processing gamma sweep data for: {metric}")
        
        # Path to gamma sweep results
        file_path = f"{config.output_path}/modularity_gamma_sweep_{metric}.csv"
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                
                # Check if required columns exist
                required_cols = ['gamma', 'modularity_mean', 'zrand_mean']
                if all(col in df.columns for col in required_cols):
                    # Filter out invalid values
                    df_clean = df.dropna(subset=['modularity_mean', 'zrand_mean'])
                    df_clean = df_clean[df_clean['n_valid_runs'] > 0]
                    
                    if len(df_clean) > 0:
                        gamma_sweep_data[metric] = df_clean
                        print(f"  Found {len(df_clean)} valid gamma points")
                    else:
                        print(f"  Warning: No valid data points for {metric}")
                else:
                    print(f"  Warning: Missing required columns for {metric}")
                    print(f"  Available columns: {list(df.columns)}")
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"  File not found: {file_path}")
    
    return gamma_sweep_data

def create_gamma_sweep_plots(gamma_sweep_data, output_dir):
    """
    Create plots showing how Q and zrand change with gamma for each metric
    """
    print("Creating gamma sweep plots...")
    
    if not gamma_sweep_data:
        print("No gamma sweep data to plot")
        return
    
    # Create a 2x1 subplot figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: Modularity Q vs Gamma
    for metric in config.metrics:
        if metric in gamma_sweep_data:
            df = gamma_sweep_data[metric]
            color = metric_colors.get(metric, '#888888')
            
            # Plot Q vs gamma
            ax1.plot(df['gamma'], df['modularity_mean'], 
                    color=color, linewidth=2, marker='o', markersize=4,
                    label=metric, alpha=0.8)
            
            # Add error bars if available
            if 'modularity_std' in df.columns:
                ax1.fill_between(df['gamma'], 
                               df['modularity_mean'] - df['modularity_std'],
                               df['modularity_mean'] + df['modularity_std'],
                               color=color, alpha=0.2)
    
    ax1.set_xlabel('Gamma', fontsize=12)
    ax1.set_ylabel('Modularity Q', fontsize=12)
    ax1.set_title('Modularity Q vs Gamma (Group Averaged Connectomes)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Z-Rand vs Gamma
    for metric in config.metrics:
        if metric in gamma_sweep_data:
            df = gamma_sweep_data[metric]
            color = metric_colors.get(metric, '#888888')
            
            # Plot zrand vs gamma
            ax2.plot(df['gamma'], df['zrand_mean'], 
                    color=color, linewidth=2, marker='s', markersize=4,
                    label=metric, alpha=0.8)
            
            # Add error bars if available
            if 'zrand_std' in df.columns:
                ax2.fill_between(df['gamma'], 
                               df['zrand_mean'] - df['zrand_std'],
                               df['zrand_mean'] + df['zrand_std'],
                               color=color, alpha=0.2)
    
    ax2.set_xlabel('Gamma', fontsize=12)
    ax2.set_ylabel('Z-Rand Score', fontsize=12)
    ax2.set_title('Z-Rand Score vs Gamma (Group Averaged Connectomes)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = f"{output_dir}/modularity_gamma_sweep_plots_group_averaged.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved gamma sweep plots: {output_file}")
    
    # Also create individual plots for better visibility
    create_individual_gamma_sweep_plots(gamma_sweep_data, output_dir)
    
def create_individual_gamma_sweep_plots(gamma_sweep_data, output_dir):
    """
    Create individual plots for each metric showing Q and zrand vs gamma
    """
    print("Creating individual gamma sweep plots...")
    
    for metric in config.metrics:
        if metric not in gamma_sweep_data:
            continue
            
        df = gamma_sweep_data[metric]
        color = metric_colors.get(metric, '#888888')
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Modularity Q
        ax1.plot(df['gamma'], df['modularity_mean'], 
                color=color, linewidth=3, marker='o', markersize=6)
        
        if 'modularity_std' in df.columns:
            ax1.fill_between(df['gamma'], 
                           df['modularity_mean'] - df['modularity_std'],
                           df['modularity_mean'] + df['modularity_std'],
                           color=color, alpha=0.3)
        
        ax1.set_xlabel('Gamma', fontsize=14)
        ax1.set_ylabel('Modularity Q', fontsize=14)
        ax1.set_title(f'Modularity Q vs Gamma\n{metric}', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=12)
        
        # Plot 2: Z-Rand
        ax2.plot(df['gamma'], df['zrand_mean'], 
                color=color, linewidth=3, marker='s', markersize=6)
        
        if 'zrand_std' in df.columns:
            ax2.fill_between(df['gamma'], 
                           df['zrand_mean'] - df['zrand_std'],
                           df['zrand_mean'] + df['zrand_std'],
                           color=color, alpha=0.3)
        
        ax2.set_xlabel('Gamma', fontsize=14)
        ax2.set_ylabel('Z-Rand Score', fontsize=14)
        ax2.set_title(f'Z-Rand Score vs Gamma\n{metric}', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=12)
        
        # Highlight best gamma point if available
        optimal_data = collect_group_averaged_optimal_parameters()
        if metric in optimal_data:
            best_gamma = optimal_data[metric]['gamma']
            best_q = optimal_data[metric]['Q']
            best_zrand = optimal_data[metric]['zrand']
            
            ax1.axvline(x=best_gamma, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax1.plot(best_gamma, best_q, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
            ax1.text(best_gamma, best_q, f'  Best γ={best_gamma:.1f}', 
                    verticalalignment='bottom', fontsize=12, fontweight='bold', color='red')
            
            ax2.axvline(x=best_gamma, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax2.plot(best_gamma, best_zrand, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
            ax2.text(best_gamma, best_zrand, f'  Best γ={best_gamma:.1f}', 
                    verticalalignment='bottom', fontsize=12, fontweight='bold', color='red')
        
        plt.tight_layout()
        
        # Save individual plot
        output_file = f"{output_dir}/gamma_sweep_{metric}_individual.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved individual plot: {output_file}")

def collect_group_averaged_optimal_parameters():
    """
    Collect optimal gamma, number of modules, zrand, and Q values for group averaged connectomes
    
    Returns:
        Dictionary: {weighted_metric: {'gamma': float, 'modules': int, 'zrand': float, 'Q': float}}
    """
    optimal_data = {}
    
    print("Collecting optimal parameters for group averaged connectomes...")
    
    # Process each weighted metric
    for weighted_metric in config.metrics:
        print(f"Processing group averaged data for: {weighted_metric}")
        
        # Path to group averaged analysis results
        file_path = f"{config.output_path}/modularity_best_gamma_zrand_{weighted_metric}.csv"
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                
                # Check if required columns exist
                required_cols = ['best_gamma', 'best_num_communities', 'best_zrand_mean', 'best_modularity_mean']
                if all(col in df.columns for col in required_cols):
                    # For group averaged, we should have one row of results
                    if len(df) > 0:
                        row = df.iloc[0]  # Take the first (and likely only) row
                        gamma_val = row['best_gamma']
                        modules_val = row['best_num_communities'] 
                        zrand_val = row['best_zrand_mean']
                        q_val = row['best_modularity_mean']
                        
                        # Check for valid values
                        if not any(np.isnan([gamma_val, modules_val, zrand_val, q_val])):
                            optimal_data[weighted_metric] = {
                                'gamma': gamma_val,
                                'modules': int(modules_val),
                                'zrand': zrand_val,
                                'Q': q_val
                            }
                            print(f"  Found: gamma={gamma_val:.3f}, modules={int(modules_val)}, zrand={zrand_val:.3f}, Q={q_val:.3f}")
                        else:
                            print(f"  Warning: Invalid values found for {weighted_metric}")
                    else:
                        print(f"  Warning: No data rows found for {weighted_metric}")
                else:
                    print(f"  Warning: Missing required columns for {weighted_metric}")
                    print(f"  Available columns: {list(df.columns)}")
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"  File not found: {file_path}")
    
    return optimal_data

def create_optimal_parameters_comparison_plot(optimal_data, output_dir):
    """
    Create comparison plots showing optimal gamma, number of modules, zrand, and Q for each metric
    For group averaged data
    """
    print("Creating optimal parameters comparison plots...")
    
    if not optimal_data:
        print("No optimal parameters data to plot")
        return
    
    # Create a 2x2 subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics_list = list(optimal_data.keys())
    colors = [metric_colors.get(metric, '#888888') for metric in metrics_list]
    
    # Extract values for each parameter
    gamma_values = [optimal_data[metric]['gamma'] for metric in metrics_list]
    modules_values = [optimal_data[metric]['modules'] for metric in metrics_list]
    zrand_values = [optimal_data[metric]['zrand'] for metric in metrics_list]
    q_values = [optimal_data[metric]['Q'] for metric in metrics_list]
    
    # Plot 1: Optimal Gamma
    bars1 = ax1.bar(range(len(metrics_list)), gamma_values, color=colors, alpha=0.8)
    ax1.set_title('Optimal Gamma Values by Metric (Group Averaged)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Gamma', fontsize=12)
    ax1.set_xticks(range(len(metrics_list)))
    ax1.set_xticklabels(metrics_list, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, value in zip(bars1, gamma_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(gamma_values)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Number of Modules
    bars2 = ax2.bar(range(len(metrics_list)), modules_values, color=colors, alpha=0.8)
    ax2.set_title('Number of Modules by Metric (Group Averaged)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Modules', fontsize=12)
    ax2.set_xticks(range(len(metrics_list)))
    ax2.set_xticklabels(metrics_list, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, value in zip(bars2, modules_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(modules_values)*0.01,
                f'{value}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Z-Rand
    bars3 = ax3.bar(range(len(metrics_list)), zrand_values, color=colors, alpha=0.8)
    ax3.set_title('Z-Rand Values by Metric (Group Averaged)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Z-Rand', fontsize=12)
    ax3.set_xticks(range(len(metrics_list)))
    ax3.set_xticklabels(metrics_list, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, value in zip(bars3, zrand_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(zrand_values)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Modularity Q
    bars4 = ax4.bar(range(len(metrics_list)), q_values, color=colors, alpha=0.8)
    ax4.set_title('Modularity Q Values by Metric (Group Averaged)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Modularity Q', fontsize=12)
    ax4.set_xticks(range(len(metrics_list)))
    ax4.set_xticklabels(metrics_list, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, value in zip(bars4, q_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(q_values)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = f"{output_dir}/optimal_parameters_comparison_group_averaged_84_regions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved optimal parameters comparison plot: {output_file}")
    
    # Also create a summary table
    summary_data = []
    for metric in metrics_list:
        data = optimal_data[metric]
        summary_data.append({
            'Metric': metric,
            'Optimal_Gamma': data['gamma'],
            'Number_of_Modules': data['modules'],
            'ZRand': data['zrand'],
            'Modularity_Q': data['Q']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = f"{output_dir}/optimal_parameters_group_averaged_84_regions.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved optimal parameters summary: {summary_file}")
    
def collect_group_averaged_community_assignments():
    """
    Collect community assignments for group averaged connectomes
    Using 84-region atlas structure
    
    Returns:
        Dictionary: {weighted_metric: {'assignments': [node_assignments], 'num_communities': int}}
    """
    community_data = {}
    
    print("Collecting community assignments for group averaged connectomes...")
    
    for weighted_metric in config.metrics:
        print(f"Processing community assignments for: {weighted_metric}")
        
        # Path to group averaged community assignments
        partition_file = f"{config.output_path}/community_assignments_zrand_{weighted_metric}.csv"
        
        if os.path.exists(partition_file):
            try:
                df = pd.read_csv(partition_file)
                
                # Check if we have node and community columns
                if len(df.columns) >= 2:
                    node_ids = df.iloc[:, 0].values
                    assignments = df.iloc[:, 1].values
                    
                    # Create full assignment array for 84 regions
                    full_assignments = np.zeros(84, dtype=int)
                    for node_id, assignment in zip(node_ids, assignments):
                        if 0 <= node_id < 84:
                            full_assignments[int(node_id)] = int(assignment)
                    
                    num_communities = len(np.unique(assignments[assignments > 0]))
                    
                    community_data[weighted_metric] = {
                        'assignments': full_assignments,
                        'num_communities': num_communities
                    }
                    
                    print(f"  Found {num_communities} communities with {np.sum(full_assignments > 0)} assigned nodes")
                else:
                    print(f"  Warning: Insufficient columns in {partition_file}")
            
            except Exception as e:
                print(f"Error reading {partition_file}: {e}")
        else:
            print(f"  File not found: {partition_file}")
    
    return community_data

def create_community_brain_visualization(community_data, output_dir):
    """
    Create brain surface visualizations showing community assignments for group averaged data
    Using dynamic colormaps based on the actual number of communities found
    Updated for 84 regions
    """
    print("="*60)
    print("CREATING COMMUNITY ASSIGNMENT BRAIN VISUALIZATIONS (GROUP AVERAGED - 84 REGIONS)")
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
    metric_community_info = {}
    
    for metric_idx, metric in enumerate(config.metrics):
        if metric not in community_data:
            print(f"Skipping {metric} - no community data")
            continue
            
        print(f"Processing {metric}...")
        
        community_assignments = community_data[metric]['assignments']
        unique_communities = np.unique(community_assignments[community_assignments > 0])
        num_communities = len(unique_communities)
        
        if num_communities == 0:
            print(f"No communities found for {metric}")
            continue
        
        print(f"  Found {num_communities} communities: {unique_communities}")
        metric_community_info[metric] = {'num_communities': num_communities, 'communities': unique_communities}
        
        # Generate dynamic colormap based on actual number of communities
        custom_cmap = generate_dynamic_colormap(num_communities)
        
        # Map to surface
        lh_data, rh_data = map_data_to_surface(
            community_assignments, lh_labels, rh_labels, lh_names, rh_names
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
                              max=np.max(unique_communities),
                              colormap=custom_cmap,
                              alpha=1.0,
                              smoothing_steps=0,  # No smoothing for discrete data
                              thresh=0.5,  # Only show assigned regions
                              colorbar=False)
                
                brain.show_view(view)
                
                temp_file = f"{output_dir}/temp_communities_{metric}_{view_title.replace(' ', '_')}.png"
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
    print("Assembling final community visualization...")
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
                # Add metric name and community count
                if metric in metric_community_info:
                    num_comm = metric_community_info[metric]['num_communities']
                    label_text = f"{metric}\n({num_comm} communities)"
                else:
                    label_text = metric
                    
                ax.text(-0.15, 0.5, label_text, transform=ax.transAxes, 
                       fontsize=11, fontweight='bold', rotation=90, 
                       verticalalignment='center', horizontalalignment='center')
    
    # Add title
    fig.suptitle('Community Assignments: Group Averaged Connectomes (7 Weighted Metrics × 4 Brain Views)\n84-Region DK Atlas with Dynamic Coloring', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout to use full width
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.05, hspace=0.1, wspace=0.02)
    
    output_file = f"{output_dir}/community_assignments_group_averaged_84_regions.png"
    plt.savefig(output_file, dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"Saved community visualization: {output_file}")
    
    # Print summary of communities per metric
    print("\nCommunity Summary:")
    for metric in config.metrics:
        if metric in metric_community_info:
            info = metric_community_info[metric]
            print(f"  {metric}: {info['num_communities']} communities {list(info['communities'])}")
    
    return output_file

def save_community_assignments(community_data, output_dir):
    """
    Save detailed community assignment data for 84 regions (group averaged)
    """
    print("Saving community assignment data...")
    
    # Save community assignments
    assignment_df_data = []
    node_to_region = get_corrected_dk_atlas_mapping()
    
    for metric in config.metrics:
        if metric in community_data:
            assignments = community_data[metric]['assignments']
            for node_idx, community_id in enumerate(assignments):
                if node_idx < 84:  # Only process 84 regions
                    fs_node_idx = node_idx + 1
                    region_name = node_to_region.get(fs_node_idx, f"Unknown_Region_{node_idx}")
                    assignment_df_data.append({
                        'Metric': metric,
                        'Node_ID': node_idx,
                        'FS_Node_ID': fs_node_idx,  # Add the 1-based FreeSurfer index
                        'Region_Name': region_name,
                        'Community_Assignment': int(community_id)
                    })
    
    assignment_df = pd.DataFrame(assignment_df_data)
    assignment_file = f"{output_dir}/community_assignments_group_averaged_84_regions.csv"
    assignment_df.to_csv(assignment_file, index=False)
    print(f"Saved community assignments: {assignment_file}")
    
    # Save summary statistics
    summary_data = []
    for metric in config.metrics:
        if metric in community_data:
            assignments = community_data[metric]['assignments']
            unique_communities = np.unique(assignments[assignments > 0])
            num_communities = len(unique_communities)
            num_assigned_nodes = np.sum(assignments > 0)
            
            summary_data.append({
                'Metric': metric,
                'Num_Communities': num_communities,
                'Assigned_Nodes': num_assigned_nodes,
                'Unassigned_Nodes': 84 - num_assigned_nodes,
                'Community_IDs': ','.join(map(str, unique_communities))
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = f"{output_dir}/community_summary_group_averaged_84_regions.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary: {summary_file}")

def process_connectome_modularity_analysis(connectome_path, output_dir, metric):
    """
    Process a single connectome with modularity analysis
    """
    try:
        print(f"\nProcessing Modularity Analysis: Group Average Metric {metric}")
        
        # Load connectome
        W = pd.read_csv(connectome_path, header=None, index_col=None).values
        print(f"  Loaded connectome: {W.shape}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # MODULARITY ANALYSIS
        print("  Computing modularity analysis...")
        try:
            mod_results = compute_modularity_analysis(W)
            if mod_results is not None:
                # Save results
                save_modularity_results(mod_results, output_dir, metric)
                print("Modularity analysis completed")
                return mod_results
            else:
                print("Modularity analysis skipped (library not available)")
                return None
        except Exception as e:
            print(f"Modularity analysis failed: {e}")
            return None
        
    except Exception as e:
        print(f"Error processing Group Average Metric {metric}: {e}")
        return None

def main():
    """
    Main function for modularity analysis
    """
    print("="*80)
    print("MODULARITY ANALYSIS - GROUP AVERAGED CONNECTOMES")
    print("="*80)
    print(f"Configuration:")
    print(f"  Base directory: {config.group_connectomes_dir}")
    print(f"  Output directory: {config.output_path}")
    print(f"  Metrics: {config.metrics}")
    
    # Create output directory
    os.makedirs(config.output_path, exist_ok=True)
    
    processed = 0
    failed = 0
    
    print(f"\nStarting modularity analysis for {len(config.metrics)} group averaged connectome files...")
    print("="*80)
    
    # Process each metric
    for metric in config.metrics:
        connectome_path = os.path.join(
            config.group_connectomes_dir,
            f"group_{metric}_median.csv"
        )
        
        if os.path.exists(connectome_path):
            results = process_connectome_modularity_analysis(
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
    print("MODULARITY ANALYSIS COMPLETED!")
    print("="*80)
    print(f"Successfully processed: {processed}/{len(config.metrics)} files")
    print(f"Failed: {failed}/{len(config.metrics)} files")
    
    # Step 1: Collect optimal parameters for visualization
    print("\nStep 1: Creating optimal parameters comparison plot...")
    optimal_data = collect_group_averaged_optimal_parameters()
    
    if optimal_data:
        create_optimal_parameters_comparison_plot(optimal_data, config.output_path)
    
    # Step 2: Collect community assignments for visualization
    print("Step 2: Creating community assignment visualizations...")
    community_data = collect_group_averaged_community_assignments()
    
    if community_data:
        save_community_assignments(community_data, config.output_path)
        create_community_brain_visualization(community_data, config.output_path)
        
if __name__ == "__main__":
    main()