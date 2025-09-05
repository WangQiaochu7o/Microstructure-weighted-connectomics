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
        self.group_connectomes_dir = "/home/localadmin/Desktop/SP2/164output/pro_group_average"
        self.output_path = "/home/localadmin/Desktop/SP2/164output/group_averaged_output"
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

def get_corrected_a2009s_atlas_mapping():
    """
    CORRECTED mapping for MRtrix3 FreeSurfer a2009s parcellation (Destrieux atlas)
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
        node_to_region[i + 1] = f"lh.{region}"
    for i, region in enumerate(cortical_regions):
        node_to_region[90 + i] = f"rh.{region}"
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
                'num_communities': results['num_communities'],
                'zrand_mean': np.mean(valid_zrand) if len(valid_zrand) > 0 else np.nan,
                'zrand_std': np.std(valid_zrand) if len(valid_zrand) > 0 else np.nan,
            }
        else:
            stats = {'gamma': gamma, 'n_valid_runs': 0, 'modularity_mean': np.nan, 'modularity_std': np.nan, 'num_communities': np.nan, 'zrand_mean': np.nan, 'zrand_std': np.nan}
        
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    valid_rows = summary_df[(~summary_df['modularity_mean'].isna()) & (summary_df['num_communities'] > 1)]
        
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
    
    if results['best_gamma_zrand_info'] is not None:
        comprehensive_summary = {
            'metric': metric,
            **{k: v for k, v in results['best_gamma_zrand_info'].items() if k != 'consensus_community'}
        }
        
        comprehensive_df = pd.DataFrame([comprehensive_summary])
        comprehensive_path = os.path.join(output_dir, f'modularity_best_gamma_zrand_{metric}.csv')
        comprehensive_df.to_csv(comprehensive_path, index=False)
        
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
        gamma_sweep_df = results['summary_stats']
        gamma_sweep_path = os.path.join(output_dir, f'modularity_gamma_sweep_{metric}.csv')
        gamma_sweep_df.to_csv(gamma_sweep_path, index=False)
        print(f"    Saved gamma sweep data: {gamma_sweep_path}")

def generate_dynamic_colormap(n_communities):
    if n_communities == 0: return ListedColormap(['#000000'])
    if n_communities <= 10:
        base_colors = ['#FF0000', '#0000FF', '#00AA00', '#FF66CC', '#FF8800', '#8800CC', '#996633', '#00CCCC', '#FFFF00', '#888888']
        return ListedColormap(base_colors[:n_communities])
    else:
        if n_communities <= 20:
            return ListedColormap([mcolors.hsv_to_rgb([i / n_communities, 0.8, 0.9]) for i in range(n_communities)])
        else:
            colors = []
            cmaps = ['tab20', 'tab20b', 'tab20c']
            for i in range(n_communities):
                cmap = plt.get_cmap(cmaps[i // 20 % len(cmaps)])
                colors.append(cmap(i % 20 / 19))
            return ListedColormap(colors)

def load_freesurfer_destrieux_atlas():
    """
    Load FreeSurfer Destrieux atlas (a2009s) for 164 regions
    """
    try:
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
        lh_annot_path = f"{config.freesurfer_home}/subjects/fsaverage/label/lh.aparc.a2009s.annot"
        rh_annot_path = f"{config.freesurfer_home}/subjects/fsaverage/label/rh.aparc.a2009s.annot"
        
        if os.path.exists(lh_annot_path) and os.path.exists(rh_annot_path):
            lh_labels, _, lh_names = nib.freesurfer.read_annot(lh_annot_path)
            rh_labels, _, rh_names = nib.freesurfer.read_annot(rh_annot_path)
            
            if len(lh_names) > 0 and isinstance(lh_names[0], bytes):
                lh_names = [name.decode('utf-8') for name in lh_names]
            if len(rh_names) > 0 and isinstance(rh_names[0], bytes):
                rh_names = [name.decode('utf-8') for name in rh_names]
                
            return lh_labels, rh_labels, fsaverage, lh_names, rh_names
        else:
            from nilearn.datasets import fetch_atlas_surf_destrieux
            atlas = fetch_atlas_surf_destrieux()
            return atlas['map_left'], atlas['map_right'], fsaverage, [f"lh_region_{i}" for i in range(148)], [f"rh_region_{i}" for i in range(148)]
    except Exception as e:
        print(f"Error loading Destrieux atlas: {e}")
        return None, None, None, None, None

def map_data_to_surface(data_values, lh_labels, rh_labels, lh_names, rh_names):
    """
    Map data values to brain surface using Destrieux atlas (164 regions)
    """
    lh_data, rh_data = np.zeros(len(lh_labels)), np.zeros(len(rh_labels))
    node_to_region = get_corrected_a2009s_atlas_mapping()
    
    lh_name_to_label = {name if name.startswith('lh.') else f"lh.{name}": i for i, name in enumerate(lh_names)}
    rh_name_to_label = {name if name.startswith('rh.') else f"rh.{name}": i for i, name in enumerate(rh_names)}
    
    for connectome_idx, data_value in enumerate(data_values):
        fs_node_idx = connectome_idx + 1
        if fs_node_idx not in node_to_region or data_value == 0: continue
        
        region_name = node_to_region[fs_node_idx]
        
        if region_name.startswith('lh.'):
            if region_name in lh_name_to_label:
                lh_data[lh_labels == lh_name_to_label[region_name]] = float(data_value)
        else:
            if region_name in rh_name_to_label:
                rh_data[rh_labels == rh_name_to_label[region_name]] = float(data_value)
                
    return np.nan_to_num(lh_data), np.nan_to_num(rh_data)

def collect_group_averaged_gamma_sweep_data():
    """
    Collect gamma sweep data for all metrics
    """
    gamma_sweep_data = {}
    print("Collecting gamma sweep data...")
    for metric in config.metrics:
        file_path = f"{config.output_path}/modularity_gamma_sweep_{metric}.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if all(col in df.columns for col in ['gamma', 'modularity_mean', 'zrand_mean']):
                    gamma_sweep_data[metric] = df.dropna(subset=['modularity_mean', 'zrand_mean'])
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return gamma_sweep_data

def create_gamma_sweep_plots(gamma_sweep_data, output_dir):
    """
    Create combined and individual plots for gamma sweep data
    """
    if not gamma_sweep_data:
        print("No gamma sweep data to plot.")
        return
    
    # Combined plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    for metric, df in gamma_sweep_data.items():
        color = metric_colors.get(metric, '#888888')
        ax1.plot(df['gamma'], df['modularity_mean'], color=color, lw=2, marker='o', ms=4, label=metric, alpha=0.8)
        ax2.plot(df['gamma'], df['zrand_mean'], color=color, lw=2, marker='s', ms=4, label=metric, alpha=0.8)
    ax1.set(title='Modularity Q vs Gamma', ylabel='Modularity Q', xlabel='Gamma')
    ax2.set(title='Z-Rand Score vs Gamma', ylabel='Z-Rand Score', xlabel='Gamma')
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/modularity_gamma_sweep_plots_group_averaged.png", dpi=300)
    plt.close()
    
    # Individual plots
    for metric, df in gamma_sweep_data.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        color = metric_colors.get(metric, '#888888')
        ax1.plot(df['gamma'], df['modularity_mean'], color=color, lw=3, marker='o')
        ax2.plot(df['gamma'], df['zrand_mean'], color=color, lw=3, marker='s')
        ax1.set(title=f'Modularity Q vs Gamma\n{metric}', xlabel='Gamma', ylabel='Modularity Q')
        ax2.set(title=f'Z-Rand Score vs Gamma\n{metric}', xlabel='Gamma', ylabel='Z-Rand Score')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gamma_sweep_{metric}_individual.png", dpi=300)
        plt.close()

def collect_group_averaged_optimal_parameters():
    """
    Collect optimal parameters for group averaged connectomes (164 regions)
    """
    optimal_data = {}
    print("Collecting optimal parameters for group averaged connectomes...")
    for metric in config.metrics:
        file_path = f"{config.output_path}/modularity_best_gamma_zrand_{metric}.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path).iloc[0]
                optimal_data[metric] = {
                    'gamma': df['best_gamma'],
                    'modules': int(df['best_num_communities']),
                    'zrand': df['best_zrand_mean'],
                    'Q': df['best_modularity_mean']
                }
            except Exception as e:
                print(f"Error reading or parsing {file_path}: {e}")
    return optimal_data

def create_optimal_parameters_comparison_plot(optimal_data, output_dir):
    """
    Create comparison bar plots for optimal parameters (164 regions)
    """
    if not optimal_data:
        print("No optimal parameters data to plot.")
        return
        
    metrics = list(optimal_data.keys())
    colors = [metric_colors.get(m, '#888888') for m in metrics]
    params = ['gamma', 'modules', 'zrand', 'Q']
    titles = ['Optimal Gamma', 'Number of Modules', 'Z-Rand Score', 'Modularity Q']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, param, title in zip(axes.flatten(), params, titles):
        values = [optimal_data[m][param] for m in metrics]
        # Fix: Use numerical positions for bars (following 84-region script approach)
        ax.bar(range(len(metrics)), values, color=colors)
        ax.set_title(title, fontweight='bold')
        # Fix: Use set_xticks() and set_xticklabels() approach (following 84-region script)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/optimal_parameters_comparison_group_averaged_164_regions.png", dpi=300)
    plt.close()
    
    # Fix: Create summary_data from optimal_data
    summary_data = []
    for metric in metrics:
        summary_data.append({
            'Metric': metric,
            'Optimal_Gamma': optimal_data[metric]['gamma'],
            'Num_Modules': optimal_data[metric]['modules'],
            'ZRand_Score': optimal_data[metric]['zrand'],
            'Modularity_Q': optimal_data[metric]['Q']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}/optimal_parameters_group_averaged_164_regions.csv", index=False)


def collect_group_averaged_community_assignments():
    """
    Collect community assignments for group averaged connectomes (164 regions)
    """
    community_data = {}
    print("Collecting community assignments for group averaged connectomes (164 regions)...")
    for metric in config.metrics:
        partition_file = f"{config.output_path}/community_assignments_zrand_{metric}.csv"
        if os.path.exists(partition_file):
            try:
                df = pd.read_csv(partition_file)
                assignments = df.iloc[:, 1].values
                full_assignments = np.zeros(164, dtype=int)
                full_assignments[:len(assignments)] = assignments
                community_data[metric] = {
                    'assignments': full_assignments,
                    'num_communities': len(np.unique(assignments[assignments > 0]))
                }
            except Exception as e:
                print(f"Error reading {partition_file}: {e}")
    return community_data

def create_community_brain_visualization(community_data, output_dir):
    """
    Create brain surface visualizations for community assignments (164 regions)
    """
    print("="*60)
    print("CREATING COMMUNITY BRAIN VISUALIZATIONS (GROUP AVERAGED - 164 REGIONS)")
    print("="*60)
    
    lh_labels, rh_labels, _, lh_names, rh_names = load_freesurfer_destrieux_atlas()
    if lh_labels is None:
        print("Failed to load Destrieux atlas. Skipping visualization.")
        return
    
    views = [('lateral', 'lh', 'Left Lateral'), ('medial', 'lh', 'Left Medial'), 
             ('lateral', 'rh', 'Right Lateral'), ('medial', 'rh', 'Right Medial')]
    
    plot_images = {}
    for metric, data in community_data.items():
        print(f"Processing {metric}...")
        assignments = data['assignments']
        num_communities = data['num_communities']
        if num_communities == 0: continue
        
        custom_cmap = generate_dynamic_colormap(num_communities)
        lh_data, rh_data = map_data_to_surface(assignments, lh_labels, rh_labels, lh_names, rh_names)
        
        plot_images[metric] = {}
        for view, hemi, view_title in views:
            try:
                brain = Brain('fsaverage', hemi, 'pial', background='white', size=(800, 600))
                brain.add_data(lh_data if hemi == 'lh' else rh_data, 
                              min=0.5, max=np.max(assignments), colormap=custom_cmap, 
                              thresh=0.5, colorbar=False)
                temp_file = f"{output_dir}/temp_{metric}_{view}.png"
                brain.save_image(temp_file)
                plot_images[metric][view_title] = np.array(Image.open(temp_file))
                brain.close()
                os.remove(temp_file)
            except Exception as e:
                print(f"Error creating {view_title} for {metric}: {e}")
                plot_images[metric][view_title] = np.ones((600, 800, 3), dtype=np.uint8) * 128
    
    fig, axes = plt.subplots(len(config.metrics), len(views), figsize=(20, 6 * len(config.metrics)))
    if len(config.metrics) == 1: axes = axes.reshape(1, -1)
    
    for row, metric in enumerate(config.metrics):
        for col, (_, _, view_title) in enumerate(views):
            ax = axes[row, col]
            if metric in plot_images and view_title in plot_images.get(metric, {}):
                ax.imshow(plot_images[metric][view_title])
            ax.axis('off')
            if row == 0: ax.set_title(view_title, fontweight='bold', pad=15)
            if col == 0:
                num_comm = community_data.get(metric, {}).get('num_communities', 0)
                ax.text(-0.15, 0.5, f"{metric}\n({num_comm} comm.)", transform=ax.transAxes, 
                       fontsize=11, fontweight='bold', rotation=90, va='center', ha='center')

    fig.suptitle('Community Assignments: Group Averaged Connectomes (164-Region Destrieux Atlas)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.05, hspace=0.1, wspace=0.02)
    output_file = f"{output_dir}/community_assignments_group_averaged_164_regions.png"
    plt.savefig(output_file, dpi=400, facecolor='white')
    plt.close(fig)
    print(f"Saved community visualization: {output_file}")

def save_community_assignments(community_data, output_dir):
    """
    Save detailed community assignment data for 164 regions
    """
    print("Saving community assignment data (164 regions)...")
    
    assignment_data = []
    node_to_region = get_corrected_a2009s_atlas_mapping()
    for metric, data in community_data.items():
        for node_idx, comm_id in enumerate(data['assignments']):
            fs_node_idx = node_idx + 1
            region_name = node_to_region.get(fs_node_idx, f"Unknown_Region_{fs_node_idx}")
            assignment_data.append({
                'Metric': metric, 'Node_ID': node_idx, 'FS_Node_ID': fs_node_idx,
                'Region_Name': region_name, 'Community_Assignment': int(comm_id)
            })
    pd.DataFrame(assignment_data).to_csv(f"{output_dir}/community_assignments_group_averaged_164_regions.csv", index=False)
    
    summary_data = []
    for metric, data in community_data.items():
        assignments = data['assignments']
        unique_comms = np.unique(assignments[assignments > 0])
        summary_data.append({
            'Metric': metric, 'Num_Communities': data['num_communities'],
            'Assigned_Nodes': np.sum(assignments > 0),
            'Unassigned_Nodes': 164 - np.sum(assignments > 0),
            'Community_IDs': ','.join(map(str, unique_comms))
        })
    pd.DataFrame(summary_data).to_csv(f"{output_dir}/community_summary_group_averaged_164_regions.csv", index=False)

def process_connectome_modularity_analysis(connectome_path, output_dir, metric):
    """
    Process a single connectome with modularity analysis
    """
    try:
        print(f"\nProcessing Modularity Analysis: {metric}")
        W = pd.read_csv(connectome_path, header=None, index_col=None).values
        print(f"  Loaded connectome: {W.shape}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("  Computing modularity analysis...")
        mod_results = compute_modularity_analysis(W)
        if mod_results:
            save_modularity_results(mod_results, output_dir, metric)
            print("  Modularity analysis completed.")
            return mod_results
        else:
            print("  Modularity analysis failed or skipped.")
            return None
    except Exception as e:
        print(f"Error processing {metric}: {e}")
        return None

def main():
    """
    Main function for modularity analysis on 164-region connectomes
    """
    print("="*80)
    print("MODULARITY ANALYSIS - GROUP AVERAGED CONNECTOMES (164 REGIONS)")
    print("="*80)
    
    os.makedirs(config.output_path, exist_ok=True)
    
    processed = 0
    failed = 0
    
    for metric in config.metrics:
        connectome_path = os.path.join(config.group_connectomes_dir, f"group_{metric}_median.csv")
        if os.path.exists(connectome_path):
            if process_connectome_modularity_analysis(connectome_path, config.output_path, metric):
                processed += 1
            else:
                failed += 1
        else:
            print(f"Connectome not found: {connectome_path}")
            failed += 1
            
    print("\n" + "="*80)
    print("MODULARITY ANALYSIS COMPLETED!")
    print(f"Successfully processed: {processed}/{len(config.metrics)}")
    print(f"Failed: {failed}/{len(config.metrics)}")
    
    if processed > 0:
        print("\nCreating visualizations...")
        optimal_data = collect_group_averaged_optimal_parameters()
        if optimal_data:
            create_optimal_parameters_comparison_plot(optimal_data, config.output_path)
        
        community_data = collect_group_averaged_community_assignments()
        if community_data:
            save_community_assignments(community_data, config.output_path)
            create_community_brain_visualization(community_data, config.output_path)
        
        gamma_data = collect_group_averaged_gamma_sweep_data()
        if gamma_data:
            create_gamma_sweep_plots(gamma_data, config.output_path)
            
if __name__ == "__main__":
    main()