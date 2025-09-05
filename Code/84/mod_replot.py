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

# --- [核心配置] ---
class AnalysisConfig:
    """Configuration for analysis parameters"""
    def __init__(self):
        self.base_path = "/home/localadmin/Desktop/SP2"
        self.group_connectomes_dir = "/home/localadmin/Desktop/SP2/84output/pro_group_average"
        self.output_path = "/home/localadmin/Desktop/SP2/84output/group_averaged_output"
        self.freesurfer_home = '/home/localadmin/freesurfer'
        self.subjects_dir = os.path.join(self.freesurfer_home, 'subjects')
        self.metrics = ['R1', 'RK_DKI', 'RD_DTI_reciprocal', 'NOS', 'F_SMI', 'FA_DTI', 'DePerp_SMI_reciprocal']

config = AnalysisConfig()

# --- [脚本设置] ---
os.environ['FREESURFER_HOME'] = config.freesurfer_home
os.environ['SUBJECTS_DIR'] = config.subjects_dir

metric_colors = {
    'R1': '#1E90FF', 'DePerp_SMI_reciprocal': '#FF8C00', 'FA_DTI': '#228B22',
    'RD_DTI_reciprocal': '#DC143C', 'NOS': '#9932CC', 'RK_DKI': '#FF69B4', 'F_SMI': '#8B4513'
}

# --- MODIFIED: Added LaTeX name mapping at global scope ---
metric_name_map_latex = {
    'NOS': r'$nos$', 'RK_DKI': r'$rk$', 'FA_DTI': r'$fa$',
    'RD_DTI_reciprocal': r'$1/rd$', 'DePerp_SMI_reciprocal': r'$1/D_{\perp e}$',
    'F_SMI': r'$f$', 'R1': r'$R1$'
}

if not hasattr(mpl_cm, 'get_cmap'):
    def get_cmap(name):
        try: return plt.get_cmap(name)
        except: return plt.colormaps[name]
    mpl_cm.get_cmap = get_cmap

# --- [辅助函数] ---
def get_corrected_dk_atlas_mapping():
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
    for i, region in enumerate(cortical_regions):
        node_to_region[i + 1] = f"lh.{region}"
    for i, region in enumerate(cortical_regions):
        node_to_region[50 + i] = f"rh.{region}"
    return node_to_region

def generate_dynamic_colormap(n_communities):
    if n_communities == 0: return ListedColormap(['#000000'])
    if n_communities <= 10:
        base_colors = ['#FF0000', '#0000FF', '#00AA00', '#FF66CC', '#FF8800', '#8800CC', '#996633', '#00CCCC', '#FFFF00', '#888888']
        return ListedColormap(base_colors[:n_communities])
    elif n_communities <= 20:
        return ListedColormap([mcolors.hsv_to_rgb([i / n_communities, 0.8, 0.9]) for i in range(n_communities)])
    else:
        base_cmap = plt.get_cmap('tab20')
        colors = [base_cmap(i % 20 / 19) for i in range(n_communities)]
        return ListedColormap(colors)

def load_freesurfer_dk_atlas():
    try:
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
        lh_annot_path = os.path.join(config.subjects_dir, 'fsaverage', 'label', 'lh.aparc.annot')
        rh_annot_path = os.path.join(config.subjects_dir, 'fsaverage', 'label', 'rh.aparc.annot')
        if os.path.exists(lh_annot_path) and os.path.exists(rh_annot_path):
            lh_labels, _, lh_names = nib.freesurfer.read_annot(lh_annot_path)
            rh_labels, _, rh_names = nib.freesurfer.read_annot(rh_annot_path)
            lh_names = [name.decode('utf-8') for name in lh_names]
            rh_names = [name.decode('utf-8') for name in rh_names]
            return lh_labels, rh_labels, fsaverage, lh_names, rh_names
        else: return None, None, None, None, None
    except Exception as e:
        print(f"Error loading atlas: {e}")
        return None, None, None, None, None

def map_data_to_surface(data_values, lh_labels, rh_labels, lh_names, rh_names):
    lh_data, rh_data = np.zeros(len(lh_labels)), np.zeros(len(rh_labels))
    node_to_region = get_corrected_dk_atlas_mapping()
    lh_name_to_label = {f"lh.{name}": i for i, name in enumerate(lh_names)}
    rh_name_to_label = {f"rh.{name}": i for i, name in enumerate(rh_names)}
    
    for connectome_idx, data_value in enumerate(data_values):
        fs_node_idx = connectome_idx + 1
        if fs_node_idx not in node_to_region or data_value == 0: continue
        region_name = node_to_region[fs_node_idx]
        
        name_to_label, labels, surface_data = (lh_name_to_label, lh_labels, lh_data) if region_name.startswith('lh.') else (rh_name_to_label, rh_labels, rh_data)
        
        simple_name = region_name.split('.')[-1]
        full_name_lh = f"lh.{simple_name}"
        full_name_rh = f"rh.{simple_name}"

        if region_name in name_to_label:
            label_val = name_to_label[region_name]
            surface_data[labels == label_val] = float(data_value)
        elif full_name_lh in lh_name_to_label and region_name.startswith('lh.'):
             label_val = lh_name_to_label[full_name_lh]
             lh_data[lh_labels == label_val] = float(data_value)
        elif full_name_rh in rh_name_to_label and region_name.startswith('rh.'):
             label_val = rh_name_to_label[full_name_rh]
             rh_data[rh_labels == label_val] = float(data_value)
             
    return np.nan_to_num(lh_data), np.nan_to_num(rh_data)

# --- [数据收集函数] ---
def collect_group_averaged_gamma_sweep_data():
    gamma_sweep_data = {}
    print("Collecting gamma sweep data...")
    for metric in config.metrics:
        file_path = f"{config.output_path}/modularity_gamma_sweep_{metric}.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                gamma_sweep_data[metric] = df.dropna().copy()
            except Exception as e:
                print(f"  Could not read {file_path}: {e}")
    return gamma_sweep_data

def collect_group_averaged_optimal_parameters():
    optimal_data = {}
    print("Collecting optimal parameters...")
    for metric in config.metrics:
        file_path = f"{config.output_path}/modularity_best_gamma_zrand_{metric}.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if len(df) > 0:
                    row = df.iloc[0]
                    optimal_data[metric] = {
                        'gamma': row['best_gamma'], 'modules': int(row['best_num_communities']),
                        'zrand': row['best_zrand_mean'], 'Q': row['best_modularity_mean']
                    }
            except Exception as e:
                print(f"  Could not read {file_path}: {e}")
    return optimal_data

def collect_group_averaged_community_assignments():
    community_data = {}
    print("Collecting community assignments...")
    for metric in config.metrics:
        file_path = f"{config.output_path}/community_assignments_zrand_{metric}.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                assignments = df.iloc[:, 1].values
                full_assignments = np.zeros(84, dtype=int)
                full_assignments[:len(assignments)] = assignments
                community_data[metric] = {
                    'assignments': full_assignments,
                    'num_communities': len(np.unique(assignments[assignments > 0]))
                }
            except Exception as e:
                print(f"  Could not read {file_path}: {e}")
    return community_data

# --- [核心绘图函数] ---

# --- MODIFIED: This function is now the primary sweep plot ---
def create_gamma_sweep_plots(gamma_sweep_data, optimal_data, output_dir):
    print("Creating aggregated gamma sweep plot...")
    if not gamma_sweep_data: return
    
    plt.rcParams['mathtext.fontset'] = 'cm' # For proper LaTeX rendering
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot data for each metric
    for metric, df in gamma_sweep_data.items():
        color = metric_colors.get(metric, '#888888')
        label = metric_name_map_latex.get(metric, metric)
        ax1.plot(df['gamma'], df['modularity_mean'], color=color, linewidth=2, marker='o', markersize=4, label=label)
        ax2.plot(df['gamma'], df['zrand_mean'], color=color, linewidth=2, marker='s', markersize=4, label=label)
    
    ax1.set(ylabel='Modularity Q', title='Modularity Q vs Gamma')
    ax2.set(xlabel='Gamma', ylabel='Z-Rand Score', title='Z-Rand Score vs Gamma')
    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    
    # Only show legend on the top plot
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # --- MODIFIED: Add a text box with optimal values ---
    text_lines = ["Optimal Values:"]
    for metric in config.metrics:
        if metric in optimal_data:
            data = optimal_data[metric]
            latex_label = metric_name_map_latex.get(metric, metric).ljust(18) # Padded for alignment
            text_lines.append(f"{latex_label} Q = {data['Q']:.2f}, Z-Rand = {data['zrand']:.2f}")
    
    info_text = "\n".join(text_lines)
    fig.text(0.92, 0.45, info_text, transform=fig.transFigure, fontsize=11,
             verticalalignment='top', horizontalalignment='left', 
             bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.6))
    
    fig.subplots_adjust(right=0.88) # Make space for the legend and text box
    plt.savefig(f"{output_dir}/modularity_gamma_sweep_aggregated.png", dpi=300, bbox_inches='tight')
    plt.close()

# --- MODIFIED: This function is now the primary brain visualization ---
def create_community_brain_visualization(community_data, output_dir):
    print("Creating aggregated 3D brain community visualization...")
    if not community_data: return
    
    plt.rcParams['mathtext.fontset'] = 'cm' # For proper LaTeX rendering
    lh_labels, rh_labels, _, lh_names, rh_names = load_freesurfer_dk_atlas()
    if lh_labels is None: 
        print("  Atlas not found. Skipping brain visualization.")
        return
        
    views = [('lateral', 'lh', 'Left Lateral'), ('medial', 'lh', 'Left Medial'), 
             ('lateral', 'rh', 'Right Lateral'), ('medial', 'rh', 'Right Medial')]
    plot_images = {}
    
    for metric, data in community_data.items():
        print(f"  Rendering brain for: {metric}")
        plot_images[metric] = {}
        assignments = data['assignments']
        num_communities = data['num_communities']
        if num_communities == 0: continue
        
        custom_cmap = generate_dynamic_colormap(num_communities)
        lh_data, rh_data = map_data_to_surface(assignments, lh_labels, rh_labels, lh_names, rh_names)
        
        for view, hemi, view_title in views:
            brain_data = lh_data if hemi == 'lh' else rh_data
            try:
                brain = Brain('fsaverage', hemi, 'pial', background='white', size=(800, 600))
                brain.add_data(brain_data, min=0.5, max=num_communities, colormap=custom_cmap, thresh=0.5, colorbar=False)
                temp_file = f"{output_dir}/temp_{metric}_{hemi}_{view}.png"
                brain.save_image(temp_file)
                plot_images[metric][view_title] = np.array(Image.open(temp_file))
                brain.close()
                os.remove(temp_file)
            except Exception as e:
                print(f"    ERROR rendering {metric} {view_title}: {e}")
                plot_images[metric][view_title] = np.ones((600, 800, 3), dtype=np.uint8) * 128
    
    # Assemble final grid image
    print("  Assembling final grid image...")
    fig, axes = plt.subplots(len(config.metrics), len(views), figsize=(20, 5 * len(config.metrics)), 
                             gridspec_kw={'wspace': 0.01, 'hspace': 0.05})
    if len(config.metrics) == 1: axes = axes.reshape(1, -1)

    for row, metric in enumerate(config.metrics):
        for col, (_, _, view_title) in enumerate(views):
            ax = axes[row, col]
            if metric in plot_images and view_title in plot_images[metric]:
                ax.imshow(plot_images[metric][view_title])
            ax.axis('off')
            if row == 0: ax.set_title(view_title, fontweight='bold', fontsize=16, pad=20)
            if col == 0:
                num_comm = community_data.get(metric, {}).get('num_communities', 0)
                # --- MODIFIED: Use LaTeX name for row label ---
                latex_label = metric_name_map_latex.get(metric, metric)
                ax.text(-0.1, 0.5, f"{latex_label}\n({num_comm} modules)", transform=ax.transAxes, 
                        ha='center', va='center', rotation=90, fontweight='bold', fontsize=14)
    
    # --- MODIFIED: Updated the main title ---
    fig.suptitle('Optimal modularity assignment on Desikan-Killiany Atlas', fontsize=22, fontweight='bold', y=0.98)
    plt.savefig(f"{output_dir}/community_assignments_aggregated.png", dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()

# --- [主执行函数] ---
def main_plotting():
    """
    Main function to call data collection and plotting functions.
    This version is simplified to only produce the two aggregated plots.
    """
    print("="*60)
    print("STARTING SIMPLIFIED PLOTTING SCRIPT")
    print(f"Reading results from: {config.output_path}")
    print("="*60)

    os.makedirs(config.output_path, exist_ok=True)
    
    # 1. Aggregated Gamma Sweep Plot
    gamma_data = collect_group_averaged_gamma_sweep_data()
    optimal_data = collect_group_averaged_optimal_parameters()
    if gamma_data and optimal_data:
        # Pass optimal_data to be displayed on the plot
        create_gamma_sweep_plots(gamma_data, optimal_data, config.output_path)
    else:
        print("\nSkipping Aggregated Gamma Sweep plot: Missing gamma or optimal data.")

    # 2. Aggregated Brain Community Visualization
    community_data = collect_group_averaged_community_assignments()
    if community_data:
        create_community_brain_visualization(community_data, config.output_path)
    else:
        print("\nSkipping Brain Visualization: No community data found.")
        
    print("\n" + "="*60)
    print("PLOTTING SCRIPT FINISHED!")
    print("="*60)

if __name__ == "__main__":
    main_plotting()