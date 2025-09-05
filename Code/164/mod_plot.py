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
        # Data paths for 164-region atlas
        self.base_path = "/home/localadmin/Desktop/SP2"
        # MODIFICATION: Point to the new structured output directory
        self.modularity_results_path = "/home/localadmin/Desktop/SP2/164output/group_averaged_output/mod"
        self.plotting_output_path = self.modularity_results_path # Save plots alongside data
        
        # Freesurfer settings
        self.freesurfer_home = '/home/localadmin/freesurfer'
        self.subjects_dir = os.path.join(self.freesurfer_home, 'subjects')
        
        # MODIFICATION: Add 'BINARY_NOS' for integrated plotting
        self.metrics = ['R1', 'RK_DKI', 'RD_DTI_reciprocal', 'NOS', 'F_SMI', 'FA_DTI', 'DePerp_SMI_reciprocal', 'BINARY_NOS']

        # --- MODIFICATION: New parameter to select a specific gamma for brain visualization ---
        self.gamma_for_plotting = 0.9 # <-- CHANGE THIS VALUE TO PLOT A DIFFERENT GAMMA

config = AnalysisConfig()

# --- [脚本设置] ---
os.environ['FREESURFER_HOME'] = config.freesurfer_home
os.environ['SUBJECTS_DIR'] = config.subjects_dir

metric_colors = {
    'R1': '#1E90FF', 'DePerp_SMI_reciprocal': '#FF8C00', 'FA_DTI': '#228B22',
    'RD_DTI_reciprocal': '#DC143C', 'NOS': '#9932CC', 'RK_DKI': '#FF69B4', 'F_SMI': '#8B4513',
    'BINARY_NOS': '#000000' # Black for binary
}

metric_name_map_latex = {
    'NOS': r'$nos$', 'RK_DKI': r'$rk$', 'FA_DTI': r'$fa$',
    'RD_DTI_reciprocal': r'$1/rd$', 'DePerp_SMI_reciprocal': r'$1/D_{\perp e}$',
    'F_SMI': r'$f$', 'R1': r'$R1$',
    'BINARY_NOS': r'$binary$'
}

if not hasattr(mpl_cm, 'get_cmap'):
    def get_cmap(name):
        try: return plt.get_cmap(name)
        except: return plt.colormaps[name]
    mpl_cm.get_cmap = get_cmap

# --- [辅助函数 for 164-region Destrieux Atlas] ---
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
        'S_interm_prim-Jensen', 'S_intrapariet_and_P_trans',
        'S_oc_middle_and_Lunatus', 'S_oc_sup_and_transversal', 'S_occipital_ant',
        'S_oc-temp_lat', 'S_oc-temp_med_and_Lingual', 'S_orbital_lateral',
        'S_orbital_med-olfact', 'S_orbital-H_Shaped', 'S_parieto_occipital',
        'S_pericallosal', 'S_postcentral', 'S_precentral-inf-part',
        'S_precentral-sup-part', 'S_suborbital', 'S_subparietal',
        'S_temporal_inf', 'S_temporal_sup', 'S_temporal_transverse'
    ]
    node_to_region = {}
    for i, region in enumerate(cortical_regions): node_to_region[i + 1] = f"lh.{region}"
    for i, region in enumerate(cortical_regions): node_to_region[90 + i] = f"rh.{region}"
    return node_to_region

def generate_dynamic_colormap(n_communities):
    if n_communities == 0: return ListedColormap(['#FFFFFF'])
    if n_communities <= 10:
        base_colors = ['#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231', '#911EB4', '#46F0F0', '#F032E6', '#BCF60C', '#FABEBE']
        return ListedColormap(base_colors[:n_communities])
    elif n_communities <= 20:
        return plt.get_cmap('tab20', n_communities)
    else:
        return plt.get_cmap('gist_ncar', n_communities)

def load_freesurfer_destrieux_atlas():
    try:
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
        lh_annot_path = os.path.join(config.subjects_dir, 'fsaverage', 'label', 'lh.aparc.a2009s.annot')
        rh_annot_path = os.path.join(config.subjects_dir, 'fsaverage', 'label', 'rh.aparc.a2009s.annot')
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
    node_to_region = get_corrected_a2009s_atlas_mapping()
    lh_name_to_label = {name: i for i, name in enumerate(lh_names)}
    rh_name_to_label = {name: i for i, name in enumerate(rh_names)}
    
    for connectome_idx, data_value in enumerate(data_values):
        fs_node_idx = connectome_idx + 1
        if fs_node_idx in node_to_region and data_value != 0:
            target_region_name = node_to_region[fs_node_idx]
            simple_name = target_region_name.split('.')[-1]
            
            label_map, surface_data, labels = (lh_name_to_label, lh_data, lh_labels) if target_region_name.startswith('lh.') else (rh_name_to_label, rh_data, rh_labels)
            
            if simple_name in label_map:
                label_val = label_map[simple_name]
                surface_data[labels == label_val] = float(data_value)
             
    return np.nan_to_num(lh_data), np.nan_to_num(rh_data)

# --- [数据收集函数] ---
def collect_gamma_sweep_data():
    gamma_sweep_data = {}
    print("Collecting gamma sweep data...")
    for metric in config.metrics:
        file_path = os.path.join(config.modularity_results_path, metric, 'best', f'modularity_gamma_sweep_{metric}.csv')
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                gamma_sweep_data[metric] = df.dropna().copy()
            except Exception as e: print(f"  Could not read {file_path}: {e}")
        else: print(f"  Sweep file not found for {metric}: {file_path}")
    return gamma_sweep_data

def collect_optimal_parameters():
    optimal_data = {}
    print("Collecting optimal parameters for text box...")
    for metric in config.metrics:
        file_path = os.path.join(config.modularity_results_path, metric, 'best', f'modularity_best_gamma_summary_{metric}.csv')
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path).iloc[0]
                optimal_data[metric] = {
                    'gamma': df['best_gamma'], 'modules': int(df['best_num_communities']),
                    'zrand': df['best_zrand_mean'], 'Q': df['best_modularity_mean']
                }
            except Exception as e: print(f"  Could not read {file_path}: {e}")
        else: print(f"  Optimal summary file not found for {metric}: {file_path}")
    return optimal_data

def collect_community_assignments_for_gamma(target_gamma):
    community_data = {}
    print(f"Collecting community assignments for gamma = {target_gamma:.1f}...")
    gamma_str = f"{target_gamma:.1f}"

    for metric in config.metrics:
        file_path = os.path.join(config.modularity_results_path, metric, f'gamma_{gamma_str}', 'community_assignments.csv')
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                assignments = df['community_assignment'].values
                
                # Ensure array is the correct size for the 164-region atlas
                full_assignments = np.zeros(164, dtype=int)
                num_nodes_in_file = len(assignments)
                if num_nodes_in_file > 164:
                    print(f"  Warning for {metric}: File has {num_nodes_in_file} nodes, trimming to 164.")
                    full_assignments = assignments[:164]
                else:
                    full_assignments[:num_nodes_in_file] = assignments

                community_data[metric] = {
                    'assignments': full_assignments,
                    'num_communities': len(np.unique(assignments[assignments > 0]))
                }
            except Exception as e:
                print(f"  Could not read or process assignments for {metric}: {e}")
        else:
            print(f"  Assignments file not found for {metric} at gamma={gamma_str}: {file_path}")
            
    return community_data

# --- [核心绘图函数] ---
def create_gamma_sweep_plots(gamma_sweep_data, optimal_data, output_dir):
    print("Creating aggregated gamma sweep plot...")
    if not gamma_sweep_data: return
    
    plt.rcParams['mathtext.fontset'] = 'cm'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    for metric, df in gamma_sweep_data.items():
        color = metric_colors.get(metric, '#888888')
        label = metric_name_map_latex.get(metric, metric)
        ax1.plot(df['gamma'], df['modularity_mean'], color=color, linewidth=2, marker='o', markersize=4, label=label)
        ax2.plot(df['gamma'], df['zrand_mean'], color=color, linewidth=2, marker='s', markersize=4, label=label)
    
    ax1.set(ylabel='Modularity Q', title='Modularity Q vs Gamma (Destrieux Atlas)')
    ax2.set(xlabel='Gamma', ylabel='Z-Rand Score', title='Z-Rand Score vs Gamma (Destrieux Atlas)')
    for ax in [ax1, ax2]: ax.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    text_lines = ["Best Values (by Z-Rand):"]
    for metric in config.metrics:
        if metric in optimal_data:
            data = optimal_data[metric]
            latex_label = metric_name_map_latex.get(metric, metric).ljust(18)
            text_lines.append(f"{latex_label} γ={data['gamma']:.1f}, Q={data['Q']:.2f}, Z={data['zrand']:.2f}")
    
    info_text = "\n".join(text_lines)
    fig.text(0.92, 0.5, info_text, transform=fig.transFigure, fontsize=11,
             verticalalignment='center', horizontalalignment='left', 
             bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.7))
    
    fig.subplots_adjust(right=0.85)
    plt.savefig(os.path.join(output_dir, "modularity_gamma_sweep_aggregated_164.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Sweep plot saved.")

def create_community_brain_visualization(community_data, output_dir, target_gamma):
    print(f"Creating aggregated 3D brain visualization for gamma = {target_gamma:.1f}...")
    if not community_data: return
    
    plt.rcParams['mathtext.fontset'] = 'cm'
    lh_labels, rh_labels, _, lh_names, rh_names = load_freesurfer_destrieux_atlas()
    if lh_labels is None: 
        print("  Atlas not found. Skipping brain visualization.")
        return
        
    views = [('lateral', 'lh', 'Left Lateral'), ('medial', 'lh', 'Left Medial'), 
             ('lateral', 'rh', 'Right Lateral'), ('medial', 'rh', 'Right Medial')]
    plot_images = {}
    
    for metric, data in community_data.items():
        print(f"  Rendering brain for: {metric}")
        plot_images[metric] = {}
        assignments, num_communities = data['assignments'], data['num_communities']
        if num_communities == 0: continue
        
        custom_cmap = generate_dynamic_colormap(num_communities)
        lh_data, rh_data = map_data_to_surface(assignments, lh_labels, rh_labels, lh_names, rh_names)
        
        for view, hemi, view_title in views:
            brain_data = lh_data if hemi == 'lh' else rh_data
            try:
                brain = Brain('fsaverage', hemi, 'pial', background='white', size=(800, 600), cortex='low_contrast')
                brain.add_data(brain_data, min=0.5, max=np.max(assignments), colormap=custom_cmap, thresh=0.5, colorbar=False)
                temp_file = os.path.join(output_dir, f"temp_{metric}_{hemi}_{view}.png")
                brain.save_image(temp_file)
                plot_images[metric][view_title] = np.array(Image.open(temp_file))
                brain.close(); os.remove(temp_file)
            except Exception as e:
                print(f"    ERROR rendering {metric} {view_title}: {e}")
                plot_images[metric][view_title] = np.ones((600, 800, 3), dtype=np.uint8) * 128
    
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
                latex_label = metric_name_map_latex.get(metric, metric)
                ax.text(-0.1, 0.5, f"{latex_label}\n({num_comm} modules)", transform=ax.transAxes, 
                        ha='center', va='center', rotation=90, fontweight='bold', fontsize=14)
    
    # fig.suptitle(f'Modularity Assignment (γ={target_gamma:.1f}) on Destrieux Atlas', fontsize=22, fontweight='bold', y=0.98)
    
    output_filename = os.path.join(output_dir, f"community_assignments_aggregated_gamma_{target_gamma:.1f}_164.png")
    plt.savefig(output_filename, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Brain visualization saved to: {output_filename}")

# --- [主执行函数] ---
def main_plotting():
    print("="*60)
    print("STARTING FLEXIBLE PLOTTING SCRIPT (164 REGIONS)")
    print(f"Reading results from: {config.modularity_results_path}")
    print(f"Saving plots to: {config.plotting_output_path}")
    print("="*60)

    os.makedirs(config.plotting_output_path, exist_ok=True)
    
    # 1. Aggregated Gamma Sweep Plot
    gamma_data = collect_gamma_sweep_data()
    optimal_data = collect_optimal_parameters()
    if gamma_data:
        create_gamma_sweep_plots(gamma_data, optimal_data, config.plotting_output_path)
    else:
        print("\nSkipping Aggregated Gamma Sweep plot: Missing sweep data.")

    # 2. Aggregated Brain Community Visualization for the SPECIFIED gamma
    target_gamma_for_plot = config.gamma_for_plotting
    community_data = collect_community_assignments_for_gamma(target_gamma_for_plot)
    if community_data:
        create_community_brain_visualization(community_data, config.plotting_output_path, target_gamma_for_plot)
    else:
        print(f"\nSkipping Brain Visualization: No community data found for gamma = {target_gamma_for_plot:.1f}.")
        
    print("\n" + "="*60)
    print("PLOTTING SCRIPT FINISHED!")
    print("="*60)

if __name__ == "__main__":
    main_plotting()