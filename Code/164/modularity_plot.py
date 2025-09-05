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

config = AnalysisConfig()

os.environ['FREESURFER_HOME'] = config.freesurfer_home
os.environ['SUBJECTS_DIR'] = config.subjects_dir

metric_colors = {
    'R1': '#1E90FF', 'DePerp_SMI_reciprocal': '#FF8C00', 'FA_DTI': '#228B22',
    'RD_DTI_reciprocal': '#DC143C', 'NOS': '#9932CC', 'RK_DKI': '#FF69B4', 'F_SMI': '#8B4513'
}

if not hasattr(mpl_cm, 'get_cmap'):
    def get_cmap(name):
        try: return plt.get_cmap(name)
        except: return matplotlib.colormaps[name]
    mpl_cm.get_cmap = get_cmap

def get_corrected_a2009s_atlas_mapping():
    """Mapping for Destrieux atlas (164 regions)"""
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
    if n_communities == 0: return ListedColormap(['#000000'])
    if n_communities <= 10:
        base_colors = ['#FF0000', '#0000FF', '#00AA00', '#FF66CC', '#FF8800', '#8800CC', '#996633', '#00CCCC', '#FFFF00', '#888888']
        return ListedColormap(base_colors[:n_communities])
    elif n_communities <= 20:
        return ListedColormap([mcolors.hsv_to_rgb([i / n_communities, 0.8, 0.9]) for i in range(n_communities)])
    else:
        colors = []
        cmaps = ['tab20', 'tab20b', 'tab20c']
        for i in range(n_communities):
            cmap_idx = (i // 20) % len(cmaps)
            color_idx = i % 20
            cmap = plt.get_cmap(cmaps[cmap_idx])
            colors.append(cmap(color_idx / 19.0))
        return ListedColormap(colors)

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
    
    # Create name-to-label mappings from the loaded annotation file names
    lh_name_to_label = {name: i for i, name in enumerate(lh_names)}
    rh_name_to_label = {name: i for i, name in enumerate(rh_names)}
    
    for connectome_idx, data_value in enumerate(data_values):
        fs_node_idx = connectome_idx + 1
        if fs_node_idx not in node_to_region or data_value == 0: continue
        
        # This is the region name from our mapping function (e.g., 'lh.G_and_S_frontomargin')
        target_region_name = node_to_region[fs_node_idx]
        
        # The names from the .annot file might be slightly different (e.g., 'G_and_S_frontomargin')
        simple_name = target_region_name.split('.')[-1]
        
        # Map to the correct hemisphere's surface data
        if target_region_name.startswith('lh.'):
            if simple_name in lh_name_to_label:
                label_val = lh_name_to_label[simple_name]
                lh_data[lh_labels == label_val] = float(data_value)
        else: # rh
            if simple_name in rh_name_to_label:
                label_val = rh_name_to_label[simple_name]
                rh_data[rh_labels == label_val] = float(data_value)
             
    return np.nan_to_num(lh_data), np.nan_to_num(rh_data)

def collect_group_averaged_gamma_sweep_data():
    gamma_sweep_data = {}
    print("Collecting gamma sweep data...")
    for metric in config.metrics:
        file_path = f"{config.output_path}/modularity_gamma_sweep_{metric}.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if 'gamma' in df.columns and 'modularity_mean' in df.columns:
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
                full_assignments = np.zeros(164, dtype=int)
                full_assignments[:len(assignments)] = assignments
                community_data[metric] = {
                    'assignments': full_assignments,
                    'num_communities': len(np.unique(assignments[assignments > 0]))
                }
            except Exception as e:
                print(f"  Could not read {file_path}: {e}")
    return community_data

def create_gamma_sweep_plots(gamma_sweep_data, output_dir):
    print("Creating gamma sweep plots (combined)...")
    if not gamma_sweep_data: return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    for metric, df in gamma_sweep_data.items():
        color = metric_colors.get(metric, '#888888')
        ax1.plot(df['gamma'], df['modularity_mean'], color=color, linewidth=2, marker='o', markersize=4, label=metric)
        ax2.plot(df['gamma'], df['zrand_mean'], color=color, linewidth=2, marker='s', markersize=4, label=metric)
    ax1.set(xlabel='Gamma', ylabel='Modularity Q', title='Modularity Q vs Gamma')
    ax2.set(xlabel='Gamma', ylabel='Z-Rand Score', title='Z-Rand Score vs Gamma')
    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/modularity_gamma_sweep_plots_group_averaged.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_individual_gamma_sweep_plots(gamma_sweep_data, optimal_data, output_dir):
    print("Creating individual gamma sweep plots...")
    for metric, df in gamma_sweep_data.items():
        color = metric_colors.get(metric, '#888888')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.plot(df['gamma'], df['modularity_mean'], color=color, linewidth=3, marker='o')
        ax1.set(xlabel='Gamma', ylabel='Modularity Q', title=f'Modularity Q vs Gamma\n{metric}')
        ax2.plot(df['gamma'], df['zrand_mean'], color=color, linewidth=3, marker='s')
        ax2.set(xlabel='Gamma', ylabel='Z-Rand Score', title=f'Z-Rand Score vs Gamma\n{metric}')
        if metric in optimal_data:
            best_gamma = optimal_data[metric]['gamma']
            ax1.axvline(x=best_gamma, color='red', linestyle='--', alpha=0.7)
            ax2.axvline(x=best_gamma, color='red', linestyle='--', alpha=0.7)
        fig.tight_layout()
        plt.savefig(f"{output_dir}/gamma_sweep_{metric}_individual.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_optimal_parameters_comparison_plot(optimal_data, output_dir):
    print("Creating optimal parameters comparison bar plots...")
    if not optimal_data: return
    metrics_list = list(optimal_data.keys())
    colors = [metric_colors.get(metric, '#888888') for metric in metrics_list]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    params = ['gamma', 'modules', 'zrand', 'Q']
    titles = ['Optimal Gamma', 'Number of Modules', 'Z-Rand', 'Modularity Q']
    for ax, param, title in zip(axes.flatten(), params, titles):
        values = [optimal_data[m][param] for m in metrics_list]
        ax.bar(metrics_list, values, color=colors)
        ax.set_title(title, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/optimal_parameters_comparison_group_averaged_164_regions.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_community_brain_visualization(community_data, output_dir):
    print("Creating 3D brain community visualizations...")
    if not community_data: return
    lh_labels, rh_labels, fsaverage, lh_names, rh_names = load_freesurfer_destrieux_atlas()
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
    
    print("  Assembling final grid image...")
    fig, axes = plt.subplots(len(config.metrics), len(views), figsize=(20, 6 * len(config.metrics)), 
                             gridspec_kw={'wspace': 0.01, 'hspace': 0.1})
    if len(config.metrics) == 1: axes = axes.reshape(1, -1)
    
    for row, metric in enumerate(config.metrics):
        for col, (_, _, view_title) in enumerate(views):
            ax = axes[row, col]
            if metric in plot_images and view_title in plot_images[metric]:
                ax.imshow(plot_images[metric][view_title])
            ax.axis('off')
            if row == 0: ax.set_title(view_title, fontweight='bold', pad=15)
            if col == 0:
                num_comm = community_data.get(metric, {}).get('num_communities', 0)
                ax.text(-0.1, 0.5, f"{metric}\n({num_comm} communities)", transform=ax.transAxes, 
                        ha='center', va='center', rotation=90, fontweight='bold')
    
    fig.suptitle('Community Assignments: Group Averaged Connectomes (164-Region Atlas)', fontsize=16, fontweight='bold', y=0.95)
    plt.savefig(f"{output_dir}/community_assignments_group_averaged_164_regions.png", dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()

def main_plotting():
    """Main function to run the plotting script."""
    print("="*60)
    print("STARTING PLOTTING-ONLY SCRIPT (164 REGIONS)")
    print(f"Reading results from: {config.output_path}")
    print("="*60)

    os.makedirs(config.output_path, exist_ok=True)
    
    gamma_data = collect_group_averaged_gamma_sweep_data()
    optimal_data_for_gamma = collect_group_averaged_optimal_parameters()
    if gamma_data:
        create_gamma_sweep_plots(gamma_data, config.output_path)
        create_individual_gamma_sweep_plots(gamma_data, optimal_data_for_gamma, config.output_path)
    else:
        print("\nSkipping Gamma Sweep plots: No data found.")

    optimal_data = collect_group_averaged_optimal_parameters()
    if optimal_data:
        create_optimal_parameters_comparison_plot(optimal_data, config.output_path)
    else:
        print("\nSkipping Optimal Parameters plot: No data found.")

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