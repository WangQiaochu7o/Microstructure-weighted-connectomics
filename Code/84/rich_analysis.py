import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import bct # Brain Connectivity Toolbox
import glob # 用于文件查找

# --- 1. Configuration Module ---

class AnalysisConfig:
    """Configuration for the analysis, supporting different parcellation schemes."""
    def __init__(self, parcellation_name):
        self.parcellation_name = parcellation_name
        
        self.base_path = "/home/localadmin/Desktop/SP2"
        self.group_connectomes_dir = f"/home/localadmin/Desktop/SP2/{self.parcellation_name}output/pro_group_average" 
        self.output_path = f"/home/localadmin/Desktop/SP2/{self.parcellation_name}output/group_averaged_output/rich_club_analysis"
        
        self.null_models_dir = os.path.join(self.output_path, "null_models")
        
        self.metrics = ['R1', 'RK_DKI', 'RD_DTI_reciprocal', 'F_SMI', 'FA_DTI', 'DePerp_SMI_reciprocal']
        
        self.num_null_models = 100

# Colors for different weighted metrics
metric_colors = {
    'R1': '#1E90FF',           # Blue
    'DePerp_SMI_reciprocal': '#FF8C00',       # Orange  
    'FA_DTI': '#228B22',       # Green
    'RD_DTI_reciprocal': '#DC143C',          # Red
    # 'NOS': '#9932CC',        # Purple
    'RK_DKI': '#FF69B4',       # Pink
    'F_SMI': '#8B4513'         # Brown
}

# --- MODIFIED: LaTeX-style names for plot legends ---
metric_name_map_latex = {
    'NOS': r'$nos$',
    'RK_DKI': r'$rk$',
    'FA_DTI': r'$fa$',
    'RD_DTI_reciprocal': r'$1/rd$',
    'DePerp_SMI_reciprocal': r'$1/D_{\perp e}$',
    'F_SMI': r'$f$',
    'R1': r'$R1$'
}

# --- 2. Core Computation Module ---

def ensure_symmetric_matrix(W):
    """Ensures matrix is symmetric and removes the diagonal."""
    W_sym = (W + W.T) / 2
    np.fill_diagonal(W_sym, 0)
    return W_sym

def generate_and_save_null_models(W, metric, config):
    """Generates and saves null models for a single connectome."""
    print(f"    Generating and saving {config.num_null_models} null models for {metric}...")
    metric_null_dir = os.path.join(config.null_models_dir, metric)
    os.makedirs(metric_null_dir, exist_ok=True)
    
    null_models = []
    for i in range(config.num_null_models):
        null_matrix = bct.randmio_und(W, 10)[0] 
        null_models.append(null_matrix)
        
        null_path = os.path.join(metric_null_dir, f"null_model_{i+1:03d}.csv")
        pd.DataFrame(null_matrix).to_csv(null_path, header=False, index=False)
        
    print(f"    ✓ Null models saved to: {metric_null_dir}")
    return null_models

def load_null_models(metric, config):
    """Loads pre-generated null models from disk for a specific metric."""
    metric_null_dir = os.path.join(config.null_models_dir, metric)
    if not os.path.isdir(metric_null_dir):
        return None

    null_files = sorted(glob.glob(os.path.join(metric_null_dir, "null_model_*.csv")))
    if len(null_files) < config.num_null_models:
        print(f"    ! Warning: Found {len(null_files)} null models, fewer than expected ({config.num_null_models}). Will regenerate.")
        return None

    print(f"    Loading {len(null_files)} pre-generated null models for {metric}...")
    null_models = [pd.read_csv(f, header=None).values for f in null_files]
    
    return null_models

def compute_rich_club_with_nulls(W, null_models):
    """Computes rich-club coefficients against a list of null models."""
    W_sym = ensure_symmetric_matrix(W)
    
    original_rc = bct.rich_club_wu(W_sym)
    klevel = len(original_rc)
    
    null_rc_list = [bct.rich_club_wu(null_matrix, klevel=klevel) for null_matrix in null_models]

    if not null_rc_list:
        raise ValueError("Null model list is empty or invalid.")
    
    null_rich_clubs_array = np.array(null_rc_list)
    avg_null_rc = np.mean(null_rich_clubs_array, axis=0)
    std_null_rc = np.std(null_rich_clubs_array, axis=0)
    
    normalized_rc = np.divide(original_rc, avg_null_rc, 
                              out=np.zeros_like(original_rc, dtype=float), 
                              where=(avg_null_rc != 0))
    
    return {
        'k_levels': np.arange(1, len(original_rc) + 1),
        'original_rich_club': original_rc,
        'null_average_rich_club': avg_null_rc,
        'normalized_rich_club': normalized_rc,
        'null_std': std_null_rc
    }

def process_group_connectome_rc(connectome_path, config, metric):
    """Processes rich-club analysis for a single group-averaged connectome."""
    try:
        print(f"\nProcessing rich-club for group-averaged metric: {metric}...")
        W = pd.read_csv(connectome_path, header=None, index_col=None).values
        print(f"  Loaded group-averaged connectome: {W.shape}")
        
        null_models = load_null_models(metric, config)
        if null_models is None:
            W_sym = ensure_symmetric_matrix(W)
            null_models = generate_and_save_null_models(W_sym, metric, config)

        print("  Computing rich-club coefficients...")
        rc_results = compute_rich_club_with_nulls(W, null_models)
        
        rc_df = pd.DataFrame(rc_results).round(4)
        output_file = os.path.join(config.output_path, f"group_rich_club_{metric}.csv")
        rc_df.to_csv(output_file, index=False)
        print(f"  ✓ Rich-club data saved to: {output_file}")
        
        return rc_results
        
    except Exception as e:
        print(f"  ✗ Error processing {metric}: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- 3. Visualization Module ---

def create_rich_club_plot(all_metrics_data, output_dir, parcellation_name):
    """Creates a single rich-club plot for all weighted metrics."""
    if not all_metrics_data:
        print("No rich-club data available to plot.")
        return
    
    # --- MODIFIED: Configure matplotlib for proper LaTeX rendering ---
    plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern for math text
    plt.rcParams['axes.unicode_minus'] = False # Ensure minus signs render correctly

    fig, ax = plt.subplots(figsize=(12, 8))
    
    for metric, rc_data in all_metrics_data.items():
        color = metric_colors.get(metric, '#000000')
        k_levels = rc_data['k_levels']
        normalized_coeffs = rc_data['normalized_rich_club']
        # --- MODIFIED: Use the LaTeX map for the legend label ---
        label = metric_name_map_latex.get(metric, metric)
        ax.plot(k_levels, normalized_coeffs, 'o-', color=color, linewidth=2.5, 
                markersize=6, label=label, alpha=0.9)
        
    # --- MODIFIED: All labels are now in English ---
    ax.set_xlabel('Degree (k)', fontsize=14)
    ax.set_ylabel(r'Normalized Rich-Club Coefficient ($\Phi_{norm}$)', fontsize=14)
    ax.set_title(f'Group-Averaged Rich-Club Organization ({parcellation_name} Nodes)', fontsize=16, fontweight='bold')
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Random Network Baseline')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12) # Fontsize adjusted for LaTeX
    
    ax.set_xlim(left=0)
    ax.tick_params(labelsize=12)
    
    output_file = os.path.join(output_dir, f"group_rich_club_plot_{parcellation_name}_nodes.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Final rich-club plot saved to {output_file}")

# --- 4. Main Execution Function ---

def main():
    """Main function to run the group-averaged rich-club analysis."""
    
    # --- Set the parcellation scheme to run here ---
    PARCELLATION_TO_RUN = '84'  # Options: '84', '164', etc.
    # ----------------------------------------------------
    
    config = AnalysisConfig(PARCELLATION_TO_RUN)
    
    print("="*80)
    print(f"Group-Averaged Rich-Club Analysis ({config.parcellation_name} Nodes)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Reading connectomes from: {config.group_connectomes_dir}")
    print(f"  Saving all output to: {config.output_path}")
    print(f"  Null models directory: {config.null_models_dir}")
    
    os.makedirs(config.output_path, exist_ok=True)
    os.makedirs(config.null_models_dir, exist_ok=True)
    
    all_results = {}
    
    for metric in config.metrics:
        connectome_path = os.path.join(
            config.group_connectomes_dir,
            f"group_{metric}_median.csv"
        )
        
        if os.path.exists(connectome_path):
            result = process_group_connectome_rc(connectome_path, config, metric)
            if result:
                all_results[metric] = result
        else:
            print(f"✗ Connectome file not found, skipping: {connectome_path}")
            
    if all_results:
        print("\nCreating final summary plot for all metrics...")
        create_rich_club_plot(all_results, config.output_path, config.parcellation_name)
    else:
        print("\nNo valid data was processed, skipping plotting.")

    print("\n🎉 Rich-club analysis complete!")

if __name__ == "__main__":
    main()