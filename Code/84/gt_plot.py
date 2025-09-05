import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.stats import norm

# Define the parameters
aim1_subjects = [f"{i:02d}" for i in range(4, 30) if i != 20]
aim2_subjects = [f"{i:02d}" for i in range(5, 19)]
metrics = ['R1', 'RK_DKI', 'RD_DTI_reciprocal', 'NOS', 'F_SMI', 'FA_DTI', 'DePerp_SMI_reciprocal']

# Base paths
base_path = "/home/localadmin/Desktop/SP2"
output_path = "/home/localadmin/Desktop/SP2/84output/pro_output_v2"

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

# --- NEW: Define the LaTeX-style names for the metrics ---
metric_name_map = {
    'NOS': r'$nos$',
    'RK_DKI': r'$rk$',
    'FA_DTI': r'$fa$',
    'RD_DTI_reciprocal': r'$1/rd$',
    'DePerp_SMI_reciprocal': r'1/$D_{\perp e}$',
    'F_SMI': r'$f$',
    'R1': r'$R1$'
}

def collect_data_by_weighted_metric():
    """
    Collect data organized by weighted metric, then by network metric
    Returns:
        Nested dictionary: {network_metric: {weighted_metric: [values]}}
    """
    data_by_network_metric = {}
    # Process each weighted metric
    for weighted_metric in metrics:
        print(f"Processing weighted metric: {weighted_metric}")
        # Process Aim 1
        for subject in aim1_subjects:
            file_path = f"{base_path}/Aim1_sub-{subject}/connectomes/dual_thresholded_connectomes/analysis_v2/{weighted_metric}/z_scores_all.csv"
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    # For each network metric in this file
                    for _, row in df.iterrows():
                        network_metric = row['metric_name']
                        z_score = row['z_score']
                        # Initialize nested structure if needed
                        if network_metric not in data_by_network_metric:
                            data_by_network_metric[network_metric] = {}
                        if weighted_metric not in data_by_network_metric[network_metric]:
                            data_by_network_metric[network_metric][weighted_metric] = []
                        data_by_network_metric[network_metric][weighted_metric].append(z_score)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        # Process Aim 2
        for subject in aim2_subjects:
            file_path = f"{base_path}/Aim2_sub-{subject}/connectomes/dual_thresholded_connectomes/analysis_v2/{weighted_metric}/z_scores_all.csv"
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    # For each network metric in this file
                    for _, row in df.iterrows():
                        network_metric = row['metric_name']
                        z_score = row['z_score']
                        # Initialize nested structure if needed
                        if network_metric not in data_by_network_metric:
                            data_by_network_metric[network_metric] = {}
                        if weighted_metric not in data_by_network_metric[network_metric]:
                            data_by_network_metric[network_metric][weighted_metric] = []
                        data_by_network_metric[network_metric][weighted_metric].append(z_score)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return data_by_network_metric

def collect_swp_data_by_weighted_metric():
    """
    Collect SWP data organized by weighted metric
    Returns:
        Dictionary: {weighted_metric: [values]}
    """
    swp_data = {}
    # Process each weighted metric
    for weighted_metric in metrics:
        swp_data[weighted_metric] = []
        # Process Aim 1
        for subject in aim1_subjects:
            file_path = f"{base_path}/Aim1_sub-{subject}/connectomes/dual_thresholded_connectomes/analysis_v2/{weighted_metric}/small_world_propensity.csv"
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    if 'SWP' in df.columns:
                        values = df['SWP'].dropna().tolist()
                        swp_data[weighted_metric].extend(values)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        # Process Aim 2
        for subject in aim2_subjects:
            file_path = f"{base_path}/Aim2_sub-{subject}/connectomes/dual_thresholded_connectomes/analysis_v2/{weighted_metric}/small_world_propensity.csv"
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    if 'SWP' in df.columns:
                        values = df['SWP'].dropna().tolist()
                        swp_data[weighted_metric].extend(values)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return swp_data

def create_network_metric_plot(network_metric, weighted_metric_data, output_dir, corrected_z_threshold):
    """
    Create a plot for one network metric showing different weighted metrics as different colors
    Args:
        network_metric: Name of the network metric (e.g., 'characteristic_path_length')
        weighted_metric_data: Dictionary {weighted_metric: [values]}
        output_dir: Directory to save the plot
    """
    if not weighted_metric_data:
        print(f"No data found for {network_metric}")
        return
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # Create legend elements
    from matplotlib.lines import Line2D
    legend_elements = []
    # Plot each weighted metric
    for i, (weighted_metric, values) in enumerate(weighted_metric_data.items()):
        if not values:
            continue
        # Calculate statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        # Get color for this weighted metric
        color = metric_colors.get(weighted_metric, '#000000')
        # Plot with slight x-offset for visibility if multiple metrics
        x_pos = 1 + (i - len(weighted_metric_data)/2) * 0.05
        ax.errorbar([x_pos], [mean_val], yerr=std_val,
                    fmt='o', color=color, markersize=10, linewidth=3,
                    capsize=8, capthick=3, elinewidth=3)
        # --- MODIFIED: Create legend element with new names ---
        legend_label = metric_name_map.get(weighted_metric, weighted_metric)
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=color, markersize=8,
                                    label=legend_label))
    # Customize the plot
    ax.set_xlim(0.5, 1.5)
    
    # --- REMOVED: Y-axis label is now omitted ---
    # ax.set_ylabel(f'{network_metric}', fontsize=12)
    
    # --- MODIFIED: Clean up the title by removing underscores ---
    plot_title = network_metric.replace('_', ' ')
    ax.set_title(plot_title + "-DK", fontsize=14, fontweight='bold')
    
    # Style similar to reference
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    # Remove x-axis labels since it's meaningless
    ax.set_xticks([1])
    ax.set_xticklabels([''])
    # Add significance threshold lines
    ax.axhline(y=1.96, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label=f'p=0.05 (uncorrected)')
    ax.axhline(y=-1.96, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=corrected_z_threshold, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'p=0.05 (Bonferroni)')
    ax.axhline(y=-corrected_z_threshold, color='red', linestyle='--', linewidth=2, alpha=0.8)
    # Add legend in corner with colored dots and weighted metric names
    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
              fancybox=True, shadow=True, fontsize=12) # Increased fontsize for LaTeX
    # Save the plot
    safe_metric_name = network_metric.replace(' ', '_').replace('/', '_')
    output_file = f"{output_dir}/{safe_metric_name}_by_weighted_metrics.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot for {network_metric} to {output_file}")

def create_swp_plot(swp_data, output_dir, corrected_z_threshold):
    """
    Create a plot for SWP showing different weighted metrics as different colors
    Args:
        swp_data: Dictionary {weighted_metric: [values]}
        output_dir: Directory to save the plot
    """
    if not swp_data:
        print("No SWP data found")
        return
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # Create legend elements
    from matplotlib.lines import Line2D
    legend_elements = []
    # Plot each weighted metric
    for i, (weighted_metric, values) in enumerate(swp_data.items()):
        if not values:
            continue
        # Calculate statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        # Get color for this weighted metric
        color = metric_colors.get(weighted_metric, '#000000')
        # Plot with slight x-offset for visibility
        x_pos = 1 + (i - len(swp_data)/2) * 0.05
        ax.errorbar([x_pos], [mean_val], yerr=std_val,
                    fmt='o', color=color, markersize=10, linewidth=3,
                    capsize=8, capthick=3, elinewidth=3)
        # --- MODIFIED: Create legend element with new names ---
        legend_label = metric_name_map.get(weighted_metric, weighted_metric)
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=color, markersize=8,
                                    label=legend_label))
    # Customize the plot
    ax.set_xlim(0.5, 1.5)
    
    # --- REMOVED: Y-axis label is now omitted ---
    # ax.set_ylabel('Small World Propensity', fontsize=12)
    
    ax.set_title('Small World Propensity-DK', fontsize=14, fontweight='bold')
    # Style similar to reference
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    # Remove x-axis labels
    ax.set_xticks([1])
    ax.set_xticklabels([''])

    # Add legend
    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
              fancybox=True, shadow=True, fontsize=12) # Increased fontsize for LaTeX
    # Save the plot
    output_file = f"{output_dir}/Small_World_Propensity_by_weighted_metrics.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SWP plot to {output_file}")

def main():
    # Calculate corrected significance thresholds
    n_network_metrics = 5  # 4 network metrics + 1 SWP
    n_weighted_metrics = 7
    n_comparisons = n_network_metrics * n_weighted_metrics  # 35  
      
    # Bonferroni correction
    corrected_alpha = 0.05 / n_comparisons
    corrected_z_threshold = norm.ppf(1 - corrected_alpha/2)  # 双尾检验    
    
    print(f"Total comparisons: {n_comparisons}")
    print(f"Corrected alpha: {corrected_alpha:.6f}")
    print(f"Corrected z-threshold: ±{corrected_z_threshold:.3f}") 
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    print("Processing z_scores_all.csv files...")
    # Collect data organized by network metric, then by weighted metric
    network_data = collect_data_by_weighted_metric()
    print(f"Found network metrics: {list(network_data.keys())}")
    
    print("\nProcessing small_world_propensity.csv files...")
    # Collect SWP data
    swp_data = collect_swp_data_by_weighted_metric()
    
  
    # Create a plot for each network metric
    for network_metric, weighted_data in network_data.items():
        create_network_metric_plot(network_metric, weighted_data, output_path, corrected_z_threshold)    
    # Create SWP plot
    create_swp_plot(swp_data, output_path, corrected_z_threshold)
    
    print(f"\nAll plots saved to: {output_path}")
    print("\nLegend colors represent weighted metrics:")
    for metric, color in metric_colors.items():
        # Also show the new mapped name
        mapped_name = metric_name_map.get(metric, "")
        print(f"  {metric} ({mapped_name}): {color}")

if __name__ == "__main__":
    main()