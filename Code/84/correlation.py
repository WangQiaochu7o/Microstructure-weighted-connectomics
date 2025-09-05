import numpy as np
import pandas as pd
from scipy.stats import pearsonr, linregress
import os
from itertools import combinations

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns    # --- 新增配置：启用 LaTeX 渲染 ---
    # 前提：你的系统必须安装了 LaTeX (如 MiKTeX, TeX Live)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # ------------------------------------    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    PLOTTING_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    print("Warning: matplotlib/seaborn not available or LaTeX not found. Plotting will be disabled.")
    print(f"Error details: {e}")
    PLOTTING_AVAILABLE = False

def load_group_connectome(filepath):
    """Load group connectome from CSV file"""
    try:
        connectome = pd.read_csv(filepath, index_col=0).values
        return connectome.astype(float)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def correlate_connectomes_safe(connectome1, connectome2, return_data=False):
    """Safely correlate two connectomes maintaining correspondence"""
    
    if connectome1 is None or connectome2 is None:
        if return_data:
            return np.nan, np.nan, 0, None, None
        return np.nan
    
    if connectome1.shape != connectome2.shape:
        print(f"Shape mismatch: {connectome1.shape} vs {connectome2.shape}")
        if return_data:
            return np.nan, np.nan, 0, None, None
        return np.nan
    
    # Extract upper triangle edges (excluding diagonal)
    mask = np.triu(np.ones_like(connectome1, dtype=bool), k=1)
    edges1 = connectome1[mask].astype(float)
    edges2 = connectome2[mask].astype(float)
    
    # First remove NaN pairs (maintaining correspondence)
    nan_mask = np.isnan(edges1) | np.isnan(edges2)
    edges1_nonan = edges1[~nan_mask]
    edges2_nonan = edges2[~nan_mask]
    
    # Remove infinite values
    inf_mask = np.isinf(edges1_nonan) | np.isinf(edges2_nonan)
    edges1_noinf = edges1_nonan[~inf_mask]
    edges2_noinf = edges2_nonan[~inf_mask]
    
    # Remove pairs where either value is zero (edges that didn't pass thresholds)
    zero_mask = (edges1_noinf == 0) | (edges2_noinf == 0)
    edges1_final = edges1_noinf[~zero_mask]
    edges2_final = edges2_noinf[~zero_mask]
    
    if len(edges1_final) < 3:
        if return_data:
            return np.nan, np.nan, 0, None, None
        return np.nan, np.nan, 0
    
    try:
        r, p = pearsonr(edges1_final, edges2_final)
        if return_data:
            return r, p, len(edges1_final), edges1_final, edges2_final
        return r, p, len(edges1_final)
    except Exception as e:
        print(f"Correlation error: {e}")
        if return_data:
            return np.nan, np.nan, 0, None, None
        return np.nan, np.nan, 0

def analyze_group_collinearity(group_results_path):
    """Analyze collinearity in group-level connectomes
    
    Returns:
        tuple: (correlations, group_connectomes) for further analysis and plotting
    """
    
    # Define metrics and their group result files
    group_metrics = {
        'NOS': 'group_NOS_median.csv',
        'RK_DKI': 'group_RK_DKI_median.csv',
        'FA_DTI': 'group_FA_DTI_median.csv',
        'RD_DTI_reciprocal': 'group_RD_DTI_reciprocal_median.csv',
        'DePerp_SMI_reciprocal': 'group_DePerp_SMI_reciprocal_median.csv',
        'F_SMI': 'group_F_SMI_median.csv',
        'R1': 'group_R1_median.csv'
    }
    
    # Load all group connectomes
    print("Loading group connectomes...")
    group_connectomes = {}
    available_metrics = []
    
    for metric_name, filename in group_metrics.items():
        filepath = os.path.join(group_results_path, filename)
        connectome = load_group_connectome(filepath)
        
        if connectome is not None:
            group_connectomes[metric_name] = connectome
            available_metrics.append(metric_name)
            print(f"✓ Loaded {metric_name}")
        else:
            print(f"✗ Failed to load {metric_name}")
    
    if len(available_metrics) < 2:
        print("Error: Need at least 2 metrics for correlation analysis!")
        return None
    
    print(f"\nAnalyzing correlations between {len(available_metrics)} metrics...")
    
    # Calculate all pairwise correlations
    correlations = []
    metric_names = available_metrics
    
    for metric1, metric2 in combinations(metric_names, 2):
        result = correlate_connectomes_safe(group_connectomes[metric1], 
                                          group_connectomes[metric2])
        
        if len(result) == 3:
            r, p, n_edges = result
            correlations.append({
                'metric1': metric1,
                'metric2': metric2,
                'r': r,
                'p': p,
                'n_edges': n_edges,
                'abs_r': abs(r) if not np.isnan(r) else 0
            })
        else:
            r = result
            correlations.append({
                'metric1': metric1,
                'metric2': metric2,
                'r': r,
                'p': np.nan,
                'n_edges': 0,
                'abs_r': abs(r) if not np.isnan(r) else 0
            })
    
    return correlations, group_connectomes

def print_correlation_results(correlations, threshold=0.2):
    """Print formatted correlation results"""
    
    print("\n" + "=" * 90)
    print("GROUP-LEVEL CONNECTOME CORRELATION ANALYSIS")
    print("=" * 90)
    
    # Filter valid correlations
    valid_correlations = [c for c in correlations if not np.isnan(c['r'])]
    
    if not valid_correlations:
        print("No valid correlations found!")
        return
    
    # Sort by absolute correlation (highest first)
    valid_correlations.sort(key=lambda x: x['abs_r'], reverse=True)
    
    print(f"{'Rank':<4} {'Metric 1':<20} {'Metric 2':<20} {'r':<8} {'|r|':<8} {'p-value':<10} {'Edges':<6} {'Assessment'}")
    print("-" * 90)
    
    problematic_count = 0
    
    for i, corr in enumerate(valid_correlations, 1):
        # Determine assessment
        abs_r = corr['abs_r']
        if abs_r > 0.9:
            assessment = "🚨 SEVERE"
            problematic = True
        elif abs_r > 0.7:
            assessment = "⚠️  HIGH"
            problematic = True
        elif abs_r > 0.5:
            assessment = "⚠️  MODERATE"
            problematic = True
        elif abs_r > threshold:
            assessment = "ℹ️  MILD"
            problematic = True
        else:
            assessment = "✓ ACCEPTABLE"
            problematic = False
        
        if problematic:
            problematic_count += 1
        
        # Format p-value
        p_str = f"{corr['p']:.2e}" if not np.isnan(corr['p']) else "N/A"
        
        print(f"{i:<4} {corr['metric1']:<20} {corr['metric2']:<20} "
              f"{corr['r']:+.3f}  {abs_r:.3f}   {p_str:<10} {corr['n_edges']:<6} {assessment}")
    
    # Summary statistics
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    
    total_pairs = len(valid_correlations)
    severe = sum(1 for c in valid_correlations if c['abs_r'] > 0.9)
    high = sum(1 for c in valid_correlations if 0.7 < c['abs_r'] <= 0.9)
    moderate = sum(1 for c in valid_correlations if 0.5 < c['abs_r'] <= 0.7)
    mild = sum(1 for c in valid_correlations if threshold < c['abs_r'] <= 0.5)
    acceptable = sum(1 for c in valid_correlations if c['abs_r'] <= threshold)
    
    print(f"Total metric pairs analyzed: {total_pairs}")
    print(f"Problematic pairs (|r| > {threshold}): {problematic_count}")
    print()
    print("Severity breakdown:")
    print(f"  🚨 Severe (|r| > 0.9):     {severe}")
    print(f"  ⚠️  High (|r| 0.7-0.9):      {high}")
    print(f"  ⚠️  Moderate (|r| 0.5-0.7):  {moderate}")
    print(f"  ℹ️  Mild (|r| {threshold}-0.5):        {mild}")
    print(f"  ✓ Acceptable (|r| ≤ {threshold}):   {acceptable}")
    
    # Recommendations
    print("\n" + "=" * 90)
    print("RECOMMENDATIONS")
    print("=" * 90)
    
    severe_pairs = [c for c in valid_correlations if c['abs_r'] > 0.9]
    high_pairs = [c for c in valid_correlations if 0.7 < c['abs_r'] <= 0.9]
    
    if severe_pairs:
        print("🚨 SEVERE COLLINEARITY detected:")
        for corr in severe_pairs:
            print(f"   • {corr['metric1']} vs {corr['metric2']} (r = {corr['r']:+.3f})")
        print("   → Use only ONE metric from each severely correlated pair")
        print()
    
    if high_pairs:
        print("⚠️  HIGH COLLINEARITY detected:")
        for corr in high_pairs:
            print(f"   • {corr['metric1']} vs {corr['metric2']} (r = {corr['r']:+.3f})")
        print("   → Consider using only one metric or combine via PCA")
        print()
    
    # Suggest independent metrics
    all_metrics = set()
    correlated_metrics = set()
    
    for corr in valid_correlations:
        all_metrics.add(corr['metric1'])
        all_metrics.add(corr['metric2'])
        if corr['abs_r'] > 0.7:  # High correlation threshold
            correlated_metrics.add(corr['metric1'])
            correlated_metrics.add(corr['metric2'])
    
    if correlated_metrics:
        independent_metrics = all_metrics - correlated_metrics
        if independent_metrics:
            print("✓ INDEPENDENT metrics (safe to use together):")
            for metric in sorted(independent_metrics):
                print(f"   • {metric}")
        else:
            print("⚠️  All metrics show some level of high collinearity")
            print("   → Consider PCA or choose representative metrics")

def save_correlation_results(correlations, output_path):
    """Save correlation results to CSV"""
    
    # Convert to DataFrame
    df = pd.DataFrame(correlations)
    
    # Save detailed results
    detailed_file = os.path.join(output_path, "group_correlation_analysis.csv")
    df.to_csv(detailed_file, index=False)
    print(f"\n💾 Detailed results saved to: {detailed_file}")
    
    # Save summary (problematic pairs only)
    problematic = df[df['abs_r'] > 0.2].copy()
    if not problematic.empty:
        problematic = problematic.sort_values('abs_r', ascending=False)
        summary_file = os.path.join(output_path, "group_problematic_correlations.csv")
        problematic.to_csv(summary_file, index=False)
        print(f"💾 Problematic pairs saved to: {summary_file}")

def plot_correlations(group_connectomes, correlations, metric_name_map, output_path, 
                     plot_threshold=0.3, max_plots=20, sample_size=5000):
    """Plot scatter plots with fitted lines for metric correlations"""
    
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping plot generation.")
        return
    
    print(f"\nCreating correlation scatter plots...")
    print(f"Plotting correlations with |r| > {plot_threshold}")
    
    # Filter correlations to plot (significant ones)
    plot_correlations_list = [c for c in correlations 
                        if not np.isnan(c['r']) and c['abs_r'] > plot_threshold]
    
    # Sort by absolute correlation (highest first)
    plot_correlations_list.sort(key=lambda x: x['abs_r'], reverse=True)
    
    # Limit number of plots
    if len(plot_correlations_list) > max_plots:
        print(f"Limiting to top {max_plots} correlations to avoid too many plots")
        plot_correlations_list = plot_correlations_list[:max_plots]
    
    if not plot_correlations_list:
        print(f"No correlations above threshold {plot_threshold} found for plotting")
        return
    
    # Create plots directory
    plots_dir = os.path.join(output_path, "correlation_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    for i, corr in enumerate(plot_correlations_list, 1):
        metric1, metric2 = corr['metric1'], corr['metric2']
        
        # Get simplified names for plotting
        simple_name1 = metric_name_map.get(metric1, metric1)
        simple_name2 = metric_name_map.get(metric2, metric2)
        
        print(f"  Creating plot {i}/{len(plot_correlations_list)}: {metric1} vs {metric2}")
        
        # Get the cleaned data for plotting
        result = correlate_connectomes_safe(
            group_connectomes[metric1], 
            group_connectomes[metric2], 
            return_data=True
        )
        
        if len(result) == 5:
            r, p, n_edges, x_data, y_data = result
        else:
            continue
            
        if x_data is None or y_data is None:
            continue
        
        # Sample data if too many points (for plotting performance)
        if len(x_data) > sample_size:
            indices = np.random.choice(len(x_data), sample_size, replace=False)
            x_plot = x_data[indices]
            y_plot = y_data[indices]
            n_plot = sample_size
        else:
            x_plot = x_data
            y_plot = y_data
            n_plot = len(x_data)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Main scatter plot
        ax1.scatter(x_plot, y_plot, alpha=0.6, s=20, color='steelblue', edgecolors='none')
        
        # Fit regression line
        try:
            slope, intercept, r_val, p_val, std_err = linregress(x_plot, y_plot)
            
            # Plot regression line
            x_line = np.array([x_plot.min(), x_plot.max()])
            y_line = slope * x_line + intercept
            ax1.plot(x_line, y_line, 'r-', linewidth=2, label=f'Fitted line (r={r:.3f})')
            
            # Add confidence interval (rough approximation)
            y_pred = slope * x_plot + intercept
            residuals = y_plot - y_pred
            mse = np.mean(residuals**2)
            confidence_interval = 1.96 * np.sqrt(mse)  # ~95% CI
            
            ax1.fill_between(x_line, y_line - confidence_interval, y_line + confidence_interval, 
                           alpha=0.2, color='red', label='95% CI')
            
        except Exception as e:
            print(f"    Warning: Could not fit regression line: {e}")
        
        # Formatting
        ax1.set_xlabel(f'{simple_name1}', fontsize=12)
        ax1.set_ylabel(f'{simple_name2}', fontsize=12)
        ax1.set_title(f'{simple_name1} vs {simple_name2}\nr = {r:+.3f}, p = {p:.2e}, n = {n_edges:,}', 
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residual plot
        if 'slope' in locals():
            y_pred_all = slope * x_plot + intercept
            residuals_all = y_plot - y_pred_all
            
            ax2.scatter(y_pred_all, residuals_all, alpha=0.6, s=20, color='orange', edgecolors='none')
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Fitted values', fontsize=12)
            ax2.set_ylabel('Residuals', fontsize=12)
            ax2.set_title('Residual Plot\n(Check for outliers & linearity)', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Identify potential outliers (points > 3 std devs from mean)
            std_residuals = np.std(residuals_all)
            outlier_threshold = 3 * std_residuals
            outliers = np.abs(residuals_all) > outlier_threshold
            
            if np.any(outliers):
                ax2.scatter(y_pred_all[outliers], residuals_all[outliers], 
                          color='red', s=50, marker='x', linewidth=2, 
                          label=f'Outliers (n={np.sum(outliers)})')
                ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        safe_name1 = simple_name1.replace('/', '-')
        safe_name2 = simple_name2.replace('/', '-')
        filename = f"correlation_{safe_name1}_vs_{safe_name2}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved {len(plot_correlations_list)} correlation plots to: {plots_dir}")
    
    # Create summary plot with all correlations
    create_correlation_matrix_plot(correlations, metric_name_map, plots_dir)

def create_correlation_matrix_plot(correlations, metric_name_map, plots_dir):
    """Create a correlation matrix heatmap"""
    
    if not PLOTTING_AVAILABLE:
        return
    
    print("  Creating correlation matrix heatmap...")
    
    # Convert correlations to matrix format
    all_metrics = set()
    for corr in correlations:
        all_metrics.add(corr['metric1'])
        all_metrics.add(corr['metric2'])
    
    all_metrics = sorted(list(all_metrics))
    n_metrics = len(all_metrics)
    
    # Create simplified labels for the plot
    plot_labels = [metric_name_map.get(m, m) for m in all_metrics]
    
    # Initialize correlation matrix
    corr_matrix = np.eye(n_metrics)  # 1s on diagonal
    p_matrix = np.zeros((n_metrics, n_metrics))
    
    # Fill in the correlation values
    metric_to_idx = {metric: i for i, metric in enumerate(all_metrics)}
    
    for corr in correlations:
        if not np.isnan(corr['r']):
            i = metric_to_idx[corr['metric1']]
            j = metric_to_idx[corr['metric2']]
            corr_matrix[i, j] = corr['r']
            corr_matrix[j, i] = corr['r']  # Symmetric
            p_matrix[i, j] = corr['p']
            p_matrix[j, i] = corr['p']
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mask for p-values (non-significant correlations)
    mask = p_matrix > 0.05
    mask[np.eye(n_metrics, dtype=bool)] = True  # Mask diagonal
    
    # Plot heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r', 
                center=0, 
                vmin=-1, 
                vmax=1,
                xticklabels=plot_labels, 
                yticklabels=plot_labels,
                mask=mask,
                square=True,
                cbar_kws={'label': 'Pearson correlation coefficient'},
                ax=ax)
    
    ax.set_title('Connectome Metric Correlations-DK', 
                fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save correlation matrix
    matrix_path = os.path.join(plots_dir, "correlation_matrix.png")
    plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved correlation matrix to: {matrix_path}")

def main():
    """Main analysis function"""
    
    # Path to group analysis results
    group_results_path = "/home/localadmin/Desktop/SP2/84output/pro_group_average"
    
    print("GROUP-LEVEL CONNECTOME COLLINEARITY ANALYSIS")
    print("=" * 50)
    print(f"Reading from: {group_results_path}")
    
    # Check if directory exists
    if not os.path.exists(group_results_path):
        print(f"Error: Directory {group_results_path} does not exist!")
        print("Please run the group thresholding analysis first.")
        return
    
    # --- NEW: Define metric name mapping ---
    metric_name_map = {
        'NOS': r'$NOS$',
        'RK_DKI': r'$RK$',
        'FA_DTI': r'$fa$',
        'RD_DTI_reciprocal': r'$1/rd$',
        'DePerp_SMI_reciprocal': r'1/$D_{\perp e}$',
        'F_SMI': r'$f$',
        'R1': r'$R1$'
    }
    
    # Analyze correlations
    result = analyze_group_collinearity(group_results_path)
    
    if result is None:
        return
    
    correlations, group_connectomes = result
    
    # Print results
    print_correlation_results(correlations, threshold=0.2)
    
    # Save results
    save_correlation_results(correlations, group_results_path)
    
    # Create correlation plots (if plotting libraries available)
    if PLOTTING_AVAILABLE:
        plot_correlations(group_connectomes, correlations, metric_name_map, group_results_path, 
                         plot_threshold=0.3, max_plots=15)
        print(f"\n🎉 Group correlation analysis complete!")
        print(f"📊 Check the 'correlation_plots' folder for scatter plots and correlation matrix")
    else:
        print(f"\n🎉 Group correlation analysis complete!")
        print("📊 Install matplotlib and seaborn to enable correlation plotting")
    
    return correlations

if __name__ == "__main__":
    results = main()