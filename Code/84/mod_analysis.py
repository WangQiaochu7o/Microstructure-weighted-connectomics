import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
from netneurotools.modularity import consensus_modularity

class AnalysisConfig:
    """Configuration for analysis parameters"""
    
    def __init__(self):
        # Data paths
        self.base_path = "/home/localadmin/Desktop/SP2"
        # MODIFICATION: Point to the correct input directory for 164-region data
        self.group_connectomes_dir = "/home/localadmin/Desktop/SP2/164output/pro_group_average"
        # MODIFICATION: Set the new root output path for all structured results
        self.output_path = "/home/localadmin/Desktop/SP2/164output/group_averaged_output/mod"
        
        # Metrics
        # MODIFICATION: Integrate 'BINARY_NOS' into the metrics list
        self.metrics = ['R1', 'RK_DKI', 'RD_DTI_reciprocal', 'NOS', 'F_SMI', 'FA_DTI', 'DePerp_SMI_reciprocal', 'BINARY_NOS']
        
        # Analysis parameters
        self.modularity_gamma_range = np.arange(0.5, 3.5, 0.1)
        self.modularity_n_runs = 1000

config = AnalysisConfig()

def ensure_symmetric_matrix(W):
    """Ensure matrix is symmetric and remove diagonal"""
    W_sym = (W + W.T) / 2
    np.fill_diagonal(W_sym, 0)
    W_sym = np.round(W_sym, decimals=3)
    return W_sym

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
        print(f"Warning: Failed for gamma={gamma:.1f}: {e}")
        return {'success': False, 'gamma': gamma}

def compute_and_save_modularity_analysis(W, metric_output_path):
    """
    Compute modularity analysis across gamma range and save results iteratively.
    """
    all_results_for_best = {}
    summary_stats = []
    
    for gamma in config.modularity_gamma_range:
        results = compute_modularity_consensus(W, gamma=gamma, n_runs=config.modularity_n_runs)
        
        # --- MODIFICATION: Save community assignment for each gamma ---
        gamma_dir = os.path.join(metric_output_path, f"gamma_{gamma:.1f}")
        os.makedirs(gamma_dir, exist_ok=True)
        
        if results['success']:
            all_results_for_best[gamma] = results # Store full results for finding the best one later
            
            # Save the community assignment for the current gamma
            community_df = pd.DataFrame({
                'node_id': range(len(results['consensus_community'])),
                'community_assignment': results['consensus_community']
            })
            community_path = os.path.join(gamma_dir, 'community_assignments.csv')
            community_df.to_csv(community_path, index=False)
            
            # Compute statistics for summary
            modularity_values = np.array(results['modularity_values'])
            zrand_scores = np.array(results['zrand_scores'])
            valid_mod = modularity_values[~np.isnan(modularity_values)]
            valid_zrand = zrand_scores[~np.isnan(zrand_scores)] if len(zrand_scores) > 0 else np.array([])
            
            stats = {
                'gamma': gamma, 'n_valid_runs': len(valid_mod),
                'modularity_mean': np.mean(valid_mod) if len(valid_mod) > 0 else np.nan,
                'modularity_std': np.std(valid_mod) if len(valid_mod) > 0 else np.nan,
                'num_communities': results['num_communities'],
                'zrand_mean': np.mean(valid_zrand) if len(valid_zrand) > 0 else np.nan,
                'zrand_std': np.std(valid_zrand) if len(valid_zrand) > 0 else np.nan
            }
        else:
            stats = {
                'gamma': gamma, 'n_valid_runs': 0, 'modularity_mean': np.nan,
                'modularity_std': np.nan, 'num_communities': np.nan,
                'zrand_mean': np.nan, 'zrand_std': np.nan
            }
        
        summary_stats.append(stats)
    
    # Find best gamma (highest zrand with >1 community)
    summary_df = pd.DataFrame(summary_stats)
    valid_rows = summary_df[(~summary_df['modularity_mean'].isna()) & (summary_df['num_communities'] > 1)]
        
    best_gamma_zrand_info = None
    if not valid_rows.empty:
        best_gamma_idx = valid_rows['zrand_mean'].idxmax()
        best_gamma = valid_rows.loc[best_gamma_idx, 'gamma']
        best_gamma_zrand_info = {
            'best_gamma': best_gamma,
            'best_modularity_mean': valid_rows.loc[best_gamma_idx, 'modularity_mean'],
            'best_modularity_std': valid_rows.loc[best_gamma_idx, 'modularity_std'],
            'best_num_communities': valid_rows.loc[best_gamma_idx, 'num_communities'],
            'best_zrand_mean': valid_rows.loc[best_gamma_idx, 'zrand_mean'],
            'consensus_community': all_results_for_best[best_gamma]['consensus_community']
        }
    
    return {'summary_stats': summary_df, 'best_gamma_zrand_info': best_gamma_zrand_info}

def save_best_results(results, metric_output_path, metric):
    """
    Save the sweep summary and best results to the 'best' subfolder.
    """
    best_dir = os.path.join(metric_output_path, "best")
    os.makedirs(best_dir, exist_ok=True)
    
    # Save the full gamma sweep data to the 'best' directory
    if 'summary_stats' in results and results['summary_stats'] is not None:
        gamma_sweep_df = results['summary_stats'].round(4)
        gamma_sweep_path = os.path.join(best_dir, f'modularity_gamma_sweep_{metric}.csv')
        gamma_sweep_df.to_csv(gamma_sweep_path, index=False)
        print(f"    Saved full gamma sweep data to: {gamma_sweep_path}")

    # Save best gamma results to the 'best' directory
    if results['best_gamma_zrand_info'] is not None:
        best_info = results['best_gamma_zrand_info']
        
        # Save summary of the best result
        comprehensive_summary = {'metric': metric, **{k: v for k, v in best_info.items() if k != 'consensus_community'}}
        comprehensive_df = pd.DataFrame([comprehensive_summary]).round(4)
        comprehensive_path = os.path.join(best_dir, f'modularity_best_gamma_summary_{metric}.csv')
        comprehensive_df.to_csv(comprehensive_path, index=False)
        
        # Save community assignments of the best result
        community_df = pd.DataFrame({
            'node_id': range(len(best_info['consensus_community'])),
            'community_assignment': best_info['consensus_community']
        })
        community_path = os.path.join(best_dir, f'community_assignments_best_{metric}.csv')
        community_df.to_csv(community_path, index=False)
        
        print(f"    Best gamma zrand: {best_info['best_gamma']:.1f} "
              f"(modularity: {best_info['best_modularity_mean']:.3f}, "
              f"communities: {int(best_info['best_num_communities'])})")
        print(f"    Saved best results to: {best_dir}")

def process_single_connectome(connectome_path, output_dir, metric):
    """
    Process a single connectome: load, analyze, and save results.
    """
    try:
        print(f"\nProcessing Metric: {metric}")
        
        W = pd.read_csv(connectome_path, header=None, index_col=None).values
        # MODIFICATION: Check for 164 regions
        if W.shape[0] != 164 or W.shape[1] != 164:
            print(f"  [ERROR] Connectome matrix is not 164x164. Shape is {W.shape}. Skipping.")
            return False
            
        print(f"  Loaded connectome: {W.shape}")
        
        metric_output_path = os.path.join(output_dir, metric)
        os.makedirs(metric_output_path, exist_ok=True)
        
        print("  Computing modularity and saving results iteratively...")
        mod_results = compute_and_save_modularity_analysis(W, metric_output_path)
        
        if mod_results:
            save_best_results(mod_results, metric_output_path, metric)
            print(f"  Successfully completed analysis for {metric}.")
            return True
        else:
            print(f"  Modularity analysis failed for {metric}.")
            return False
        
    except Exception as e:
        print(f"  [FATAL ERROR] processing {metric}: {e}")
        return False

def main():
    """
    Main function for modularity analysis with structured output.
    """
    print("="*80)
    print("MODULARITY ANALYSIS - INTEGRATED (WEIGHTED & BINARY) WITH STRUCTURED OUTPUT")
    print("="*80)
    
    os.makedirs(config.output_path, exist_ok=True)
    
    processed = 0
    failed = 0
    
    for metric in config.metrics:
        # MODIFICATION: Logic to handle both weighted and binary files
        if metric == 'BINARY_NOS':
            connectome_path = os.path.join(config.group_connectomes_dir, "group_NOS_binary.csv")
        else:
            connectome_path = os.path.join(config.group_connectomes_dir, f"group_{metric}_median.csv")
        
        if os.path.exists(connectome_path):
            success = process_single_connectome(connectome_path, config.output_path, metric)
            if success:
                processed += 1
            else:
                failed += 1
        else:
            print(f"\nConnectome not found, skipping: {connectome_path}")
            failed += 1
    
    print("\n" + "="*80)
    print("ANALYSIS SCRIPT FINISHED!")
    print(f"Successfully processed: {processed}/{len(config.metrics)} metrics")
    print(f"Failed or skipped: {failed}/{len(config.metrics)} metrics")
    print("="*80)
        
if __name__ == "__main__":
    main()