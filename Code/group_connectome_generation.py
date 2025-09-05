import numpy as np
import pandas as pd
import os
from collections import defaultdict

def load_connectome(filepath):
    """Load connectome from CSV file without headers/index"""
    try:
        # Load without assuming headers or index columns
        connectome = pd.read_csv(filepath, header=None).values
        connectome = connectome.astype(float)
        # Check for and replace NaN values with zeros
        if np.isnan(connectome).any():
            nan_count = np.sum(np.isnan(connectome))
            print(f"  Found {nan_count} NaN values, replacing with zeros")
            connectome = np.nan_to_num(connectome, nan=0.0)
            
        return connectome
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def apply_nos_threshold(connectome_nos, threshold=5):
    """Apply NOS threshold to individual connectome"""
    if connectome_nos is None:
        return None
    
    # Create binary mask for edges above threshold
    thresholded = connectome_nos.copy()
    thresholded[thresholded <= threshold] = 0
    
    return thresholded

def create_group_connectome(base_path, metric_paths, nos_threshold=5, group_threshold=0.5):
    """
    Create group-level connectome with dual thresholds
    
    Parameters:
    -----------
    base_path : str
        Base directory path
    metric_paths : dict
        Dictionary of metric names and their relative file paths
    nos_threshold : int
        Individual-level threshold for NOS (default: 5)
    group_threshold : float
        Group-level threshold (proportion of subjects, default: 0.5)
    
    Returns:
    --------
    dict with group connectomes and statistics
    """
    
    # Define subjects for each aim
    aim1_subjects = [i for i in range(4, 30) if i != 20]  # 04-29 excluding 20
    aim2_subjects = list(range(5, 19))  # 05-18
    all_subjects = [(1, sub) for sub in aim1_subjects] + [(2, sub) for sub in aim2_subjects]
    
    print(f"Processing {len(all_subjects)} subjects...")
    print(f"Individual threshold: NOS > {nos_threshold}")
    print(f"Group threshold: Present in ≥{group_threshold*100:.0f}% of subjects")
    print("=" * 60)
    
    # Storage for all subject data
    subject_data = {}
    edge_counts = None
    total_subjects = len(all_subjects)
    
    # Process each subject
    valid_subjects = 0
    for i, (aim, sub) in enumerate(all_subjects):
        subject_folder = os.path.join(base_path, f"Aim{aim}_sub-{sub:02d}", "connectomes")
        
        print(f"Processing Aim {aim}, Subject {sub:02d}... ", end="")
        
        # Load NOS connectome for thresholding
        nos_path = os.path.join(subject_folder, "NOS/connectome_NOS.csv")
        connectome_nos = load_connectome(nos_path)
        
        if connectome_nos is None:
            print("FAILED (NOS not found)")
            continue
        
        # Check matrix dimensions
        if valid_subjects == 0:
            print(f"Matrix shape: {connectome_nos.shape} ", end="")
        
        # Apply NOS threshold
        nos_thresholded = apply_nos_threshold(connectome_nos, nos_threshold)
        
        # Create binary mask of valid edges (NOS > threshold)
        edge_mask = (nos_thresholded > 0).astype(int)
        
        # Initialize edge counter on first valid subject
        if edge_counts is None:
            edge_counts = np.zeros_like(edge_mask)
        
        # Add to edge count
        edge_counts += edge_mask
        
        # Load other metrics and apply same mask
        subject_metrics = {}
        all_metrics_loaded = True
        
        for metric_name, relative_path in metric_paths.items():
            if metric_name == 'NOS':
                # Use thresholded NOS
                subject_metrics[metric_name] = nos_thresholded
            else:
                # Load other metric and apply NOS-based mask
                metric_path = os.path.join(subject_folder, relative_path)
                metric_connectome = load_connectome(metric_path)
                
                if metric_connectome is None:
                    print(f"FAILED ({metric_name} not found)")
                    all_metrics_loaded = False
                    break
                
                # Check that dimensions match
                if metric_connectome.shape != connectome_nos.shape:
                    print(f"FAILED ({metric_name} shape mismatch: {metric_connectome.shape} vs {connectome_nos.shape})")
                    all_metrics_loaded = False
                    break
                
                # Apply NOS-based mask to other metrics
                masked_metric = metric_connectome.copy()
                masked_metric[edge_mask == 0] = 0
                subject_metrics[metric_name] = masked_metric
        
        if all_metrics_loaded:
            subject_data[f"Aim{aim}_sub{sub:02d}"] = subject_metrics
            valid_subjects += 1
            print(f"SUCCESS ({np.sum(edge_mask)} edges)")
        
    print(f"\nSuccessfully processed {valid_subjects} subjects")
    
    # Print final matrix dimensions
    if edge_counts is not None:
        print(f"Final matrix dimensions: {edge_counts.shape}")
    
    # Apply group threshold
    min_subjects = int(np.ceil(valid_subjects * group_threshold))
    group_edge_mask = (edge_counts >= min_subjects).astype(int)
    
    print(f"Group threshold: {min_subjects}/{valid_subjects} subjects")
    print(f"Edges passing group threshold: {np.sum(group_edge_mask)}")
    
    # Create group-level connectomes
    group_connectomes = {}
    
    for metric_name in metric_paths.keys():
        print(f"\nCreating group connectome for {metric_name}...")
        
        # Collect all valid subject data for this metric
        metric_matrices = []
        for subject_id, metrics in subject_data.items():
            metric_matrices.append(metrics[metric_name])
        
        metric_stack = np.stack(metric_matrices, axis=0)  # Shape: (subjects, nodes, nodes)
        
        # Calculate statistics only for edges that pass group threshold
        group_connectomes[metric_name] = {
            'binary': group_edge_mask,
            'mean': np.zeros_like(group_edge_mask, dtype=float),
            'median': np.zeros_like(group_edge_mask, dtype=float),
            'std': np.zeros_like(group_edge_mask, dtype=float),
            'edge_count': edge_counts.copy()
        }
        
        # Calculate statistics only for valid edges
        for i in range(group_edge_mask.shape[0]):
            for j in range(group_edge_mask.shape[1]):
                if group_edge_mask[i, j] == 1:
                    # Get values from all subjects for this edge
                    edge_values = metric_stack[:, i, j]
                    # Remove zeros (subjects where this edge wasn't present)
                    edge_values_nonzero = edge_values[edge_values > 0]
                    
                    if len(edge_values_nonzero) > 0:
                        group_connectomes[metric_name]['mean'][i, j] = np.mean(edge_values_nonzero)
                        group_connectomes[metric_name]['median'][i, j] = np.median(edge_values_nonzero)
                        group_connectomes[metric_name]['std'][i, j] = np.std(edge_values_nonzero)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Valid subjects: {valid_subjects}")
    print(f"Matrix dimensions: {group_edge_mask.shape}")
    print(f"Total possible edges: {np.sum(np.triu(np.ones_like(group_edge_mask), k=1))}")
    print(f"Edges after individual threshold (NOS > {nos_threshold}): varies by subject")
    print(f"Edges after group threshold (≥{group_threshold*100:.0f}% subjects): {np.sum(group_edge_mask)}")
    
    # Edge density
    total_possible = np.sum(np.triu(np.ones_like(group_edge_mask), k=1))
    final_edges = np.sum(np.triu(group_edge_mask, k=1))
    density = final_edges / total_possible
    print(f"Final network density: {density:.3f} ({final_edges}/{total_possible})")
    
    return {
        'group_connectomes': group_connectomes,
        'subject_data': subject_data,
        'edge_counts': edge_counts,
        'group_mask': group_edge_mask,
        'valid_subjects': valid_subjects,
        'thresholds': {
            'nos_threshold': nos_threshold,
            'group_threshold': group_threshold,
            'min_subjects': min_subjects
        }
    }

def save_group_connectomes(results, output_dir):
    """Save group connectomes to CSV files without headers/index"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for metric_name, connectome_dict in results['group_connectomes'].items():
        for stat_type, matrix in connectome_dict.items():
            if stat_type != 'edge_count':  # Save edge_count separately
                filename = f"group_{metric_name}_{stat_type}.csv"
                filepath = os.path.join(output_dir, filename)
                # Save without headers or index to match input format
                pd.DataFrame(matrix).to_csv(filepath, header=False, index=False)
                print(f"Saved: {filename} (shape: {matrix.shape})")
    
    # Save edge counts
    edge_count_path = os.path.join(output_dir, "edge_counts.csv")
    pd.DataFrame(results['edge_counts']).to_csv(edge_count_path, header=False, index=False)
    print(f"Saved: edge_counts.csv (shape: {results['edge_counts'].shape})")
    
    # Save summary info
    summary_path = os.path.join(output_dir, "group_analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Group Connectome Analysis Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Valid subjects: {results['valid_subjects']}\n")
        f.write(f"Matrix dimensions: {results['group_mask'].shape}\n")
        f.write(f"NOS threshold: {results['thresholds']['nos_threshold']}\n")
        f.write(f"Group threshold: {results['thresholds']['group_threshold']} ({results['thresholds']['min_subjects']} subjects)\n")
        f.write(f"Final edges: {np.sum(results['group_mask'])}\n")
        
        total_possible = np.sum(np.triu(np.ones_like(results['group_mask']), k=1))
        final_edges = np.sum(np.triu(results['group_mask'], k=1))
        density = final_edges / total_possible
        f.write(f"Network density: {density:.3f}\n")
    
    print(f"Saved: group_analysis_summary.txt")

def main():
    """Main function to run group analysis"""
    
    base_path = "/home/localadmin/Desktop/SP2"
    
    # Define metrics to analyze
    metric_paths = {
        'NOS': 'NOS/connectome_NOS.csv',
        'RK_DKI': 'DWI_weighted/median_rk_dki_connectome.csv',
        'FA_DTI': 'DWI_weighted/median_fa_dti_connectome.csv',
        'RD_DTI_reciprocal': 'reciprocal_rd_dti/median_rd_dti_reciprocal_connectome.csv',
        'DePerp_SMI_reciprocal': 'reciprocal_DePerp_smi/median_DePerp_smi_reciprocal_connectome.csv',
        'F_SMI': 'SM_weighted/median_f_smi_connectome.csv',
        'R1': 'R1_weighted/median_R1_connectome.csv'
    }
        
    # Run analysis
    results = create_group_connectome(
        base_path=base_path,
        metric_paths=metric_paths,
        nos_threshold=5,
        group_threshold=0.5
    )
    
    # Save results
    output_dir = "/home/localadmin/Desktop/SP2/84output/pro_group_average"
    save_group_connectomes(results, output_dir)
    
    print(f"\n🎉 Group analysis complete! Results saved to: {output_dir}")
    
    # Return for further analysis
    return results

if __name__ == "__main__":
    results = main()