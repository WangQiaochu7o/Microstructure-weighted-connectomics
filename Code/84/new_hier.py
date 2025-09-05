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
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 为较新版本的matplotlib修复
if not hasattr(mpl_cm, 'get_cmap'):
    def get_cmap(name):
        """为新版matplotlib提供的兼容性函数"""
        try:
            return plt.get_cmap(name)
        except:
            return matplotlib.colormaps[name]
    
    mpl_cm.get_cmap = get_cmap

class AnalysisConfig:
    """分析参数配置"""
    
    def __init__(self):
        # 数据路径
        self.base_path = "/home/localadmin/Desktop/SP2"
        # [已修正] 将输入目录从164output改为84output
        self.group_connectomes_dir = "/home/localadmin/Desktop/SP2/84output/pro_group_average" 
        self.output_path = "/home/localadmin/Desktop/SP2/84output/group_averaged_output"
        self.freesurfer_home = '/home/localadmin/freesurfer'
        self.subjects_dir = os.path.join(self.freesurfer_home, 'subjects')
        
        # 度量指标
        self.metrics = ['R1', 'RK_DKI', 'RD_DTI_reciprocal', 'NOS', 'F_SMI', 'FA_DTI', 'DePerp_SMI_reciprocal']
        
        # 层次聚类参数
        self.hierarchical_max_clusters = 20
        self.hierarchical_linkage_methods = ['ward'] # 仅使用 'ward' 方法

config = AnalysisConfig()

# 设置环境变量
os.environ['FREESURFER_HOME'] = config.freesurfer_home
os.environ['SUBJECTS_DIR'] = config.subjects_dir

# 为不同加权度量定义颜色
metric_colors = {
    'R1': '#1E90FF',           # 蓝色
    'DePerp_SMI_reciprocal': '#FF8C00',       # 橙色  
    'FA_DTI': '#228B22',  # 绿色
    'RD_DTI_reciprocal': '#DC143C',          # 红色
    'NOS': '#9932CC',        # 紫色
    'RK_DKI': '#FF69B4',       # 粉色
    'F_SMI': '#8B4513'  # 棕色
}

def ensure_symmetric_matrix(W):
    """确保矩阵对称并移除对角线"""
    W_sym = (W + W.T) / 2
    np.fill_diagonal(W_sym, 0)
    W_sym = np.round(W_sym, decimals=3)
    return W_sym

def get_dk_atlas_mapping():
    """
    为84分区的 Desikan-Killiany (DK) 图谱提供映射。
    此函数主要为可视化目的，返回68个皮层区域的映射。
    """
    # FreeSurfer aparc (DK) atlas cortical regions (34 per hemisphere)
    cortical_regions = [
        'bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus',
        'entorhinal', 'fusiform', 'inferiorparietal', 'inferiortemporal',
        'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal',
        'lingual', 'medialorbitofrontal', 'middletemporal', 'parahippocampal',
        'paracentral', 'parsopercularis', 'parsorbitalis', 'parstriangularis',
        'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral',
        'precuneus', 'rostralanteriorcingulate', 'rostralmiddlefrontal',
        'superiorfrontal', 'superiorparietal', 'superiortemporal', 'supramarginal',
        'frontalpole', 'temporalpole', 'transversetemporal', 'insula'
    ]
    
    node_to_region = {}
    # 节点 1-34: 左半球皮层
    for i, region in enumerate(cortical_regions):
        node_idx = i + 1 
        node_to_region[node_idx] = f"lh.{region}"
    
    # 节点 51-84: 右半球皮层 (在MRtrix索引中通常跳过了皮层下区域)
    for i, region in enumerate(cortical_regions):
        node_idx = 51 + i 
        node_to_region[node_idx] = f"rh.{region}"
        
    return node_to_region

def compute_uncentered_correlation_distance(W):
    """使用非中心化皮尔逊相关系数计算距离矩阵"""
    W_sym = ensure_symmetric_matrix(W)
    n_regions = W_sym.shape[0]
    correlation_matrix = np.zeros((n_regions, n_regions))
    
    for i in range(n_regions):
        for j in range(i, n_regions):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                x, y = W_sym[i, :], W_sym[j, :]
                numerator = np.sum(x * y)
                denominator = np.sqrt(np.sum(x**2) * np.sum(y**2))
                corr = numerator / denominator if denominator > 1e-10 else 0.0
                correlation_matrix[i, j] = correlation_matrix[j, i] = corr
                
    distance_matrix = 1 - np.abs(correlation_matrix)
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix, correlation_matrix

def perform_hierarchical_clustering(distance_matrix, linkage_method='ward'):
    """使用指定的连接方法执行层次聚类"""
    distance_condensed = squareform(distance_matrix)
    linkage_matrix = linkage(distance_condensed, method=linkage_method)
    return linkage_matrix

def find_optimal_clusters_hierarchical(linkage_matrix, W_sym, max_clusters=20):
    """使用轮廓分析寻找最优聚类数"""
    silhouette_scores = []
    cluster_range = range(2, min(max_clusters + 1, W_sym.shape[0]))
    
    for n_clusters in cluster_range:
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        if len(np.unique(cluster_labels)) > 1:
            try:
                score = silhouette_score(W_sym, cluster_labels, metric='euclidean')
                silhouette_scores.append(score)
            except Exception:
                silhouette_scores.append(-1)
        else:
            silhouette_scores.append(-1)
    
    optimal_n_clusters, optimal_silhouette = 2, -1
    if len(silhouette_scores) > 0 and max(silhouette_scores) > -1:
        optimal_idx = np.argmax(silhouette_scores)
        optimal_n_clusters = cluster_range[optimal_idx]
        optimal_silhouette = silhouette_scores[optimal_idx]
    
    return optimal_n_clusters, optimal_silhouette, list(cluster_range), silhouette_scores

def compute_hierarchical_clustering_analysis(W):
    """仅使用'ward'方法计算层次聚类分析"""
    print("    开始层次聚类计算...")
    W_sym = ensure_symmetric_matrix(W)
    
    print("    计算非中心化相关距离矩阵...")
    distance_matrix, _ = compute_uncentered_correlation_distance(W_sym)
    
    results = {}
    linkage_method = 'ward'
    
    try:
        print(f"      运行连接方法: {linkage_method}")
        linkage_matrix = perform_hierarchical_clustering(distance_matrix, linkage_method)
        
        print(f"        寻找最优聚类数 (最大: {config.hierarchical_max_clusters})...")
        optimal_n_clusters, optimal_silhouette, cluster_range, silhouette_scores = find_optimal_clusters_hierarchical(
            linkage_matrix, W_sym, config.hierarchical_max_clusters
        )
        
        method_result = {
            'linkage_method': linkage_method,
            'linkage_matrix': linkage_matrix,
            'optimal_n_clusters': optimal_n_clusters,
            'cluster_range': cluster_range,
            'silhouette_scores': silhouette_scores,
            'success': True
        }
        results[linkage_method] = method_result
        print(f"        结果: {optimal_n_clusters} 个聚类, 轮廓系数: {optimal_silhouette:.3f}")
        
    except Exception as e:
        print(f"        失败: {e}")
        results[linkage_method] = {'success': False, 'error': str(e)}
    
    return {'best_result': results.get('ward')}

def save_hierarchical_results(results, output_dir, metric):
    """保存层次聚类分析结果"""
    best_result = results.get('best_result')
    if not best_result or not best_result.get('success'):
        print(f"    [错误] {metric} 没有有效的层次聚类结果")
        return

    print(f"    为 {metric} 保存结果...")
    
    # 保存轮廓系数扫描数据
    sweep_df = pd.DataFrame({
        'n_clusters': best_result['cluster_range'],
        'silhouette_score': best_result['silhouette_scores']
    })
    sweep_path = os.path.join(output_dir, f'hierarchical_silhouette_sweep_ward_{metric}_84regions.csv')
    sweep_df.to_csv(sweep_path, index=False)
    print(f"      已保存轮廓系数扫描数据: {sweep_path}")

def generate_dynamic_colormap(n_communities):
    """根据社群数量生成动态颜色映射"""
    if n_communities <= 10:
        base_colors = ['#FF0000', '#0000FF', '#00AA00', '#FF66CC', '#FF8800', '#8B00CC', '#996633', '#00CCCC', '#FFFF00', '#888888']
        return ListedColormap(base_colors[:n_communities])
    elif n_communities <= 20:
        return plt.get_cmap('tab20')
    else:
        colors = []
        cmaps = ['tab20', 'tab20b', 'tab20c']
        for i in range(n_communities):
            cmap_idx, color_idx = divmod(i, 20)
            cmap = plt.get_cmap(cmaps[cmap_idx % len(cmaps)])
            colors.append(cmap(color_idx / 19.0))
        return ListedColormap(colors)

def load_freesurfer_dk_atlas():
    """加载FreeSurfer Desikan-Killiany (aparc)图谱"""
    try:
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
        # 使用标准的 DK 图谱 (aparc)
        lh_annot_path = os.path.join(config.subjects_dir, 'fsaverage', 'label', 'lh.aparc.annot')
        rh_annot_path = os.path.join(config.subjects_dir, 'fsaverage', 'label', 'rh.aparc.annot')
        
        lh_labels, _, lh_names = nib.freesurfer.read_annot(lh_annot_path)
        rh_labels, _, rh_names = nib.freesurfer.read_annot(rh_annot_path)
        
        # 将名称从 bytes 解码为 str
        lh_names = [name.decode('utf-8') for name in lh_names]
        rh_names = [name.decode('utf-8') for name in rh_names]
            
        return lh_labels, rh_labels, lh_names, rh_names
    except Exception as e:
        print(f"加载DK图谱时出错: {e}")
        return None, None, None, None

def map_data_to_surface_dk(data_values, lh_labels, rh_labels, lh_names, rh_names):
    """使用DK图谱将84分区数据映射到大脑皮层表面"""
    lh_data = np.zeros(len(lh_labels), dtype=float)
    rh_data = np.zeros(len(rh_labels), dtype=float)
    
    node_to_region = get_dk_atlas_mapping() # 获取皮层区域映射
    
    # FreeSurfer的 .annot 文件中的名称直接对应于DK图谱区域
    lh_name_to_label_val = {name: i for i, name in enumerate(lh_names)}
    rh_name_to_label_val = {name: i for i, name in enumerate(rh_names)}

    for node_idx, data_value in enumerate(data_values):
        mrtrix_idx = node_idx + 1 # 从0-based转为1-based
        
        # 仅处理在皮层映射中定义的节点
        if mrtrix_idx in node_to_region and data_value != 0:
            region_name_full = node_to_region[mrtrix_idx]
            hemi_prefix, region_name = region_name_full.split('.', 1)

            if hemi_prefix == 'lh':
                label_val = lh_name_to_label_val.get(region_name)
                if label_val is not None:
                    lh_data[lh_labels == label_val] = float(data_value)
            else: # rh
                label_val = rh_name_to_label_val.get(region_name)
                if label_val is not None:
                    rh_data[rh_labels == label_val] = float(data_value)
                
    return np.nan_to_num(lh_data), np.nan_to_num(rh_data)

def create_ward_silhouette_sweep_plot(sweep_data, metric, output_dir):
    """为给定度量创建'ward'方法的轮廓系数扫描图"""
    print(f"为 {metric} 创建轮廓系数扫描图...")
    if not isinstance(sweep_data, pd.DataFrame) or sweep_data.empty:
        print(f"  {metric} 无扫描数据。")
        return

    color = metric_colors.get(metric, '#888888')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sweep_data['n_clusters'], sweep_data['silhouette_score'], 
            color=color, linewidth=3, marker='o', markersize=8, label=metric)
    
    if not sweep_data.empty:
        best_idx = sweep_data['silhouette_score'].idxmax()
        optimal_n = sweep_data.loc[best_idx, 'n_clusters']
        optimal_score = sweep_data.loc[best_idx, 'silhouette_score']
        
        ax.axvline(x=optimal_n, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.plot(optimal_n, optimal_score, 'D', color='red', markersize=10, 
                markeredgecolor='black', markeredgewidth=1.5,
                label=f'最优 k={optimal_n} ({optimal_score:.3f})')
    
    ax.set_xlabel('聚类数 (k)', fontsize=14)
    ax.set_ylabel('轮廓系数', fontsize=14)
    ax.set_title(f'{metric} 的轮廓系数 vs. 聚类数 (84分区 DK)\n(Ward 方法)', fontsize=16, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=12)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"hierarchical_silhouette_sweep_ward_{metric}_84regions.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存图像: {output_file}")

def create_brain_visualization_for_cluster_numbers(metric, linkage_matrix, output_dir):
    """创建脑图可视化，展示聚类分配如何随聚类数变化"""
    print(f"\n为 {metric} 创建跨聚类数的脑图可视化 (84分区 DK)...")
    
    lh_labels, rh_labels, lh_names, rh_names = load_freesurfer_dk_atlas()
    if lh_labels is None:
        print("  加载FreeSurfer DK图谱失败，跳过可视化。")
        return

    views = [('lateral', 'lh', '左外侧'), ('medial', 'lh', '左内侧'), 
             ('lateral', 'rh', '右外侧'), ('medial', 'rh', '右内侧')]
    
    cluster_numbers_to_plot = range(2, 21) # 绘制k=2, 3, 4, 5, 6
    
    fig, axes = plt.subplots(len(cluster_numbers_to_plot), len(views), 
                             figsize=(16, 4 * len(cluster_numbers_to_plot)),
                             gridspec_kw={'wspace': 0.01, 'hspace': 0.15})
    
    for row, n_clusters in enumerate(cluster_numbers_to_plot):
        print(f"  处理 k={n_clusters} 个聚类...")
        cluster_assignments = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        unique_clusters = np.unique(cluster_assignments[cluster_assignments > 0])
        num_clusters_found = len(unique_clusters)
        
        if num_clusters_found == 0:
            print(f"    k={n_clusters} 未找到聚类")
            for col in range(len(views)):
                axes[row, col].text(0.5, 0.5, '无聚类', ha='center', va='center')
                axes[row, col].set_axis_off()
            continue

        custom_cmap = generate_dynamic_colormap(num_clusters_found)
        # 使用适配DK图谱的映射函数
        lh_data, rh_data = map_data_to_surface_dk(cluster_assignments, lh_labels, rh_labels, lh_names, rh_names)
        
        for col, (view, hemi, view_title) in enumerate(views):
            ax = axes[row, col]
            try:
                brain = Brain('fsaverage', hemi, 'pial', cortex='low_contrast', background='white', size=(800, 600))
                surface_data = lh_data if hemi == 'lh' else rh_data
                
                brain.add_data(surface_data, min=0.5, max=np.max(unique_clusters), colormap=custom_cmap,
                               alpha=1.0, smoothing_steps=0, thresh=0.5, colorbar=False)
                brain.show_view(view)
                
                temp_file = f"{output_dir}/temp_{metric}_{n_clusters}_{hemi}_{view}_84regions.png"
                brain.save_image(temp_file, mode='rgb', antialiased=True)
                img = Image.open(temp_file)
                ax.imshow(np.array(img))
                brain.close()
                os.remove(temp_file)
            except Exception as e:
                print(f"    为 k={n_clusters} 创建视图 {view_title} 时出错: {e}")
                ax.text(0.5, 0.5, '错误', ha='center', va='center', color='red')
            
            ax.set_axis_off()

            if row == 0:
                ax.set_title(view_title, fontsize=14, fontweight='bold', pad=10)
            if col == 0:
                ax.set_ylabel(f'k = {n_clusters}', rotation=0, labelpad=40, fontsize=12, fontweight='bold', ha='right', va='center')

    fig.suptitle(f'{metric} 的层次聚类分配 (84分区 DK, Ward 方法)', fontsize=16, fontweight='bold', y=0.98)
    
    output_file = f"{output_dir}/hierarchical_assignments_by_n_clusters_{metric}_84regions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  已保存脑图可视化网格: {output_file}")

def process_connectome_hierarchical_analysis(connectome_path, output_dir, metric):
    """使用层次聚类分析处理单个连接组"""
    try:
        print(f"\n处理度量 {metric} 的层次聚类分析 (84分区)")
        W = pd.read_csv(connectome_path, header=None, index_col=None).values
        if W.shape[0] != 84 or W.shape[1] != 84:
            print(f"  [错误] 连接组矩阵不是84x84。形状为 {W.shape}")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        
        hierarchical_results = compute_hierarchical_clustering_analysis(W)
        
        if hierarchical_results and hierarchical_results.get('best_result'):
            print("  层次聚类分析成功完成。")
            save_hierarchical_results(hierarchical_results, output_dir, metric)
            return hierarchical_results
        else:
            print("  [错误] 层次聚类分析失败。")
            return None
            
    except Exception as e:
        print(f"  [致命错误] 处理度量 {metric} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """层次聚类分析的主函数 (84分区 DK)"""
    print("="*80)
    print("层次聚类分析 - WARD 方法 (84分区 Desikan-Killiany)")
    print("="*80)
    
    os.makedirs(config.output_path, exist_ok=True)
    processed, failed = 0, 0
    
    for metric in config.metrics:
        # [已简化] 直接从config中指定的84分区目录加载文件
        connectome_path = os.path.join(config.group_connectomes_dir, f"group_{metric}_median.csv")

        if os.path.exists(connectome_path):
            results = process_connectome_hierarchical_analysis(connectome_path, config.output_path, metric)
            
            if results and results.get('best_result'):
                # 图 1: 轮廓系数扫描图
                sweep_data_path = os.path.join(config.output_path, f'hierarchical_silhouette_sweep_ward_{metric}_84regions.csv')
                if os.path.exists(sweep_data_path):
                    sweep_df = pd.read_csv(sweep_data_path)
                    create_ward_silhouette_sweep_plot(sweep_df, metric, config.output_path)
                
                # 图 2: 脑图可视化网格
                linkage_matrix = results['best_result']['linkage_matrix']
                create_brain_visualization_for_cluster_numbers(metric, linkage_matrix, config.output_path)
                
                processed += 1
            else:
                failed += 1
                print(f"[失败] {metric} 分析未成功。")
        else:
            print(f"[错误] 连接组文件未找到: {connectome_path}")
            failed += 1
            
    print("\n" + "="*80)
    print("分析完成")
    print(f"成功处理: {processed}/{len(config.metrics)} 个度量")
    print(f"失败: {failed}/{len(config.metrics)} 个度量")
    print("="*80)

if __name__ == "__main__":
    main()