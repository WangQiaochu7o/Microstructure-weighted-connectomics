import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 核心配置 (已修改为动态路径) ---

class AnalysisConfig:
    """分析参数配置"""
    def __init__(self, parcellation_name):
        # [新增] 保存当前正在分析的分区方案名称
        self.parcellation_name = parcellation_name
        
        # 数据路径
        self.base_path = "/home/localadmin/Desktop/SP2"
        # [已修改] 输出/读取路径现在基于分区方案动态生成
        self.output_path = f"/home/localadmin/Desktop/SP2/{self.parcellation_name}output/group_averaged_output"
        
        # 度量指标
        self.metrics = ['R1', 'RK_DKI', 'RD_DTI_reciprocal', 'NOS', 'F_SMI', 'FA_DTI', 'DePerp_SMI_reciprocal']
        
        # 我们要分析的特定gamma值
        self.target_gamma = 0.8

# 定义绘图颜色 (与您的脚本保持一致)
metric_colors = {
    'R1': '#1E90FF',           # 蓝色
    'DePerp_SMI_reciprocal': '#FF8C00',       # 橙色  
    'FA_DTI': '#228B22',       # 绿色
    'RD_DTI_reciprocal': '#DC143C',          # 红色
    'NOS': '#9932CC',        # 紫色
    'RK_DKI': '#FF69B4',       # 粉色
    'F_SMI': '#8B4513'         # 棕色
}

# --- 2. 数据提取函数 (无需修改) ---

def collect_data_at_gamma(target_gamma):
    """
    从每个度量的gamma sweep文件中收集特定gamma值下的Q和Z-Rand数据。
    """
    results = []
    print(f"正在从gamma sweep文件中收集 gamma = {target_gamma} 的数据...")
    
    for metric in config.metrics:
        file_path = os.path.join(config.output_path, f"modularity_gamma_sweep_{metric}.csv")
        
        if not os.path.exists(file_path):
            print(f"  - 警告: 未找到文件 {file_path}，跳过 {metric}。")
            continue
            
        try:
            df = pd.read_csv(file_path)
            # 使用np.isclose来处理浮点数比较的精度问题
            target_row = df[np.isclose(df['gamma'], target_gamma)]
            
            if target_row.empty:
                print(f"  - 警告: 在 {metric} 的文件中未找到 gamma = {target_gamma} 的数据。")
                continue
            
            q_mean = target_row['modularity_mean'].iloc[0]
            zrand_mean = target_row['zrand_mean'].iloc[0]
            
            results.append({
                'Metric': metric,
                'Q_mean': q_mean,
                'ZRand_mean': zrand_mean
            })
            print(f"  - 成功: 为 {metric} 找到数据 (Q={q_mean:.4f}, Z-Rand={zrand_mean:.4f})")
            
        except Exception as e:
            print(f"  - 错误: 读取或处理文件 {file_path} 时出错: {e}")

    return pd.DataFrame(results) if results else pd.DataFrame()

# --- 3. 核心绘图函数 (已修改以包含分区信息) ---

def create_comparison_plot_at_gamma(data_df, target_gamma, parcellation_name):
    """
    创建一张图，包含两条曲线，对比在特定gamma值下各度量的Q和Z-Rand值。
    """
    if data_df.empty:
        print("没有可供绘图的数据。")
        return
        
    print("\n正在创建对比图...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    metrics = data_df['Metric']
    x = np.arange(len(metrics))

    ax.plot(x, data_df['Q_mean'], 'o-', color='royalblue', linewidth=3, markersize=8, label='模块度 Q 值')
    ax.plot(x, data_df['ZRand_mean'], 's--', color='darkorange', linewidth=3, markersize=8, label='Z-Rand 分数 (稳定性)')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=12)
    
    # [已修改] 标题现在包含分区信
    ax.set_title(f'模块化指标对比 ({parcellation_name} 分区, Gamma = {target_gamma})', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('数值', fontsize=14)
    ax.set_xlabel('加权度量 (Weighting Metric)', fontsize=14)
    
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
    
    for i, row in data_df.iterrows():
        ax.text(i, row['Q_mean'] + 0.01, f"{row['Q_mean']:.3f}", ha='center', va='bottom', color='royalblue', fontsize=10, fontweight='bold')
        ax.text(i, row['ZRand_mean'] - 0.02, f"{row['ZRand_mean']:.3f}", ha='center', va='top', color='darkorange', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # [已修改] 输出文件名现在包含分区信息
    gamma_str = str(target_gamma).replace('.', 'p')
    output_file = os.path.join(config.output_path, f"modularity_comparison_at_gamma_{gamma_str}_{parcellation_name}_regions.png")
    plt.savefig(output_file, dpi=300)
    
    print(f"✓ 对比图已成功保存至: {output_file}")
    plt.close()


# --- 4. 主执行函数 (已修改为可配置) ---

def main():
    """主函数，负责执行数据收集和绘图"""
    
    # --- 在此处设置要运行的分区方案 ---
    # 将此值更改为 '164' 或 '84'
    PARCELLATION_TO_RUN = '84' 
    # ------------------------------------

    # 基于选择初始化全局配置
    global config
    config = AnalysisConfig(PARCELLATION_TO_RUN)
    
    print("="*60)
    # [已修改] 日志信息现在包含分区方案
    print(f"开始分析 {config.parcellation_name} 分区在 Gamma = {config.target_gamma} 时的模块化指标")
    print(f"将从以下目录读取结果: {config.output_path}")
    print("="*60)

    # 1. 收集数据
    gamma_data_df = collect_data_at_gamma(config.target_gamma)

    # 2. 如果成功收集到数据，则进行绘图
    if not gamma_data_df.empty:
        create_comparison_plot_at_gamma(gamma_data_df, config.target_gamma, config.parcellation_name)
    else:
        print("\n未能收集到任何数据，无法生成图表。请检查文件路径和内容。")

    print("\n" + "="*60)
    print("脚本执行完毕！")
    print("="*60)


if __name__ == "__main__":
    main()