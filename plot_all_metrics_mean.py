import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_all_metrics(sit_file, repa_file, out_dir):
    df_sit = pd.read_csv(sit_file, sep='\t')
    df_repa = pd.read_csv(repa_file, sep='\t')
    
    # -------------------------------------------------------------
    # 1. Plot Mean by Layer
    # -------------------------------------------------------------
    mean_sit_layer = df_sit.groupby(['metric', 'layer'])['value'].mean().reset_index()
    mean_repa_layer = df_repa.groupby(['metric', 'layer'])['value'].mean().reset_index()
    
    metrics = sorted(set(mean_sit_layer['metric'].unique()).union(set(mean_repa_layer['metric'].unique())))
    num_metrics = len(metrics)
    
    cols = 4
    rows = (num_metrics + cols - 1) // cols
    
    color_sit = '#3498db'
    color_repa = '#2ecc71'

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.5))
    axes = axes.flatten()
    
    for i, m in enumerate(metrics):
        ax = axes[i]
        
        if m in mean_sit_layer['metric'].values:
            sit_subset = mean_sit_layer[mean_sit_layer['metric'] == m].sort_values('layer')
            ax.plot(sit_subset['layer'], sit_subset['value'], marker='s', markersize=4, linestyle='-', label='SiT', color=color_sit, linewidth=2)
            
        if m in mean_repa_layer['metric'].values:
            repa_subset = mean_repa_layer[mean_repa_layer['metric'] == m].sort_values('layer')
            ax.plot(repa_subset['layer'], repa_subset['value'], marker='o', markersize=4, linestyle='-', label='REPA', color=color_repa, linewidth=2)
        
        ax.set_title(m.upper(), fontsize=13, fontweight='bold')
        ax.set_xlabel("Layer", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=10)
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("Averaged Metrics over Layers (SiT vs REPA) - All Configurations", fontsize=20, fontweight='bold')
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path_layer = out_dir / "all_metrics_layer_mean.png"
    plt.savefig(out_path_layer, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Layer trend plot saved to: {out_path_layer}")

    # -------------------------------------------------------------
    # 2. Plot Mean by Timestep
    # -------------------------------------------------------------
    mean_sit_step = df_sit.groupby(['metric', 'timestep'])['value'].mean().reset_index()
    mean_repa_step = df_repa.groupby(['metric', 'timestep'])['value'].mean().reset_index()
    
    fig_t, axes_t = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.5))
    axes_t = axes_t.flatten()
    
    for i, m in enumerate(metrics):
        ax = axes_t[i]
        
        if m in mean_sit_step['metric'].values:
            sit_subset = mean_sit_step[mean_sit_step['metric'] == m].sort_values('timestep')
            ax.plot(sit_subset['timestep'], sit_subset['value'], marker='s', markersize=6, linestyle='-', label='SiT', color=color_sit, linewidth=2)
            
        if m in mean_repa_step['metric'].values:
            repa_subset = mean_repa_step[mean_repa_step['metric'] == m].sort_values('timestep')
            ax.plot(repa_subset['timestep'], repa_subset['value'], marker='o', markersize=6, linestyle='-', label='REPA', color=color_repa, linewidth=2)
        
        ax.set_title(m.upper(), fontsize=13, fontweight='bold')
        ax.set_xlabel("Timestep (t)", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=10)
        # Force x-axis to show timesteps nicely if there's only one point
        ax.margins(x=0.1)

    for j in range(i + 1, len(axes_t)):
        fig_t.delaxes(axes_t[j])
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_t.suptitle("Averaged Metrics over Timesteps (SiT vs REPA) - All Configurations", fontsize=20, fontweight='bold')
    
    out_path_step = out_dir / "all_metrics_timestep_mean.png"
    plt.savefig(out_path_step, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Timestep trend plot saved to: {out_path_step}")

if __name__ == "__main__":
    sit_file = "/Users/nguyenthanhlam/experiemnts/outputs/sit_imagenet_metrics/metrics.tsv"
    repa_file = "/Users/nguyenthanhlam/experiemnts/outputs/repa_imagenet_metrics/metrics.tsv"
    out_dir = Path("/Users/nguyenthanhlam/experiemnts/outputs")
    plot_all_metrics(sit_file, repa_file, out_dir)
