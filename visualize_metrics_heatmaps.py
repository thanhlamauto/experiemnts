import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Categories mapping
COARSE = ['cka', 'cknna', 'linear_top1', 'ncm_acc', 'participation_ratio', 'effective_rank']
FINE = ['hf_ratio', 'decay_slope']
SPATIAL = ['mad', 'entropy']
LOWER_IS_BETTER = ['decay_slope']

def normalize_metric(df):
    df_norm = df.copy()
    for m in df_norm['metric'].unique():
        mask = df_norm['metric'] == m
        vals = df_norm.loc[mask, 'value']
        v_min, v_max = vals.min(), vals.max()
        if v_max > v_min:
            scores = (vals - v_min) / (v_max - v_min)
            if m in LOWER_IS_BETTER: scores = 1.0 - scores
            df_norm.loc[mask, 'score'] = scores
        else:
            df_norm.loc[mask, 'score'] = 1.0
    return df_norm

def create_heatmaps(tsv_file, model_name, out_dir):
    path = Path(tsv_file)
    if not path.exists(): return
    df = pd.read_csv(path, sep='\t')
    if df.empty: return
    
    df = normalize_metric(df)
    
    groups = {
        'COARSE (Semantic)': [m for m in COARSE if m in df['metric'].unique()],
        'FINE (Texture/Details)': [m for m in FINE if m in df['metric'].unique()],
        'SPATIAL (Context)': [m for m in SPATIAL if m in df['metric'].unique()]
    }

    plt.style.use('dark_background')
    
    for g_name, m_list in groups.items():
        if not m_list: continue
        
        num_metrics = len(m_list)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 8))
        if num_metrics == 1: axes = [axes]
        
        fig.patch.set_facecolor('#050505')
        
        for idx, m in enumerate(m_list):
            ax = axes[idx]
            # Pivot to matrix: Rows=Layer, Cols=Timestep
            sub = df[df['metric'] == m].pivot(index='layer', columns='timestep', values='score')
            # Sort layers descending (top layer at top of plot)
            sub = sub.sort_index(ascending=False)
            
            sns.heatmap(sub, ax=ax, cmap='magma', cbar_kws={'label': 'Normalized Score'}, 
                        annot=False, linewidths=0.05)
            
            ax.set_title(f"{m.upper()}", fontweight='bold', fontsize=14, color='cyan')
            ax.set_ylabel("Layer Index" if idx == 0 else "")
            ax.set_xlabel("Timestep (t)")

        plt.suptitle(f"{model_name}: {g_name} Heatmap Atlas", fontsize=20, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        save_path = out_dir / f"heatmap_{model_name.lower().replace(' ', '_')}_{g_name.split(' ')[0].lower()}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#050505')
        plt.close()
        print(f"Saved heatmap atlas: {save_path}")

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "outputs/visualizations/heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    create_heatmaps(base_dir / "outputs/sit_imagenet_metrics/metrics.tsv", "SiT Vanilla", out_dir)
    create_heatmaps(base_dir / "outputs/repa_imagenet_metrics/metrics.tsv", "REPA", out_dir)
