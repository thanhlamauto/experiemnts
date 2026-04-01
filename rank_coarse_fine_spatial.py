import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Categories mapping
COARSE = ['cka', 'cknna', 'linear_top1', 'ncm_acc', 'participation_ratio', 'effective_rank']
FINE = ['hf_ratio', 'decay_slope']
SPATIAL = ['mad', 'entropy']

# Which metrics need inverted normalization (lower value meant stronger expression of that property)
# decay_slope is negative, lower (more negative) means stronger local detail decay.
LOWER_IS_BETTER = ['decay_slope']

def normalize_group(df_group):
    # Normalize each metric to 0-1 across all its rows
    df_norm = df_group.copy()
    
    for m in df_norm['metric'].unique():
        mask = df_norm['metric'] == m
        vals = df_norm.loc[mask, 'value']
        v_min, v_max = vals.min(), vals.max()
        
        if v_max > v_min:
            scores = (vals - v_min) / (v_max - v_min)
            if m in LOWER_IS_BETTER:
                scores = 1.0 - scores
            df_norm.loc[mask, 'score'] = scores
        else:
            df_norm.loc[mask, 'score'] = 1.0 if m not in LOWER_IS_BETTER else 0.0

    return df_norm

def compute_rankings(tsv_file, model_name, out_dir):
    path = Path(tsv_file)
    if not path.exists():
        print(f"[Warning] {path} not found.")
        return
    df = pd.read_csv(path, sep='\t')
    if df.empty: return
    
    df['score'] = 0.0
    df = normalize_group(df)
    
    def get_category(m):
        if m in COARSE: return 'Coarse (Semantic)'
        if m in FINE: return 'Fine (Texture/Local)'
        if m in SPATIAL: return 'Spatial (Global Context)'
        return None
        
    df['category'] = df['metric'].apply(get_category)
    df = df.dropna(subset=['category'])
    categories = ['Coarse (Semantic)', 'Fine (Texture/Local)', 'Spatial (Global Context)']
    
    # 1. Ranking by Layer
    layer_scores = df.groupby(['category', 'layer'])['score'].mean().reset_index()
    print(f"\n{'='*60}\n MODEL: {model_name} \n{'='*60}")
    print("\n--- TOP LAYERS FOR EACH PROPERTY ---")
    for cat in categories:
        subset = layer_scores[layer_scores['category'] == cat].sort_values('score', ascending=False)
        top = subset.head(3)
        top_str = " | ".join([f"L{int(row['layer']):02d} ({row['score']*100:.1f}%)" for _, row in top.iterrows()])
        print(f"➤ BEST for {cat.upper()}: {top_str}")

    # 2. Ranking by Timestep
    step_scores = df.groupby(['category', 'timestep'])['score'].mean().reset_index()
    print("\n--- TOP TIMESTEPS FOR EACH PROPERTY ---")
    for cat in categories:
        subset = step_scores[step_scores['category'] == cat].sort_values('score', ascending=False)
        top = subset.head(3)
        top_str = " | ".join([f"t={row['timestep']:.2f} ({row['score']*100:.1f}%)" for _, row in top.iterrows()])
        print(f"➤ BEST for {cat.upper()}: {top_str}")

    # 3. Comprehensive Breakdown Plotting (3 rows x 2 cols)
    plt.style.use('dark_background')
    fig, axes = plt.subplots(num_cats := len(categories), 2, figsize=(20, 6 * num_cats))
    fig.patch.set_facecolor('#050505')
    
    # Custom color palette for metrics
    colors_pool = ['#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4']

    for i, cat in enumerate(categories):
        # --- LEFT: METRIC vs LAYER ---
        ax_l = axes[i, 0]
        cat_metrics = df[df['category'] == cat]['metric'].unique()
        
        for m_idx, m in enumerate(cat_metrics):
            sub = df[(df['category'] == cat) & (df['metric'] == m)].groupby('layer')['score'].mean().reset_index().sort_values('layer')
            ax_l.plot(sub['layer'], sub['score'] * 100, label=m.upper(), 
                     marker='o', alpha=0.9, linewidth=3, color=colors_pool[m_idx % len(colors_pool)])
        
        ax_l.set_title(f"{cat}: Metrics vs Layer", fontweight='bold', fontsize=16, color='white')
        ax_l.set_ylabel("Normalized Score (%)", fontsize=12)
        ax_l.set_xlabel("Layer Index", fontsize=12)
        ax_l.grid(True, alpha=0.15, linestyle='--')
        ax_l.legend(loc='best', frameon=True, fontsize=10)
        ax_l.set_ylim(-5, 105)

        # --- RIGHT: METRIC vs TIMESTEP ---
        ax_r = axes[i, 1]
        for m_idx, m in enumerate(cat_metrics):
            sub = df[(df['category'] == cat) & (df['metric'] == m)].groupby('timestep')['score'].mean().reset_index().sort_values('timestep')
            ax_r.plot(sub['timestep'], sub['score'] * 100, label=m.upper(), 
                     marker='s', alpha=0.9, linewidth=3, color=colors_pool[m_idx % len(colors_pool)])
        
        ax_r.set_title(f"{cat}: Metrics vs Timestep (Noise)", fontweight='bold', fontsize=16, color='cyan')
        ax_r.set_xlabel("Timestep (t)", fontsize=12)
        ax_r.grid(True, alpha=0.15, linestyle='--')
        ax_r.legend(loc='best', frameon=True, fontsize=10)
        ax_r.set_ylim(-5, 105)

    plt.suptitle(f"Surgical Metric Analysis: {model_name.upper()}", fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_img_detail = out_dir / f"surgical_breakdown_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(out_img_detail, dpi=150, bbox_inches='tight', facecolor='#050505')
    plt.close()
    print(f"Surgical breakdown saved to {out_img_detail}")

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    sit_file = base_dir / "outputs/sit_imagenet_metrics/metrics.tsv"
    repa_file = base_dir / "outputs/repa_imagenet_metrics/metrics.tsv"
    
    out_dir_local = base_dir / "outputs/visualizations"
    out_dir_local.mkdir(parents=True, exist_ok=True)
    
    compute_rankings(sit_file, "SiT Vanilla", out_dir_local)
    compute_rankings(repa_file, "REPA", out_dir_local)
