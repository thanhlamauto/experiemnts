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

    # 3. Detailed Breakdown Plotting (New)
    fig, axes = plt.subplots(num_cats := len(categories), 1, figsize=(12, 5 * num_cats))
    plt.style.use('dark_background')
    fig.set_facecolor('#0a0a0a')
    
    for i, cat in enumerate(categories):
        ax = axes[i]
        cat_metrics = df[df['category'] == cat]['metric'].unique()
        
        for m in cat_metrics:
            sub = df[(df['category'] == cat) & (df['metric'] == m)].groupby('layer')['score'].mean().reset_index().sort_values('layer')
            ax.plot(sub['layer'], sub['score'] * 100, label=m.upper(), marker='o', alpha=0.8, linewidth=2)
        
        ax.set_title(f"{model_name}: {cat} Detailed Metrics", fontweight='bold', fontsize=14)
        ax.set_ylabel("Normalized Score (%)")
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', frameon=True, alpha=0.5)
        ax.set_ylim(0, 105)

    plt.tight_layout()
    out_img_detail = out_dir / f"detailed_breakdown_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(out_img_detail, dpi=200, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print(f"Detailed breakdown saved to {out_img_detail}")

    # 4. Summary Plotting (Original style but improved)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = {'Coarse (Semantic)': '#e74c3c', 'Fine (Texture/Local)': '#2ecc71', 'Spatial (Global Context)': '#3498db'}
    
    for cat in categories:
        sub = layer_scores[layer_scores['category'] == cat].sort_values('layer')
        axes[0].plot(sub['layer'], sub['score'] * 100, label=cat, color=colors[cat], marker='o', linewidth=3)
    axes[0].set_title(f"{model_name}: Roles over Layers (Aggregated)", fontweight='bold')
    axes[0].set_ylabel("Composite Score (%)")
    axes[0].set_xlabel("Layer")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for cat in categories:
        sub = step_scores[step_scores['category'] == cat].sort_values('timestep')
        axes[1].plot(sub['timestep'], sub['score'] * 100, label=cat, color=colors[cat], marker='s', linewidth=3)
    axes[1].set_title(f"{model_name}: Roles over Timesteps (Aggregated)", fontweight='bold')
    axes[1].set_xlabel("Timestep (t)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    out_img = out_dir / f"coarse_fine_spatial_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(out_img, dpi=200, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print(f"\nPlot saved to {out_img}")

if __name__ == '__main__':
    # Use relative paths so it works on both local and Colab
    base_dir = Path(__file__).resolve().parent
    sit_file = base_dir / "outputs/sit_imagenet_metrics/metrics.tsv"
    repa_file = base_dir / "outputs/repa_imagenet_metrics/metrics.tsv"
    
    out_dir_local = base_dir / "outputs"
    out_dir_local.mkdir(parents=True, exist_ok=True)
    
    # Run for local outputs
    compute_rankings(sit_file, "SiT Vanilla", out_dir_local)
    compute_rankings(repa_file, "REPA", out_dir_local)
    
    # Save mirror to artifacts ONLY if the directory exists (local development)
    artifact_dir = Path("/Users/nguyenthanhlam/.gemini/antigravity/brain/7eb263dd-b701-408d-9cbb-2a643483db90")
    if artifact_dir.exists():
        compute_rankings(sit_file, "SiT Vanilla", artifact_dir)
        compute_rankings(repa_file, "REPA", artifact_dir)
