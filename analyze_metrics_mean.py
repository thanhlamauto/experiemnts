import pandas as pd
from pathlib import Path

def analyze_tsv(path_str, model_name):
    path = Path(path_str)
    if not path.exists():
        print(f"\n[Warning] File not found: {path}")
        return
        
    df = pd.read_csv(path, sep='\t')
    if df.empty:
        print(f"\n[Warning] File is empty: {path}")
        return

    # Aggregate by layer
    mean_by_layer = df.groupby(['metric', 'layer'])['value'].mean().reset_index()
    # Aggregate by timestep
    mean_by_step = df.groupby(['metric', 'timestep'])['value'].mean().reset_index()

    print(f"\n{'='*70}")
    print(f"                     MODEL: {model_name}")
    print(f"{'='*70}")

    metrics = sorted(df['metric'].unique())

    print("\n[ 1. LAYER TRENDS (Averaged across all timesteps) ]")
    for m in metrics:
        subset = mean_by_layer[mean_by_layer['metric'] == m]
        sorted_asc = subset.sort_values('layer', ascending=True)
        all_layers = " | ".join([f"L{int(row['layer']):02d}({row['value']:.3g})" for _, row in sorted_asc.iterrows()])
        
        print(f"➤ {m.upper()}")
        print(f"    All Layers: {all_layers}")

    print("\n[ 2. TIMESTEP TRENDS (Averaged across all layers) ]")
    for m in metrics:
        subset = mean_by_step[mean_by_step['metric'] == m]
        sorted_asc = subset.sort_values('timestep', ascending=True)
        all_steps = " | ".join([f"t={row['timestep']:.2f}({row['value']:.3g})" for _, row in sorted_asc.iterrows()])
        
        print(f"➤ {m.upper()}")
        print(f"    All Timesteps: {all_steps}")

if __name__ == "__main__":
    sit_file = "/Users/nguyenthanhlam/experiemnts/outputs/sit_imagenet_metrics/metrics.tsv"
    repa_file = "/Users/nguyenthanhlam/experiemnts/outputs/repa_imagenet_metrics/metrics.tsv"
    
    analyze_tsv(sit_file, "SiT (Vanilla)")
    analyze_tsv(repa_file, "REPA")
