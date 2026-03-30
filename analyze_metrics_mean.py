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
        sorted_desc = subset.sort_values('value', ascending=False)
        top3 = " | ".join([f"L{int(row['layer']):02d} ({row['value']:.3g})" for _, row in sorted_desc.head(3).iterrows()])
        bottom3 = " | ".join([f"L{int(row['layer']):02d} ({row['value']:.3g})" for _, row in sorted_desc.tail(3).iterrows()])
        
        print(f"➤ {m.upper()}")
        print(f"    Highest (Top 3) : {top3}")
        print(f"    Lowest (Bot 3)  : {bottom3}")

    print("\n[ 2. TIMESTEP TRENDS (Averaged across all layers) ]")
    for m in metrics:
        subset = mean_by_step[mean_by_step['metric'] == m]
        sorted_desc = subset.sort_values('value', ascending=False)
        top3 = " | ".join([f"t={row['timestep']:.2f} ({row['value']:.3g})" for _, row in sorted_desc.head(3).iterrows()])
        bottom3 = " | ".join([f"t={row['timestep']:.2f} ({row['value']:.3g})" for _, row in sorted_desc.tail(3).iterrows()])
        
        print(f"➤ {m.upper()}")
        print(f"    Highest (Top 3) : {top3}")
        print(f"    Lowest (Bot 3)  : {bottom3}")

if __name__ == "__main__":
    sit_file = "/Users/nguyenthanhlam/experiemnts/outputs/sit_imagenet_metrics/metrics.tsv"
    repa_file = "/Users/nguyenthanhlam/experiemnts/outputs/repa_imagenet_metrics/metrics.tsv"
    
    analyze_tsv(sit_file, "SiT (Vanilla)")
    analyze_tsv(repa_file, "REPA")
