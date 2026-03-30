import sys
from pathlib import Path

# Try to import necessary viz libs
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from IPython.display import display
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install them: !pip install pandas matplotlib numpy pillow ipython")
    sys.exit(1)

# Default path assuming script is in the repo root
_SCRIPT_DIR = Path(__file__).resolve().parent
EXP1_DIR = _SCRIPT_DIR / "outputs" / "exp1"

def set_exp1_dir(path_str: str):
    """Override the default outputs/exp1 directory path if needed."""
    global EXP1_DIR
    EXP1_DIR = Path(path_str)


def plot_loss_curves(timesteps=None):
    """
    Plots the MSE loss vs Layer index for SiT and REPA.
    If timesteps is None, plots all timesteps found in the CSV.
    Creates a grid of subplots (one for each timestep) to side-by-side compare SiT and REPA.
    """
    sit_csv = EXP1_DIR / "sit_frozen_ffn_loss.csv"
    repa_csv = EXP1_DIR / "repa_frozen_ffn_loss.csv"
    
    if not sit_csv.exists() or not repa_csv.exists():
        print(f"Error: CSV files not found in {EXP1_DIR}. Please run exp1 first.")
        return
        
    df_sit = pd.read_csv(sit_csv, index_col=0)
    df_repa = pd.read_csv(repa_csv, index_col=0)
    
    # Identify timesteps from columns if not provided
    if timesteps is None:
        timesteps_str = [c for c in df_sit.columns if c.startswith("t=")]
    else:
        timesteps_str = [f"t={t}" for t in timesteps]
        
    n_plots = len(timesteps_str)
    cols = min(n_plots, 3)
    rows = int(np.ceil(n_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()
    
    for i, t_col in enumerate(timesteps_str):
        ax = axes[i]
        
        if t_col in df_sit.columns:
            ax.plot(df_sit.index, df_sit[t_col], label="SiT", marker='o', markersize=4, linestyle='-', linewidth=2)
        if t_col in df_repa.columns:
            ax.plot(df_repa.index, df_repa[t_col], label="REPA", marker='s', markersize=4, linestyle='-', linewidth=2)
            
        ax.set_title(f"Loss at {t_col}")
        ax.set_xlabel("Layer Index (0 to 27)")
        ax.set_ylabel("MSE (pred vs velocity target)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.show()


def show_clean_reference(backend="sit"):
    """Displays the original clean input images (x0) that were passed into the noisy forward pass."""
    path = EXP1_DIR / "images" / backend / "00_reference_clean_x0.png"
    if path.exists():
        print(f"Clean reference images (x0):")
        display(Image.open(str(path)))
    else:
        print(f"Error: {path} not found.")


def compare_layer(layer: int, t: float):
    """
    Displays the decoded image prediction of a specific layer at a specific timestep,
    horizontally concatenating SiT (left) and REPA (right) for direct visual comparison.
    """
    p_sit = EXP1_DIR / "images" / "sit" / f"t{t:.1f}" / f"layer{layer:02d}.png"
    p_repa = EXP1_DIR / "images" / "repa" / f"t{t:.1f}" / f"layer{layer:02d}.png"
    
    img_sit = Image.open(str(p_sit)) if p_sit.exists() else None
    img_repa = Image.open(str(p_repa)) if p_repa.exists() else None
    
    if img_sit is None and img_repa is None:
        print(f"Error: Neither SiT nor REPA images found for L{layer} at t={t}")
        return
        
    # Create combined image
    w_sit, h_sit = img_sit.size if img_sit else (0, 0)
    w_repa, h_repa = img_repa.size if img_repa else (0, 0)
    
    total_w = w_sit + w_repa
    max_h = max(h_sit, h_repa)
    
    canvas = Image.new('RGB', (total_w, max_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    if img_sit:
        canvas.paste(img_sit, (0, 0))
        draw.text((4, 4), "SiT", fill=(255, 0, 0)) # Red text for visibility
        
    if img_repa:
        canvas.paste(img_repa, (w_sit, 0))
        draw.text((w_sit + 4, 4), "REPA", fill=(0, 255, 0)) # Green text for visibility
        
    print(f"Comparison at Layer {layer}, t = {t} (Left: SiT, Right: REPA)")
    display(canvas)
    
    
def compare_summary(t: float):
    """
    Displays the ENTIRE layers summary grid at a specific timestep t.
    Horizontally concatenates SiT (left) and REPA (right) so you can scroll
    down through all 28 layers and see SiT vs REPA side-by-side.
    """
    p_sit = EXP1_DIR / "images" / "sit" / f"summary_t{t:.1f}.png"
    p_repa = EXP1_DIR / "images" / "repa" / f"summary_t{t:.1f}.png"
    
    img_sit = Image.open(str(p_sit)) if p_sit.exists() else None
    img_repa = Image.open(str(p_repa)) if p_repa.exists() else None
    
    if img_sit is None and img_repa is None:
        print(f"Error: Neither SiT nor REPA summary grids found for t={t}")
        return
        
    w_sit, h_sit = img_sit.size if img_sit else (0, 0)
    w_repa, h_repa = img_repa.size if img_repa else (0, 0)
    
    total_w = w_sit + w_repa + 20 # 20px gap
    max_h = max(h_sit, h_repa)
    
    canvas = Image.new('RGB', (total_w, max_h), (200, 200, 200)) # Grey gap
    draw = ImageDraw.Draw(canvas)
    
    if img_sit:
        canvas.paste(img_sit, (0, 0))
    if img_repa:
        canvas.paste(img_repa, (w_sit + 20, 0))
        
    print(f"Summary Grid at t = {t}")
    print(f"{' '*10}SiT{' '*60}REPA")
    display(canvas)


def show_layer_evolution(layer: int, backend: str = "sit"):
    """
    Shows how a specific layer evolves across all evaluated timesteps.
    Displays the 'layerXX_all_timesteps.png' summary.
    """
    path = EXP1_DIR / "images" / backend / f"layer{layer:02d}_all_timesteps.png"
    if path.exists():
        print(f"Evolution of Layer {layer:02d} for {backend.upper()} across timesteps:")
        display(Image.open(str(path)))
    else:
        print(f"Error: {path} not found.")


def _compute_laplacian_variance(img_path: Path) -> float:
    """Computes the variance of the Laplacian (proxy for high-frequency details/sharpness)."""
    img = Image.open(str(img_path)).convert('L')
    gray = np.array(img, dtype=np.float32)
    # 3x3 Discrete Laplacian convolution (edges)
    lap = (gray[0:-2, 1:-1] + gray[2:, 1:-1] + 
           gray[1:-1, 0:-2] + gray[1:-1, 2:] - 
           4 * gray[1:-1, 1:-1])
    return float(np.var(lap))


def plot_hf_curves(timesteps=None):
    """
    Measures and plots High-Frequency Detail (Laplacian Variance) for SiT vs REPA
    by directly processing the saved output images.
    If timesteps is None, it scans all 'tX.X' directories found.
    """
    backends = ["sit", "repa"]
    data = {"sit": {}, "repa": {}}
    
    # Scan available timesteps from directory structure if not provided
    if timesteps is None:
        t_dirs = set()
        for b in backends:
            b_dir = EXP1_DIR / "images" / b
            if b_dir.exists():
                t_dirs.update([d.name for d in b_dir.glob("t*") if d.is_dir()])
        timesteps_str = sorted(list(t_dirs))
    else:
        timesteps_str = [f"t{t:.1f}" for t in timesteps]

    if not timesteps_str:
        print(f"Error: No image directories found in {EXP1_DIR}/images")
        return

    # Compute scores
    for b in backends:
        for t_str in timesteps_str:
            t_dir = EXP1_DIR / "images" / b / t_str
            if not t_dir.exists():
                continue
                
            layer_files = sorted(list(t_dir.glob("layer*.png")))
            if not layer_files:
                continue
                
            data[b][t_str] = {}
            for lf in layer_files:
                try:
                    # Extract layer integer from filename 'layer05.png'
                    layer_idx = int(lf.stem.replace("layer", ""))
                    data[b][t_str][layer_idx] = _compute_laplacian_variance(lf)
                except ValueError:
                    pass

    # Plot
    n_plots = len(timesteps_str)
    cols = min(n_plots, 3)
    rows = int(np.ceil(n_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()
    
    for i, t_str in enumerate(timesteps_str):
        ax = axes[i]
        
        if t_str in data["sit"] and data["sit"][t_str]:
            layers = sorted(data["sit"][t_str].keys())
            scores = [data["sit"][t_str][l] for l in layers]
            ax.plot(layers, scores, label="SiT", marker='o', markersize=4, linestyle='-', linewidth=2)
            
        if t_str in data["repa"] and data["repa"][t_str]:
            layers = sorted(data["repa"][t_str].keys())
            scores = [data["repa"][t_str][l] for l in layers]
            ax.plot(layers, scores, label="REPA", marker='s', markersize=4, linestyle='-', linewidth=2)
            
        ax.set_title(f"HF Detail (Sharpness) at {t_str}")
        ax.set_xlabel("Layer Index (0 to 27)")
        ax.set_ylabel("Laplacian Variance (Higher = Sharper)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
    for j in range(len(timesteps_str), len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("This module is intended to be imported into a Jupyter Notebook / Colab cell:")
    print("  import analyze_exp1 as viz")
    print("  viz.plot_loss_curves()")
    print("  viz.plot_hf_curves()            # <--- New: plot sharpness/HF details from saved images")
    print("  viz.compare_summary(t=0.5)")
    print("  viz.compare_layer(layer=14, t=0.5)")

