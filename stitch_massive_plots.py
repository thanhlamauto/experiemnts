#!/usr/bin/env python3
"""
Stitch together massive activation plots for SiT and REPA into a grid.
"""
from PIL import Image
import os
from pathlib import Path
from typing import List

def create_summary_grid(outdir: str, layers: List[int], output_name="massive_summary_grid.png"):
    outdir = Path(outdir)
    backends = ['sit', 'repa']
    
    # Collect available images
    rows = []
    for l_idx in layers:
        row_imgs = []
        for backend in backends:
            img_path = outdir / f"massive_{backend}_L{l_idx:02d}.png"
            if img_path.exists():
                row_imgs.append(Image.open(img_path))
            else:
                print(f"[warn] Missing {img_path}")
        if row_imgs:
            rows.append(row_imgs)
            
    if not rows:
        print("No images found to stitch.")
        return
    
    # Calculate grid size
    num_rows = len(rows)
    num_cols = len(rows[0])
    
    w, h = rows[0][0].size
    grid_img = Image.new('RGB', (w * num_cols, h * num_rows), (0, 0, 0))
    
    for r_idx, row in enumerate(rows):
        for c_idx, img in enumerate(row):
            grid_img.paste(img, (c_idx * w, r_idx * h))
            
    save_path = outdir / output_name
    grid_img.save(save_path)
    print(f"Summary grid saved to {save_path}")
    return save_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs/massive_activations")
    parser.add_argument("--layers", type=str, default="2,12,25")
    args = parser.parse_args()
    
    layers = [int(x) for x in args.layers.split(",")]
    create_summary_grid(args.outdir, layers)
