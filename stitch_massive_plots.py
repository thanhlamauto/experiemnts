#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="outputs/massive_activations")
    parser.add_argument("--backends", type=str, default="sit,repa", help="comma separated backends")
    parser.add_argument("--layers", type=str, default=",".join(map(str, range(28))))
    parser.add_argument("--timesteps", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(args.dir)
    layers = [int(l.strip()) for l in args.layers.split(",")]
    t_vals = [float(t.strip()) for t in args.timesteps.split(",")]
    backends = [b.strip().lower() for b in args.backends.split(",")]

    if not args.output:
        tags = "_vs_".join(backends)
        args.output = base_dir / f"final_massive_comparison_{tags}.png"

    print(f"Stitching {len(layers)} layers x {len(t_vals)} timesteps for {backends}...")
    
    # 1. Identify sample image and sizing
    test_img_path = None
    for b in backends:
        for l in layers:
            for t in t_vals:
                t_str = f"{t:.1f}".replace(".", "p")
                p = base_dir / f"massive_{b}_L{l:02d}_t{t_str}.png"
                if p.exists():
                    test_img_path = p
                    break
            if test_img_path: break
        if test_img_path: break
    
    if not test_img_path:
        print(f"[error] No images found in {base_dir}")
        return

    sample_img = Image.open(test_img_path)
    w, h = sample_img.size
    
    # Dynamic Downscaling for safety
    max_dim = 15000 
    total_w_raw = len(t_vals) * len(backends) * w
    if total_w_raw > max_dim:
        scale = max_dim / total_w_raw
        print(f"[warn] Downscaling images to {scale:.2f}x for safety.")
        w, h = int(w * scale), int(h * scale)

    # 2. Layout Params
    margin_top = 250
    margin_left = 350
    row_h = h
    col_w = w * len(backends)
    
    canvas_w = margin_left + len(t_vals) * col_w
    canvas_h = margin_top + len(layers) * row_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 100)
        font_mid = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 70)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 50)
    except:
        font_large = font_mid = font_small = ImageFont.load_default()

    # 3. Draw Headers
    draw.text((margin_left + 50, 40), f"MASSIVE ACTIVATIONS COMPARISON: {' vs '.join(backends).upper()}", fill="white", font=font_large)
    
    for i, t in enumerate(t_vals):
        # Timestep Header
        center_x = margin_left + i * col_w + col_w // 2 - 100
        draw.text((center_x, margin_top - 180), f"Timestep t = {t}", fill="cyan", font=font_mid)
        
        # Backend Sub-headers
        for b_idx, b in enumerate(backends):
            bx = margin_left + i * col_w + b_idx * w + w // 2 - 50
            draw.text((bx, margin_top - 80), b.upper(), fill="white", font=font_small)

    # 4. Paste Images
    for r, l in enumerate(layers):
        draw.text((40, margin_top + r * h + h // 2 - 40), f"LAYER {l}", fill="yellow", font=font_mid)
        
        for i, t in enumerate(t_vals):
            t_str = f"{t:.1f}".replace(".", "p")
            for b_idx, b in enumerate(backends):
                img_path = base_dir / f"massive_{b}_L{l:02d}_t{t_str}.png"
                if img_path.exists():
                    img = Image.open(img_path)
                    if img.size[0] != w:
                        img = img.resize((w, h), Image.Resampling.LANCZOS)
                    canvas.paste(img, (margin_left + i * col_w + b_idx * w, margin_top + r * h))
                else:
                    draw.text((margin_left + i * col_w + b_idx * w + 50, margin_top + r * h + 50), "MISSING", fill="red")

    # 5. Save
    print(f"Saving final comparison to {args.output}...")
    canvas.save(args.output, optimize=True, quality=80)
    print("Done!")

if __name__ == "__main__":
    main()
