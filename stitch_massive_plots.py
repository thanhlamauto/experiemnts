#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="outputs/massive_activations")
    parser.add_argument("--backend", type=str, default="repa", help="sit or repa")
    parser.add_argument("--layers", type=str, default=",".join(map(str, range(28))))
    parser.add_argument("--timesteps", type=str, default="0.1,0.2,0.3,0.5,0.7,0.8,0.9")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(args.dir)
    layers = [int(l.strip()) for l in args.layers.split(",")]
    t_vals = [float(t.strip()) for t in args.timesteps.split(",")]
    backend = args.backend.lower()

    if not args.output:
        args.output = base_dir / f"super_composite_{backend}.png"

    print(f"Stitching {len(layers)} layers x {len(t_vals)} timesteps for {backend}...")
    
    # 1. Peek at one image to get dimensions
    test_img_path = None
    for l in layers:
        for t in t_vals:
            t_str = f"{t:.1f}".replace(".", "p")
            p = base_dir / f"massive_{backend}_L{l:02d}_t{t_str}.png"
            if p.exists():
                test_img_path = p
                break
        if test_img_path: break
    
    if not test_img_path:
        print(f"[error] No images found for backend {backend} in {base_dir}")
        return

    sample_img = Image.open(test_img_path)
    w, h = sample_img.size
    print(f"Sample image size: {w}x{h}")

    # 2. Create Canvas (Black background)
    margin_top = 200
    margin_left = 300
    canvas_w = margin_left + len(t_vals) * w
    canvas_h = margin_top + len(layers) * h
    
    # Limit size if too large for PIL/Memory
    if canvas_w * canvas_h > 20000 * 20000:
        print("[warn] Canvas too large, downscaling sample images...")
        scale = 0.5
        w, h = int(w * scale), int(h * scale)
        canvas_w = margin_left + len(t_vals) * w
        canvas_h = margin_top + len(layers) * h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a font, fallback to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 80)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 60)
    except:
        font_large = font_small = ImageFont.load_default()

    # 3. Draw Headers
    draw.text((margin_left + 20, 50), f"MASSIVE ACTIVATIONS: {backend.upper()} (All Layers & Timesteps)", fill="white", font=font_large)
    
    for i, t in enumerate(t_vals):
        tx = margin_left + i * w + w // 2 - 50
        draw.text((tx, margin_top - 100), f"t = {t}", fill="cyan", font=font_small)

    # 4. Paste Images
    for r, l in enumerate(layers):
        # Draw Layer Label
        draw.text((50, margin_top + r * h + h // 2 - 30), f"LAYER {l}", fill="yellow", font=font_small)
        
        for c, t in enumerate(t_vals):
            t_str = f"{t:.1f}".replace(".", "p")
            img_path = base_dir / f"massive_{backend}_L{l:02d}_t{t_str}.png"
            
            if img_path.exists():
                img = Image.open(img_path)
                if w != img.size[0]:
                    img = img.resize((w, h), Image.Resampling.LANCZOS)
                canvas.paste(img, (margin_left + c * w, margin_top + r * h))
            else:
                draw.text((margin_left + c * w + 50, margin_top + r * h + 50), "MISSING", fill="red")

    # 5. Save
    print(f"Saving super composite to {args.output}...")
    canvas.save(args.output, optimize=True, quality=85)
    print("Done!")

if __name__ == "__main__":
    main()
