#!/usr/bin/env bash
# Chạy heatmap activation cho SiT vanilla + REPA, rồi ghép ảnh so sánh.
set -euo pipefail

ROOT="${ROOT:-/workspace}"
PYTHON="${PYTHON:-python3}"

SIT_CKPT="${SIT_CKPT:-$ROOT/SiT/pretrained_models/SiT-XL-2-256x256.pt}"
REPA_CKPT="${REPA_CKPT:-$ROOT/REPA/pretrained_models/last.pt}"

OUT_SIT="${OUT_SIT:-$ROOT/outputs/sit_activation_heatmap}"
OUT_REPA="${OUT_REPA:-$ROOT/outputs/repa_activation_heatmap}"
OUT_CMP="${OUT_CMP:-$ROOT/outputs/activation_heatmap_compare}"

# Thêm ví dụ: EXTRA_ARGS='--fast' hoặc EXTRA_ARGS='--per-noise-heatmaps'
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "== SiT vanilla → $OUT_SIT"
$PYTHON "$ROOT/plot_sit_xl2_activation_layers.py" \
  --ckpt "$SIT_CKPT" \
  --outdir "$OUT_SIT" \
  --backend sit \
  --sit-root "$ROOT/SiT" \
  --resolution 256 \
  --num-classes 1000 \
  $EXTRA_ARGS

echo "== REPA → $OUT_REPA"
$PYTHON "$ROOT/plot_sit_xl2_activation_layers.py" \
  --ckpt "$REPA_CKPT" \
  --outdir "$OUT_REPA" \
  --backend repa \
  --repa-root "$ROOT/REPA" \
  --resolution 256 \
  --num-classes 1000 \
  --encoder-depth 8 \
  --projector-embed-dims 768 \
  $EXTRA_ARGS

echo "== So sánh (side-by-side) → $OUT_CMP"
$PYTHON "$ROOT/plot_activation_heatmap_compare.py" \
  --sit "$OUT_SIT/activations_sit_xl2.png" \
  --repa "$OUT_REPA/activations_sit_xl2.png" \
  --out "$OUT_CMP/sit_vs_repa_activations.png"

echo "Xem: $OUT_CMP/sit_vs_repa_activations.png (và .pdf)"
