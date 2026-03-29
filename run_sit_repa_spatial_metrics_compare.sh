#!/usr/bin/env bash
# Chay spatial metrics per-layer cho SiT vanilla + REPA, roi ve so sanh.
set -euo pipefail

ROOT="${ROOT:-/workspace}"
IMAGENET_ROOT="${IMAGENET_ROOT:-$HOME/data/mini_imagenet_folder}"
SEED="${SEED:-0}"
METRICS="${METRICS:-lds,cds,rmsc,lgr,msdr,graph_gap,ubc,hf_ratio}"
NOISE_LEVELS="${NOISE_LEVELS:-1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.0}"
PSEUDO_MASK_NPZ="${PSEUDO_MASK_NPZ:-}"
PSEUDO_BACKGROUND_LABEL="${PSEUDO_BACKGROUND_LABEL:-}"

SIT_CKPT="${SIT_CKPT:-$ROOT/SiT/pretrained_models/SiT-XL-2-256x256.pt}"
REPA_CKPT="${REPA_CKPT:-$ROOT/REPA/pretrained_models/last.pt}"

OUT_SIT="${OUT_SIT:-$ROOT/outputs/sit_imagenet_spatial_metrics}"
OUT_REPA="${OUT_REPA:-$ROOT/outputs/repa_imagenet_spatial_metrics}"
OUT_PLOTS="${OUT_PLOTS:-$ROOT/outputs/spatial_metrics_compare_plots}"

PYTHON="${PYTHON:-/venv/main/bin/python}"

EXTRA_ARGS=()
if [[ -n "$PSEUDO_MASK_NPZ" ]]; then
  EXTRA_ARGS+=(--pseudo-mask-npz "$PSEUDO_MASK_NPZ")
fi
if [[ -n "$PSEUDO_BACKGROUND_LABEL" ]]; then
  EXTRA_ARGS+=(--pseudo-background-label "$PSEUDO_BACKGROUND_LABEL")
fi

echo "== SiT (vanilla) spatial metrics -> $OUT_SIT"
rm -f "$OUT_SIT/metrics.tsv"
"$PYTHON" "$ROOT/run_sit_imagenet_spatial_metrics.py" \
  --backend sit \
  --ckpt "$SIT_CKPT" \
  --sit-root "$ROOT/SiT" \
  --imagenet-root "$IMAGENET_ROOT" \
  --outdir "$OUT_SIT" \
  --metrics "$METRICS" \
  --noise-levels "$NOISE_LEVELS" \
  --model-num-classes 1000 \
  --resolution 256 \
  --vae mse \
  --layers all \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"

echo "== REPA spatial metrics -> $OUT_REPA"
rm -f "$OUT_REPA/metrics.tsv"
"$PYTHON" "$ROOT/run_sit_imagenet_spatial_metrics.py" \
  --backend repa \
  --ckpt "$REPA_CKPT" \
  --repa-root "$ROOT/REPA" \
  --imagenet-root "$IMAGENET_ROOT" \
  --outdir "$OUT_REPA" \
  --metrics "$METRICS" \
  --noise-levels "$NOISE_LEVELS" \
  --model-num-classes 1000 \
  --resolution 256 \
  --encoder-depth 8 \
  --projector-embed-dims 768 \
  --vae mse \
  --layers all \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"

echo "== Compare plots -> $OUT_PLOTS"
"$PYTHON" "$ROOT/plot_spatial_metrics_compare.py" \
  --repa "$OUT_REPA/metrics.tsv" \
  --sit "$OUT_SIT/metrics.tsv" \
  --outdir "$OUT_PLOTS"

echo "Done. Xem:"
echo "  $OUT_PLOTS/repa_vs_sit_spatial_grid.png"
echo "  $OUT_PLOTS/repa_vs_sit_spatial_delta_heatmap.png"
