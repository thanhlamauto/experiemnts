#!/usr/bin/env bash
# Chạy metrics SiT vanilla + REPA, sau đó vẽ so sánh (PNG/PDF).
set -euo pipefail

ROOT="${ROOT:-/workspace}"
IMAGENET_ROOT="${IMAGENET_ROOT:-$HOME/data/mini_imagenet_folder}"
SEED="${SEED:-0}"
METRICS_PRESET="${METRICS_PRESET:-full}"
FULL_METRICS="sanity,linear,knn,cka,cknna,nc1,ncm_acc,etf_dev,participation_ratio,effective_rank,mad,entropy,decay,hf"
GEOMETRY_ONLY_METRICS="nc1,ncm_acc,etf_dev,participation_ratio,effective_rank"
if [[ -z "${METRICS:-}" ]]; then
  case "$METRICS_PRESET" in
    full)
      METRICS="$FULL_METRICS"
      ;;
    geometry-only)
      METRICS="$GEOMETRY_ONLY_METRICS"
      ;;
    *)
      echo "[error] Unknown METRICS_PRESET='$METRICS_PRESET'. Use 'full' or 'geometry-only'." >&2
      exit 1
      ;;
  esac
fi
CLASS_GEOMETRY_TRAIN_PER_CLASS="${CLASS_GEOMETRY_TRAIN_PER_CLASS:-50}"
CLASS_GEOMETRY_VAL_PER_CLASS="${CLASS_GEOMETRY_VAL_PER_CLASS:-20}"
CLASS_GEOMETRY_SUBSET_SEED="${CLASS_GEOMETRY_SUBSET_SEED:-0}"
NCM_METRIC="${NCM_METRIC:-cosine}"

SIT_CKPT="${SIT_CKPT:-$ROOT/SiT/pretrained_models/SiT-XL-2-256x256.pt}"
REPA_CKPT="${REPA_CKPT:-$ROOT/REPA/pretrained_models/last.pt}"

OUT_SIT="${OUT_SIT:-$ROOT/outputs/sit_imagenet_metrics}"
OUT_REPA="${OUT_REPA:-$ROOT/outputs/repa_imagenet_metrics}"
OUT_PLOTS="${OUT_PLOTS:-$ROOT/outputs/metrics_compare_plots}"

PYTHON="${PYTHON:-python3}"

echo "== Compare preset: $METRICS_PRESET"
echo "== Metrics: $METRICS"

echo "== SiT (vanilla) → $OUT_SIT"
rm -f "$OUT_SIT/metrics.tsv"
$PYTHON "$ROOT/run_sit_imagenet_metrics.py" \
  --backend sit \
  --ckpt "$SIT_CKPT" \
  --sit-root "$ROOT/SiT" \
  --imagenet-root "$IMAGENET_ROOT" \
  --outdir "$OUT_SIT" \
  --metrics "$METRICS" \
  --model-num-classes 1000 \
  --num-classes 64 \
  --resolution 256 \
  --vae mse \
  --class-geometry-train-per-class "$CLASS_GEOMETRY_TRAIN_PER_CLASS" \
  --class-geometry-val-per-class "$CLASS_GEOMETRY_VAL_PER_CLASS" \
  --class-geometry-subset-seed "$CLASS_GEOMETRY_SUBSET_SEED" \
  --ncm-metric "$NCM_METRIC" \
  --layers all \
  --probe-layers all \
  --seed "$SEED"

echo "== REPA → $OUT_REPA"
rm -f "$OUT_REPA/metrics.tsv"
$PYTHON "$ROOT/run_sit_imagenet_metrics.py" \
  --backend repa \
  --ckpt "$REPA_CKPT" \
  --repa-root "$ROOT/REPA" \
  --imagenet-root "$IMAGENET_ROOT" \
  --outdir "$OUT_REPA" \
  --metrics "$METRICS" \
  --model-num-classes 1000 \
  --num-classes 64 \
  --resolution 256 \
  --encoder-depth 8 \
  --projector-embed-dims 768 \
  --vae mse \
  --class-geometry-train-per-class "$CLASS_GEOMETRY_TRAIN_PER_CLASS" \
  --class-geometry-val-per-class "$CLASS_GEOMETRY_VAL_PER_CLASS" \
  --class-geometry-subset-seed "$CLASS_GEOMETRY_SUBSET_SEED" \
  --ncm-metric "$NCM_METRIC" \
  --layers all \
  --probe-layers all \
  --seed "$SEED"

echo "== Plots → $OUT_PLOTS"
$PYTHON "$ROOT/plot_imagenet_metrics_compare.py" \
  --repa "$OUT_REPA/metrics.tsv" \
  --sit "$OUT_SIT/metrics.tsv" \
  --outdir "$OUT_PLOTS"

echo "Done. Xem: $OUT_PLOTS/repa_vs_sit_metrics_layers.png (và _probes, _geometry, _overview)"
