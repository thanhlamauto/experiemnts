#!/usr/bin/env bash
# Chạy metrics SiT vanilla + REPA trên NHIỀU timesteps để đo đạc sự thay đổi qua thời gian.
set -euo pipefail

ROOT="${ROOT:-$PWD}"
IMAGENET_ROOT="${IMAGENET_ROOT:-$HOME/data/mini_imagenet_folder}"
SEED="${SEED:-0}"
METRICS_PRESET="${METRICS_PRESET:-full}"

# Multi-timesteps configuration
TIMESTEPS="0.1,0.3,0.5,0.7,0.9"

FULL_METRICS="sanity,linear,knn,cka,cknna,nc1,ncm_acc,etf_dev,participation_ratio,effective_rank,mad,entropy,decay,hf"
GEOMETRY_ONLY_METRICS="nc1,ncm_acc,etf_dev,participation_ratio,effective_rank"
if [[ -z "${METRICS:-}" ]]; then
  case "$METRICS_PRESET" in
    full) METRICS="$FULL_METRICS" ;;
    geometry-only) METRICS="$GEOMETRY_ONLY_METRICS" ;;
    *) echo "[error] Unknown METRICS_PRESET" >&2; exit 1 ;;
  esac
fi

CLASS_GEOMETRY_TRAIN_PER_CLASS="${CLASS_GEOMETRY_TRAIN_PER_CLASS:-50}"
CLASS_GEOMETRY_VAL_PER_CLASS="${CLASS_GEOMETRY_VAL_PER_CLASS:-20}"
CLASS_GEOMETRY_SUBSET_SEED="${CLASS_GEOMETRY_SUBSET_SEED:-0}"
NCM_METRIC="${NCM_METRIC:-cosine}"

SIT_CKPT="${SIT_CKPT:-$ROOT/SiT/pretrained_models/SiT-XL-2-256x256.pt}"
REPA_CKPT="${REPA_CKPT:-$ROOT/REPA/pretrained_models/last.pt}"

echo "== Checking checkpoints =="
# Xoá file bị lỗi (kích thước quá nhỏ) do phiên bản wget dỏm tải nhầm HTML
if [ -f "$SIT_CKPT" ]; then
  SIT_SIZE=$(wc -c < "$SIT_CKPT" || stat -f%z "$SIT_CKPT")
  if [ "$SIT_SIZE" -lt 1000000 ]; then rm -f "$SIT_CKPT"; fi
fi

if [ -f "$REPA_CKPT" ]; then
  REPA_SIZE=$(wc -c < "$REPA_CKPT" || stat -f%z "$REPA_CKPT")
  if [ "$REPA_SIZE" -lt 1000000 ]; then rm -f "$REPA_CKPT"; fi
fi

if [ ! -f "$SIT_CKPT" ]; then
  echo "Downloading SiT checkpoint to $SIT_CKPT..."
  mkdir -p "$(dirname "$SIT_CKPT")"
  $PYTHON -c "import urllib.request; urllib.request.urlretrieve('https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=1', '$SIT_CKPT')"
fi

if [ ! -f "$REPA_CKPT" ]; then
  echo "Downloading REPA checkpoint to $REPA_CKPT..."
  mkdir -p "$(dirname "$REPA_CKPT")"
  $PYTHON -c "import urllib.request; urllib.request.urlretrieve('https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=1', '$REPA_CKPT')"
fi

OUT_SIT="${OUT_SIT:-$ROOT/outputs/sit_imagenet_metrics}"
OUT_REPA="${OUT_REPA:-$ROOT/outputs/repa_imagenet_metrics}"

PYTHON="${PYTHON:-python3}"

echo "== SiT (vanilla) [MULTI-TIMESTEPS] -> $OUT_SIT"
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
  --timesteps "$TIMESTEPS" \
  --seed "$SEED"

echo "== REPA [MULTI-TIMESTEPS] -> $OUT_REPA"
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
  --timesteps "$TIMESTEPS" \
  --seed "$SEED"

echo "Done multi-timestep generations!"
