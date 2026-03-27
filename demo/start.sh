#!/bin/bash
set -eu

# Build the Svelte frontend, then launch the Python demo backend.
# Override HOST, PORT, GPU_IDS, STEP, and MODEL_TYPE via environment variables.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
GPU_IDS="${GPU_IDS:-0}"
STEP="${STEP:-1}"
MODEL_TYPE="${MODEL_TYPE:-T2V-1.3B}"
USE_TAEHV="${USE_TAEHV:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
LOCAL_GPU_IDS="$(seq 0 $((${#GPU_ARRAY[@]} - 1)) | paste -sd, -)"

cd "$FRONTEND_DIR"
npm install
npm run build
echo "frontend build success"

cd "$SCRIPT_DIR"
TAEHV_FLAG=""
case "$(printf '%s' "$USE_TAEHV" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    TAEHV_FLAG="--use_taehv"
    ;;
esac

CUDA_VISIBLE_DEVICES="$GPU_IDS" python main.py \
  --port "$PORT" \
  --host "$HOST" \
  --num_gpus "$(printf '%s' "$GPU_IDS" | awk -F',' '{print NF}')" \
  --gpu_ids "$LOCAL_GPU_IDS" \
  --step "$STEP" \
  --model_type "$MODEL_TYPE" \
  $TAEHV_FLAG
