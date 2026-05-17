#!/bin/bash
set -eu

# Build the Svelte frontend, then launch the Python demo backend.
# Override HOST, PORT, GPU_IDS, STEP, MODEL_TYPE, CONFIG_PATH,
# CHECKPOINT_FOLDER, ONLINE_BATCHING_MODE, and ONLINE_SLO_WAIT_THRESHOLD via
# environment variables.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
GPU_IDS="${GPU_IDS:-0}"
MODEL_TYPE="${MODEL_TYPE:-T2V-1.3B}"
USE_TAEHV="${USE_TAEHV:-0}"
USE_TENSORRT="${USE_TENSORRT:-0}"
FAST="${FAST:-0}"
ONLINE_BATCHING_MODE="${ONLINE_BATCHING_MODE:-batch}"
ONLINE_SLO_WAIT_THRESHOLD="${ONLINE_SLO_WAIT_THRESHOLD:-0.5}"

case "$MODEL_TYPE" in
  T2V-14B|14B|t2v-14b|14b)
    MODEL_TYPE="T2V-14B"
    CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/wan_causal_dmd_v2v_14b.yaml}"
    CHECKPOINT_FOLDER="${CHECKPOINT_FOLDER:-$PROJECT_ROOT/ckpts/wan_causal_dmd_v2v_14b}"
    STEP="${STEP:-1}"
    ;;
  *)
    CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/wan_causal_dmd_v2v.yaml}"
    CHECKPOINT_FOLDER="${CHECKPOINT_FOLDER:-$PROJECT_ROOT/ckpts/wan_causal_dmd_v2v}"
    STEP="${STEP:-2}"
    ;;
esac

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

TENSORRT_FLAG=""
case "$(printf '%s' "$USE_TENSORRT" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    TENSORRT_FLAG="--use_tensorrt"
    ;;
esac

FAST_FLAG=""
case "$(printf '%s' "$FAST" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    FAST_FLAG="--fast"
    ;;
esac

CUDA_VISIBLE_DEVICES="$GPU_IDS" python main.py \
  --port "$PORT" \
  --host "$HOST" \
  --num_gpus "$(printf '%s' "$GPU_IDS" | awk -F',' '{print NF}')" \
  --gpu_ids "$LOCAL_GPU_IDS" \
  --config_path "$CONFIG_PATH" \
  --checkpoint_folder "$CHECKPOINT_FOLDER" \
  --step "$STEP" \
  --model_type "$MODEL_TYPE" \
  --online_batching_mode "$ONLINE_BATCHING_MODE" \
  --online_slo_wait_threshold "$ONLINE_SLO_WAIT_THRESHOLD" \
  $TAEHV_FLAG \
  $TENSORRT_FLAG \
  $FAST_FLAG
