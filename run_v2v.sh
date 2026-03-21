#!/bin/bash
set -eu

# Unified offline launcher for all supported video-to-video inference modes.
# Environment variables can override the default config, checkpoint, output, GPU,
# and launch parameters without having to edit the script itself.
MODE="${1:-single}"
if [ "$#" -gt 0 ]; then
  shift
fi

ROOT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-configs/wan_causal_dmd_v2v.yaml}"
CHECKPOINT_FOLDER="${CHECKPOINT_FOLDER:-ckpts/wan_causal_dmd_v2v}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-outputs/}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
FPS="${FPS:-16}"
STEP="${STEP:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"

case "$MODE" in
  single|batch)
    # Standard single-GPU batched inference.
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" python -m streamv2v.inference \
      --config_path "$ROOT_DIR/$CONFIG_PATH" \
      --checkpoint_folder "$ROOT_DIR/$CHECKPOINT_FOLDER" \
      --output_folder "$ROOT_DIR/$OUTPUT_FOLDER" \
      --prompt_file_path "$ROOT_DIR/${PROMPT_FILE_PATH:-examples/prompt.txt}" \
      --video_path "$ROOT_DIR/${VIDEO_PATH:-examples/original.mp4}" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --fps "$FPS" \
      --step "$STEP" \
      "$@"
    ;;
  single-wo|wo|wo-batch)
    # Single-GPU inference without batched denoising.
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}" python -m streamv2v.inference_wo_batch \
      --config_path "$ROOT_DIR/$CONFIG_PATH" \
      --checkpoint_folder "$ROOT_DIR/$CHECKPOINT_FOLDER" \
      --output_folder "$ROOT_DIR/$OUTPUT_FOLDER" \
      --prompt_file_path "$ROOT_DIR/${PROMPT_FILE_PATH:-examples/prompt.txt}" \
      --video_path "$ROOT_DIR/${VIDEO_PATH:-examples/original.mp4}" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --fps "$FPS" \
      --step "$STEP" \
      "$@"
    ;;
  pipe|parallel)
    # Pipeline-parallel inference across multiple GPUs on one node.
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}" torchrun \
      --nproc_per_node="${NPROC_PER_NODE:-2}" \
      --master_port="$MASTER_PORT" \
      --module streamv2v.inference_pipe \
      --config_path "$ROOT_DIR/$CONFIG_PATH" \
      --checkpoint_folder "$ROOT_DIR/$CHECKPOINT_FOLDER" \
      --output_folder "$ROOT_DIR/$OUTPUT_FOLDER" \
      --prompt_file_path "$ROOT_DIR/${PROMPT_FILE_PATH:-examples/prompt.txt}" \
      --video_path "$ROOT_DIR/${VIDEO_PATH:-examples/original.mp4}" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --fps "$FPS" \
      --step "$STEP" \
      "$@"
    ;;
  *)
    echo "Unknown mode '$MODE'. Expected one of: single, single-wo, pipe" >&2
    exit 1
    ;;
esac
