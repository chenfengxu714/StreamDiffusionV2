#!/usr/bin/env bash

# ----------------------
# User-defined variables
# ----------------------
CHECKPOINT_DIR="/mnt/localssd/sdxl_logs/2025-01-23-15-16-31.765725_seed228885"
PROMPT_PATH="captions_coco10k.txt"
REF_DIR="/mnt/localssd/coco10k/subset/"
DENOSING_STEPS="999 749 499 249"

# Adjust this if you have a different number of GPUs available
NUM_GPUS=8

# -------------
# Main script
# -------------
# Grab all checkpoints in the folder
CHECKPOINTS=(${CHECKPOINT_DIR}/checkpoint_model_*)

# Print how many checkpoints were found
echo "Found ${#CHECKPOINTS[@]} checkpoints in ${CHECKPOINT_DIR}"

# Loop over each checkpoint and launch a job
for ((i=0; i<${#CHECKPOINTS[@]}; i++)); do
  
  # GPU to use (round-robin assignment)
  GPU_ID=$(( i % NUM_GPUS ))

  # Pick a unique port for each process. For example, offset from 29500.
  # Feel free to choose any range that won't collide with other applications.
  MASTER_PORT=$((29500 + i))

  echo "Launching eval for checkpoint: ${CHECKPOINTS[$i]} on GPU ${GPU_ID}, master_port ${MASTER_PORT}"
  
  # Run eval on GPU_ID, put the process in the background
  CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --nproc_per_node 1 \
    --master_port ${MASTER_PORT} \
    causvid/evaluation/eval_sdxl_coco.py \
      --denoising_step_list $DENOSING_STEPS \
      --prompt_path "$PROMPT_PATH" \
      --checkpoint_path "${CHECKPOINTS[$i]}" \
      --ref_dir "$REF_DIR" &

  # If we've launched as many parallel tasks as GPUs, wait for this batch to finish
  if (( (i+1) % NUM_GPUS == 0 )); then
    echo "Waiting for batch of $NUM_GPUS processes to finish..."
    wait
  fi
done

# If there are leftover tasks that didn't perfectly divide into NUM_GPUS, wait again
wait

echo "All evaluations have completed."
