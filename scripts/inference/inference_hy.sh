#!/bin/bash
script_name=$(basename "$0")
script_name_no_ext="${script_name%.sh}"

num_gpus=1
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29519 \
    fastvideo/sample/inference_hy.py \
    --height 720 \
    --width 1280 \
    --num_frames 29 \
    --num_inference_steps 4 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 7 \
    --flow-reverse \
    --prompt_path path_to_prompt_txt \
    --seed 1024 \
    --output_path outputs_video/${script_name_no_ext}/cfg6/ \
    --model_path ckpt/DCM_HY/ 
