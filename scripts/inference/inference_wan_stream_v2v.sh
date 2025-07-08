#!/bin/bash
script_name=$(basename "$0")
script_name_no_ext="${script_name%.sh}"

num_gpus=1
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 12345 \
    streamv2v/inference_wan_stream_v2v.py \
    --height 480 \
    --width 832 \
    --num_frames 17 \
    --num_inference_steps 4 \
    --t_start 1 \
    --guidance_scale 1 \
    --embedded_cfg_scale 1 \
    --flow_shift 3 \
    --flow-reverse \
    --prompt_path prompt_v2v.txt \
    --seed 0 \
    --reference_video_path output_17.mp4  \
    --output_path outputs_video/${script_name_no_ext}/res/ \
    --model_path ckpt/DCM_WAN/
