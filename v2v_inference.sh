CUDA_VISIBLE_DEVICES=0 python streamv2v/inference.py \
--config_path configs/wan_causal_dmd_v2v.yaml \
--checkpoint_folder wan_causal_dmd/2025-08-14-09-44-27.384892_seed290041/checkpoint_model_001000 \
--output_folder outputs/ \
--prompt_file_path prompt.txt \
--video_path original.mp4 \
--height 480 \
--width 832 \
--fps 16