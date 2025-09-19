torchrun --nproc_per_node=2 --master_port=29501 streamv2v/inference_pipe.py \
--config_path configs/wan_causal_dmd_v2v.yaml \
--checkpoint_folder ckpts/wan_causal_dmd_v2v \
--output_folder outputs/ \
--prompt_file_path prompt.txt \
--video_path original.mp4 \
--height 480 \
--width 832 \
--fps 16 
# --schedule_block #optional for scheduling blocks