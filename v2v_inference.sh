CUDA_VISIBLE_DEVICES=1 python streamv2v/inference.py \
--config_path configs/wan_causal_dmd_v2v.yaml \
--checkpoint_folder ckpts/autoregressive_checkpoint \
--output_folder outputs/ \
--prompt_file_path prompt.txt \
--video_path original.mp4