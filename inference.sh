CUDA_VISIBLE_DEVICES=1 python minimal_inference/autoregressive_inference.py \
--config_path configs/wan_causal_dmd.yaml \
--checkpoint_folder ckpts/autoregressive_checkpoint \
--output_folder outputs/ \
--prompt_file_path prompt.txt \