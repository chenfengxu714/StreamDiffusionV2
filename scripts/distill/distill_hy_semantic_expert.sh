export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

torchrun --nnodes 3 --nproc_per_node 8\
    --node_rank=0 \
    --rdzv_id=456 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=10.140.54.18:23209 \
    fastvideo/distill_dcm_hy_semantic_expert.py \
    --seed 42\
    --pretrained_model_name_or_path pretrained/hunyuanvideo-community/HunyuanVideo \
    --model_type hunyuan_hf \
    --cache_dir data/.cache \
    --data_json_path data/HD-Mixkit-Finetune-Hunyuan/videos2caption.json \
    --validation_prompt_dir data/HD-Mixkit-Finetune-Hunyuan/validation \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 32 \
    --sp_size 4 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 1500 \
    --learning_rate 1e-6 \
    --mixed_precision bf16 \
    --checkpointing_steps 200 \
    --validation_steps 64 \
    --validation_sampling_steps "2,4,8" \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --log_validation \
    --output_dir data/outputs/distill_hy_semantic \
    --tracker_project_name distill_hy_semantic \
    --num_height 720 \
    --num_width 1280 \
    --num_frames 29 \
    --validation_guidance_scale "1.0" \
    --shift 17 \
    --num_euler_timesteps 50 \
    --multi_phased_distill_schedule "4000-1" \
    --not_apply_cfg_solver
