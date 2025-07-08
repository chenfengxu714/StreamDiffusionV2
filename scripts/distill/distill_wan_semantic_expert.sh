export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline

torchrun --nnodes 1 --nproc_per_node 1 --master_port 29903 \
    fastvideo/distill_dcm_wan_semantic_expert.py \
    --seed 42\
    --pretrained_model_name_or_path pretrained/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/ \
    --model_type wan \
    --cache_dir data/.cache \
    --data_json_path data/HD-Mixkit-Finetune-Hunyuan/videos2caption.json \
    --validation_prompt_dir data/HD-Mixkit-Finetune-Hunyuan/validation \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 21 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 1000 \
    --learning_rate 1e-6 \
    --mixed_precision bf16 \
    --checkpointing_steps 64 \
    --validation_steps 64 \
    --validation_sampling_steps "2,4,8" \
    --checkpoints_total_limit 1 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --log_validation \
    --output_dir data/outputs/distill_wan_semantic \
    --tracker_project_name distill_wan_semantic \
    --num_height 480 \
    --num_width 832 \
    --num_frames 81 \
    --validation_guidance_scale "1.0" \
    --shift 17 \
    --num_euler_timesteps 50 \
    --multi_phased_distill_schedule "4000-1" 
