"""
Single GPU Inference Pipeline - Refactored from inference_pipe.py

This file extracts core logic from multi-GPU inference code to implement a complete 
inference pipeline on a single GPU:
1. VAE encode input video
2. DiT inference (using input mode, processing all 30 blocks)
3. VAE decode output video
"""

from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
import argparse
import torch
import os
import time
import numpy as np
import logging

import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange
from streamv2v.inference import compute_noise_scale_and_step


def load_mp4_as_tensor(
    video_path: str,
    max_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Load an .mp4 video and return it as a PyTorch tensor with shape [C, T, H, W].

    Args:
        video_path (str): Path to the input .mp4 video file
        max_frames (int, optional): Maximum number of frames to load. If None, loads all frames
        resize_hw (tuple, optional): Target (height, width) to resize each frame. If None, no resizing
        normalize (bool, optional): Whether to normalize pixel values to [-1, 1]

    Returns:
        torch.Tensor: Tensor of shape [C, T, H, W], dtype=torch.float32
    """
    assert os.path.exists(video_path), f"Video file not found: {video_path}"

    video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW")
    if max_frames is not None:
        video = video[:max_frames]

    video = rearrange(video, "t c h w -> c t h w")
    if resize_hw is not None:
        c, t, h0, w0 = video.shape
        video = torch.stack([
            TF.resize(video[:, i], resize_hw, antialias=True)
            for i in range(t)
        ], dim=1)
    if video.dtype != torch.float32:
        video = video.float()
    if normalize:
        video = video / 127.5 - 1.0

    return video  # [C, T, H, W]


class SingleGPUInferencePipeline:
    """
    Single GPU Inference Pipeline Manager
    
    This class encapsulates the complete inference logic on a single GPU, 
    including encoding, inference, and decoding.
    """
    
    def __init__(self, config, device: torch.device):
        """
        Initialize the single GPU inference pipeline manager.
        
        Args:
            config: Configuration object
            device: GPU device
        """
        self.config = config
        self.device = device
        
        # Setup logging
        self.logger = logging.getLogger("SingleGPUInference")
        self.logger.setLevel(logging.INFO)
        # Prevent messages from propagating to the root logger (avoid double prints)
        self.logger.propagate = False
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Initialize pipeline
        self.pipeline = CausalStreamInferencePipeline(config, device=str(device))
        self.pipeline.to(device=str(device), dtype=torch.bfloat16)
        
        # Performance tracking
        self.t_dit = 100.0
        self.t_total = 100.0
        self.processed = 0
        self.processed_offset = 3
        self.base_chunk_size = 4
        self.t_refresh = 50
        
        self.logger.info("Single GPU inference pipeline manager initialized")
    
    def load_model(self, checkpoint_folder: str):
        """Load the model from checkpoint."""
        state_dict = torch.load(os.path.join(checkpoint_folder, "model.pt"), map_location="cpu")["generator"]
        self.pipeline.generator.load_state_dict(state_dict, strict=True)
        self.logger.info("Model loaded successfully")
    
    def prepare_pipeline(self, text_prompts: list, noise: torch.Tensor, 
                        current_start: int, current_end: int, batch_denoise: bool=True):
        """Prepare the pipeline for inference."""
        # Use the original prepare method which now handles distributed environment gracefully
        denoised_pred = self.pipeline.prepare(
            text_prompts=text_prompts,
            device=self.device,
            dtype=torch.bfloat16,
            block_mode='input',
            noise=noise,
            current_start=current_start,
            current_end=current_end,
            batch_denoise=batch_denoise,
        )
        return denoised_pred
    
    def run_inference(self, input_video_original: torch.Tensor, prompts: list, 
                     num_chunks: int, chunk_size: int, noise_scale: float, 
                     output_folder: str, fps: int, num_steps: int):
        """
        Run the complete single GPU inference pipeline.
        
        This method integrates the complete encoding, inference, and decoding pipeline.
        """
        self.logger.info("Starting single GPU inference pipeline")
        
        os.makedirs(output_folder, exist_ok=True)
        results = {}
        save_results = 0

        fps_list = []
        dit_fps_list = []
        
        # Initialize variables
        start_idx = 0
        end_idx = 1 + chunk_size
        current_start = 0
        current_end = self.pipeline.frame_seq_length * (1+chunk_size//4)
        init_noise_scale = noise_scale
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Process first chunk (initialization)
        if end_idx <= input_video_original.shape[2]:
            inp = input_video_original[:, :, start_idx:end_idx]
            
            # VAE encoding
            latents = self.pipeline.vae.stream_encode(inp)
            latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
            
            noise = torch.randn_like(latents)
            noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
            
            # Prepare pipeline
            denoised_pred = self.prepare_pipeline(
                text_prompts=prompts,
                noise=noisy_latents,
                current_start=current_start,
                current_end=current_end,
                batch_denoise=False,
            )
            
            # Save first result - only start decoding after num_steps
            video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = video[0].permute(0, 2, 3, 1).contiguous()
            results[save_results] = video.cpu().float().numpy()
            save_results += 1
        
        # Process remaining chunks
        while self.processed < num_chunks + num_steps - 1:
            # Update indices
            start_idx = end_idx
            end_idx = end_idx + chunk_size
            current_start = current_end
            current_end = current_end + (chunk_size // 4) * self.pipeline.frame_seq_length
            
            if end_idx <= input_video_original.shape[2]:
                inp = input_video_original[:, :, start_idx:end_idx]
                
                noise_scale, current_step = compute_noise_scale_and_step(
                    input_video_original, end_idx, chunk_size, noise_scale, init_noise_scale
                )
                
                # VAE encoding
                latents = self.pipeline.vae.stream_encode(inp)
                latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
                
                noise = torch.randn_like(latents)
                noisy_latents = noise * noise_scale + latents * (1 - noise_scale)

            if current_start//self.pipeline.frame_seq_length >= 50:
                current_start = self.pipeline.kv_cache_length - self.pipeline.frame_seq_length
                current_end = current_start + (chunk_size // 4) * self.pipeline.frame_seq_length
                
            torch.cuda.synchronize()
            dit_start_time = time.time()
            # DiT inference - using input mode to process all 30 blocks
            denoised_pred = self.pipeline.inference_wo_batch(
                noise=noisy_latents,
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
            )
            
            if self.processed >self.processed_offset:
                torch.cuda.synchronize()
                dit_fps_list.append(chunk_size/(time.time()-dit_start_time))
            
            self.processed += 1
            
            # VAE decoding - only start decoding after num_steps
            video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred[[-1]])
            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = video[0].permute(0, 2, 3, 1).contiguous()
            
            results[save_results] = video.cpu().float().numpy()
            save_results += 1
        
            # Update timing
            torch.cuda.synchronize()
            end_time = time.time()
            t = end_time - start_time
            fps_test = inp.shape[2]/t
            fps_list.append(fps_test)
            self.logger.info(f"Processed {self.processed}, time: {t:.4f} s, FPS: {fps_test:.4f}")
            start_time = end_time
        
        # Save final video
        video_list = [results[i] for i in range(num_chunks)]
        video = np.concatenate(video_list, axis=0)
        fps_avg = np.mean(np.array(fps_list))
        self.logger.info(f"DiT Average FPS: {np.mean(np.array(dit_fps_list)):.4f}")
        self.logger.info(f"Video shape: {video.shape}, Average FPS: {fps_avg:.4f}")
        
        output_path = os.path.join(output_folder, f"output_{0:03d}.mp4")
        export_to_video(video, output_path, fps=fps)
        self.logger.info(f"Video saved to: {output_path}")
        
        self.logger.info("Single GPU inference pipeline completed")


def main():
    """Main function for the single GPU inference pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Configuration file path")
    parser.add_argument("--checkpoint_folder", type=str, required=True, help="Checkpoint folder path")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder path")
    parser.add_argument("--prompt_file_path", type=str, required=True, help="Prompt file path")
    parser.add_argument("--video_path", type=str, required=True, help="Input video path")
    parser.add_argument("--noise_scale", type=float, default=0.700, help="Noise scale")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--fps", type=int, default=16, help="Output video fps")
    parser.add_argument("--step", type=int, default=2, help="Step")
    parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type (e.g., T2V-1.3B)")
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    
    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    config = OmegaConf.load(args.config_path)
    for k, v in vars(args).items():
        config[k] = v
    # Derive denoising_step_list from step if provided
    # Always base on the canonical full list to ensure --step overrides YAML
    full_denoising_list = [700, 600, 500, 400, 0]
    step_value = args.step
    if step_value <= 1:
        config.denoising_step_list = [700, 0]
    elif step_value == 2:
        config.denoising_step_list = [700, 500, 0]
    elif step_value == 3:
        config.denoising_step_list = [700, 600, 400, 0]
    else:
        config.denoising_step_list = full_denoising_list
    
    # Load input video
    input_video_original = load_mp4_as_tensor(args.video_path, resize_hw=(args.height, args.width)).unsqueeze(0)
    if input_video_original.dtype != torch.bfloat16:
        input_video_original = input_video_original.to(dtype=torch.bfloat16).to(device)
    
    print(f"Input video tensor shape: {input_video_original.shape}")
    b, c, t, h, w = input_video_original.shape
    
    # Calculate number of chunks
    chunk_size = 4 * config.num_frame_per_block
    num_chunks = (t - 1) // chunk_size
    
    # Initialize pipeline manager
    pipeline_manager = SingleGPUInferencePipeline(config, device)
    pipeline_manager.load_model(args.checkpoint_folder)
    
    # Load prompts
    dataset = TextDataset(args.prompt_file_path)
    prompts = [dataset[0]]
    num_steps = len(pipeline_manager.pipeline.denoising_step_list)
    
    # Run inference
    try:
        pipeline_manager.run_inference(
            input_video_original, prompts, num_chunks, chunk_size, 
            args.noise_scale, args.output_folder, args.fps, num_steps
        )
    except Exception as e:
        print(f"Error occurred during inference: {e}")
        raise


if __name__ == "__main__":
    main()