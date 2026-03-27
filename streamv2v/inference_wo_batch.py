"""
Single GPU Inference Pipeline - Refactored from inference_pipe.py

This file extracts core logic from multi-GPU inference code to implement a complete 
inference pipeline on a single GPU:
1. VAE encode input video
2. DiT inference (using input mode, processing all 30 blocks)
3. VAE decode output video
"""

from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from causvid.util import set_seed
from diffusers.utils import export_to_video
from causvid.data import TextDataset
import argparse
import torch
import os
import time
import numpy as np
import logging
from typing import List

try:
    from streamv2v.inference import compute_noise_scale_and_step, SingleGPUStreamSession
    from streamv2v.inference_common import (
        load_generator_state_dict,
        load_mp4_as_tensor,
        merge_cli_config,
    )
except ModuleNotFoundError:
    from inference import compute_noise_scale_and_step, SingleGPUStreamSession
    from inference_common import load_generator_state_dict, load_mp4_as_tensor, merge_cli_config

LOGGER = logging.getLogger(__name__)


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
        self.profile = bool(config.get("profile", False))
        
        self.logger.info("Single GPU inference pipeline manager initialized")
    
    def load_model(self, checkpoint_folder: str):
        """Load the model from checkpoint."""
        ckpt_path, state_dict = load_generator_state_dict(checkpoint_folder)
        try:
            self.pipeline.generator.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            self.logger.warning(f"Strict load_state_dict failed: {exc}; retrying with strict=False")
            self.pipeline.generator.load_state_dict(state_dict, strict=False)
        self.logger.info(f"Model loaded successfully from {ckpt_path}")
    
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

    def _sync_for_timing(self):
        if self.profile:
            torch.cuda.synchronize()

    def reset_stream_state(self, reset_vae_flags: bool = True) -> None:
        """Reset cached state before starting a new no-batch stream session."""
        if reset_vae_flags:
            self.pipeline.vae.model.first_encode = True
            self.pipeline.vae.model.first_decode = True

        self.pipeline.kv_cache1 = None
        self.pipeline.crossattn_cache = None
        self.pipeline.hidden_states = None
        self.pipeline.block_x = None
        self.processed = 0

    def _encode_noisy_latents(self, images: torch.Tensor, noise_scale: float) -> torch.Tensor:
        latents = self.pipeline.vae.stream_encode(images)
        latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
        noise = torch.randn_like(latents)
        return noise * noise_scale + latents * (1 - noise_scale)

    def _decode_video_array(self, denoised_pred: torch.Tensor, last_frame_only: bool = False) -> np.ndarray:
        if last_frame_only:
            denoised_pred = denoised_pred[[-1]]

        video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video[0].permute(0, 2, 3, 1).contiguous()
        return video.cpu().float().numpy()

    def start_stream_session(self, prompt: str, images: torch.Tensor, noise_scale: float) -> tuple[SingleGPUStreamSession, np.ndarray]:
        """Initialize a no-batch streaming session and return the first decoded frames."""
        self.reset_stream_state(reset_vae_flags=True)

        chunk_size = self.base_chunk_size * self.pipeline.num_frame_per_block
        current_start = 0
        current_end = self.pipeline.frame_seq_length * (1 + chunk_size // self.base_chunk_size)

        noisy_latents = self._encode_noisy_latents(images, noise_scale)
        denoised_pred = self.prepare_pipeline(
            text_prompts=[prompt],
            noise=noisy_latents,
            current_start=current_start,
            current_end=current_end,
            batch_denoise=False,
        )
        initial_video = self._decode_video_array(denoised_pred, last_frame_only=False)

        session = SingleGPUStreamSession(
            prompt=prompt,
            noise_scale=noise_scale,
            init_noise_scale=noise_scale,
            chunk_size=chunk_size,
            current_start=current_end,
            current_end=current_end + (chunk_size // self.base_chunk_size) * self.pipeline.frame_seq_length,
            last_image=images[:, :, [-1]],
            processed=0,
        )
        return session, initial_video

    def run_stream_batch(self, session: SingleGPUStreamSession, images: torch.Tensor) -> List[np.ndarray]:
        """Process one or more chunk-aligned frame groups for an active no-batch stream session."""
        num_frames = images.shape[2]
        input_batch = num_frames // session.chunk_size
        noise_scale, current_step = compute_noise_scale_and_step(
            input_video_original=torch.cat([session.last_image, images], dim=2),
            end_idx=num_frames + 1,
            chunk_size=num_frames,
            noise_scale=float(session.noise_scale),
            init_noise_scale=float(session.init_noise_scale),
        )
        noisy_latents = self._encode_noisy_latents(images, noise_scale)

        outputs: List[np.ndarray] = []

        for batch_idx in range(input_batch):
            if session.current_start // self.pipeline.frame_seq_length >= self.t_refresh:
                session.current_start = self.pipeline.kv_cache_length - self.pipeline.frame_seq_length
                session.current_end = session.current_start + (session.chunk_size // self.base_chunk_size) * self.pipeline.frame_seq_length

            denoised_pred = self.pipeline.inference_wo_batch(
                noise=noisy_latents[:, batch_idx].unsqueeze(1),
                current_start=session.current_start,
                current_end=session.current_end,
                current_step=current_step,
            )

            session.processed += 1
            self.processed = session.processed
            outputs.append(self._decode_video_array(denoised_pred, last_frame_only=True))

            session.current_start = session.current_end
            session.current_end += (session.chunk_size // self.base_chunk_size) * self.pipeline.frame_seq_length

        session.last_image = images[:, :, [-1]]
        session.noise_scale = noise_scale
        return outputs
    
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
        
        self._sync_for_timing()
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
                
            self._sync_for_timing()
            dit_start_time = time.time()
            # DiT inference - using input mode to process all 30 blocks
            denoised_pred = self.pipeline.inference_wo_batch(
                noise=noisy_latents,
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
            )
            
            if self.processed >self.processed_offset:
                self._sync_for_timing()
                if self.profile:
                    dit_fps_list.append(chunk_size / (time.time() - dit_start_time))
            
            self.processed += 1
            
            # VAE decoding - only start decoding after num_steps
            video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred[[-1]])
            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = video[0].permute(0, 2, 3, 1).contiguous()
            
            results[save_results] = video.cpu().float().numpy()
            save_results += 1
        
            # Update timing
            if self.profile:
                self._sync_for_timing()
                end_time = time.time()
                t = end_time - start_time
                fps_test = inp.shape[2] / t
                fps_list.append(fps_test)
                self.logger.info(f"Processed {self.processed}, time: {t:.4f} s, FPS: {fps_test:.4f}")
                start_time = end_time
        
        # Save final video
        video_list = [results[i] for i in range(num_chunks)]
        video = np.concatenate(video_list, axis=0)
        if self.profile and fps_list:
            fps_avg = np.mean(np.array(fps_list))
            dit_avg = np.mean(np.array(dit_fps_list)) if dit_fps_list else 0.0
            self.logger.info(f"DiT Average FPS: {dit_avg:.4f}")
            self.logger.info(f"Video shape: {video.shape}, Average FPS: {fps_avg:.4f}")
        else:
            self.logger.info(f"Video shape: {video.shape}")
        
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--t2v", action="store_true", default=False)
    parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type (e.g., T2V-1.3B)")
    parser.add_argument("--profile", action="store_true", default=False, help="Enable synchronized throughput logging")
    parser.add_argument("--use_taehv", action="store_true", default=False, help="Use the lightweight TAEHV VAE for encode/decode")
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    
    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    config = merge_cli_config(args.config_path, args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    LOGGER.info("Denoising Step List: %s", list(config.denoising_step_list))

    set_seed(args.seed)
    
    # Load input video
    input_video_original = load_mp4_as_tensor(args.video_path, resize_hw=(args.height, args.width)).unsqueeze(0)
    if input_video_original.dtype != torch.bfloat16:
        input_video_original = input_video_original.to(dtype=torch.bfloat16).to(device)
    
    LOGGER.info("Input video tensor shape: %s", tuple(input_video_original.shape))
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
        LOGGER.exception("Error occurred during inference: %s", e)
        raise


if __name__ == "__main__":
    main()
