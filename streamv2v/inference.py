"""
Single GPU Inference Pipeline - Refactored from inference_pipe.py

This file extracts core logic from multi-GPU inference code to implement a complete 
inference pipeline on a single GPU:
1. VAE encode input video
2. DiT inference (using input mode, processing all 30 blocks)
3. VAE decode output video
"""

from models.wan.causal_stream_inference import CausalStreamInferencePipeline
from models.util import set_seed
from diffusers.utils import export_to_video
from models.data import TextDataset
import argparse
from dataclasses import dataclass
import torch
import os
import time
import numpy as np
import logging
from typing import List

try:
    from streamv2v.inference_common import (
        load_generator_state_dict,
        load_mp4_as_tensor,
        merge_cli_config,
    )
except ModuleNotFoundError:
    from inference_common import (
        load_generator_state_dict,
        load_mp4_as_tensor,
        merge_cli_config,
    )

LOGGER = logging.getLogger(__name__)


@dataclass
class SingleGPUStreamSession:
    prompt: str
    noise_scale: float
    init_noise_scale: float
    chunk_size: int
    current_start: int
    current_end: int
    last_image: torch.Tensor
    processed: int = 0

def compute_noise_scale_and_step(input_video_original: torch.Tensor, end_idx: int, chunk_size: int, noise_scale: float, init_noise_scale: float):
    """Compute adaptive noise scale and current step based on video content."""
    l2_dist=(input_video_original[:,:,end_idx-chunk_size:end_idx]-input_video_original[:,:,end_idx-chunk_size-1:end_idx-1])**2
    l2_dist = (torch.sqrt(l2_dist.mean(dim=(0,1,3,4))).max()/0.2).clamp(0,1)
    new_noise_scale = (init_noise_scale-0.1*l2_dist.item())*0.9+noise_scale*0.1
    current_step = int(1000*new_noise_scale)-100
    return new_noise_scale, current_step

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

        self.t2v = config.t2v
        self.profile = bool(config.get("profile", False))
        self.encode_fps_list: list[float] = []
        self.decode_fps_list: list[float] = []

        
        self.logger.info("Single GPU inference pipeline manager initialized")
    
    def load_model(self, checkpoint_folder: str):
        """Load the model from checkpoint."""
        ckpt_path, state_dict = load_generator_state_dict(checkpoint_folder)
        self.logger.info(f"Loading checkpoint from {ckpt_path}")

        # Load into the pipeline generator
        try:
            self.pipeline.generator.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            # Try non-strict load as a fallback and report
            self.logger.warning(f"Strict load_state_dict failed: {e}; retrying with strict=False")
            self.pipeline.generator.load_state_dict(state_dict, strict=False)
    
    def prepare_pipeline(self, text_prompts: list, noise: torch.Tensor, current_start: int, current_end: int):
        """Prepare the pipeline for inference."""
        # Use the original prepare method which now handles distributed environment gracefully
        denoised_pred = self.pipeline.prepare(
            text_prompts=text_prompts,
            device=self.device,
            dtype=torch.bfloat16,
            block_mode='input',
            noise=noise,
            current_start=current_start,
            current_end=current_end
        )
        return denoised_pred

    def _sync_for_timing(self):
        if self.profile:
            torch.cuda.synchronize()

    def _record_stage_fps(self, values: list[float], num_frames: int, elapsed: float) -> None:
        if self.profile and elapsed > 0 and num_frames > 0:
            values.append(num_frames / elapsed)

    def _timed_stream_encode(self, images: torch.Tensor) -> torch.Tensor:
        self._sync_for_timing()
        start_time = time.time()
        latents = self.pipeline.vae.stream_encode(images)
        self._sync_for_timing()
        self._record_stage_fps(self.encode_fps_list, int(images.shape[2]), time.time() - start_time)
        return latents

    def _timed_stream_decode(self, denoised_pred: torch.Tensor) -> torch.Tensor:
        self._sync_for_timing()
        start_time = time.time()
        video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred)
        self._sync_for_timing()
        self._record_stage_fps(self.decode_fps_list, int(video.shape[1]), time.time() - start_time)
        return video

    def reset_stream_state(self, reset_vae_flags: bool = True) -> None:
        """Reset cached model state before starting a new streaming session."""
        if reset_vae_flags:
            self.pipeline.vae.model.first_encode = True
            self.pipeline.vae.model.first_decode = True

        self.pipeline.kv_cache1 = None
        self.pipeline.crossattn_cache = None
        self.pipeline.block_x = None
        self.pipeline.hidden_states = None
        self.processed = 0

    def _encode_noisy_latents(self, images: torch.Tensor, noise_scale: float) -> torch.Tensor:
        latents = self._timed_stream_encode(images)
        latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
        noise = torch.randn_like(latents)
        return noise * noise_scale + latents * (1 - noise_scale)

    def _decode_video_array(self, denoised_pred: torch.Tensor, last_frame_only: bool = False) -> np.ndarray:
        if last_frame_only:
            denoised_pred = denoised_pred[[-1]]

        video = self._timed_stream_decode(denoised_pred)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video[0].permute(0, 2, 3, 1).contiguous()
        return video.detach().cpu().float().numpy()

    def start_stream_session(self, prompt: str, images: torch.Tensor, noise_scale: float) -> tuple[SingleGPUStreamSession, np.ndarray]:
        """Initialize a streaming session and return the first decoded frames."""
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
        """Process one or more chunk-aligned frame groups for an active streaming session."""
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
        num_steps = len(self.pipeline.denoising_step_list)

        for batch_idx in range(input_batch):
            if session.current_start // self.pipeline.frame_seq_length >= self.t_refresh:
                session.current_start = self.pipeline.kv_cache_length - self.pipeline.frame_seq_length
                session.current_end = session.current_start + (session.chunk_size // self.base_chunk_size) * self.pipeline.frame_seq_length

            denoised_pred = self.pipeline.inference_stream(
                noise=noisy_latents[:, batch_idx].unsqueeze(1),
                current_start=session.current_start,
                current_end=session.current_end,
                current_step=current_step,
            )

            session.processed += 1
            self.processed = session.processed

            if session.processed >= num_steps:
                outputs.append(self._decode_video_array(denoised_pred, last_frame_only=True))

            session.current_start = session.current_end
            session.current_end += (session.chunk_size // self.base_chunk_size) * self.pipeline.frame_seq_length

        session.last_image = images[:, :, [-1]]
        session.noise_scale = noise_scale
        return outputs
    
    def run_inference(
        self, 
        input_video_original: torch.Tensor, 
        prompts: list, 
        num_chunks: int, 
        chunk_size: int, 
        noise_scale: float, 
        output_folder: str, 
        fps: int, 
        target_fps:int,  
        num_steps: int,
        ):
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
        self.encode_fps_list = []
        self.decode_fps_list = []
        
        # Initialize variables
        start_idx = 0
        if self.t2v:
            end_idx = 1 + chunk_size - 4
        else:
            end_idx = 1 + chunk_size
        current_start = 0
        current_end = self.pipeline.frame_seq_length * (1+(end_idx-1)//4)
        
        self._sync_for_timing()
        start_time = time.time()
        
        # Process first chunk (initialization)
        if not self.t2v:
            inp = input_video_original[:, :, start_idx:end_idx]
            
            # VAE encoding
            latents = self._timed_stream_encode(inp)
            latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
            
            noise = torch.randn_like(latents)
            noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
        else:
            noisy_latents = torch.randn(1,self.pipeline.num_frame_per_block,16,self.pipeline.height,self.pipeline.width, device=self.device, dtype=torch.bfloat16)
        
            
        # Prepare pipeline
        denoised_pred = self.prepare_pipeline(
            text_prompts=prompts,
            noise=noisy_latents,
            current_start=current_start,
            current_end=current_end
        )
        
        # Save first result - only start decoding after num_steps
        video = self._timed_stream_decode(denoised_pred)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video[0].permute(0, 2, 3, 1).contiguous()
        results[save_results] = video.cpu().float().numpy()
        self.logger.info(
            "Prepared initial chunk: start=%s, end=%s, start_idx=%s, save_results=%s, frames=%s",
            current_start,
            current_end,
            start_idx,
            save_results,
            video.shape[0],
        )
        save_results += 1
        
        init_noise_scale = noise_scale

        # Process remaining chunks
        while self.processed < num_chunks + num_steps - 1:
            # Update indices
            start_idx = end_idx
            end_idx = end_idx + chunk_size
            current_start = current_end
            current_end = current_end + (chunk_size // 4) * self.pipeline.frame_seq_length

            if not self.t2v and end_idx <= input_video_original.shape[2]:
                inp = input_video_original[:, :, start_idx:end_idx]
                
                noise_scale, current_step = compute_noise_scale_and_step(
                    input_video_original, end_idx, chunk_size, noise_scale, init_noise_scale
                )
                
                # VAE encoding
                latents = self._timed_stream_encode(inp)
                latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
                
                noise = torch.randn_like(latents)
                noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
            else:
                noisy_latents = torch.randn(1,self.pipeline.num_frame_per_block,16,self.pipeline.height,self.pipeline.width, device=self.device, dtype=torch.bfloat16)
                current_step = None # Use default steps

            self._sync_for_timing()
            dit_start_time = time.time()
                
            # DiT inference - using input mode to process all 30 blocks
            denoised_pred = self.pipeline.inference_stream(
                noise=noisy_latents,
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
            )

            if self.processed > self.processed_offset:
                self._sync_for_timing()
                if self.profile:
                    dit_fps_list.append(chunk_size / (time.time() - dit_start_time))
            
            self.processed += 1
            
            # VAE decoding - only start decoding after num_steps
            if self.processed >= num_steps:
                if self.t2v and self.processed == num_steps:
                    continue
                video = self._timed_stream_decode(denoised_pred[[-1]])
                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video[0].permute(0, 2, 3, 1).contiguous()
                results[save_results] = video.cpu().float().numpy()
                save_results += 1
            
                # Update timing
                if self.profile:
                    self._sync_for_timing()
                    end_time = time.time()
                    t = end_time - start_time
                    fps_test = chunk_size / t
                    fps_list.append(fps_test)
                    self.logger.info(f"Processed {self.processed}, time: {t:.4f} s, FPS: {fps_test:.4f}")
                else:
                    fps_test = None

                if self.processed == num_steps + self.processed_offset and target_fps is not None and fps_test is not None and fps_test < target_fps:
                    max_chunk_size = (self.pipeline.num_kv_cache - self.pipeline.num_sink_tokens - 1) * self.base_chunk_size
                    num_chunks=(num_chunks-self.processed-num_steps+1)//(max_chunk_size//chunk_size)+self.processed-num_steps+1
                    self.pipeline.hidden_states=self.pipeline.hidden_states.repeat(1,max_chunk_size//chunk_size,1,1,1)
                    chunk_size = max_chunk_size
                    self.logger.info(f"Adjust chunk size to {chunk_size}")

                if self.profile:
                    start_time = end_time
        

        # Save final video
        video_list = [results[i] for i in range(num_chunks)]
        video = np.concatenate(video_list, axis=0)
        if self.profile and fps_list:
            fps_avg = np.mean(np.array(fps_list))
            dit_avg = np.mean(np.array(dit_fps_list)) if dit_fps_list else 0.0
            encode_avg = np.mean(np.array(self.encode_fps_list)) if self.encode_fps_list else 0.0
            decode_avg = np.mean(np.array(self.decode_fps_list)) if self.decode_fps_list else 0.0
            self.logger.info(f"VAE Encode Average FPS: {encode_avg:.4f}")
            self.logger.info(f"DiT Average FPS: {dit_avg:.4f}")
            self.logger.info(f"VAE Decode Average FPS: {decode_avg:.4f}")
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
    parser.add_argument("--video_path", type=str, required=False, default=None, help="Input video path")
    parser.add_argument("--noise_scale", type=float, default=0.8, help="Noise scale")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--fps", type=int, default=16, help="Output video fps")
    parser.add_argument("--step", type=int, default=2, help="Step")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gpu_id", type=int, default=None, help="CUDA device index for single-GPU inference")
    parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type (e.g., T2V-1.3B)")
    parser.add_argument("--num_frames", type=int, default=81, help="Video length (number of frames)")
    parser.add_argument("--fixed_noise_scale", action="store_true", default=False)
    parser.add_argument("--t2v", action="store_true", default=False)
    parser.add_argument("--target_fps", type=int, required=False, default=None, help="Video length (number of frames)")
    parser.add_argument("--profile", action="store_true", default=False, help="Enable synchronized throughput logging")
    parser.add_argument("--use_taehv", action="store_true", default=False, help="Use the lightweight TAEHV VAE for encode/decode")
    parser.add_argument("--use_tensorrt", "--use_taehv_tensorrt", dest="use_tensorrt", action="store_true", default=False, help="Enable available TensorRT acceleration paths")
    parser.add_argument("--fast", action="store_true", default=False, help="Enable the fast path: --use_taehv --use_tensorrt")
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    
    # Auto-detect device
    if torch.cuda.is_available():
        if args.gpu_id is not None:
            torch.cuda.set_device(args.gpu_id)
            device = torch.device(f"cuda:{args.gpu_id}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load configuration
    config = merge_cli_config(args.config_path, args)

    set_seed(args.seed)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    LOGGER.info("Denoising Step List: %s", list(config.denoising_step_list))
    
    # Load input video
    if not args.t2v:
        input_video_original = load_mp4_as_tensor(args.video_path, resize_hw=(args.height, args.width)).unsqueeze(0)
        LOGGER.info("Input video tensor shape: %s", tuple(input_video_original.shape))
        b, c, t, h, w = input_video_original.shape
        if input_video_original.dtype != torch.bfloat16:
            input_video_original = input_video_original.to(dtype=torch.bfloat16).to(device)
    else:
        input_video_original = None
        t = args.num_frames
    
    # Calculate number of chunks
    chunk_size = 4 * config.num_frame_per_block
    num_chunks = (t - 1) // chunk_size

    if args.t2v:
        num_chunks+=1
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
            input_video_original, 
            prompts, 
            num_chunks, 
            chunk_size, 
            args.noise_scale, 
            args.output_folder, 
            args.fps, 
            args.target_fps, 
            num_steps,
        )
    except Exception as e:
        LOGGER.exception("Error occurred during inference: %s", e)
        raise


if __name__ == "__main__":
    main()
