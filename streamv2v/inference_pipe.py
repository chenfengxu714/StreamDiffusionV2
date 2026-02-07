"""
Refactored multi-rank inference pipeline with communication abstractions.

This is a refactored version of inference_pipe_multi.py that uses the new
communication abstraction layers for better code organization and maintainability.
"""

from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from streamv2v.inference import compute_noise_scale_and_step
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
import numpy as np

import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange

# Import our new communication abstractions
from streamv2v.communication import (
    DistributedCommunicator,
    ModelDataTransfer,
    BufferManager,
    KVCacheManager,
    CommunicationConfig,
    init_distributed,
    setup_logging,
    compute_balanced_split
)


def load_mp4_as_tensor(
    video_path: str,
    max_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Loads an .mp4 video and returns it as a PyTorch tensor with shape [C, T, H, W].

    Args:
        video_path (str): Path to the input .mp4 video file.
        max_frames (int, optional): Maximum number of frames to load. If None, loads all.
        resize_hw (tuple, optional): Target (height, width) to resize each frame. If None, no resizing.
        normalize (bool, optional): Whether to normalize pixel values to [-1, 1].

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



class InferencePipelineManager:
    """
    Manages the inference pipeline with communication abstractions.
    
    This class encapsulates the main inference logic and uses the communication
    abstractions for distributed operations.
    """
    
    def __init__(self, config, device: torch.device, rank: int, world_size: int):
        """
        Initialize the inference pipeline manager.
        
        Args:
            config: Configuration object
            device: GPU device
            rank: Current rank
            world_size: Total number of ranks
        """
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size

        self.com_stream = torch.cuda.Stream()
        self.control_stream = torch.cuda.Stream()
        
        # Setup logging
        self.logger = setup_logging(rank)
        
        # Initialize communication components
        comm_config = CommunicationConfig(
            max_outstanding=config.get('max_outstanding', 1),
            buffer_pool_size=config.get('buffer_pool_size', 10),
            enable_buffer_reuse=config.get('enable_buffer_reuse', True)
        )
        
        self.communicator = DistributedCommunicator(rank, world_size, device, comm_config)
        self.buffer_manager = BufferManager(device, comm_config)
        
        # Initialize pipeline
        self.pipeline = CausalStreamInferencePipeline(config, device=str(device))
        self.pipeline.to(device=str(device), dtype=torch.bfloat16)
        
        # Initialize KV cache manager
        self.kv_cache_manager = KVCacheManager(self.pipeline, device)
        
        # Initialize model data transfer
        self.data_transfer = ModelDataTransfer(
            self.communicator, 
            self.buffer_manager, 
            self.kv_cache_manager, 
            comm_config
        )
        
        # Performance tracking
        self.t_dit = 100.0
        self.t_total = 100.0
        self.processed = 0
        self.schedule_step = (self.world_size + len(config.denoising_step_list)) * 2
        self.processed_offset = 3
        self.base_chunk_size = 4
        self.t_refresh = 50
        
        self.logger.info(f"Initialized InferencePipelineManager for rank {rank}")
    
    def load_model(self, checkpoint_folder: str):
        """Load the model from checkpoint."""
        state_dict = torch.load(os.path.join(checkpoint_folder, "model.pt"), map_location="cpu")["generator"]
        self.pipeline.generator.load_state_dict(state_dict, strict=True)
        self.logger.info("Model loaded successfully")
    
    def prepare_pipeline(self, text_prompts: list, noise: torch.Tensor, 
                        block_mode: str, current_start: int, current_end: int, block_num: torch.Tensor):
        """Prepare the pipeline for inference."""
        denoised_pred = self.pipeline.prepare(
            text_prompts=text_prompts, 
            device=self.device, 
            dtype=torch.bfloat16, 
            noise=noise, 
            block_mode=block_mode, 
            current_start=current_start, 
            current_end=current_end,
            block_num=block_num
        )
        
        # Broadcast the prepared result from rank 0
        self.data_transfer.broadcast_tensor(denoised_pred, src=0)
        return denoised_pred
    
    def run_rank_0_loop(self, input_video_original: torch.Tensor, prompts: list, 
                       num_chunks: int, num_steps: int, chunk_size: int,
                       block_num: torch.Tensor, noise_scale: float, 
                       schedule_block: bool, total_blocks: int):
        """
        Run the main loop for rank 0 (encoder + async send).
        
        This method encapsulates the rank 0 logic using the communication abstractions.
        """
        self.logger.info("Starting rank 0 inference loop")
        
        # Initialize variables
        start_idx = 0
        end_idx = 1 + chunk_size
        current_start = 0
        current_end = self.pipeline.frame_seq_length * (1+chunk_size//self.base_chunk_size)
        init_noise_scale = noise_scale
        
        outstanding = []
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        while True:
            # Process new chunk if available
            start_idx = end_idx
            end_idx = end_idx + chunk_size
            current_start = current_end
            current_end = current_end + (chunk_size // self.base_chunk_size) * self.pipeline.frame_seq_length

            if schedule_block:
                torch.cuda.synchronize()
                start_vae = time.time()
                
            if end_idx <= input_video_original.shape[2]:
                inp = input_video_original[:, :, start_idx:end_idx]
                
                noise_scale, current_step = compute_noise_scale_and_step(
                    input_video_original, end_idx, chunk_size, noise_scale, init_noise_scale
                )
                
                latents = self.pipeline.vae.stream_encode(inp)
                latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
                
                noise = torch.randn_like(latents)
                noisy_latents = noise * noise_scale + latents * (1 - noise_scale)

            # if current_start//self.pipeline.frame_seq_length >= self.t_refresh:
            #     current_start = self.pipeline.kv_cache_length - self.pipeline.frame_seq_length
            #     current_end = current_start + (chunk_size // self.base_chunk_size) * self.pipeline.frame_seq_length
            
            # Measure DiT time if scheduling is enabled
            if schedule_block:
                torch.cuda.synchronize()
                start_dit = time.time()
                t_vae = start_dit - start_vae
            
            # Run inference
            denoised_pred, patched_x_shape = self.pipeline.inference(
                noise=noisy_latents,
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
                block_mode='input',
                block_num=block_num[self.rank],
            )
            
            # Update DiT timing
            if schedule_block:
                torch.cuda.synchronize()
                temp = time.time() - start_dit
                if temp < self.t_dit:
                    self.t_dit = temp
            
            self.processed += 1
            
            with torch.cuda.stream(self.com_stream):
                if self.processed >= self.world_size:
                    if 'latent_data' in locals():
                        self.buffer_manager.return_buffer(latent_data.latents, "latent")
                        self.buffer_manager.return_buffer(latent_data.original_latents, "origin")

                        if hasattr(latent_data, 'patched_x_shape') and latent_data.patched_x_shape is not None:
                            self.buffer_manager.return_buffer(latent_data.patched_x_shape, "misc")
                        if hasattr(latent_data, 'current_start') and latent_data.current_start is not None:
                            self.buffer_manager.return_buffer(latent_data.current_start, "misc")
                        if hasattr(latent_data, 'current_end') and latent_data.current_end is not None:
                            self.buffer_manager.return_buffer(latent_data.current_end, "misc")

                    # Receive data from previous rank
                    latent_data = self.data_transfer.receive_latent_data_async(num_steps)
            
            torch.cuda.current_stream().wait_stream(self.com_stream)
            
            # Wait for outstanding operations
            while len(outstanding) >= self.config.get('max_outstanding', 1):
                oldest = outstanding.pop(0)
                for work in oldest:
                    work.wait()
            
            # Send data to next rank
            with torch.cuda.stream(self.com_stream):
                work_objects = self.data_transfer.send_latent_data_async(
                    chunk_idx=start_idx,
                    latents=denoised_pred,
                    original_latents=self.pipeline.hidden_states,
                    patched_x_shape=patched_x_shape,
                    current_start=self.pipeline.kv_cache_starts,
                    current_end=self.pipeline.kv_cache_ends,
                    current_step=current_step
                )
                outstanding.append(work_objects)
                # Handle block scheduling
                if schedule_block and self.processed >= self.schedule_step:
                    self._handle_block_scheduling(block_num, total_blocks)
                    schedule_block = False

            # Update timing and check completion
            torch.cuda.synchronize()
            end_time = time.time()
            t = end_time - start_time
            self.logger.info(f"Encode {self.processed}, time: {t:.4f} s, fps: {inp.shape[2]/t:.4f}")
            
            if schedule_block:
                t_total = self.t_dit + t_vae
                if t_total < self.t_total:
                    self.t_total = t_total

            if self.processed >= self.world_size:
                self.pipeline.hidden_states.copy_(latent_data.original_latents)
                self.pipeline.kv_cache_starts.copy_(latent_data.current_start)
                self.pipeline.kv_cache_ends.copy_(latent_data.current_end)
            
            start_time = end_time

            if self.processed + self.processed_offset >= num_chunks + num_steps * self.world_size + self.world_size - self.rank - 1:
                break
        
        self.logger.info("Rank 0 inference loop completed")
    
    def run_final_rank_loop(self, num_chunks: int, num_steps: int, chunk_size: int,
                           block_num: torch.Tensor, output_folder: str, fps: int,
                           schedule_block: bool, total_blocks: int, results: dict):
        """
        Run the main loop for the final rank (async receiver + decode).
        
        This method encapsulates the final rank logic using the communication abstractions.
        """
        self.logger.info("Starting final rank inference loop")
        
        os.makedirs(output_folder, exist_ok=True)
        save_results = 1
        
        outstanding = []

        fps_list = []
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        while save_results < num_chunks:
            # Receive data from previous rank
            with torch.cuda.stream(self.com_stream):
                if 'latent_data' in locals():
                    self.buffer_manager.return_buffer(latent_data.latents, "latent")
                    self.buffer_manager.return_buffer(latent_data.original_latents, "origin")

                    if hasattr(latent_data, 'patched_x_shape') and latent_data.patched_x_shape is not None:
                        self.buffer_manager.return_buffer(latent_data.patched_x_shape, "misc")
                    if hasattr(latent_data, 'current_start') and latent_data.current_start is not None:
                        self.buffer_manager.return_buffer(latent_data.current_start, "misc")
                    if hasattr(latent_data, 'current_end') and latent_data.current_end is not None:
                        self.buffer_manager.return_buffer(latent_data.current_end, "misc")

                latent_data = self.data_transfer.receive_latent_data_async(num_steps)
                # Handle block scheduling
                if schedule_block and self.processed >= self.schedule_step - self.rank:
                    self._handle_block_scheduling(block_num, total_blocks)
                    schedule_block = False
            torch.cuda.current_stream().wait_stream(self.com_stream)
            
            # Measure DiT time if scheduling is enabled
            if schedule_block:
                torch.cuda.synchronize()
                start_dit = time.time()
            
            # Run inference
            denoised_pred, _ = self.pipeline.inference(
                noise=latent_data.original_latents,
                current_start=latent_data.current_start,
                current_end=latent_data.current_end,
                current_step=latent_data.current_step,
                block_mode='output',
                block_num=block_num[self.rank],
                patched_x_shape=latent_data.patched_x_shape,
                block_x=latent_data.latents,
            )
            
            # Update DiT timing
            if schedule_block:
                torch.cuda.synchronize()
                temp = time.time() - start_dit
                if temp < self.t_dit:
                    self.t_dit = temp
            
            self.processed += 1
            
            # Wait for outstanding operations
            while len(outstanding) >= self.config.get('max_outstanding', 1):
                oldest = outstanding.pop(0)
                for work in oldest:
                    work.wait()
            
            # Send data to next rank (if not the last rank)
            with torch.cuda.stream(self.com_stream):
                work_objects = self.data_transfer.send_latent_data_async(
                    chunk_idx=latent_data.chunk_idx,
                    latents=latent_data.latents,
                    original_latents=denoised_pred,
                    patched_x_shape=latent_data.patched_x_shape,
                    current_start=latent_data.current_start,
                    current_end=latent_data.current_end,
                    current_step=latent_data.current_step
                )
                outstanding.append(work_objects)

            # Decode and save video
            if self.processed >= num_steps * self.world_size - 1:
                if schedule_block:
                    torch.cuda.synchronize()
                    start_vae = time.time()

                video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred[[-1]])
                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video[0].permute(0, 2, 3, 1).contiguous()
                
                results[save_results] = video.cpu().float().numpy()
                
                torch.cuda.synchronize()
                end_time = time.time()
                t = end_time - start_time
                fps_test = video.shape[0]/t
                if self.processed > self.schedule_step:
                    fps_list.append(fps_test)
                self.logger.info(f"Decode {self.processed}, time: {t:.4f} s, FPS: {fps_test:.4f}")
                
                if schedule_block:
                    t_vae = end_time - start_vae
                    t_total = t_vae + self.t_dit
                    if t_total < self.t_total:
                        self.t_total = t_total
                
                save_results += 1
                start_time = end_time
                
            if save_results >= num_chunks:
                break
        
        # Save final video
        video_list = [results[i] for i in range(num_chunks)]
        video = np.concatenate(video_list, axis=0)

        fps_list = np.array(fps_list)
        fps_avg = np.mean(fps_list)
        self.logger.info(f"Video shape: {video.shape}, Average FPS: {fps_avg:.4f}")
        
        output_path = os.path.join(output_folder, f"output_{0:03d}.mp4")
        export_to_video(video, output_path, fps=fps)
        self.logger.info(f"Video saved to: {output_path} (Press Ctrl+C to force exit)")
    
    def run_middle_rank_loop(self, num_chunks: int, num_steps: int, chunk_size: int,
                            block_num: torch.Tensor, schedule_block: bool, total_blocks: int):
        """
        Run the main loop for middle ranks (async receiver + dit blocks + sender).
        
        This method encapsulates the middle rank logic using the communication abstractions.
        """
        self.logger.info("Starting middle rank inference loop")
        
        outstanding = []
        
        torch.cuda.synchronize()
        start_time = time.time()

        fps_list = []
        
        while True:
            # Receive data from previous rank
            with torch.cuda.stream(self.com_stream):
                if 'latent_data' in locals():
                    self.buffer_manager.return_buffer(latent_data.latents, "latent")
                    self.buffer_manager.return_buffer(latent_data.original_latents, "origin")
                    if hasattr(latent_data, 'patched_x_shape') and latent_data.patched_x_shape is not None:
                        self.buffer_manager.return_buffer(latent_data.patched_x_shape, "misc")
                    if hasattr(latent_data, 'current_start') and latent_data.current_start is not None:
                        self.buffer_manager.return_buffer(latent_data.current_start, "misc")
                    if hasattr(latent_data, 'current_end') and latent_data.current_end is not None:
                        self.buffer_manager.return_buffer(latent_data.current_end, "misc")
                latent_data = self.data_transfer.receive_latent_data_async(num_steps)

                # Handle block scheduling
                if schedule_block and self.processed >= self.schedule_step - self.rank:
                    self._handle_block_scheduling(block_num, total_blocks)
                    schedule_block = False

            torch.cuda.current_stream().wait_stream(self.com_stream)

            if schedule_block:
                torch.cuda.synchronize()
                start_dit = time.time()
            
            # Run inference
            denoised_pred, _ = self.pipeline.inference(
                noise=latent_data.original_latents,
                current_start=latent_data.current_start,
                current_end=latent_data.current_end,
                current_step=latent_data.current_step,
                block_mode='middle',
                block_num=block_num[self.rank],
                patched_x_shape=latent_data.patched_x_shape,
                block_x=latent_data.latents,
            )
            
            if schedule_block:
                torch.cuda.synchronize()
                temp = time.time() - start_dit
                if temp < self.t_dit:
                    self.t_dit = temp
            
            self.processed += 1

            # Wait for outstanding operations
            while len(outstanding) >= self.config.get('max_outstanding', 1):
                oldest = outstanding.pop(0)
                for work in oldest:
                    work.wait()
            
            # Send data to next rank
            with torch.cuda.stream(self.com_stream):
                work_objects = self.data_transfer.send_latent_data_async(
                    chunk_idx=latent_data.chunk_idx,
                    latents=denoised_pred,
                    original_latents=latent_data.original_latents,
                    patched_x_shape=latent_data.patched_x_shape,
                    current_start=latent_data.current_start,
                    current_end=latent_data.current_end,
                    current_step=latent_data.current_step
                )
                outstanding.append(work_objects)
            
            # Update timing
            torch.cuda.synchronize()
            end_time = time.time()
            t = end_time - start_time

            if self.processed > self.schedule_step:
                fps_list.append(chunk_size/t)

            if schedule_block:
                t_total = self.t_dit
                if t_total < self.t_total:
                    self.t_total = t_total
            
            self.logger.info(f"Middle {self.processed}, time: {t:.4f} s, fps: {chunk_size/t:.4f}")

            start_time = end_time

            if self.processed + self.processed_offset >= num_chunks + num_steps * self.world_size + self.world_size - self.rank - 1:
                break
        
        self.logger.info(f"DiT Average FPS: {np.mean(fps_list):.4f}")
        self.logger.info(f"Rank {self.rank} inference loop completed")
    
    def _handle_block_scheduling(self, block_num: torch.Tensor, total_blocks: int):
        """Handle block scheduling and rebalancing."""
        self.logger.info(f"Scheduling block in {self.processed}")
        
        # Gather timing information from all ranks
        t_total_tensor = torch.tensor(self.t_total, dtype=torch.float32, device=self.device)
        t_dit_tensor = torch.tensor(self.t_dit, dtype=torch.float32, device=self.device)
        
        gather_blocks = [torch.zeros_like(t_dit_tensor, dtype=torch.float32, device=self.device) 
                        for _ in range(self.world_size)]
        
        dist.all_gather(gather_blocks, t_dit_tensor)
        t_dit_list = [t_dit_i.item() for t_dit_i in gather_blocks]
        
        dist.all_gather(gather_blocks, t_total_tensor)
        t_list = [t_i.item() for t_i in gather_blocks]
        
        # Compute new block distribution
        new_block_num = torch.tensor(
            compute_balanced_split(total_blocks, t_list, t_dit_list, block_num.tolist()),
            dtype=torch.int64, device=self.device
        )

        self.logger.info(f"New block distribution: {new_block_num[self.rank].tolist()}")
        
        # Broadcast new block distribution
        dist.broadcast(new_block_num, src=self.world_size - 1)
        
        # Rebalance KV cache
        self.data_transfer.rebalance_kv_cache(block_num, new_block_num, total_blocks)
        
        # Update block_num
        block_num.copy_(new_block_num)

        start_block, end_block = block_num[self.rank][0].item(), block_num[self.rank][1].item()
        blocks_to_keep = list(range(start_block, end_block))
        for i in range(self.pipeline.num_transformer_blocks):
            if i not in blocks_to_keep:
                self.pipeline.kv_cache1[i]['k'] = self.pipeline.kv_cache1[i]['k'].cpu()
                self.pipeline.kv_cache1[i]['v'] = self.pipeline.kv_cache1[i]['v'].cpu()

        self.logger.info("Block scheduling completed")
    
    def cleanup(self):
        """Clean up resources."""
        self.data_transfer.cleanup()
        self.logger.info("InferencePipelineManager cleanup completed")


def main():
    """Main function for the refactored inference pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--checkpoint_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--prompt_file_path", type=str)
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--noise_scale", type=float, default=0.8)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_outstanding", type=int, default=1, help="max number of outstanding sends/recv to keep")
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--schedule_block", action="store_true", default=False)
    parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type (e.g., T2V-1.3B)")
    
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    init_distributed()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    
    assert world_size >= 2, "world_size must be at least 2"
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Load configuration
    config = OmegaConf.load(args.config_path)
    for k, v in vars(args).items():
        config[k] = v
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
    if rank == 0:
        num_chunks = (t - 1) // chunk_size
    else:
        num_chunks = 0
    num_chunks_tensor = torch.tensor([num_chunks], dtype=torch.int64, device=device)
    dist.broadcast(num_chunks_tensor, src=0)
    num_chunks = int(num_chunks_tensor.item())
    
    # Initialize pipeline manager
    pipeline_manager = InferencePipelineManager(config, device, rank, world_size)
    pipeline_manager.load_model(args.checkpoint_folder)

    # Load prompts
    dataset = TextDataset(args.prompt_file_path)
    prompts = [dataset[0]]
    num_steps = len(pipeline_manager.pipeline.denoising_step_list)
    
    # Determine block mode and setup block distribution
    if rank == 0:
        block_mode = 'input'
    elif rank == world_size - 1:
        block_mode = 'output'
    else:
        block_mode = 'middle'
    
    # Setup block distribution
    total_blocks = pipeline_manager.pipeline.num_transformer_blocks
    if world_size == 2:
        total_block_num = [[0, 15], [15, total_blocks]]
    else:
        base = total_blocks // world_size
        rem = total_blocks % world_size
        start = 0
        total_block_num = []
        for r in range(world_size):
            size = base + (1 if r < rem else 0)
            end = start + size if r < world_size - 1 else total_blocks
            total_block_num.append([start, end])
            start = end
    
    block_num = torch.tensor(total_block_num, dtype=torch.int64, device=device)
    
    # Prepare pipeline
    start_idx = 0
    end_idx = 5
    current_start = 0
    current_end = pipeline_manager.pipeline.frame_seq_length * 2
    
    inp = input_video_original[:, :, start_idx:end_idx]
    
    # Only rank 0 performs VAE encoding operation
    if rank == 0:
        latents = pipeline_manager.pipeline.vae.stream_encode(inp)
        latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
        noise = torch.randn_like(latents)
        noisy_latents = noise * args.noise_scale + latents * (1 - args.noise_scale)
        
        # First broadcast the shape information
        latents_shape = torch.tensor(latents.shape, dtype=torch.int64, device=device)
        pipeline_manager.communicator.broadcast_tensor(latents_shape, src=0)
        # Then broadcast noisy_latents
        pipeline_manager.communicator.broadcast_tensor(noisy_latents, src=0)
    else:
        # Other ranks receive shape info first
        latents_shape = torch.zeros(5, dtype=torch.int64, device=device)
        pipeline_manager.communicator.broadcast_tensor(latents_shape, src=0)
        # Create tensor with same shape for receiving broadcast data
        noisy_latents = torch.zeros(tuple(latents_shape.tolist()), dtype=torch.bfloat16, device=device)
        # Receive the broadcasted noisy_latents
        pipeline_manager.communicator.broadcast_tensor(noisy_latents, src=0)
    
    denoised_pred = pipeline_manager.prepare_pipeline(
        text_prompts=prompts,
        noise=noisy_latents,
        block_mode=block_mode,
        current_start=current_start,
        current_end=current_end,
        block_num=block_num[rank],
    )

    # Clear unused GPU memory
    torch.cuda.empty_cache()

    # Save initial result for final rank
    if rank == world_size - 1:
        results = {}
        video = pipeline_manager.pipeline.vae.stream_decode_to_pixel(denoised_pred)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video[0].permute(0, 2, 3, 1).contiguous()
        results[0] = video.cpu().float().numpy()
    
    dist.barrier()
    pipeline_manager.logger.info(f"Prepared, Block num: {block_num[rank].tolist()}")

    used_mem = torch.cuda.memory_allocated(device) / 1024 / 1024 / 1024
    total_mem = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024 / 1024
    pipeline_manager.logger.info(f"Current GPU memory usage: {used_mem:.2f} GB / {total_mem:.2f} GB")
    
    # Run appropriate loop based on rank
    try:
        if rank == 0:
            pipeline_manager.run_rank_0_loop(
                input_video_original, prompts, num_chunks, num_steps, chunk_size,
                block_num, args.noise_scale, args.schedule_block, total_blocks
            )
        elif rank == world_size - 1:
            pipeline_manager.run_final_rank_loop(
                num_chunks, num_steps, chunk_size, block_num, args.output_folder,
                args.fps, args.schedule_block, total_blocks, results
            )
        else:
            pipeline_manager.run_middle_rank_loop(
                num_chunks, num_steps, chunk_size, block_num, args.schedule_block, total_blocks
            )
    finally:
        # Cleanup
        pipeline_manager.cleanup()
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()