"""
Refactored multi-rank inference pipeline with communication abstractions.

This is a refactored version of inference_pipe_multi.py that uses the new
communication abstraction layers for better code organization and maintainability.
"""

from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from causvid.util import set_seed
from diffusers.utils import export_to_video
from causvid.data import TextDataset
import argparse
from dataclasses import dataclass
import torch
import torch.distributed as dist
import os
import time
import numpy as np
import logging

try:
    from streamv2v.inference import compute_noise_scale_and_step
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
    from streamv2v.inference_common import (
        load_generator_state_dict,
        load_mp4_as_tensor,
        merge_cli_config,
    )
except ModuleNotFoundError:
    from inference import compute_noise_scale_and_step
    from communication import (
        DistributedCommunicator,
        ModelDataTransfer,
        BufferManager,
        KVCacheManager,
        CommunicationConfig,
        init_distributed,
        setup_logging,
        compute_balanced_split
    )
    from inference_common import load_generator_state_dict, load_mp4_as_tensor, merge_cli_config

LOGGER = logging.getLogger(__name__)


def compute_default_block_distribution(total_blocks: int, world_size: int) -> list[list[int]]:
    """Split transformer blocks into contiguous ranges for each rank."""
    if world_size == 2:
        midpoint = total_blocks // 2
        return [[0, midpoint], [midpoint, total_blocks]]

    base = total_blocks // world_size
    rem = total_blocks % world_size
    start = 0
    block_ranges = []
    for rank in range(world_size):
        size = base + (1 if rank < rem else 0)
        end = start + size if rank < world_size - 1 else total_blocks
        block_ranges.append([start, end])
        start = end
    return block_ranges


@dataclass
class MultiGPUDemoInputSession:
    prompt: str
    noise_scale: float
    init_noise_scale: float
    chunk_size: int
    current_start: int
    current_end: int
    last_image: torch.Tensor
    chunk_idx: int = 0
    input_batch: int = 0
    current_step: int = 0
    noisy_latents: torch.Tensor | None = None


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
        self.profile = bool(config.get('profile', False))
        
        self.logger.info(f"Initialized InferencePipelineManager for rank {rank}")
    
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

    def _wait_for_outstanding(self, outstanding: list) -> None:
        """Keep the number of queued async sends bounded."""
        while len(outstanding) >= self.config.get('max_outstanding', 1):
            oldest = outstanding.pop(0)
            for work in oldest:
                work.wait()

    def _drain_outstanding(self, outstanding: list) -> None:
        """Wait for all queued async sends to complete."""
        while outstanding:
            oldest = outstanding.pop(0)
            for work in oldest:
                work.wait()

    def _maybe_schedule_blocks(self, schedule_block: bool, threshold: int, block_num: torch.Tensor, total_blocks: int) -> bool:
        """Run one-time block rebalancing when the warmup threshold is reached."""
        if schedule_block and self.processed >= threshold:
            self._handle_block_scheduling(block_num, total_blocks)
            return False
        return schedule_block

    def _receive_latent_data(self, previous_latent_data, num_steps: int):
        """Release the previous payload and receive the next one from the upstream rank."""
        with torch.cuda.stream(self.com_stream):
            if previous_latent_data is not None:
                self.data_transfer.release_latent_data(previous_latent_data)
            latent_data = self.data_transfer.receive_latent_data_async(num_steps)
        torch.cuda.current_stream().wait_stream(self.com_stream)
        return latent_data

    def _run_worker_stage(self, role: str, latent_data, block_num: torch.Tensor):
        """Execute the local DiT blocks for a middle or output rank."""
        return self.pipeline.inference(
            noise=latent_data.original_latents,
            current_start=latent_data.current_start,
            current_end=latent_data.current_end,
            current_step=latent_data.current_step,
            block_mode=role,
            block_num=block_num,
            patched_x_shape=latent_data.patched_x_shape,
            block_x=latent_data.latents,
        )

    def _send_worker_result(self, role: str, outstanding: list, latent_data, denoised_pred: torch.Tensor) -> None:
        """Forward the payload that should continue around the pipeline ring."""
        if role == 'output':
            latents = latent_data.latents
            original_latents = denoised_pred
        else:
            latents = denoised_pred
            original_latents = latent_data.original_latents

        with torch.cuda.stream(self.com_stream):
            work_objects = self.data_transfer.send_latent_data_async(
                chunk_idx=latent_data.chunk_idx,
                latents=latents,
                original_latents=original_latents,
                patched_x_shape=latent_data.patched_x_shape,
                current_start=latent_data.current_start,
                current_end=latent_data.current_end,
                current_step=latent_data.current_step
            )
            outstanding.append(work_objects)

    def _decode_prediction(self, denoised_pred: torch.Tensor) -> np.ndarray:
        """Decode the newest latent prediction into pixel-space frames."""
        video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred[[-1]])
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video[0].permute(0, 2, 3, 1).contiguous()
        return video.cpu().float().numpy()

    def _rank_loop_complete(self, num_chunks: int, num_steps: int) -> bool:
        """Return whether a non-output rank has processed all required chunks."""
        return (
            self.processed + self.processed_offset
            >= num_chunks + num_steps * self.world_size + self.world_size - self.rank - 1
        )

    def _safe_mean(self, values: list) -> float:
        if not values:
            return 0.0
        return float(np.mean(np.array(values)))

    def _timing_enabled(self, schedule_block: bool = False) -> bool:
        """Only force GPU synchronization when profiling or schedule calibration needs it."""
        return self.profile or schedule_block

    def _sync_for_timing(self, schedule_block: bool = False) -> None:
        if self._timing_enabled(schedule_block):
            torch.cuda.synchronize()

    def reset_stream_state(self, reset_encode: bool = False, reset_decode: bool = False) -> None:
        """Reset cached inference state before starting a new prompt/session."""
        self.pipeline.kv_cache1 = None
        self.pipeline.crossattn_cache = None
        self.pipeline.block_x = None
        self.pipeline.hidden_states = None
        self.processed = 0

        if reset_encode:
            self.pipeline.vae.model.first_encode = True
        if reset_decode:
            self.pipeline.vae.model.first_decode = True

    def _broadcast_initial_noise(self, noisy_latents: torch.Tensor) -> None:
        latents_shape = torch.tensor(noisy_latents.shape, dtype=torch.int64, device=self.device)
        self.communicator.broadcast_tensor(latents_shape, src=0)
        self.communicator.broadcast_tensor(noisy_latents, src=0)

    def _receive_initial_noise(self) -> torch.Tensor:
        latents_shape = torch.zeros(5, dtype=torch.int64, device=self.device)
        self.communicator.broadcast_tensor(latents_shape, src=0)
        noisy_latents = torch.zeros(tuple(latents_shape.tolist()), dtype=torch.bfloat16, device=self.device)
        self.communicator.broadcast_tensor(noisy_latents, src=0)
        return noisy_latents

    def get_demo_chunk_size(self) -> int:
        """Return the demo stream chunk size in frames."""
        return self.base_chunk_size * self.pipeline.num_frame_per_block

    def get_demo_first_batch_num_frames(self) -> int:
        """Return the number of frames required to initialize a demo stream."""
        return 1 + self.get_demo_chunk_size()

    def prepare_demo_input_session(self, images: torch.Tensor, prompt: str, block_num: torch.Tensor, noise_scale: float) -> None:
        """Initialize rank 0 for demo streaming and broadcast the first noisy latents."""
        self.reset_stream_state(reset_encode=True)
        torch.cuda.empty_cache()

        latents = self.pipeline.vae.stream_encode(images)
        latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
        noise = torch.randn_like(latents)
        noisy_latents = noise * noise_scale + latents * (1 - noise_scale)

        self._broadcast_initial_noise(noisy_latents)
        self.prepare_pipeline(
            text_prompts=[prompt],
            noise=noisy_latents,
            block_mode='input',
            current_start=0,
            current_end=self.pipeline.frame_seq_length * 2,
            block_num=block_num,
        )
        torch.cuda.empty_cache()
        dist.barrier()

    def start_demo_input_stream_session(
        self,
        prompt: str,
        images: torch.Tensor,
        block_num: torch.Tensor,
        noise_scale: float,
    ) -> MultiGPUDemoInputSession:
        """Initialize rank 0 and return the demo stream session state."""
        chunk_size = self.get_demo_chunk_size()
        self.prepare_demo_input_session(images, prompt, block_num, noise_scale)
        current_start = self.pipeline.frame_seq_length * (1 + chunk_size // self.base_chunk_size)
        current_end = current_start + (chunk_size // self.base_chunk_size) * self.pipeline.frame_seq_length
        return MultiGPUDemoInputSession(
            prompt=prompt,
            noise_scale=noise_scale,
            init_noise_scale=noise_scale,
            chunk_size=chunk_size,
            current_start=current_start,
            current_end=current_end,
            last_image=images[:, :, [-1]],
        )

    def prepare_demo_worker_session(self, prompt: str, block_mode: str, block_num: torch.Tensor, decode_initial: bool = False):
        """Initialize a non-input rank for demo streaming from the broadcast first chunk."""
        self.reset_stream_state(reset_decode=(block_mode == 'output'))
        torch.cuda.empty_cache()

        noisy_latents = self._receive_initial_noise()
        denoised_pred = self.prepare_pipeline(
            text_prompts=[prompt],
            noise=noisy_latents,
            block_mode=block_mode,
            current_start=0,
            current_end=self.pipeline.frame_seq_length * 2,
            block_num=block_num,
        )
        torch.cuda.empty_cache()
        dist.barrier()

        if decode_initial:
            return self._decode_prediction(denoised_pred)
        return None

    def maybe_refresh_demo_input_window(self, session: MultiGPUDemoInputSession) -> None:
        """Wrap the KV-cache window once the streaming refresh threshold is reached."""
        if session.current_start // self.pipeline.frame_seq_length >= self.t_refresh:
            session.current_start = self.pipeline.kv_cache_length - self.pipeline.frame_seq_length
            session.current_end = session.current_start + (session.chunk_size // self.base_chunk_size) * self.pipeline.frame_seq_length

    def prepare_demo_input_batch(self, session: MultiGPUDemoInputSession, images: torch.Tensor) -> None:
        """Encode one demo chunk and update the session with the current denoising step."""
        num_frames = images.shape[2]
        session.input_batch = num_frames // session.chunk_size
        session.noise_scale, session.current_step = compute_noise_scale_and_step(
            input_video_original=torch.cat([session.last_image, images], dim=2),
            end_idx=num_frames + 1,
            chunk_size=num_frames,
            noise_scale=float(session.noise_scale),
            init_noise_scale=float(session.init_noise_scale),
        )

        latents = self.pipeline.vae.stream_encode(images)
        latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
        noise = torch.randn_like(latents)
        session.noisy_latents = noise * session.noise_scale + latents * (1 - session.noise_scale)

    def run_demo_input_step(
        self,
        session: MultiGPUDemoInputSession,
        block_num: torch.Tensor,
        previous_latent_data=None,
    ):
        """Run one rank-0 demo step from the current session batch."""
        if session.noisy_latents is None or session.input_batch <= 0:
            raise RuntimeError("demo input batch was not prepared before run_demo_input_step")

        denoised_pred, patched_x_shape = self.run_input_stage(
            noisy_latents=session.noisy_latents[:, -session.input_batch].unsqueeze(1),
            current_start=session.current_start,
            current_end=session.current_end,
            current_step=session.current_step,
            block_num=block_num,
            previous_latent_data=previous_latent_data,
        )
        session.input_batch -= 1
        return denoised_pred, patched_x_shape

    def advance_demo_input_stream_session(self, session: MultiGPUDemoInputSession, images: torch.Tensor) -> None:
        """Advance the demo stream session after a chunk has been queued downstream."""
        session.last_image = images[:, :, [-1]]
        session.chunk_idx += 1
        session.current_start = session.current_end
        session.current_end += (session.chunk_size // self.base_chunk_size) * self.pipeline.frame_seq_length

    def send_demo_input_prompt_update(
        self,
        prompt: str,
        device: torch.device,
        num_steps: int,
        chunk_idx: int,
        denoised_pred: torch.Tensor,
        patched_x_shape: torch.Tensor,
        current_step: int,
    ) -> None:
        """Signal a prompt restart from rank 0 and drain in-flight returns from downstream ranks."""
        with torch.cuda.stream(self.com_stream):
            self.data_transfer.send_latent_data_async(
                chunk_idx=-1,
                latents=denoised_pred.new_zeros([1] * denoised_pred.ndim),
                original_latents=self.pipeline.hidden_states.new_zeros([1] * self.pipeline.hidden_states.ndim),
                patched_x_shape=patched_x_shape,
                current_start=self.pipeline.kv_cache_starts,
                current_end=self.pipeline.kv_cache_ends,
                current_step=int(current_step),
            )
            self.data_transfer.send_prompt_async(prompt, device)
            for _ in range(min(chunk_idx, self.world_size - 1)):
                pending_data = self.data_transfer.receive_latent_data_async(num_steps)
                self.data_transfer.release_latent_data(pending_data)

    def send_demo_middle_prompt_update(
        self,
        prompt: str,
        device: torch.device,
        denoised_pred: torch.Tensor,
        latent_data,
    ) -> None:
        """Forward a prompt restart from a middle rank to the next rank."""
        with torch.cuda.stream(self.com_stream):
            self.data_transfer.send_latent_data_async(
                chunk_idx=-1,
                latents=denoised_pred.new_zeros([1] * denoised_pred.ndim),
                original_latents=latent_data.original_latents,
                patched_x_shape=latent_data.patched_x_shape,
                current_start=latent_data.current_start,
                current_end=latent_data.current_end,
                current_step=int(latent_data.current_step),
            )
            self.data_transfer.send_prompt_async(prompt, device)

    def run_input_stage(self, noisy_latents: torch.Tensor, current_start: int, current_end: int, current_step: int, block_num: torch.Tensor, previous_latent_data=None):
        """Run the rank-0 stage for one streaming chunk."""
        if previous_latent_data is not None and self.processed >= self.world_size:
            self.pipeline.hidden_states.copy_(previous_latent_data.original_latents)
            self.pipeline.kv_cache_starts.copy_(previous_latent_data.current_start)
            self.pipeline.kv_cache_ends.copy_(previous_latent_data.current_end)

        return self.pipeline.inference(
            noise=noisy_latents,
            current_start=current_start,
            current_end=current_end,
            current_step=current_step,
            block_mode='input',
            block_num=block_num,
        )
    
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
        
        self._sync_for_timing(schedule_block)
        start_time = time.time()
        
        while True:
            # Process new chunk if available
            start_idx = end_idx
            end_idx = end_idx + chunk_size
            current_start = current_end
            current_end = current_end + (chunk_size // self.base_chunk_size) * self.pipeline.frame_seq_length

            if schedule_block:
                self._sync_for_timing(schedule_block)
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
                self._sync_for_timing(schedule_block)
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
                self._sync_for_timing(schedule_block)
                temp = time.time() - start_dit
                if temp < self.t_dit:
                    self.t_dit = temp
            
            self.processed += 1
            
            with torch.cuda.stream(self.com_stream):
                if self.processed >= self.world_size:
                    if 'latent_data' in locals():
                        self.data_transfer.release_latent_data(latent_data)

                    # Receive data from previous rank
                    latent_data = self.data_transfer.receive_latent_data_async(num_steps)
            
            torch.cuda.current_stream().wait_stream(self.com_stream)
            
            # Wait for outstanding operations
            self._wait_for_outstanding(outstanding)
            
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
            if self._timing_enabled(schedule_block):
                self._sync_for_timing(schedule_block)
                end_time = time.time()
                t = end_time - start_time
                self.logger.info(f"Encode {self.processed}, time: {t:.4f} s, fps: {inp.shape[2]/t:.4f}")

                if schedule_block:
                    t_total = self.t_dit + t_vae
                    if t_total < self.t_total:
                        self.t_total = t_total
                start_time = end_time

            if self.processed >= self.world_size:
                self.pipeline.hidden_states.copy_(latent_data.original_latents)
                self.pipeline.kv_cache_starts.copy_(latent_data.current_start)
                self.pipeline.kv_cache_ends.copy_(latent_data.current_end)

            if self.processed + self.processed_offset >= num_chunks + num_steps * self.world_size + self.world_size - self.rank - 1:
                break

        if 'latent_data' in locals():
            self.data_transfer.release_latent_data(latent_data)
        self._drain_outstanding(outstanding)
        
        self.logger.info("Rank 0 inference loop completed")
    
    def run_final_rank_loop(self, num_chunks: int, num_steps: int, chunk_size: int,
                           block_num: torch.Tensor, output_folder: str, fps: int,
                           schedule_block: bool, total_blocks: int, results: dict):
        """Run the worker loop for the output rank."""
        self.run_worker_rank_loop(
            role='output',
            num_chunks=num_chunks,
            num_steps=num_steps,
            chunk_size=chunk_size,
            block_num=block_num,
            schedule_block=schedule_block,
            total_blocks=total_blocks,
            output_folder=output_folder,
            fps=fps,
            results=results,
        )
    
    def run_middle_rank_loop(self, num_chunks: int, num_steps: int, chunk_size: int,
                            block_num: torch.Tensor, schedule_block: bool, total_blocks: int):
        """Run the worker loop for a middle rank."""
        self.run_worker_rank_loop(
            role='middle',
            num_chunks=num_chunks,
            num_steps=num_steps,
            chunk_size=chunk_size,
            block_num=block_num,
            schedule_block=schedule_block,
            total_blocks=total_blocks,
        )

    def run_worker_rank_loop(
        self,
        role: str,
        num_chunks: int,
        num_steps: int,
        chunk_size: int,
        block_num: torch.Tensor,
        schedule_block: bool,
        total_blocks: int,
        output_folder: str = None,
        fps: int = None,
        results: dict = None,
    ):
        """Run the shared receive -> infer -> forward loop for middle and output ranks."""
        if role not in {'middle', 'output'}:
            raise ValueError(f"Unsupported worker role: {role}")

        self.logger.info(f"Starting {role} rank inference loop")

        if role == 'output':
            if output_folder is None or fps is None or results is None:
                raise ValueError("output rank requires output_folder, fps, and results")
            os.makedirs(output_folder, exist_ok=True)
            save_results = 1

        outstanding = []
        fps_list = []
        latent_data = None

        self._sync_for_timing(schedule_block)
        start_time = time.time()

        while True:
            latent_data = self._receive_latent_data(latent_data, num_steps)
            schedule_block = self._maybe_schedule_blocks(
                schedule_block,
                self.schedule_step - self.rank,
                block_num,
                total_blocks,
            )

            if schedule_block:
                self._sync_for_timing(schedule_block)
                start_dit = time.time()

            denoised_pred, _ = self._run_worker_stage(role, latent_data, block_num[self.rank])

            if schedule_block:
                self._sync_for_timing(schedule_block)
                temp = time.time() - start_dit
                if temp < self.t_dit:
                    self.t_dit = temp

            self.processed += 1
            self._wait_for_outstanding(outstanding)
            self._send_worker_result(role, outstanding, latent_data, denoised_pred)

            if role == 'output':
                if self.processed >= num_steps * self.world_size - 1:
                    if schedule_block:
                        self._sync_for_timing(schedule_block)
                        start_vae = time.time()

                    video = self._decode_prediction(denoised_pred)
                    results[save_results] = video

                    if self._timing_enabled(schedule_block):
                        self._sync_for_timing(schedule_block)
                        end_time = time.time()
                        elapsed = end_time - start_time
                        fps_test = video.shape[0] / elapsed
                        if self.processed > self.schedule_step:
                            fps_list.append(fps_test)
                        self.logger.info(f"Decode {self.processed}, time: {elapsed:.4f} s, FPS: {fps_test:.4f}")

                        if schedule_block:
                            t_vae = end_time - start_vae
                            t_total = t_vae + self.t_dit
                            if t_total < self.t_total:
                                self.t_total = t_total
                        start_time = end_time
                    save_results += 1

                if save_results >= num_chunks:
                    break
            else:
                if self._timing_enabled(schedule_block):
                    self._sync_for_timing(schedule_block)
                    end_time = time.time()
                    elapsed = end_time - start_time
                    fps_test = chunk_size / elapsed

                    if self.processed > self.schedule_step:
                        fps_list.append(fps_test)

                    if schedule_block:
                        t_total = self.t_dit
                        if t_total < self.t_total:
                            self.t_total = t_total

                    self.logger.info(f"Middle {self.processed}, time: {elapsed:.4f} s, fps: {fps_test:.4f}")
                    start_time = end_time

                if self._rank_loop_complete(num_chunks, num_steps):
                    break

        if latent_data is not None:
            self.data_transfer.release_latent_data(latent_data)
        self._drain_outstanding(outstanding)

        if role == 'output':
            video_list = [results[i] for i in range(num_chunks)]
            video = np.concatenate(video_list, axis=0)
            fps_avg = self._safe_mean(fps_list)
            self.logger.info(f"Video shape: {video.shape}, Average FPS: {fps_avg:.4f}")

            output_path = os.path.join(output_folder, f"output_{0:03d}.mp4")
            export_to_video(video, output_path, fps=fps)
            self.logger.info(f"Video saved to: {output_path} (Press Ctrl+C to force exit)")
            return

        self.logger.info(f"DiT Average FPS: {self._safe_mean(fps_list):.4f}")
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--schedule_block", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False, help="Enable synchronized throughput logging")
    parser.add_argument("--t2v", action="store_true", default=False)
    parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type (e.g., T2V-1.3B)")
    parser.add_argument("--use_taehv", action="store_true", default=False, help="Use the lightweight TAEHV VAE for encode/decode")
    
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
    total_block_num = compute_default_block_distribution(total_blocks, world_size)
    
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
