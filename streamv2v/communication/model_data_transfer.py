"""
Model data transfer abstraction layer.

This module provides high-level interfaces for transferring model data
between distributed ranks during inference.
"""

import torch
from typing import List, Tuple, Optional, Any
import logging
from .distributed_communicator import DistributedCommunicator
from .buffer_manager import BufferManager
from .kv_cache_manager import KVCacheManager
from .data_containers import LatentData, CommunicationConfig, PerformanceMetrics
from .utils import CommunicationTimer


class ModelDataTransfer:
    """
    High-level interface for model data transfer operations.
    
    This class encapsulates all model-related data transfer operations,
    providing a clean interface for sending and receiving latent data,
    KV caches, and other model state between ranks.
    """
    
    def __init__(self, communicator: DistributedCommunicator, 
                 buffer_manager: BufferManager,
                 kv_cache_manager: Optional[KVCacheManager] = None,
                 config: Optional[CommunicationConfig] = None):
        """
        Initialize the model data transfer manager.
        
        Args:
            communicator: Distributed communicator instance
            buffer_manager: Buffer manager for tensor allocation
            kv_cache_manager: KV cache manager (optional)
            config: Communication configuration
        """
        self.comm = communicator
        self.buffer_mgr = buffer_manager
        self.kv_cache_mgr = kv_cache_manager
        self.config = config or CommunicationConfig()
        
        # Setup logging
        self.logger = logging.getLogger(f"ModelDataTransfer_rank_{communicator.rank}")
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[Rank {communicator.rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Performance tracking
        self.transfer_count = 0
        self.total_transfer_time = 0.0
    
    def send_latent_data_async(self, chunk_idx: int, latents: torch.Tensor,
                             original_latents: torch.Tensor, patched_x_shape: torch.Tensor,
                             current_start: torch.Tensor, current_end: torch.Tensor,
                             current_step: int) -> List[Any]:
        """
        Asynchronously send latent data to the next rank.
        
        Args:
            chunk_idx: Chunk index
            latents: Latent tensor
            original_latents: Original latent tensor
            patched_x_shape: Patched x shape tensor
            current_start: Current start indices
            current_end: Current end indices
            current_step: Current step
            
        Returns:
            List of work objects for all send operations
        """
        with CommunicationTimer(f"send_latent_data_async chunk_{chunk_idx}", self.logger):
            work_objects = self.comm.send_latent_data_async(
                chunk_idx=chunk_idx,
                latents=latents,
                original_latents=original_latents,
                patched_x_shape=patched_x_shape,
                current_start=current_start,
                current_end=current_end,
                current_step=current_step
            )
        
        self.transfer_count += 1
        self.logger.debug(f"Sent latent data for chunk {chunk_idx}")
        return work_objects
    
    def receive_latent_data_async(self, num_steps: int) -> LatentData:
        """
        Asynchronously receive latent data from the previous rank.
        
        Args:
            num_steps: Number of denoising steps
            
        Returns:
            LatentData object containing all received data
        """
        with CommunicationTimer("receive_latent_data_async", self.logger):
            chunk_idx, latents, original_latents, current_start, current_end, current_step, patched_x_shape = \
                self.comm.recv_latent_data_async(num_steps, self.buffer_mgr)
        
        self.transfer_count += 1
        self.logger.debug(f"Received latent data for chunk {chunk_idx}")
        
        return LatentData(
            chunk_idx=chunk_idx,
            latents=latents,
            original_latents=original_latents,
            current_start=current_start,
            current_end=current_end,
            current_step=current_step,
            patched_x_shape=patched_x_shape
        )

    def send_prompt_async(self, prompt: str, device: torch.device) -> List[Any]:
        return self.comm.send_prompt_async(prompt, device)

    def recv_prompt_async(self) -> str:
        return self.comm.recv_prompt_async()
    
    def send_kv_cache_blocks(self, block_indices: List[int], donor_rank: int) -> None:
        """
        Send KV cache blocks to all ranks.
        
        Args:
            block_indices: List of block indices to send
            donor_rank: Rank that owns the KV cache data
        """
        if self.kv_cache_mgr is None:
            raise RuntimeError("KV cache manager not initialized")
        
        with CommunicationTimer(f"send_kv_cache_blocks {len(block_indices)} blocks", self.logger):
            self.kv_cache_mgr.broadcast_kv_blocks(block_indices, donor_rank)
        
        self.logger.debug(f"Sent KV cache blocks {block_indices} from rank {donor_rank}")
    
    def rebalance_kv_cache(self, old_intervals: torch.Tensor, 
                          new_intervals: torch.Tensor, total_blocks: int) -> None:
        """
        Rebalance KV cache ownership based on new block intervals.
        
        Args:
            old_intervals: Previous block intervals [world_size, 2]
            new_intervals: New block intervals [world_size, 2]
            total_blocks: Total number of blocks
        """
        if self.kv_cache_mgr is None:
            raise RuntimeError("KV cache manager not initialized")
        
        with CommunicationTimer("rebalance_kv_cache", self.logger):
            self.kv_cache_mgr.rebalance_kv_cache_by_diff(old_intervals, new_intervals, total_blocks)
        
        self.logger.info("Rebalanced KV cache ownership")
    
    def broadcast_tensor(self, tensor: torch.Tensor, src: int) -> None:
        """
        Broadcast a tensor from source to all ranks.
        
        Args:
            tensor: Tensor to broadcast
            src: Source rank
        """
        with CommunicationTimer(f"broadcast_tensor from rank {src}", self.logger):
            self.comm.broadcast_tensor(tensor, src)
        
        self.logger.debug(f"Broadcasted tensor from rank {src}, shape: {tensor.shape}")
    
    def all_gather_tensors(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Gather tensors from all ranks.
        
        Args:
            tensor: Local tensor to gather
            
        Returns:
            List of tensors from all ranks
        """
        with CommunicationTimer("all_gather_tensors", self.logger):
            gather_list = self.comm.all_gather_tensors(tensor)
        
        self.logger.debug(f"Gathered tensors from all ranks, local shape: {tensor.shape}")
        return gather_list
    
    def wait_for_outstanding(self, max_outstanding: Optional[int] = None) -> None:
        """
        Wait for outstanding operations to complete.
        
        Args:
            max_outstanding: Maximum number of outstanding operations to keep
        """
        with CommunicationTimer("wait_for_outstanding", self.logger):
            self.comm.wait_for_outstanding(max_outstanding)
    
    def barrier(self) -> None:
        """Synchronize all ranks."""
        with CommunicationTimer("barrier", self.logger):
            self.comm.barrier()
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Get performance metrics for data transfer operations.
        
        Returns:
            PerformanceMetrics object containing timing information
        """
        # This is a simplified version - in practice, you'd want to track
        # more detailed timing information
        avg_transfer_time = self.total_transfer_time / max(1, self.transfer_count)
        
        return PerformanceMetrics(
            dit_time=0.0,  # Would be filled by caller
            total_time=0.0,  # Would be filled by caller
            communication_time=avg_transfer_time,
            buffer_allocation_time=0.0  # Would be tracked by buffer manager
        )
    
    def get_statistics(self) -> dict:
        """
        Get transfer statistics.
        
        Returns:
            Dictionary containing transfer statistics
        """
        return {
            "transfer_count": self.transfer_count,
            "total_transfer_time": self.total_transfer_time,
            "avg_transfer_time": self.total_transfer_time / max(1, self.transfer_count),
            "communicator_stats": self.comm.get_statistics(),
            "buffer_manager_stats": self.buffer_mgr.get_statistics() if self.buffer_mgr else None
        }
    
    def print_statistics(self) -> None:
        """Print transfer statistics."""
        stats = self.get_statistics()
        self.logger.info("Model Data Transfer Statistics:")
        for key, value in stats.items():
            if key == "communicator_stats" or key == "buffer_manager_stats":
                if value:
                    self.logger.info(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        self.logger.info(f"    {sub_key}: {sub_value}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.buffer_mgr:
            self.buffer_mgr.clear_buffers()
        self.logger.info("Model data transfer cleanup completed")
    
    def __del__(self):
        """Cleanup when the transfer manager is destroyed."""
        self.cleanup()
