"""
KV Cache management for distributed inference.

This module provides functionality for managing and rebalancing KV caches
across distributed ranks during inference.
"""

import torch
import torch.distributed as dist
from typing import List, Dict, Tuple, Optional
import logging
from .utils import CommunicationTags, CommunicationTimer
from .data_containers import KVCacheData, BlockInterval


class KVCacheManager:
    """
    Manages KV cache operations for distributed inference.
    
    This class handles KV cache broadcasting, rebalancing, and ownership
    management across distributed ranks.
    """
    
    def __init__(self, pipeline, device: torch.device):
        """
        Initialize the KV cache manager.
        
        Args:
            pipeline: The inference pipeline containing KV caches
            device: GPU device for operations
        """
        self.pipeline = pipeline
        self.device = device
        self.frame_seq_length = pipeline.frame_seq_length
        self.time_step_length = len(pipeline.denoising_step_list)
        
        # Setup logging
        self.logger = logging.getLogger(f"KVCacheManager_{device}")
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[KVCacheManager {device}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def broadcast_kv_blocks(self, block_indices: List[int], donor_rank: int) -> None:
        """
        Broadcast kv_cache1 entries for the specified block indices from donor_rank to all ranks.
        
        This ensures the receiver rank has the up-to-date KV cache when ownership moves.
        
        Args:
            block_indices: List of block indices to broadcast
            donor_rank: Rank that owns the KV cache data
        """
        if len(block_indices) == 0:
            return
        
        rank = dist.get_rank()
        
        with CommunicationTimer(f"broadcast_kv_blocks from rank {donor_rank}", self.logger):
            for bi in block_indices:
                # Broadcast key cache
                if self.pipeline.kv_cache1[bi]['k'].device != self.device:
                    self.pipeline.kv_cache1[bi]['k'] = self.pipeline.kv_cache1[bi]['k'].to(self.device)
                    self.pipeline.kv_cache1[bi]['v'] = self.pipeline.kv_cache1[bi]['v'].to(self.device)
                
                dist.barrier()

                dist.broadcast(self.pipeline.kv_cache1[bi]['k'], src=donor_rank)
                # Broadcast value cache
                dist.broadcast(self.pipeline.kv_cache1[bi]['v'], src=donor_rank)
                # Broadcast global end index
                dist.broadcast(self.pipeline.kv_cache1[bi]['global_end_index'], src=donor_rank)
                # Broadcast local end index
                dist.broadcast(self.pipeline.kv_cache1[bi]['local_end_index'], src=donor_rank)
                
                # Adjust global_end_index for the receiving rank
                if donor_rank > rank:
                    self.pipeline.kv_cache1[bi]['global_end_index'] += self.frame_seq_length * (donor_rank - rank) * self.time_step_length
        
        self.logger.debug(f"Broadcasted KV cache for blocks {block_indices} from rank {donor_rank}")
    
    def compute_block_owners(self, block_intervals: torch.Tensor, total_blocks: int) -> torch.Tensor:
        """
        Given block intervals in [start, end) format for all ranks, return a tensor
        where each entry is the owner rank of that block index.
        
        Args:
            block_intervals: Block intervals for all ranks [world_size, 2]
            total_blocks: Total number of blocks
            
        Returns:
            Tensor of length total_blocks with owner ranks
        """
        world_size = block_intervals.shape[0]
        owners = torch.full((total_blocks,), -1, dtype=torch.int64, device=block_intervals.device)
        
        for r in range(world_size):
            s = int(block_intervals[r, 0].item())
            e = int(block_intervals[r, 1].item())
            if e > s:
                owners[s:e] = r
        
        self.logger.debug(f"Computed block owners: {owners.tolist()}")
        return owners
    
    def rebalance_kv_cache_by_diff(self, old_block_intervals: torch.Tensor, 
                                  new_block_intervals: torch.Tensor, total_blocks: int) -> None:
        """
        Compare ownership from old to new intervals and broadcast KV cache for blocks whose owner changes.
        
        For each moved block i, use the previous owner's rank as src to broadcast
        pipeline.kv_cache1[i]['k'/'v'/...] to all ranks so the new owner has the correct state.
        
        Args:
            old_block_intervals: Previous block intervals [world_size, 2]
            new_block_intervals: New block intervals [world_size, 2]
            total_blocks: Total number of blocks
        """
        with CommunicationTimer("rebalance_kv_cache_by_diff", self.logger):
            old_owners = self.compute_block_owners(old_block_intervals, total_blocks)
            new_owners = self.compute_block_owners(new_block_intervals, total_blocks)
            
            # Find blocks that changed ownership
            moved_by_src = {}
            for i in range(total_blocks):
                o = int(old_owners[i].item())
                n = int(new_owners[i].item())
                if o != n and o >= 0:
                    if o not in moved_by_src:
                        moved_by_src[o] = []
                    moved_by_src[o].append(i)
            
            # Synchronize before broadcasting
            dist.barrier()
            
            # Broadcast per donor rank (can batch multiple blocks per src)
            for src, blocks in moved_by_src.items():
                self.broadcast_kv_blocks(blocks, donor_rank=src)
        
        self.logger.info(f"Rebalanced KV cache: {len(moved_by_src)} ranks had ownership changes")
    
    def get_kv_cache_statistics(self, block_intervals: torch.Tensor, total_blocks: int) -> Dict[str, any]:
        """
        Get statistics about KV cache distribution.
        
        Args:
            block_intervals: Current block intervals [world_size, 2]
            total_blocks: Total number of blocks
            
        Returns:
            Dictionary containing KV cache statistics
        """
        owners = self.compute_block_owners(block_intervals, total_blocks)
        
        # Count blocks per rank
        block_counts = {}
        for rank in range(block_intervals.shape[0]):
            block_counts[rank] = int((owners == rank).sum().item())
        
        # Calculate memory usage per rank (approximate)
        memory_per_block = 0
        if hasattr(self.pipeline, 'kv_cache1') and len(self.pipeline.kv_cache1) > 0:
            # Estimate memory per block based on first block
            first_block = self.pipeline.kv_cache1[0]
            if 'k' in first_block and 'v' in first_block:
                k_memory = first_block['k'].numel() * first_block['k'].element_size()
                v_memory = first_block['v'].numel() * first_block['v'].element_size()
                memory_per_block = k_memory + v_memory
        
        memory_usage = {rank: block_counts[rank] * memory_per_block for rank in block_counts}
        
        return {
            "block_counts": block_counts,
            "memory_usage_bytes": memory_usage,
            "total_blocks": total_blocks,
            "memory_per_block_bytes": memory_per_block,
            "frame_seq_length": self.frame_seq_length
        }
    
    def print_kv_cache_statistics(self, block_intervals: torch.Tensor, total_blocks: int) -> None:
        """
        Print KV cache statistics.
        
        Args:
            block_intervals: Current block intervals [world_size, 2]
            total_blocks: Total number of blocks
        """
        stats = self.get_kv_cache_statistics(block_intervals, total_blocks)
        
        self.logger.info("KV Cache Statistics:")
        self.logger.info(f"  Total blocks: {stats['total_blocks']}")
        self.logger.info(f"  Memory per block: {stats['memory_per_block_bytes']} bytes")
        self.logger.info(f"  Frame sequence length: {stats['frame_seq_length']}")
        
        self.logger.info("  Block distribution:")
        for rank, count in stats['block_counts'].items():
            memory_mb = stats['memory_usage_bytes'][rank] / (1024 * 1024)
            self.logger.info(f"    Rank {rank}: {count} blocks, {memory_mb:.2f} MB")
    
    def validate_kv_cache_consistency(self, block_intervals: torch.Tensor, total_blocks: int) -> bool:
        """
        Validate that KV cache ownership is consistent with block intervals.
        
        Args:
            block_intervals: Current block intervals [world_size, 2]
            total_blocks: Total number of blocks
            
        Returns:
            True if consistent, False otherwise
        """
        owners = self.compute_block_owners(block_intervals, total_blocks)
        
        # Check that all blocks have owners
        unowned_blocks = (owners == -1).sum().item()
        if unowned_blocks > 0:
            self.logger.error(f"Found {unowned_blocks} unowned blocks")
            return False
        
        # Check that block intervals are contiguous and non-overlapping
        for rank in range(block_intervals.shape[0]):
            start = int(block_intervals[rank, 0].item())
            end = int(block_intervals[rank, 1].item())
            
            if start < 0 or end > total_blocks or start >= end:
                self.logger.error(f"Invalid block interval for rank {rank}: [{start}, {end})")
                return False
            
            # Check that all blocks in this interval are owned by this rank
            for block_idx in range(start, end):
                if int(owners[block_idx].item()) != rank:
                    self.logger.error(f"Block {block_idx} not owned by rank {rank}")
                    return False
        
        self.logger.debug("KV cache consistency validation passed")
        return True
    
    def cleanup_kv_cache(self, block_intervals: torch.Tensor, total_blocks: int) -> None:
        """
        Clean up KV cache for blocks not owned by current rank.
        
        Args:
            block_intervals: Current block intervals [world_size, 2]
            total_blocks: Total number of blocks
        """
        rank = dist.get_rank()
        owners = self.compute_block_owners(block_intervals, total_blocks)
        
        cleaned_blocks = 0
        for block_idx in range(total_blocks):
            if int(owners[block_idx].item()) != rank:
                # Clear KV cache for blocks not owned by this rank
                if hasattr(self.pipeline, 'kv_cache1') and block_idx < len(self.pipeline.kv_cache1):
                    if 'k' in self.pipeline.kv_cache1[block_idx]:
                        self.pipeline.kv_cache1[block_idx]['k'].zero_()
                    if 'v' in self.pipeline.kv_cache1[block_idx]:
                        self.pipeline.kv_cache1[block_idx]['v'].zero_()
                    cleaned_blocks += 1
        
        self.logger.info(f"Cleaned up KV cache for {cleaned_blocks} blocks not owned by rank {rank}")
