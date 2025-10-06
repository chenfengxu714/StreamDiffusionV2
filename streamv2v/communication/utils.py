"""
Utility functions and constants for communication operations.

This module provides utility functions and constants used across the communication module.
"""

import torch
import torch.distributed as dist
from typing import List, Tuple, Optional
import time
import logging

# Communication tags for different types of data
class CommunicationTags:
    """Constants for communication tags."""
    LATENT_HDR = 11001
    LATENT_PAY = 11002
    START_END_STEP = 11003
    PATCHED_X_SHAPE = 11004
    LATENT_ORIGIN_HDR = 11005
    LATENT_ORIGIN_PAY = 11006
    KV_CACHE_K = 11007
    KV_CACHE_V = 11008
    KV_CACHE_GLOBAL_END = 11009
    KV_CACHE_LOCAL_END = 11010
    BLOCK_INTERVALS = 11011
    PERFORMANCE_METRICS = 11012
    UPDATED_PROMPT_LENGTH = 11013
    UPDATED_PROMPT = 11014


def init_distributed():
    """
    Initialize distributed communication.
    
    This function initializes the distributed process group if not already initialized.
    """
    if not dist.is_initialized():
        backend = "nccl"
        dist.init_process_group(backend=backend)


def get_rank_info() -> Tuple[int, int]:
    """
    Get current rank and world size.
    
    Returns:
        Tuple of (rank, world_size)
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed not initialized")
    return dist.get_rank(), dist.get_world_size()


def get_next_rank(rank: int, world_size: int) -> int:
    """
    Get the next rank in the ring topology.
    
    Args:
        rank: Current rank
        world_size: Total number of ranks
        
    Returns:
        Next rank in the ring
    """
    return (rank + 1) % world_size


def get_prev_rank(rank: int, world_size: int) -> int:
    """
    Get the previous rank in the ring topology.
    
    Args:
        rank: Current rank
        world_size: Total number of ranks
        
    Returns:
        Previous rank in the ring
    """
    return (rank - 1) % world_size


def create_tensor_header(shape: Tuple[int, ...], dtype: torch.dtype, 
                        chunk_idx: int, device: torch.device) -> torch.Tensor:
    """
    Create a header tensor for communication.
    
    Args:
        shape: Shape of the tensor to be sent
        dtype: Data type of the tensor
        chunk_idx: Chunk index
        device: Device where the header will be created
        
    Returns:
        Header tensor containing metadata
    """
    header_data = [chunk_idx] + list(shape)
    return torch.tensor(header_data, dtype=torch.int64, device=device)


def parse_tensor_header(header: torch.Tensor) -> Tuple[int, Tuple[int, ...]]:
    """
    Parse a header tensor to extract metadata.
    
    Args:
        header: Header tensor
        
    Returns:
        Tuple of (chunk_idx, shape)
    """
    header_list = header.tolist()
    chunk_idx = int(header_list[0])
    shape = tuple(int(x) for x in header_list[1:])
    return chunk_idx, shape


def validate_tensor_for_communication(tensor: torch.Tensor, 
                                    expected_device: torch.device,
                                    expected_dtype: torch.dtype) -> None:
    """
    Validate tensor properties for communication.
    
    Args:
        tensor: Tensor to validate
        expected_device: Expected device
        expected_dtype: Expected data type
        
    Raises:
        ValueError: If tensor properties don't match expectations
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    
    if tensor.device != expected_device:
        raise ValueError(f"Tensor device {tensor.device} doesn't match expected {expected_device}")
    
    if tensor.dtype != expected_dtype:
        raise ValueError(f"Tensor dtype {tensor.dtype} doesn't match expected {expected_dtype}")


def compute_balanced_split(total_blocks: int, rank_times: List[float], 
                          dit_times: List[float], 
                          current_block_nums: List[List[int]]) -> List[List[int]]:
    """
    Compute new block splits for all ranks to balance total rank times.
    
    This function is moved from the original file to provide better organization.
    
    Args:
        total_blocks: Total number of DiT blocks
        rank_times: List of total iteration times for each rank [t_rank0, t_rank1, ..., t_rankN] (DiT + VAE time)
        dit_times: List of pure DiT inference times for each rank [dit_rank0, dit_rank1, ..., dit_rankN] (DiT time only)
        current_block_nums: List of current block_num format for each rank [[rank0_blocks], [rank1_blocks], ...]
        
    Returns:
        List of new block_num format for each rank, matching the original format:
        - For world_size == 2: [[end_idx_rank0], [start_idx_rank1]]
        - For world_size > 2: [[end_idx_rank0], [start1, end1], [start2, end2], ..., [start_idx_last]]
        Note: Numbers are shared across ranks (rank0_end = rank1_start, rank1_end = rank2_start, etc.)
    """
    num_ranks = len(rank_times)
    if num_ranks == 0 or num_ranks != len(current_block_nums) or num_ranks != len(dit_times):
        return current_block_nums
    
    # Edge case: if we have more ranks than blocks, we can't guarantee 1 block per rank
    if num_ranks > total_blocks:
        # Fall back to original behavior for this edge case
        return current_block_nums

    # Step 1: Calculate total DiT time and per-block DiT time
    total_dit_time = sum(dit_times)
    dit_time_per_block = total_dit_time / total_blocks
    
    # Step 2: Calculate average rank time
    avg_rank_time = sum(rank_times) / num_ranks
    
    # Step 3: Extract current block counts from current_block_nums (all ranks use [start, end) now)
    current_block_counts = []
    for block_num in current_block_nums:
        # block_num: [start, end) exclusive end
        start_idx, end_idx = int(block_num[0]), int(block_num[1])
        current_block_counts.append(max(0, end_idx - start_idx))
    
    # Step 4: Calculate target block counts based on time differences
    target_blocks = []
    for i in range(num_ranks):
        time_diff = avg_rank_time - rank_times[i]  # positive = needs more time, negative = needs less time
        block_adjustment = time_diff / dit_time_per_block  # convert time difference to block count
        target_count = current_block_counts[i] + block_adjustment
        # Ensure each rank gets at least 1 block (minimum allocation)
        target_count = max(1, int(round(target_count)))
        target_blocks.append(target_count)
    
    # Step 5: Adjust to ensure total blocks sum to total_blocks while maintaining minimum 1 block per rank
    current_total = sum(target_blocks)
    if current_total != total_blocks:
        diff = total_blocks - current_total
        # When adding, give to ranks with smallest counts first; when removing, take from largest counts first
        if diff > 0:
            order = sorted(range(num_ranks), key=lambda i: (target_blocks[i], i))
        else:
            order = sorted(range(num_ranks), key=lambda i: (target_blocks[i], i), reverse=True)
        i = 0
        while diff != 0 and num_ranks > 0:
            idx = order[i % num_ranks]
            if diff > 0:
                target_blocks[idx] += 1
                diff -= 1
            else:
                # Only remove blocks if rank has more than 1 block (maintain minimum allocation)
                if target_blocks[idx] > 1:
                    target_blocks[idx] -= 1
                    diff += 1
            i += 1
    
    # Step 6: Convert target block counts to contiguous [start, end) intervals from 0 to total_blocks
    new_block_nums = []
    running_start = 0
    for i in range(num_ranks):
        block_count = int(target_blocks[i])
        start_idx = running_start
        end_idx = start_idx + block_count
        # Guard (should not trigger if sums are correct)
        if end_idx > total_blocks:
            end_idx = total_blocks
        new_block_nums.append([start_idx, end_idx])
        running_start = end_idx
    
    return new_block_nums


def setup_logging(rank: int, log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging for the current rank.
    
    Args:
        rank: Current rank
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(log_level)
    # Prevent messages from propagating to the root logger (avoid double prints)
    logger.propagate = False
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


class CommunicationTimer:
    """
    Timer for measuring communication performance.
    
    This class provides context manager functionality for timing communication operations.
    """
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if self.logger:
            self.logger.info(f"{self.operation_name} took {duration:.4f} seconds")
    
    @property
    def duration(self) -> float:
        """Get the duration of the timed operation."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
