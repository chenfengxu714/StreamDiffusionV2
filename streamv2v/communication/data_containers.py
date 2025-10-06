"""
Data containers for communication operations.

This module defines data structures used for communication between distributed ranks.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch


@dataclass
class LatentData:
    """
    Container for latent data and related information.
    
    This class encapsulates all the data that needs to be transferred between ranks
    during the inference pipeline.
    """
    chunk_idx: int
    latents: torch.Tensor
    original_latents: torch.Tensor
    current_start: torch.Tensor
    current_end: torch.Tensor
    current_step: int
    patched_x_shape: torch.Tensor
    
    def __post_init__(self):
        """Validate tensor shapes and types after initialization."""
        if not isinstance(self.latents, torch.Tensor):
            raise TypeError("latents must be a torch.Tensor")
        if not isinstance(self.original_latents, torch.Tensor):
            raise TypeError("original_latents must be a torch.Tensor")
        if not isinstance(self.current_start, torch.Tensor):
            raise TypeError("current_start must be a torch.Tensor")
        if not isinstance(self.current_end, torch.Tensor):
            raise TypeError("current_end must be a torch.Tensor")
        if not isinstance(self.patched_x_shape, torch.Tensor):
            raise TypeError("patched_x_shape must be a torch.Tensor")


@dataclass
class KVCacheData:
    """
    Container for KV cache data.
    
    This class encapsulates key-value cache information for transformer blocks.
    """
    block_index: int
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    global_end_index: torch.Tensor
    local_end_index: torch.Tensor
    
    def __post_init__(self):
        """Validate tensor shapes and types after initialization."""
        if not isinstance(self.k_cache, torch.Tensor):
            raise TypeError("k_cache must be a torch.Tensor")
        if not isinstance(self.v_cache, torch.Tensor):
            raise TypeError("v_cache must be a torch.Tensor")
        if not isinstance(self.global_end_index, torch.Tensor):
            raise TypeError("global_end_index must be a torch.Tensor")
        if not isinstance(self.local_end_index, torch.Tensor):
            raise TypeError("local_end_index must be a torch.Tensor")


@dataclass
class CommunicationConfig:
    """
    Configuration for communication operations.
    
    This class holds configuration parameters for distributed communication.
    """
    max_outstanding: int = 1
    buffer_pool_size: int = 10
    enable_buffer_reuse: bool = True
    communication_timeout: float = 30.0
    enable_async_communication: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_outstanding < 1:
            raise ValueError("max_outstanding must be at least 1")
        if self.buffer_pool_size < 1:
            raise ValueError("buffer_pool_size must be at least 1")
        if self.communication_timeout <= 0:
            raise ValueError("communication_timeout must be positive")


@dataclass
class BlockInterval:
    """
    Container for block interval information.
    
    This class represents a block interval [start, end) for a specific rank.
    """
    start: int
    end: int
    rank: int
    
    def __post_init__(self):
        """Validate block interval parameters."""
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
    
    @property
    def size(self) -> int:
        """Get the size of the block interval."""
        return self.end - self.start
    
    def contains(self, block_index: int) -> bool:
        """Check if the block interval contains the given block index."""
        return self.start <= block_index < self.end


@dataclass
class PerformanceMetrics:
    """
    Container for performance metrics.
    
    This class holds timing and performance information for communication operations.
    """
    dit_time: float
    total_time: float
    communication_time: float
    buffer_allocation_time: float
    
    def __post_init__(self):
        """Validate performance metrics."""
        if self.dit_time < 0:
            raise ValueError("dit_time must be non-negative")
        if self.total_time < 0:
            raise ValueError("total_time must be non-negative")
        if self.communication_time < 0:
            raise ValueError("communication_time must be non-negative")
        if self.buffer_allocation_time < 0:
            raise ValueError("buffer_allocation_time must be non-negative")
    
    @property
    def efficiency(self) -> float:
        """Calculate communication efficiency (computation time / total time)."""
        if self.total_time == 0:
            return 0.0
        return (self.total_time - self.communication_time) / self.total_time
