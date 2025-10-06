"""
Buffer manager for efficient GPU memory management.

This module provides a buffer pool manager to avoid repeated GPU memory allocations
during distributed communication operations.
"""

import torch
from typing import Dict, List, Tuple, Optional
import threading
import logging
from .data_containers import CommunicationConfig


class BufferManager:
    """
    Manages GPU buffer pools to avoid repeated allocations.
    
    This class maintains pools of pre-allocated GPU tensors that can be reused
    across communication operations, reducing memory allocation overhead.
    """
    
    def __init__(self, device: torch.device, config: Optional[CommunicationConfig] = None):
        """
        Initialize the buffer manager.
        
        Args:
            device: GPU device for buffer allocation
            config: Communication configuration
        """
        self.device = device
        self.config = config or CommunicationConfig()
        
        # Buffer pools: {shape: [tensor1, tensor2, ...]}
        self.free_buffers = {}  # For latent tensors
        self.free_buffers_origin = {}  # For original latent tensors
        self.free_buffers_kv = {}  # For KV cache tensors
        self.free_buffers_misc = {}  # For headers, shapes, index vectors (int64 etc.)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self.allocation_count = 0
        self.reuse_count = 0
        self.total_allocated_memory = 0
        
        # Setup logging
        self.logger = logging.getLogger(f"BufferManager_{device}")
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            # handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                f'[BufferManager {device}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        # self.logger.setLevel(logging.DEBUG)
    
    def get_buffer(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                   buffer_type: str = "latent") -> torch.Tensor:
        """
        Get or allocate a buffer with the specified shape and dtype.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            buffer_type: Type of buffer ("latent", "origin", "kv")
            
        Returns:
            Tensor buffer
        """
        with self._lock:
            # Select the appropriate buffer pool
            if buffer_type == "latent":
                buffer_pool = self.free_buffers
            elif buffer_type == "origin":
                buffer_pool = self.free_buffers_origin
            elif buffer_type == "kv":
                buffer_pool = self.free_buffers_kv
            elif buffer_type == "misc":
                buffer_pool = self.free_buffers_misc
            else:
                raise ValueError(f"Unknown buffer type: {buffer_type}")
            
            # Try to reuse existing buffer
            if self.config.enable_buffer_reuse and shape in buffer_pool and len(buffer_pool[shape]) > 0:
                buffer = buffer_pool[shape].pop()
                self.reuse_count += 1
                self.logger.debug(f"Reused buffer of shape {shape}, type {buffer_type}")
                return buffer
            
            # Allocate new buffer
            buffer = torch.empty(shape, dtype=dtype, device=self.device)
            self.allocation_count += 1
            self.total_allocated_memory += buffer.numel() * buffer.element_size()
            
            self.logger.debug(f"Allocated new buffer of shape {shape}, type {buffer_type}")
            return buffer
    
    def return_buffer(self, tensor: torch.Tensor, buffer_type: str = "latent") -> None:
        """
        Return a buffer to the pool for reuse.
        
        Args:
            tensor: Tensor to return
            buffer_type: Type of buffer ("latent", "origin", "kv")
        """
        if not self.config.enable_buffer_reuse:
            return
        
        with self._lock:
            # Select the appropriate buffer pool
            if buffer_type == "latent":
                buffer_pool = self.free_buffers
            elif buffer_type == "origin":
                buffer_pool = self.free_buffers_origin
            elif buffer_type == "kv":
                buffer_pool = self.free_buffers_kv
            elif buffer_type == "misc":
                buffer_pool = self.free_buffers_misc
            else:
                raise ValueError(f"Unknown buffer type: {buffer_type}")
            
            shape = tuple(tensor.shape)
            
            # Initialize pool for this shape if it doesn't exist
            if shape not in buffer_pool:
                buffer_pool[shape] = []
            
            # Add buffer to pool if not at capacity
            if len(buffer_pool[shape]) < self.config.buffer_pool_size:
                # Clear the tensor to free memory
                tensor.zero_()
                buffer_pool[shape].append(tensor)
                self.logger.debug(f"Returned buffer of shape {shape}, type {buffer_type}")
            else:
                self.logger.debug(f"Buffer pool full for shape {shape}, type {buffer_type}, discarding")
    
    def clear_buffers(self, buffer_type: Optional[str] = None) -> None:
        """
        Clear buffer pools to free memory.
        
        Args:
            buffer_type: Specific buffer type to clear, or None to clear all
        """
        with self._lock:
            if buffer_type is None:
                # Clear all buffer pools
                self.free_buffers.clear()
                self.free_buffers_origin.clear()
                self.free_buffers_kv.clear()
                self.free_buffers_misc.clear()
                self.logger.info("Cleared all buffer pools")
            else:
                # Clear specific buffer pool
                if buffer_type == "latent":
                    self.free_buffers.clear()
                elif buffer_type == "origin":
                    self.free_buffers_origin.clear()
                elif buffer_type == "kv":
                    self.free_buffers_kv.clear()
                elif buffer_type == "misc":
                    self.free_buffers_misc.clear()
                else:
                    raise ValueError(f"Unknown buffer type: {buffer_type}")
                self.logger.info(f"Cleared {buffer_type} buffer pool")
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get buffer manager statistics.
        
        Returns:
            Dictionary containing statistics
        """
        with self._lock:
            total_free_buffers = sum(len(pool) for pool in self.free_buffers.values())
            total_free_buffers_origin = sum(len(pool) for pool in self.free_buffers_origin.values())
            total_free_buffers_kv = sum(len(pool) for pool in self.free_buffers_kv.values())
            total_free_buffers_misc = sum(len(pool) for pool in self.free_buffers_misc.values())

            return {
                "allocation_count": self.allocation_count,
                "reuse_count": self.reuse_count,
                "total_allocated_memory_bytes": self.total_allocated_memory,
                "total_free_buffers": total_free_buffers,
                "total_free_buffers_origin": total_free_buffers_origin,
                "total_free_buffers_kv": total_free_buffers_kv,
                "total_free_buffers_misc": total_free_buffers_misc,
                "reuse_rate": self.reuse_count / max(1, self.allocation_count),
                "buffer_pool_size": self.config.buffer_pool_size,
                "enable_buffer_reuse": self.config.enable_buffer_reuse
            }
    
    def print_statistics(self) -> None:
        """Print buffer manager statistics."""
        stats = self.get_statistics()
        self.logger.info("Buffer Manager Statistics:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")
    
    def preallocate_buffers(self, common_shapes: List[Tuple[Tuple[int, ...], torch.dtype, str]], 
                          count_per_shape: int = 5) -> None:
        """
        Preallocate buffers for common shapes to reduce allocation overhead.
        
        Args:
            common_shapes: List of (shape, dtype, buffer_type) tuples
            count_per_shape: Number of buffers to preallocate per shape
        """
        with self._lock:
            for shape, dtype, buffer_type in common_shapes:
                for _ in range(count_per_shape):
                    buffer = torch.empty(shape, dtype=dtype, device=self.device)
                    
                    # Select the appropriate buffer pool
                    if buffer_type == "latent":
                        buffer_pool = self.free_buffers
                    elif buffer_type == "origin":
                        buffer_pool = self.free_buffers_origin
                    elif buffer_type == "kv":
                        buffer_pool = self.free_buffers_kv
                    elif buffer_type == "misc":
                        buffer_pool = self.free_buffers_misc
                    else:
                        raise ValueError(f"Unknown buffer type: {buffer_type}")
                    
                    # Initialize pool for this shape if it doesn't exist
                    if shape not in buffer_pool:
                        buffer_pool[shape] = []
                    
                    buffer_pool[shape].append(buffer)
                    self.allocation_count += 1
                    self.total_allocated_memory += buffer.numel() * buffer.element_size()
            
            self.logger.info(f"Preallocated {len(common_shapes) * count_per_shape} buffers")
    
    def __del__(self):
        """Cleanup when the buffer manager is destroyed."""
        self.clear_buffers()