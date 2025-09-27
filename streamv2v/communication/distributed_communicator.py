"""
Distributed communication abstraction layer.

This module provides a high-level interface for distributed communication operations,
encapsulating the low-level PyTorch distributed primitives.
"""

import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Any
import logging
import time
from .utils import CommunicationTags, get_next_rank, get_prev_rank, CommunicationTimer
from .data_containers import CommunicationConfig


class DistributedCommunicator:
    """
    High-level interface for distributed communication operations.
    
    This class encapsulates all distributed communication operations, providing
    a clean interface for sending and receiving tensors between ranks.
    """
    
    def __init__(self, rank: int, world_size: int, device: torch.device, 
                 config: Optional[CommunicationConfig] = None):
        """
        Initialize the distributed communicator.
        
        Args:
            rank: Current rank
            world_size: Total number of ranks
            device: GPU device for communication
            config: Communication configuration
        """
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.config = config or CommunicationConfig()
        
        # Track outstanding operations
        self.outstanding_operations: List[Any] = []
        
        # Setup logging
        self.logger = logging.getLogger(f"DistributedCommunicator_rank_{rank}")
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Validate distributed is initialized
        if not dist.is_initialized():
            raise RuntimeError("Distributed not initialized. Call init_distributed() first.")
    
    def send_tensor_async(self, tensor: torch.Tensor, dst: int, tag: int) -> Any:
        """
        Asynchronously send a tensor to the specified destination.
        
        Args:
            tensor: Tensor to send
            dst: Destination rank
            tag: Communication tag
            
        Returns:
            Work object for the send operation
        """
        if tensor.device != self.device:
            raise ValueError(f"Tensor device {tensor.device} doesn't match communicator device {self.device}")
        
        work = dist.isend(tensor, dst=dst, tag=tag)
        self.outstanding_operations.append(work)
        
        self.logger.debug(f"Started async send to rank {dst} with tag {tag}, tensor shape: {tensor.shape}")
        return work
    
    def recv_tensor(self, src: int, tag: int, shape: Tuple[int, ...], 
                   dtype: torch.dtype) -> torch.Tensor:
        """
        Receive a tensor from the specified source.
        
        Args:
            src: Source rank
            tag: Communication tag
            shape: Expected tensor shape
            dtype: Expected tensor dtype
            
        Returns:
            Received tensor
        """
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        
        with CommunicationTimer(f"recv_tensor from rank {src}", self.logger):
            dist.recv(tensor, src=src, tag=tag)
        
        self.logger.debug(f"Received tensor from rank {src} with tag {tag}, shape: {tensor.shape}")
        return tensor
    
    def send_header_and_tensor_async(self, header: torch.Tensor, tensor: torch.Tensor,
                                   dst: int, tag_header: int, tag_tensor: int) -> Tuple[Any, Any]:
        """
        Asynchronously send a header and tensor pair.
        
        Args:
            header: Header tensor containing metadata
            tensor: Data tensor
            dst: Destination rank
            tag_header: Tag for header
            tag_tensor: Tag for tensor
            
        Returns:
            Tuple of (header_work, tensor_work)
        """
        if header.device != self.device or tensor.device != self.device:
            raise ValueError("Header and tensor must be on the same device as communicator")
        
        header_work = dist.isend(header, dst=dst, tag=tag_header)
        tensor_work = dist.isend(tensor, dst=dst, tag=tag_tensor)
        
        self.outstanding_operations.extend([header_work, tensor_work])
        
        self.logger.debug(f"Started async send of header+tensor to rank {dst}, "
                         f"header shape: {header.shape}, tensor shape: {tensor.shape}")
        return header_work, tensor_work
    
    def recv_header_and_tensor(self, src: int, tag_header: int, tag_tensor: int, header_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Receive a header and tensor pair.
        
        Args:
            src: Source rank
            tag_header: Tag for header
            tag_tensor: Tag for tensor
            header_len: Length of header tensor to receive
            
        Returns:
            Tuple of (header, tensor)
        """
        with CommunicationTimer(f"recv_header_and_tensor from rank {src}", self.logger):
            # First receive the header to get tensor shape (length can vary)
            header = torch.empty(header_len, dtype=torch.int64, device=self.device)
            dist.recv(header, src=src, tag=tag_header)
            
            # Parse header to get tensor shape
            chunk_idx, shape = self._parse_header(header)
            
            # Receive the tensor
            tensor = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
            dist.recv(tensor, src=src, tag=tag_tensor)
        
        self.logger.debug(f"Received header+tensor from rank {src}, "
                         f"header: {header.tolist()}, tensor shape: {tensor.shape}")
        return header, tensor
    
    def send_latent_data_async(self, chunk_idx: int, latents: torch.Tensor,
                             original_latents: torch.Tensor, patched_x_shape: torch.Tensor,
                             current_start: torch.Tensor, current_end: torch.Tensor,
                             current_step: int) -> List[Any]:
        """
        Asynchronously send all latent data components.
        
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
        dst = get_next_rank(self.rank, self.world_size)
        work_objects = []
        
        # Create headers
        latent_header = self._create_header(chunk_idx, latents.shape)
        origin_header = self._create_header(chunk_idx, original_latents.shape)
        
        # Create start/end/step tensor
        start_end_step = torch.cat([
            current_start, 
            current_end, 
            torch.tensor([current_step], dtype=torch.int64, device=self.device)
        ], dim=0)
        
        # Send all components asynchronously
        work_objects.extend(self.send_header_and_tensor_async(
            latent_header, latents, dst, CommunicationTags.LATENT_HDR, CommunicationTags.LATENT_PAY
        ))
        
        work_objects.extend(self.send_header_and_tensor_async(
            origin_header, original_latents, dst, 
            CommunicationTags.LATENT_ORIGIN_HDR, CommunicationTags.LATENT_ORIGIN_PAY
        ))
        
        work_objects.append(self.send_tensor_async(
            patched_x_shape, dst, CommunicationTags.PATCHED_X_SHAPE
        ))
        
        work_objects.append(self.send_tensor_async(
            start_end_step, dst, CommunicationTags.START_END_STEP
        ))
        
        self.logger.debug(f"Started async send of latent data to rank {dst}, chunk_idx: {chunk_idx}")
        return work_objects
    
    def recv_latent_data_async(self, num_steps: int, buffer_manager) -> Tuple[int, torch.Tensor, torch.Tensor, 
                                                                           torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """
        Asynchronously receive all latent data components.
        
        Args:
            num_steps: Number of denoising steps
            buffer_manager: Buffer manager for tensor allocation
            
        Returns:
            Tuple of (chunk_idx, latents, original_latents, current_start, current_end, current_step, patched_x_shape)
        """
        src = get_prev_rank(self.rank, self.world_size)
        
        with CommunicationTimer(f"recv_latent_data_async from rank {src}", self.logger):
            # Receive latent header (length 4): [i, bsz, slen, cch]
            latent_header = buffer_manager.get_buffer((4,), torch.int64, "misc")
            dist.recv(latent_header, src=src, tag=CommunicationTags.LATENT_HDR)
            chunk_idx, latent_shape = self._parse_header(latent_header)
            # header no longer needed
            buffer_manager.return_buffer(latent_header, "misc")
            # Allocate or reuse buffer for latents: shape (bsz, slen, cch)
            latents = buffer_manager.get_buffer(tuple(latent_shape), torch.bfloat16, "latent")
            dist.recv(latents, src=src, tag=CommunicationTags.LATENT_PAY)

            # Receive original latent header (length 6): [i, bsz, cch, tlen, hh, ww]
            origin_header = buffer_manager.get_buffer((6,), torch.int64, "misc")
            dist.recv(origin_header, src=src, tag=CommunicationTags.LATENT_ORIGIN_HDR)
            _, origin_shape = self._parse_header(origin_header)
            # header no longer needed
            buffer_manager.return_buffer(origin_header, "misc")
            # Allocate or reuse buffer for original latents: shape (bsz, cch, tlen, hh, ww)
            original_latents = buffer_manager.get_buffer(tuple(origin_shape), torch.bfloat16, "origin")
            dist.recv(original_latents, src=src, tag=CommunicationTags.LATENT_ORIGIN_PAY)

            # Receive patched_x_shape (length 5, int64)
            patched_x_shape = buffer_manager.get_buffer((5,), torch.int64, "misc")
            dist.recv(patched_x_shape, src=src, tag=CommunicationTags.PATCHED_X_SHAPE)

            # Receive start_end_step (length 2*num_steps+1, int64)
            start_end_step = buffer_manager.get_buffer((2 * num_steps + 1,), torch.int64, "misc")
            dist.recv(start_end_step, src=src, tag=CommunicationTags.START_END_STEP)

            # Parse start/end/step into dedicated misc buffers, then release the combined vector
            current_start = buffer_manager.get_buffer((num_steps,), torch.int64, "misc")
            current_end = buffer_manager.get_buffer((num_steps,), torch.int64, "misc")
            current_start.copy_(start_end_step[:num_steps])
            current_end.copy_(start_end_step[num_steps:-1])
            current_step = int(start_end_step[-1].item())
            # Release the temporary combined buffer
            buffer_manager.return_buffer(start_end_step, "misc")
        
        self.logger.debug(f"Received latent data from rank {src}, chunk_idx: {chunk_idx}")
        return chunk_idx, latents, original_latents, current_start, current_end, current_step, patched_x_shape

    def send_prompt_async(self, prompt: str, device: torch.device) -> List[Any]:
        work_objects = []
        dst = get_next_rank(self.rank, self.world_size)

        # Encode to bytes
        encoded = prompt.encode("utf-8")
        data = torch.ByteTensor(list(encoded)).to(device)

        # Send length first
        length = torch.tensor([len(data)], dtype=torch.int64, device=data.device)
        work_objects.append(dist.isend(length, dst=dst, tag=CommunicationTags.UPDATED_PROMPT_LENGTH))

        # Then send the content
        work_objects.append(dist.isend(data, dst=dst, tag=CommunicationTags.UPDATED_PROMPT))

        return work_objects

    def recv_prompt_async(self) -> str:
        src = get_prev_rank(self.rank, self.world_size)

        # Receive length first
        length = torch.empty(1, dtype=torch.int64, device=self.device)
        dist.recv(length, src=src, tag=CommunicationTags.UPDATED_PROMPT_LENGTH)

        # Then receive the content
        prompt = torch.empty(length.item(), dtype=torch.uint8, device=self.device)
        dist.recv(prompt, src=src, tag=CommunicationTags.UPDATED_PROMPT)

        return bytes(prompt.cpu().tolist()).decode("utf-8")    
    
    def broadcast_tensor(self, tensor: torch.Tensor, src: int) -> None:
        """
        Broadcast a tensor from source to all ranks.
        
        Args:
            tensor: Tensor to broadcast
            src: Source rank
        """
        with CommunicationTimer(f"broadcast_tensor from rank {src}", self.logger):
            dist.broadcast(tensor, src=src)
        
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
            gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(gather_list, tensor)
        
        self.logger.debug(f"Gathered tensors from all ranks, local shape: {tensor.shape}")
        return gather_list
    
    def wait_for_outstanding(self, max_outstanding: Optional[int] = None) -> None:
        """
        Wait for outstanding operations to complete.
        
        Args:
            max_outstanding: Maximum number of outstanding operations to keep
        """
        max_outstanding = max_outstanding or self.config.max_outstanding
        
        while len(self.outstanding_operations) >= max_outstanding:
            if not self.outstanding_operations:
                break
            
            # Wait for the oldest operation
            oldest_operations = self.outstanding_operations.pop(0)
            
            # Handle both single work objects and lists of work objects
            if isinstance(oldest_operations, (list, tuple)):
                for work in oldest_operations:
                    try:
                        work.wait()
                    except Exception as e:
                        self.logger.error(f"Error waiting for outstanding operation: {e}")
                        raise
            else:
                try:
                    oldest_operations.wait()
                except Exception as e:
                    self.logger.error(f"Error waiting for outstanding operation: {e}")
                    raise
        
        self.logger.debug(f"Outstanding operations: {len(self.outstanding_operations)}")
    
    def barrier(self) -> None:
        """Synchronize all ranks."""
        with CommunicationTimer("barrier", self.logger):
            dist.barrier()
    
    def _create_header(self, chunk_idx: int, shape: Tuple[int, ...]) -> torch.Tensor:
        """Create a header tensor for communication."""
        header_data = [chunk_idx] + list(shape)
        return torch.tensor(header_data, dtype=torch.int64, device=self.device)
    
    def _parse_header(self, header: torch.Tensor) -> Tuple[int, Tuple[int, ...]]:
        """Parse a header tensor to extract metadata."""
        header_list = header.tolist()
        chunk_idx = int(header_list[0])
        shape = tuple(int(x) for x in header_list[1:])
        return chunk_idx, shape
    
    def get_statistics(self) -> dict:
        """Get communication statistics."""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "outstanding_operations": len(self.outstanding_operations),
            "max_outstanding": self.config.max_outstanding,
            "device": str(self.device)
        }
    
    def print_statistics(self) -> None:
        """Print communication statistics."""
        stats = self.get_statistics()
        self.logger.info("Distributed Communicator Statistics:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")