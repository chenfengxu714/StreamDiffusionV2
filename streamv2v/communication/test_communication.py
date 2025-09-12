"""
Test cases for the communication module.

This module provides comprehensive tests for all communication abstractions.
"""

import unittest
import torch
import torch.distributed as dist
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to the path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication.data_containers import LatentData, KVCacheData, CommunicationConfig, BlockInterval, PerformanceMetrics
from communication.buffer_manager import BufferManager
from communication.utils import CommunicationTags, setup_logging, compute_balanced_split
from communication.distributed_communicator import DistributedCommunicator
from communication.kv_cache_manager import KVCacheManager
from communication.model_data_transfer import ModelDataTransfer


class TestDataContainers(unittest.TestCase):
    """Test cases for data container classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.sample_latents = torch.randn(1, 4, 16, 16, device=self.device)
        self.sample_original_latents = torch.randn(1, 4, 16, 16, 16, device=self.device)
        self.sample_current_start = torch.tensor([0, 1, 2], device=self.device)
        self.sample_current_end = torch.tensor([1, 2, 3], device=self.device)
        self.sample_patched_x_shape = torch.tensor([1, 4, 16, 16, 16], device=self.device)
    
    def test_latent_data_creation(self):
        """Test LatentData creation and validation."""
        latent_data = LatentData(
            chunk_idx=0,
            latents=self.sample_latents,
            original_latents=self.sample_original_latents,
            current_start=self.sample_current_start,
            current_end=self.sample_current_end,
            current_step=100,
            patched_x_shape=self.sample_patched_x_shape
        )
        
        self.assertEqual(latent_data.chunk_idx, 0)
        self.assertEqual(latent_data.current_step, 100)
        self.assertTrue(torch.equal(latent_data.latents, self.sample_latents))
    
    def test_latent_data_validation(self):
        """Test LatentData validation with invalid inputs."""
        with self.assertRaises(TypeError):
            LatentData(
                chunk_idx=0,
                latents="invalid",  # Should be torch.Tensor
                original_latents=self.sample_original_latents,
                current_start=self.sample_current_start,
                current_end=self.sample_current_end,
                current_step=100,
                patched_x_shape=self.sample_patched_x_shape
            )
    
    def test_communication_config(self):
        """Test CommunicationConfig creation and validation."""
        config = CommunicationConfig(
            max_outstanding=5,
            buffer_pool_size=20,
            enable_buffer_reuse=True,
            communication_timeout=60.0
        )
        
        self.assertEqual(config.max_outstanding, 5)
        self.assertEqual(config.buffer_pool_size, 20)
        self.assertTrue(config.enable_buffer_reuse)
        self.assertEqual(config.communication_timeout, 60.0)
    
    def test_communication_config_validation(self):
        """Test CommunicationConfig validation with invalid inputs."""
        with self.assertRaises(ValueError):
            CommunicationConfig(max_outstanding=0)  # Should be at least 1
        
        with self.assertRaises(ValueError):
            CommunicationConfig(buffer_pool_size=0)  # Should be at least 1
        
        with self.assertRaises(ValueError):
            CommunicationConfig(communication_timeout=0)  # Should be positive
    
    def test_block_interval(self):
        """Test BlockInterval creation and methods."""
        interval = BlockInterval(start=0, end=10, rank=0)
        
        self.assertEqual(interval.start, 0)
        self.assertEqual(interval.end, 10)
        self.assertEqual(interval.rank, 0)
        self.assertEqual(interval.size, 10)
        self.assertTrue(interval.contains(5))
        self.assertFalse(interval.contains(10))
        self.assertFalse(interval.contains(-1))
    
    def test_block_interval_validation(self):
        """Test BlockInterval validation with invalid inputs."""
        with self.assertRaises(ValueError):
            BlockInterval(start=-1, end=10, rank=0)  # Start should be non-negative
        
        with self.assertRaises(ValueError):
            BlockInterval(start=10, end=5, rank=0)  # End should be greater than start
        
        with self.assertRaises(ValueError):
            BlockInterval(start=0, end=10, rank=-1)  # Rank should be non-negative
    
    def test_performance_metrics(self):
        """Test PerformanceMetrics creation and methods."""
        metrics = PerformanceMetrics(
            dit_time=1.0,
            total_time=2.0,
            communication_time=0.5,
            buffer_allocation_time=0.1
        )
        
        self.assertEqual(metrics.dit_time, 1.0)
        self.assertEqual(metrics.total_time, 2.0)
        self.assertEqual(metrics.communication_time, 0.5)
        self.assertEqual(metrics.buffer_allocation_time, 0.1)
        self.assertEqual(metrics.efficiency, 0.75)  # (2.0 - 0.5) / 2.0


class TestBufferManager(unittest.TestCase):
    """Test cases for BufferManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.config = CommunicationConfig(buffer_pool_size=5)
        self.buffer_manager = BufferManager(self.device, self.config)
    
    def test_buffer_allocation(self):
        """Test buffer allocation and reuse."""
        shape = (1, 4, 16, 16)
        dtype = torch.float32
        
        # Allocate a buffer
        buffer1 = self.buffer_manager.get_buffer(shape, dtype, "latent")
        self.assertEqual(buffer1.shape, shape)
        self.assertEqual(buffer1.dtype, dtype)
        self.assertEqual(buffer1.device, self.device)
        
        # Return the buffer
        self.buffer_manager.return_buffer(buffer1, "latent")
        
        # Get another buffer of the same shape - should reuse
        buffer2 = self.buffer_manager.get_buffer(shape, dtype, "latent")
        self.assertEqual(buffer2.shape, shape)
        self.assertEqual(buffer2.dtype, dtype)
    
    def test_buffer_statistics(self):
        """Test buffer manager statistics."""
        shape = (1, 4, 16, 16)
        dtype = torch.float32
        
        # Allocate and return some buffers
        buffer1 = self.buffer_manager.get_buffer(shape, dtype, "latent")
        self.buffer_manager.return_buffer(buffer1, "latent")
        
        buffer2 = self.buffer_manager.get_buffer(shape, dtype, "latent")
        self.buffer_manager.return_buffer(buffer2, "latent")
        
        stats = self.buffer_manager.get_statistics()
        self.assertEqual(stats['allocation_count'], 2)
        self.assertEqual(stats['reuse_count'], 1)
        self.assertGreater(stats['total_allocated_memory_bytes'], 0)
    
    def test_buffer_cleanup(self):
        """Test buffer cleanup."""
        shape = (1, 4, 16, 16)
        dtype = torch.float32
        
        # Allocate and return some buffers
        buffer1 = self.buffer_manager.get_buffer(shape, dtype, "latent")
        self.buffer_manager.return_buffer(buffer1, "latent")
        
        # Clear buffers
        self.buffer_manager.clear_buffers("latent")
        
        stats = self.buffer_manager.get_statistics()
        self.assertEqual(stats['total_free_buffers'], 0)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_compute_balanced_split(self):
        """Test the compute_balanced_split function."""
        total_blocks = 30
        rank_times = [1.0, 2.0, 1.5]  # Rank 1 is slower
        dit_times = [0.8, 1.6, 1.2]
        current_block_nums = [[0, 10], [10, 20], [20, 30]]
        
        new_block_nums = compute_balanced_split(total_blocks, rank_times, dit_times, current_block_nums)
        
        # Should have same number of ranks
        self.assertEqual(len(new_block_nums), len(current_block_nums))
        
        # Should sum to total_blocks
        total_allocated = sum(end - start for start, end in new_block_nums)
        self.assertEqual(total_allocated, total_blocks)
        
        # Should be contiguous
        for i in range(len(new_block_nums) - 1):
            self.assertEqual(new_block_nums[i][1], new_block_nums[i + 1][0])
    
    def test_compute_balanced_split_edge_cases(self):
        """Test compute_balanced_split with edge cases."""
        # Empty input
        result = compute_balanced_split(0, [], [], [])
        self.assertEqual(result, [])
        
        # Single rank
        result = compute_balanced_split(10, [1.0], [0.8], [[0, 10]])
        self.assertEqual(result, [[0, 10]])
        
        # Invalid input lengths
        result = compute_balanced_split(10, [1.0], [0.8], [[0, 10], [10, 20]])
        self.assertEqual(result, [[0, 10], [10, 20]])  # Should return original


class TestDistributedCommunicator(unittest.TestCase):
    """Test cases for DistributedCommunicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.config = CommunicationConfig()
        
        # Mock distributed environment
        with patch('torch.distributed.is_initialized', return_value=True):
            self.communicator = DistributedCommunicator(0, 2, self.device, self.config)
    
    def test_communicator_initialization(self):
        """Test communicator initialization."""
        self.assertEqual(self.communicator.rank, 0)
        self.assertEqual(self.communicator.world_size, 2)
        self.assertEqual(self.communicator.device, self.device)
    
    def test_communicator_initialization_without_distributed(self):
        """Test communicator initialization without distributed."""
        with patch('torch.distributed.is_initialized', return_value=False):
            with self.assertRaises(RuntimeError):
                DistributedCommunicator(0, 2, self.device, self.config)
    
    def test_create_header(self):
        """Test header creation and parsing."""
        chunk_idx = 5
        shape = (1, 4, 16, 16)
        
        header = self.communicator._create_header(chunk_idx, shape)
        self.assertEqual(header.shape, (5,))  # chunk_idx + 4 shape dimensions
        self.assertEqual(header.dtype, torch.int64)
        
        parsed_chunk_idx, parsed_shape = self.communicator._parse_header(header)
        self.assertEqual(parsed_chunk_idx, chunk_idx)
        self.assertEqual(parsed_shape, shape)
    
    def test_communicator_statistics(self):
        """Test communicator statistics."""
        stats = self.communicator.get_statistics()
        
        self.assertEqual(stats['rank'], 0)
        self.assertEqual(stats['world_size'], 2)
        self.assertEqual(stats['outstanding_operations'], 0)
        self.assertEqual(stats['max_outstanding'], 1)


class TestKVCacheManager(unittest.TestCase):
    """Test cases for KVCacheManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        
        # Mock pipeline with KV cache
        self.mock_pipeline = Mock()
        self.mock_pipeline.frame_seq_length = 16
        self.mock_pipeline.kv_cache1 = [
            {
                'k': torch.randn(1, 8, 16, 64, device=self.device),
                'v': torch.randn(1, 8, 16, 64, device=self.device),
                'global_end_index': torch.tensor([16], device=self.device),
                'local_end_index': torch.tensor([16], device=self.device)
            }
            for _ in range(30)
        ]
        
        self.kv_cache_manager = KVCacheManager(self.mock_pipeline, self.device)
    
    def test_compute_block_owners(self):
        """Test block owner computation."""
        block_intervals = torch.tensor([[0, 10], [10, 20], [20, 30]], device=self.device)
        total_blocks = 30
        
        owners = self.kv_cache_manager.compute_block_owners(block_intervals, total_blocks)
        
        self.assertEqual(owners.shape, (30,))
        self.assertTrue(torch.all(owners[:10] == 0))
        self.assertTrue(torch.all(owners[10:20] == 1))
        self.assertTrue(torch.all(owners[20:30] == 2))
    
    def test_kv_cache_statistics(self):
        """Test KV cache statistics."""
        block_intervals = torch.tensor([[0, 10], [10, 20], [20, 30]], device=self.device)
        total_blocks = 30
        
        stats = self.kv_cache_manager.get_kv_cache_statistics(block_intervals, total_blocks)
        
        self.assertEqual(stats['total_blocks'], 30)
        self.assertEqual(stats['block_counts'][0], 10)
        self.assertEqual(stats['block_counts'][1], 10)
        self.assertEqual(stats['block_counts'][2], 10)
        self.assertGreater(stats['memory_per_block_bytes'], 0)
    
    def test_validate_kv_cache_consistency(self):
        """Test KV cache consistency validation."""
        block_intervals = torch.tensor([[0, 10], [10, 20], [20, 30]], device=self.device)
        total_blocks = 30
        
        is_consistent = self.kv_cache_manager.validate_kv_cache_consistency(block_intervals, total_blocks)
        self.assertTrue(is_consistent)
        
        # Test with invalid intervals
        invalid_intervals = torch.tensor([[0, 10], [10, 20], [20, 25]], device=self.device)  # Missing blocks
        is_consistent = self.kv_cache_manager.validate_kv_cache_consistency(invalid_intervals, total_blocks)
        self.assertFalse(is_consistent)


class TestModelDataTransfer(unittest.TestCase):
    """Test cases for ModelDataTransfer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.config = CommunicationConfig()
        
        # Mock components
        with patch('torch.distributed.is_initialized', return_value=True):
            self.communicator = DistributedCommunicator(0, 2, self.device, self.config)
        
        self.buffer_manager = BufferManager(self.device, self.config)
        self.mock_pipeline = Mock()
        self.kv_cache_manager = KVCacheManager(self.mock_pipeline, self.device)
        
        self.data_transfer = ModelDataTransfer(
            self.communicator,
            self.buffer_manager,
            self.kv_cache_manager,
            self.config
        )
    
    def test_data_transfer_initialization(self):
        """Test data transfer initialization."""
        self.assertEqual(self.data_transfer.comm, self.communicator)
        self.assertEqual(self.data_transfer.buffer_mgr, self.buffer_manager)
        self.assertEqual(self.data_transfer.kv_cache_mgr, self.kv_cache_manager)
        self.assertEqual(self.data_transfer.transfer_count, 0)
    
    def test_data_transfer_statistics(self):
        """Test data transfer statistics."""
        stats = self.data_transfer.get_statistics()
        
        self.assertEqual(stats['transfer_count'], 0)
        self.assertEqual(stats['total_transfer_time'], 0.0)
        self.assertIsNotNone(stats['communicator_stats'])
        self.assertIsNotNone(stats['buffer_manager_stats'])
    
    def test_cleanup(self):
        """Test data transfer cleanup."""
        # Should not raise any exceptions
        self.data_transfer.cleanup()


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
