import asyncio
import io
import sys
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = ROOT / "demo"
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config, validate_online_batching_config
from connection_manager import ConnectionManager
from streamv2v.inference_common import merge_cli_config
from util import bytes_to_pil, read_images_from_queue
from vid2vid import Pipeline
import vid2vid
import vid2vid_pipe


def make_jpeg_bytes(size=(8, 8), color=(255, 0, 0)):
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeProcess:
    def __init__(self, target=None, args=None, daemon=None):
        self.target = target
        self.args = args or ()
        self.daemon = daemon
        self.pid = 12345
        self._alive = True

    def start(self):
        return None

    def is_alive(self):
        return self._alive


class FakeManager:
    def dict(self):
        return {}


class FakeEvent:
    def __init__(self):
        self._is_set = False
        self.wait_calls = 0

    def set(self):
        self._is_set = True

    def clear(self):
        self._is_set = False

    def is_set(self):
        return self._is_set

    def wait(self, timeout=None):
        self.wait_calls += 1
        self._is_set = True
        return True


class FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def get(self):
        return self.items.pop(0)

    def qsize(self):
        return len(self.items)


class FakeResetPipeline:
    def __init__(self):
        self.reset_count = 0

    def reset_for_idle(self):
        self.reset_count += 1


class OnlineInferenceTests(unittest.TestCase):
    def test_input_params_accept_upload_mode_fields(self):
        params = Pipeline.InputParams(input_mode="upload", upload_mode=True)
        namespace = Pipeline.params_to_namespace(params)

        self.assertEqual(namespace.input_mode, "upload")
        self.assertTrue(namespace.upload_mode)

    def test_input_params_accept_use_taehv(self):
        params = Pipeline.InputParams(use_taehv=True)
        namespace = Pipeline.params_to_namespace(params)
        self.assertTrue(namespace.use_taehv)

    def test_input_params_accept_use_tensorrt(self):
        params = Pipeline.InputParams(use_tensorrt=True)
        namespace = Pipeline.params_to_namespace(params)
        self.assertTrue(namespace.use_tensorrt)

    def test_camera_frame_bytes_decode_to_image(self):
        image = bytes_to_pil(make_jpeg_bytes(size=(6, 4)))
        self.assertEqual(image.size, (6, 4))

    def test_repeated_upload_mode_updates_do_not_reset_upload_queue(self):
        manager = ConnectionManager()
        user_id = uuid.uuid4()
        manager.active_connections[user_id] = {
            "queue": asyncio.Queue(),
            "output_queue": asyncio.Queue(),
            "video_frame_queue": asyncio.Queue(),
            "is_upload_mode": False,
            "video_upload_completed": False,
            "video_queue_index": 0,
            "video_total_frames": 0,
        }

        manager.set_upload_mode(user_id, True)
        asyncio.run(manager.add_video_frame(user_id, b"frame-1"))
        manager.set_video_upload_completed(user_id, True)
        manager.set_upload_mode(user_id, True)

        session = manager.active_connections[user_id]
        self.assertTrue(session["is_upload_mode"])
        self.assertEqual(session["video_frame_queue"].qsize(), 1)
        self.assertEqual(session["video_total_frames"], 1)
        self.assertTrue(session["video_upload_completed"])

    def test_upload_queue_does_not_replay_consumed_frames(self):
        manager = ConnectionManager()
        user_id = uuid.uuid4()
        manager.active_connections[user_id] = {
            "queue": asyncio.Queue(),
            "output_queue": asyncio.Queue(),
            "video_frame_queue": asyncio.Queue(),
            "is_upload_mode": True,
            "video_upload_completed": True,
            "video_queue_index": 0,
            "video_total_frames": 0,
        }

        asyncio.run(manager.add_video_frame(user_id, b"frame-1"))
        asyncio.run(manager.add_video_frame(user_id, b"frame-2"))

        first = asyncio.run(manager.get_next_video_frame(user_id))
        second = asyncio.run(manager.get_next_video_frame(user_id))
        third = asyncio.run(manager.get_next_video_frame(user_id))

        self.assertEqual(first, b"frame-1")
        self.assertEqual(second, b"frame-2")
        self.assertIsNone(third)

    def test_upload_queue_streams_available_frames_before_upload_done(self):
        manager = ConnectionManager()
        user_id = uuid.uuid4()
        manager.active_connections[user_id] = {
            "queue": asyncio.Queue(),
            "output_queue": asyncio.Queue(),
            "video_frame_queue": asyncio.Queue(),
            "is_upload_mode": True,
            "video_upload_completed": False,
            "video_queue_index": 0,
            "video_total_frames": 0,
        }

        asyncio.run(manager.add_video_frame(user_id, b"frame-1"))

        first = asyncio.run(manager.get_next_video_frame(user_id))
        before_done_empty = asyncio.run(manager.get_next_video_frame(user_id))
        manager.set_video_upload_completed(user_id, True)
        after_done_empty = asyncio.run(manager.get_next_video_frame(user_id))

        self.assertEqual(first, b"frame-1")
        self.assertIsNone(before_done_empty)
        self.assertIsNone(after_done_empty)

    def test_disconnect_last_user_resets_pipeline_to_idle(self):
        manager = ConnectionManager()
        user_id = uuid.uuid4()
        pipeline = FakeResetPipeline()
        manager.active_connections[user_id] = {
            "queue": asyncio.Queue(),
            "output_queue": asyncio.Queue(),
            "video_frame_queue": asyncio.Queue(),
            "is_upload_mode": False,
            "video_upload_completed": False,
            "video_queue_index": 0,
            "video_total_frames": 0,
        }

        asyncio.run(manager.disconnect(user_id, pipeline))

        self.assertEqual(pipeline.reset_count, 1)
        self.assertEqual(manager.get_user_count(), 0)

    def test_disconnect_keeps_pipeline_when_other_users_remain(self):
        manager = ConnectionManager()
        first_user_id = uuid.uuid4()
        second_user_id = uuid.uuid4()
        pipeline = FakeResetPipeline()
        for user_id in (first_user_id, second_user_id):
            manager.active_connections[user_id] = {
                "queue": asyncio.Queue(),
                "output_queue": asyncio.Queue(),
                "video_frame_queue": asyncio.Queue(),
                "is_upload_mode": False,
                "video_upload_completed": False,
                "video_queue_index": 0,
                "video_total_frames": 0,
            }

        asyncio.run(manager.disconnect(first_user_id, pipeline))

        self.assertEqual(pipeline.reset_count, 0)
        self.assertEqual(manager.get_user_count(), 1)

    def test_multi_gpu_prepare_uses_model_specific_block_count(self):
        pipeline = vid2vid_pipe.MultiGPUPipeline.__new__(vid2vid_pipe.MultiGPUPipeline)
        pipeline.prompt = "test prompt"
        pipeline.args = merge_cli_config(
            config.config_path,
            {
                **config._asdict(),
                "num_gpus": 2,
                "gpu_ids": "0,1",
                "model_type": "T2V-14B",
            },
        )

        with patch.object(vid2vid_pipe, "Queue", side_effect=list), patch.object(
            vid2vid_pipe, "Event", side_effect=FakeEvent
        ), patch.object(
            vid2vid_pipe, "Manager", return_value=FakeManager()
        ), patch.object(
            vid2vid_pipe, "Process", side_effect=FakeProcess
        ), patch.object(
            vid2vid_pipe, "wait_for_processes_ready", return_value=None
        ):
            pipeline.prepare()

        self.assertEqual(pipeline.total_blocks, 40)
        self.assertEqual(pipeline.total_block_num, [[0, 20], [20, 40]])
        self.assertEqual(len(pipeline.processes), 2)

    def test_accept_new_params_updates_runtime_taehv_and_requests_restart(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.prompt = "prompt"
        pipeline.runtime_state = {"prompt": "prompt", "use_taehv": False, "use_tensorrt": False}
        pipeline.restart_event = FakeEvent()
        pipeline.output_queue = FakeQueue()
        pipeline.input_queue = FakeQueue()

        pipeline.accept_new_params(Pipeline.InputParams(use_taehv=True))

        self.assertTrue(pipeline.runtime_state["use_taehv"])
        self.assertTrue(pipeline.restart_event.is_set())

    def test_accept_new_params_updates_runtime_tensorrt_and_requests_restart(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.prompt = "prompt"
        pipeline.runtime_state = {"prompt": "prompt", "use_taehv": False, "use_tensorrt": False}
        pipeline.restart_event = FakeEvent()
        pipeline.output_queue = FakeQueue()
        pipeline.input_queue = FakeQueue()

        pipeline.accept_new_params(Pipeline.InputParams(use_tensorrt=True))

        self.assertTrue(pipeline.runtime_state["use_tensorrt"])
        self.assertTrue(pipeline.restart_event.is_set())

    def test_reset_for_idle_soft_resets_without_recreating_worker(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.input_queue = FakeQueue()
        pipeline.output_queue = FakeQueue()
        pipeline.restart_event = FakeEvent()
        pipeline.idle_event = FakeEvent()
        pipeline.close = lambda: self.fail("reset_for_idle should not close the worker")
        pipeline.prepare = lambda: self.fail("reset_for_idle should not recreate the worker")
        pipeline.input_queue.put("stale-input")
        pipeline.output_queue.put("stale-output")

        pipeline.reset_for_idle()

        self.assertEqual(pipeline.input_queue.qsize(), 0)
        self.assertEqual(pipeline.output_queue.qsize(), 0)
        self.assertTrue(pipeline.restart_event.is_set())
        self.assertTrue(pipeline.idle_event.is_set())
        self.assertEqual(pipeline.idle_event.wait_calls, 1)

    def test_read_images_from_queue_can_return_wait_time(self):
        frame = np.zeros((4, 4, 3), dtype=np.float32)
        queue = FakeQueue()
        queue.put(frame)
        queue.put(frame)

        images, wait_time = read_images_from_queue(
            queue,
            num_frames_needed=2,
            device=torch.device("cpu"),
            return_wait_time=True,
        )

        self.assertEqual(tuple(images.shape), (1, 3, 2, 4, 4))
        self.assertGreaterEqual(wait_time, 0.0)

    def test_read_images_from_queue_uses_oldest_enqueued_frame_age(self):
        frame = np.zeros((4, 4, 3), dtype=np.float32)
        queue = FakeQueue()
        old_enqueue_time = time.monotonic() - 1.25
        queue.put((frame, old_enqueue_time))
        queue.put((frame, time.monotonic()))

        _, wait_time = read_images_from_queue(
            queue,
            num_frames_needed=2,
            device=torch.device("cpu"),
            return_wait_time=True,
        )

        self.assertGreaterEqual(wait_time, 1.0)

    def test_build_single_gpu_pipeline_manager_uses_explicit_modes(self):
        class FakePipeline:
            def __init__(self, args, device):
                self.args = args
                self.device = device
                self.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

            def load_model(self, checkpoint_folder):
                self.checkpoint_folder = checkpoint_folder

        args = SimpleNamespace(
            online_batching_mode="wo_batch",
            checkpoint_folder="/tmp/checkpoint",
            use_taehv=False,
            use_tensorrt=False,
        )

        with patch.object(vid2vid, "StreamNoBatchInferencePipeline", FakePipeline):
            pipeline, mode_info = vid2vid.build_single_gpu_pipeline_manager(args, torch.device("cpu"))

        self.assertIsInstance(pipeline, FakePipeline)
        self.assertEqual(mode_info["mode"], "wo_batch")

    def test_build_single_gpu_batch_mode_preserves_memory_fallback(self):
        class FakePipeline:
            def __init__(self, args, device):
                self.args = args
                self.device = device
                self.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

            def load_model(self, checkpoint_folder):
                self.checkpoint_folder = checkpoint_folder

        args = SimpleNamespace(
            online_batching_mode="batch",
            checkpoint_folder="/tmp/checkpoint",
            use_taehv=False,
            use_tensorrt=False,
        )

        with patch.object(
            vid2vid,
            "select_stream_execution_mode",
            return_value={"mode": "stream_wo_batch"},
        ), patch.object(vid2vid, "StreamNoBatchInferencePipeline", FakePipeline):
            pipeline, mode_info = vid2vid.build_single_gpu_pipeline_manager(args, torch.device("cpu"))

        self.assertIsInstance(pipeline, FakePipeline)
        self.assertEqual(mode_info["mode"], "stream_wo_batch")

    def test_auto_restore_replicates_self_attention_evict_state(self):
        blocks = [
            SimpleNamespace(self_attn=SimpleNamespace(evict_idx=[[1, 2], [3], [4]])),
            SimpleNamespace(self_attn=SimpleNamespace(evict_idx=[[5], [6]])),
        ]
        generator = SimpleNamespace(model=SimpleNamespace(blocks=blocks))

        vid2vid.SLOAdaptiveSingleGPUInferencePipeline._copy_first_self_attn_evict_state(
            generator,
            batch_size=3,
        )

        self.assertEqual(blocks[0].self_attn.evict_idx, [[1, 2], [1, 2], [1, 2]])
        self.assertEqual(blocks[1].self_attn.evict_idx, [[5], [5], [5]])
        self.assertIsNot(blocks[0].self_attn.evict_idx[0], blocks[0].self_attn.evict_idx[1])

    def test_invalid_multi_gpu_non_batch_mode_raises(self):
        with self.assertRaisesRegex(ValueError, "only supported when --num_gpus=1"):
            validate_online_batching_config(num_gpus=2, online_batching_mode="auto")


if __name__ == "__main__":
    unittest.main()
