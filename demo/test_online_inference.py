import asyncio
import io
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = ROOT / "demo"
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config
from connection_manager import ConnectionManager
from streamv2v.inference_common import merge_cli_config
from util import bytes_to_pil
from vid2vid import Pipeline
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

    def set(self):
        self._is_set = True

    def is_set(self):
        return self._is_set


class OnlineInferenceTests(unittest.TestCase):
    def test_input_params_accept_upload_mode_fields(self):
        params = Pipeline.InputParams(input_mode="upload", upload_mode=True)
        namespace = Pipeline.params_to_namespace(params)

        self.assertEqual(namespace.input_mode, "upload")
        self.assertTrue(namespace.upload_mode)

    def test_camera_frame_bytes_decode_to_image(self):
        image = bytes_to_pil(make_jpeg_bytes(size=(6, 4)))
        self.assertEqual(image.size, (6, 4))

    def test_repeated_upload_mode_updates_do_not_reset_cached_frames(self):
        manager = ConnectionManager()
        user_id = uuid.uuid4()
        manager.active_connections[user_id] = {
            "queue": asyncio.Queue(),
            "output_queue": asyncio.Queue(),
            "video_frame_queue": asyncio.Queue(),
            "video_frames": [],
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
        self.assertEqual(session["video_frames"], [b"frame-1"])
        self.assertTrue(session["video_upload_completed"])

    def test_upload_queue_loops_cached_video_frames(self):
        manager = ConnectionManager()
        user_id = uuid.uuid4()
        manager.active_connections[user_id] = {
            "queue": asyncio.Queue(),
            "output_queue": asyncio.Queue(),
            "video_frame_queue": asyncio.Queue(),
            "video_frames": [],
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
        self.assertEqual(third, b"frame-1")

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


if __name__ == "__main__":
    unittest.main()
