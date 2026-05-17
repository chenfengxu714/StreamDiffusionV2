import sys
import os
import logging
import queue
import time
import traceback
import threading
from multiprocessing import Queue, Manager, Event, Process
from typing import Literal

DEMO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DEMO_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from util import (
    array_to_image,
    clear_queue,
    dump_pydantic_model,
    image_to_array,
    read_images_from_queue,
    resolve_worker_device,
    select_stream_execution_mode,
)

import torch

from pydantic import BaseModel, Field
from PIL import Image
from typing import List
from streamv2v.inference import (
    SingleGPUInferencePipeline as StreamBatchInferencePipeline,
    compute_noise_scale_and_step,
)
from streamv2v.inference_wo_batch import SingleGPUInferencePipeline as StreamNoBatchInferencePipeline
from streamv2v.inference_common import merge_cli_config

LOGGER = logging.getLogger(__name__)
STARTUP_TIMEOUT_SECONDS = 180.0

default_prompt = "Cyberpunk-inspired figure, neon-lit hair highlights, augmented cybernetic facial features, glowing interface holograms floating around, futuristic cityscape reflected in eyes, vibrant neon color palette, cinematic sci-fi style"

page_content = """<h1 class="text-3xl font-bold">StreamDiffusionV2</h1>
<p class="text-sm">
    This demo showcases
    <a
    href="https://streamdiffusionv2.github.io/"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamDiffusionV2
</a>
video-to-video pipeline with a MJPEG stream server.
</p>
"""


def set_config_value(config, key: str, value) -> None:
    if isinstance(config, dict):
        config[key] = value
        return
    setattr(config, key, value)


def sync_pydantic_field_default(model_cls, field_name: str, value) -> None:
    if hasattr(model_cls, "model_fields") and field_name in model_cls.model_fields:
        model_cls.model_fields[field_name].default = value
    if hasattr(model_cls, "__fields__") and field_name in model_cls.__fields__:
        model_cls.__fields__[field_name].default = value


class SLOAdaptiveSingleGPUInferencePipeline(StreamBatchInferencePipeline):
    """Switch one loaded single-GPU stream between batch and no-batch execution."""

    def __init__(self, config, device: torch.device):
        super().__init__(config, device)
        self.online_slo_wait_threshold = float(getattr(config, "online_slo_wait_threshold", 0.5))
        self.active_mode = "batch"
        self._batch_state_refs = None
        self._canonical_denoising_step_list = self.pipeline.denoising_step_list.clone()

    def start_stream_session(self, prompt: str, images: torch.Tensor, noise_scale: float):
        self.active_mode = "batch"
        self._batch_state_refs = None
        self.pipeline.denoising_step_list = self._canonical_denoising_step_list.clone()
        return super().start_stream_session(prompt, images, noise_scale)

    def _snapshot_batch_state_refs(self) -> dict:
        conditional_dict = self.pipeline.conditional_dict or {}
        return {
            "kv_cache1": [
                {
                    "k": entry["k"],
                    "v": entry["v"],
                    "global_end_index": entry["global_end_index"],
                    "local_end_index": entry["local_end_index"],
                }
                for entry in self.pipeline.kv_cache1
            ],
            "crossattn_cache": [
                {
                    "k": entry["k"],
                    "v": entry["v"],
                    "is_init": entry.get("is_init", False),
                }
                for entry in self.pipeline.crossattn_cache
            ],
            "prompt_embeds": conditional_dict.get("prompt_embeds"),
            "hidden_states": self.pipeline.hidden_states,
            "block_x": self.pipeline.block_x,
            "kv_cache_starts": self.pipeline.kv_cache_starts,
            "kv_cache_ends": self.pipeline.kv_cache_ends,
        }

    def _enter_wo_batch(self) -> None:
        if self.active_mode == "wo_batch":
            return

        self._batch_state_refs = self._snapshot_batch_state_refs()
        for entry, saved in zip(self.pipeline.kv_cache1, self._batch_state_refs["kv_cache1"]):
            entry["k"] = saved["k"][:1]
            entry["v"] = saved["v"][:1]
            entry["global_end_index"] = saved["global_end_index"][:1]
            entry["local_end_index"] = saved["local_end_index"][:1]

        for entry, saved in zip(self.pipeline.crossattn_cache, self._batch_state_refs["crossattn_cache"]):
            entry["k"] = saved["k"][:1]
            entry["v"] = saved["v"][:1]
            entry["is_init"] = saved["is_init"]

        prompt_embeds = self._batch_state_refs["prompt_embeds"]
        if prompt_embeds is not None:
            self.pipeline.conditional_dict["prompt_embeds"] = prompt_embeds[:1]

        previous_mode = self.active_mode
        self.active_mode = "wo_batch"
        self.logger.info("Online auto batching converted from %s to %s", previous_mode, self.active_mode)

    @staticmethod
    def _copy_first_sequence_to_all(tensor: torch.Tensor) -> None:
        if tensor.shape[0] > 1 and tensor.stride(0) != 0:
            tensor[1:].copy_(tensor[:1].expand_as(tensor[1:]))

    @staticmethod
    def _copy_first_self_attn_evict_state(generator, batch_size: int) -> None:
        for block in getattr(generator.model, "blocks", []):
            self_attn = getattr(block, "self_attn", None)
            evict_idx = getattr(self_attn, "evict_idx", None)
            if isinstance(evict_idx, list) and evict_idx:
                self_attn.evict_idx = [list(evict_idx[0]) for _ in range(batch_size)]

    def _return_to_batch(self, session) -> None:
        if self.active_mode == "batch":
            return
        if self._batch_state_refs is None:
            raise RuntimeError("Cannot restore batch mode before batch state has been captured")

        for entry, saved in zip(self.pipeline.kv_cache1, self._batch_state_refs["kv_cache1"]):
            self._copy_first_sequence_to_all(saved["k"])
            self._copy_first_sequence_to_all(saved["v"])
            self._copy_first_sequence_to_all(saved["global_end_index"])
            self._copy_first_sequence_to_all(saved["local_end_index"])
            if "total_steps" in entry:
                entry["current_step"] = entry["total_steps"]
            entry["k"] = saved["k"]
            entry["v"] = saved["v"]
            entry["global_end_index"] = saved["global_end_index"]
            entry["local_end_index"] = saved["local_end_index"]

        for entry, saved in zip(self.pipeline.crossattn_cache, self._batch_state_refs["crossattn_cache"]):
            self._copy_first_sequence_to_all(saved["k"])
            self._copy_first_sequence_to_all(saved["v"])
            entry["k"] = saved["k"]
            entry["v"] = saved["v"]
            entry["is_init"] = saved["is_init"]

        if self._batch_state_refs["prompt_embeds"] is not None:
            self.pipeline.conditional_dict["prompt_embeds"] = self._batch_state_refs["prompt_embeds"]

        self.pipeline.hidden_states = torch.zeros_like(self._batch_state_refs["hidden_states"])
        self.pipeline.block_x = (
            torch.zeros_like(self._batch_state_refs["block_x"])
            if self._batch_state_refs["block_x"] is not None
            else None
        )
        self.pipeline.kv_cache_starts = self._batch_state_refs["kv_cache_starts"]
        self.pipeline.kv_cache_ends = self._batch_state_refs["kv_cache_ends"]
        self.pipeline.kv_cache_starts.fill_(int(session.current_start))
        self.pipeline.kv_cache_ends.fill_(int(session.current_end))
        self.pipeline.denoising_step_list = self._canonical_denoising_step_list.clone()
        self.pipeline.timestep = self.pipeline.denoising_step_list.clone()
        self._copy_first_self_attn_evict_state(
            self.pipeline.generator,
            batch_size=int(self.pipeline.hidden_states.shape[0]),
        )

        session.processed = 0
        self.processed = 0
        previous_mode = self.active_mode
        self.active_mode = "batch"
        self._batch_state_refs = None
        self.logger.info("Online auto batching converted from %s to %s", previous_mode, self.active_mode)

    def _run_stream_batch_wo_batch(self, session, images: torch.Tensor) -> List:
        num_frames = images.shape[2]
        input_batch = num_frames // session.chunk_size
        noise_scale, current_step = compute_noise_scale_and_step(
            input_video_original=torch.cat([session.last_image, images], dim=2),
            end_idx=num_frames + 1,
            chunk_size=num_frames,
            noise_scale=float(session.noise_scale),
            init_noise_scale=float(session.init_noise_scale),
        )
        noisy_latents = self._encode_noisy_latents(images, noise_scale)

        outputs = []
        for batch_idx in range(input_batch):
            if session.current_start // self.pipeline.frame_seq_length >= self.t_refresh:
                session.current_start = self.pipeline.kv_cache_length - self.pipeline.frame_seq_length
                session.current_end = session.current_start + (session.chunk_size // self.base_chunk_size) * self.pipeline.frame_seq_length

            denoised_pred = self.pipeline.inference_wo_batch(
                noise=noisy_latents[:, batch_idx].unsqueeze(1),
                current_start=session.current_start,
                current_end=session.current_end,
                current_step=current_step,
            )

            session.processed += 1
            self.processed = session.processed
            outputs.append(self._decode_video_array(denoised_pred, last_frame_only=True))

            session.current_start = session.current_end
            session.current_end += (session.chunk_size // self.base_chunk_size) * self.pipeline.frame_seq_length

        session.last_image = images[:, :, [-1]]
        session.noise_scale = noise_scale
        return outputs

    def run_stream_batch(self, session, images: torch.Tensor, queue_wait_time: float | None = None) -> List:
        target_mode = (
            "wo_batch"
            if queue_wait_time is not None and queue_wait_time >= self.online_slo_wait_threshold
            else "batch"
        )

        if target_mode == "wo_batch":
            self._enter_wo_batch()
            return self._run_stream_batch_wo_batch(session, images)

        self._return_to_batch(session)
        return super().run_stream_batch(session, images, queue_wait_time=queue_wait_time)


def build_single_gpu_pipeline_manager(args, device: torch.device):
    online_batching_mode = getattr(args, "online_batching_mode", "batch")
    if online_batching_mode == "batch":
        mode_info = select_stream_execution_mode(args, device)
        pipeline_cls = (
            StreamBatchInferencePipeline
            if mode_info["mode"] == "stream_batch"
            else StreamNoBatchInferencePipeline
        )
    elif online_batching_mode == "wo_batch":
        mode_info = {"mode": "wo_batch"}
        pipeline_cls = StreamNoBatchInferencePipeline
    elif online_batching_mode == "auto":
        mode_info = {"mode": "auto"}
        pipeline_cls = SLOAdaptiveSingleGPUInferencePipeline
    else:
        raise ValueError(
            f"Unsupported online_batching_mode={online_batching_mode!r}; expected batch, wo_batch, or auto"
        )
    pipeline_manager = pipeline_cls(args, device)
    pipeline_manager.load_model(args.checkpoint_folder)
    pipeline_manager.logger.info(
        "Online single-GPU worker selected online_batching_mode=%s, resolved_mode=%s, use_taehv=%s, use_tensorrt=%s",
        online_batching_mode,
        mode_info["mode"],
        bool(getattr(args, "use_taehv", False)),
        bool(getattr(args, "use_tensorrt", False)),
    )
    return pipeline_manager, mode_info

class Pipeline:
    class Info(BaseModel):
        name: str = "StreamV2V"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        
        prompt: str = Field(
            default_prompt,
            title="Update your prompt here",
            field="textarea",
            id="prompt",
        )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        restart: bool = Field(
            default=False,
            title="Restart",
            description="Restart the streaming",
        )
        input_mode: Literal["camera", "upload"] = Field(
            default="camera",
            title="Input Mode",
            hide=True,
            id="input_mode",
        )
        upload_mode: bool = Field(
            default=False,
            title="Upload Mode",
            hide=True,
            id="upload_mode",
        )
        use_taehv: bool = Field(
            default=False,
            title="Use TAEHV VAE",
            description="Use the lightweight TAEHV decoder for online inference",
            field="checkbox",
            hide=True,
            id="use_taehv",
        )
        use_tensorrt: bool = Field(
            default=False,
            title="Use TensorRT",
            description="Enable available TensorRT acceleration paths for online inference",
            field="checkbox",
            hide=True,
            id="use_tensorrt",
        )

    def __init__(self, args):
        torch.set_grad_enabled(False)

        config = merge_cli_config(args.config_path, args._asdict())
        sync_pydantic_field_default(self.InputParams, "use_taehv", bool(getattr(config, "use_taehv", False)))
        sync_pydantic_field_default(self.InputParams, "use_tensorrt", bool(getattr(config, "use_tensorrt", False)))
        params = self.InputParams()
        config["height"] = params.height
        config["width"] = params.width

        self.prompt = params.prompt
        self.args = config
        self._lifecycle_lock = threading.RLock()
        self.prepare()

    def _get_lifecycle_lock(self):
        if not hasattr(self, "_lifecycle_lock"):
            self._lifecycle_lock = threading.RLock()
        return self._lifecycle_lock

    def prepare(self):
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.prepare_event = Event()
        self.stop_event = Event()
        self.restart_event = Event()
        self.idle_event = Event()
        self.error_queue = Queue()
        self.runtime_state = Manager().dict()
        self.runtime_state["prompt"] = self.prompt
        self.runtime_state["use_taehv"] = bool(getattr(self.args, "use_taehv", False))
        self.runtime_state["use_tensorrt"] = bool(getattr(self.args, "use_tensorrt", False))
        self.process = Process(
            target=generate_process,
            args=(
                self.args,
                self.runtime_state,
                self.prepare_event,
                self.restart_event,
                self.stop_event,
                self.idle_event,
                self.input_queue,
                self.output_queue,
                self.error_queue,
            ),
            daemon=True
        )
        self.process.start()
        self.processes = [self.process]
        wait_for_processes_ready(
            processes=self.processes,
            ready_events=[self.prepare_event],
            error_queue=self.error_queue,
        )

    def accept_new_params(self, params: "Pipeline.InputParams"):
        with self._get_lifecycle_lock():
            if hasattr(params, "image"):
                image_array = image_to_array(params.image, self.args.width, self.args.height)
                self.input_queue.put((image_array, time.monotonic()))

            if hasattr(params, "prompt") and params.prompt and self.prompt != params.prompt:
                self.prompt = params.prompt
                self.runtime_state["prompt"] = self.prompt

            if hasattr(params, "use_taehv"):
                requested_use_taehv = bool(params.use_taehv)
                if requested_use_taehv != bool(self.runtime_state.get("use_taehv", False)):
                    self.runtime_state["use_taehv"] = requested_use_taehv
                    self.restart_event.set()
                    clear_queue(self.output_queue)

            if hasattr(params, "use_tensorrt"):
                requested_use_tensorrt = bool(params.use_tensorrt)
                if requested_use_tensorrt != bool(self.runtime_state.get("use_tensorrt", False)):
                    self.runtime_state["use_tensorrt"] = requested_use_tensorrt
                    self.restart_event.set()
                    clear_queue(self.output_queue)

            if hasattr(params, "restart") and params.restart:
                self.restart_event.set()
                clear_queue(self.output_queue)

    @staticmethod
    def params_to_namespace(params: "Pipeline.InputParams"):
        from types import SimpleNamespace

        return SimpleNamespace(**dump_pydantic_model(params))

    def produce_outputs(self) -> List[Image.Image]:
        with self._get_lifecycle_lock():
            qsize = self.output_queue.qsize()
            results = []
            for _ in range(qsize):
                results.append(array_to_image(self.output_queue.get()))
            return results

    def close(self):
        with self._get_lifecycle_lock():
            LOGGER.info("Setting stop event for the single-GPU demo worker")
            self.stop_event.set()

            LOGGER.info("Waiting for demo worker shutdown")
            for i, process in enumerate(self.processes):
                process.join(timeout=1.0)
                if process.is_alive():
                    LOGGER.warning("Process %s did not terminate gracefully; terminating", i)
                    process.terminate()
                    process.join(timeout=0.5)
                    if process.is_alive():
                        LOGGER.error("Force killing process %s", i)
                        process.kill()
            LOGGER.info("Pipeline closed successfully")

    def reset_for_idle(self):
        with self._get_lifecycle_lock():
            LOGGER.info("Resetting demo pipeline to idle waiting mode")
            clear_queue(self.input_queue)
            clear_queue(self.output_queue)
            self.idle_event.clear()
            self.restart_event.set()
            if not self.idle_event.wait(timeout=5.0):
                LOGGER.warning("Timed out waiting for demo worker to enter idle mode")


def _maybe_raise_worker_error(error_queue):
    try:
        worker_name, error_message = error_queue.get_nowait()
    except queue.Empty:
        return
    raise RuntimeError(f"{worker_name} failed during startup:\n{error_message}")


def wait_for_processes_ready(processes, ready_events, error_queue, timeout_seconds: float = STARTUP_TIMEOUT_SECONDS):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        _maybe_raise_worker_error(error_queue)
        if all(event.is_set() for event in ready_events):
            return
        dead_processes = [process.pid for process in processes if not process.is_alive()]
        if dead_processes:
            raise RuntimeError(f"Demo worker processes exited before becoming ready: {dead_processes}")
        time.sleep(0.1)

    _maybe_raise_worker_error(error_queue)
    raise TimeoutError(f"Timed out waiting for demo workers to become ready after {timeout_seconds:.0f}s")


def report_worker_error(error_queue, worker_name: str) -> None:
    error_queue.put((worker_name, traceback.format_exc()))


def generate_process(args, runtime_state, prepare_event, restart_event, stop_event, idle_event, input_queue, output_queue, error_queue):
    torch.set_grad_enabled(False)
    try:
        class ResetOrStopEvent:
            def is_set(self):
                return stop_event.is_set() or restart_event.is_set()

        read_interrupt_event = ResetOrStopEvent()

        def reset_to_waiting_mode():
            clear_queue(input_queue)
            clear_queue(output_queue)
            restart_event.clear()
            idle_event.set()
            return None, False

        device = resolve_worker_device(args.gpu_ids, rank=0)
        torch.cuda.set_device(device)

        current_use_taehv = bool(runtime_state.get("use_taehv", getattr(args, "use_taehv", False)))
        current_use_tensorrt = bool(runtime_state.get("use_tensorrt", getattr(args, "use_tensorrt", False)))
        set_config_value(args, "use_taehv", current_use_taehv)
        set_config_value(args, "use_tensorrt", current_use_tensorrt)
        pipeline_manager, _ = build_single_gpu_pipeline_manager(args, device)
        chunk_size = pipeline_manager.base_chunk_size * args.num_frame_per_block
        first_batch_num_frames = 1 + chunk_size
        is_running = False
        prompt = runtime_state["prompt"]
        session = None

        prepare_event.set()
        idle_event.set()

        while not stop_event.is_set():
            requested_use_taehv = bool(runtime_state.get("use_taehv", current_use_taehv))
            requested_use_tensorrt = bool(runtime_state.get("use_tensorrt", current_use_tensorrt))
            if requested_use_taehv != current_use_taehv or requested_use_tensorrt != current_use_tensorrt:
                pipeline_manager.logger.info(
                    "Rebuilding online single-GPU worker for use_taehv=%s, use_tensorrt=%s",
                    requested_use_taehv,
                    requested_use_tensorrt,
                )
                current_use_taehv = requested_use_taehv
                current_use_tensorrt = requested_use_tensorrt
                set_config_value(args, "use_taehv", current_use_taehv)
                set_config_value(args, "use_tensorrt", current_use_tensorrt)
                clear_queue(input_queue)
                clear_queue(output_queue)
                del pipeline_manager
                torch.cuda.empty_cache()
                pipeline_manager, _ = build_single_gpu_pipeline_manager(args, device)
                chunk_size = pipeline_manager.base_chunk_size * args.num_frame_per_block
                first_batch_num_frames = 1 + chunk_size
                prompt = runtime_state["prompt"]
                session = None
                is_running = False
                restart_event.clear()
                idle_event.set()
                continue

            # Prepare first batch
            if not is_running or runtime_state["prompt"] != prompt or restart_event.is_set():
                prompt = runtime_state["prompt"]
                if restart_event.is_set():
                    session, is_running = reset_to_waiting_mode()
                images = read_images_from_queue(input_queue, first_batch_num_frames, device, read_interrupt_event)
                if images is None:
                    if stop_event.is_set():
                        break
                    if restart_event.is_set():
                        session, is_running = reset_to_waiting_mode()
                        continue

                idle_event.clear()
                session, initial_video = pipeline_manager.start_stream_session(
                    prompt=prompt,
                    images=images,
                    noise_scale=args.noise_scale,
                )
                for image in initial_video:
                    output_queue.put(image)
                is_running = True

            images, queue_wait_time = read_images_from_queue(
                input_queue,
                chunk_size,
                device,
                read_interrupt_event,
                return_wait_time=True,
            )
            if images is None:
                if stop_event.is_set():
                    break
                if restart_event.is_set():
                    session, is_running = reset_to_waiting_mode()
                    continue

            for decoded_video in pipeline_manager.run_stream_batch(session, images, queue_wait_time=queue_wait_time):
                for image in decoded_video:
                    output_queue.put(image)
    except Exception:
        report_worker_error(error_queue, "single_gpu_demo_worker")
        raise
