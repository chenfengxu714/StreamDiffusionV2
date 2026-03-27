"""Public Python API for simple offline video-to-video inference."""

from __future__ import annotations

from contextlib import ExitStack
from importlib.resources import as_file, files
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
from typing import Literal, Sequence

import torch

from streamv2v.inference_common import load_mp4_as_tensor, normalize_acceleration_flags


InferenceMode = Literal["single", "single-wo", "pipe"]

_SINGLE_MODE_TO_MODULE = {
    "single": "streamv2v.inference",
    "single-wo": "streamv2v.inference_wo_batch",
}


def _resolve_default_config_path(resource_stack: ExitStack) -> str:
    resource = files("streamv2v.configs").joinpath("wan_causal_dmd_v2v.yaml")
    return str(resource_stack.enter_context(as_file(resource)))


def _normalize_gpu_ids(gpu_ids: int | Sequence[int] | None) -> list[int] | None:
    if gpu_ids is None:
        return None
    if isinstance(gpu_ids, int):
        return [gpu_ids]
    return [int(gpu_id) for gpu_id in gpu_ids]


def _normalize_device_gpu_id(device: str | torch.device | None) -> list[int] | None:
    if device is None:
        return None
    device_str = str(device)
    if not device_str.startswith("cuda:"):
        return None
    return [int(device_str.split(":", 1)[1])]


def _resolve_single_gpu_id(gpu_ids: list[int] | None) -> int | None:
    if gpu_ids is None:
        return None
    if len(gpu_ids) != 1:
        raise ValueError("single and single-wo modes accept exactly one GPU id")
    return int(gpu_ids[0])


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_common_args(
    *,
    config_path: str,
    checkpoint_folder: str,
    video_path: str,
    prompt_file_path: str,
    output_folder: str,
    noise_scale: float,
    height: int,
    width: int,
    fps: int,
    step: int,
    seed: int,
    model_type: str,
    profile: bool,
    use_taehv: bool,
    use_tensorrt: bool,
    fast: bool,
) -> list[str]:
    args = [
        "--config_path",
        config_path,
        "--checkpoint_folder",
        checkpoint_folder,
        "--output_folder",
        output_folder,
        "--prompt_file_path",
        prompt_file_path,
        "--video_path",
        video_path,
        "--noise_scale",
        str(noise_scale),
        "--height",
        str(height),
        "--width",
        str(width),
        "--fps",
        str(fps),
        "--step",
        str(step),
        "--seed",
        str(seed),
        "--model_type",
        model_type,
    ]
    if profile:
        args.append("--profile")
    if use_taehv:
        args.append("--use_taehv")
    if use_tensorrt:
        args.append("--use_tensorrt")
    if fast:
        args.append("--fast")
    return args


class StreamVideoToVideo:
    """Convenience wrapper around the offline Python entrypoints."""

    def __init__(
        self,
        checkpoint_folder: str,
        mode: InferenceMode = "single",
        *,
        config_path: str | None = None,
        device: str | torch.device | None = None,
        gpu_ids: int | Sequence[int] | None = None,
        num_gpus: int | None = None,
        noise_scale: float = 0.8,
        height: int = 480,
        width: int = 832,
        fps: int = 16,
        step: int = 2,
        seed: int = 0,
        model_type: str = "T2V-1.3B",
        use_taehv: bool = False,
        use_tensorrt: bool = False,
        fast: bool = False,
        profile: bool = False,
        schedule_block: bool = False,
    ) -> None:
        self.checkpoint_folder = checkpoint_folder
        self.mode = mode
        self.config_path = config_path
        self.device = device
        self.gpu_ids = gpu_ids
        self.num_gpus = num_gpus
        self.noise_scale = noise_scale
        self.height = height
        self.width = width
        self.fps = fps
        self.step = step
        self.seed = seed
        self.model_type = model_type
        self.use_taehv = use_taehv
        self.use_tensorrt = use_tensorrt
        self.fast = fast
        self.profile = profile
        self.schedule_block = schedule_block

    def generate(self, video_path: str, prompt: str) -> torch.Tensor:
        with tempfile.TemporaryDirectory(prefix="streamv2v_generate_") as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")
            self.run_video(video_path=video_path, prompt=prompt, output_path=output_path)
            return load_mp4_as_tensor(output_path, normalize=False)

    def run_video(self, video_path: str, prompt: str, output_path: str) -> str:
        return run_video_to_video(
            checkpoint_folder=self.checkpoint_folder,
            video_path=video_path,
            prompt=prompt,
            output_path=output_path,
            mode=self.mode,
            config_path=self.config_path,
            device=self.device,
            gpu_ids=self.gpu_ids,
            num_gpus=self.num_gpus,
            noise_scale=self.noise_scale,
            height=self.height,
            width=self.width,
            fps=self.fps,
            step=self.step,
            seed=self.seed,
            model_type=self.model_type,
            use_taehv=self.use_taehv,
            use_tensorrt=self.use_tensorrt,
            fast=self.fast,
            profile=self.profile,
            schedule_block=self.schedule_block,
        )


def run_video_to_video(
    *,
    checkpoint_folder: str,
    video_path: str,
    prompt: str,
    output_path: str,
    mode: InferenceMode = "single",
    config_path: str | None = None,
    device: str | torch.device | None = None,
    gpu_ids: int | Sequence[int] | None = None,
    num_gpus: int | None = None,
    noise_scale: float = 0.8,
    height: int = 480,
    width: int = 832,
    fps: int = 16,
    step: int = 2,
    seed: int = 0,
    model_type: str = "T2V-1.3B",
    use_taehv: bool = False,
    use_tensorrt: bool = False,
    fast: bool = False,
    profile: bool = False,
    schedule_block: bool = False,
) -> str:
    """Run offline video-to-video inference from Python."""
    flags = normalize_acceleration_flags(
        {
            "use_taehv": use_taehv,
            "use_tensorrt": use_tensorrt,
            "fast": fast,
        }
    )
    use_taehv = bool(flags["use_taehv"])
    use_tensorrt = bool(flags["use_tensorrt"])
    fast = bool(flags["fast"])

    requested_gpu_ids = _normalize_gpu_ids(gpu_ids)
    device_gpu_ids = _normalize_device_gpu_id(device)
    if requested_gpu_ids is None and device_gpu_ids is not None:
        requested_gpu_ids = device_gpu_ids

    if mode == "pipe":
        if num_gpus is None:
            num_gpus = len(requested_gpu_ids) if requested_gpu_ids is not None else 2
        if requested_gpu_ids is not None and len(requested_gpu_ids) != num_gpus:
            raise ValueError("num_gpus must match len(gpu_ids) for pipe mode")
    elif num_gpus is not None and num_gpus != 1:
        raise ValueError("num_gpus is only used for pipe mode")

    resource_stack = ExitStack()
    try:
        resolved_config_path = config_path or _resolve_default_config_path(resource_stack)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="streamv2v_api_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            prompt_path = temp_dir_path / "prompt.txt"
            prompt_path.write_text(prompt + "\n", encoding="utf-8")
            temp_output_dir = temp_dir_path / "outputs"
            temp_output_dir.mkdir(parents=True, exist_ok=True)

            common_args = _build_common_args(
                config_path=resolved_config_path,
                checkpoint_folder=checkpoint_folder,
                video_path=video_path,
                prompt_file_path=str(prompt_path),
                output_folder=str(temp_output_dir),
                noise_scale=noise_scale,
                height=height,
                width=width,
                fps=fps,
                step=step,
                seed=seed,
                model_type=model_type,
                profile=profile,
                use_taehv=use_taehv,
                use_tensorrt=use_tensorrt,
                fast=fast,
            )

            env = os.environ.copy()
            if requested_gpu_ids is not None and mode == "pipe":
                env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in requested_gpu_ids)

            if mode == "pipe":
                cmd = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    f"--nproc_per_node={num_gpus}",
                    f"--master_port={_pick_free_port()}",
                    "-m",
                    "streamv2v.inference_pipe",
                    *common_args,
                ]
                if schedule_block:
                    cmd.append("--schedule_block")
            else:
                module_name = _SINGLE_MODE_TO_MODULE.get(mode)
                if module_name is None:
                    raise ValueError(f"Unsupported mode: {mode}")
                cmd = [sys.executable, "-m", module_name, *common_args]
                single_gpu_id = _resolve_single_gpu_id(requested_gpu_ids)
                if single_gpu_id is not None:
                    cmd.extend(["--gpu_id", str(single_gpu_id)])

            subprocess.run(cmd, env=env, check=True)
            generated_path = temp_output_dir / "output_000.mp4"
            shutil.copy2(generated_path, output_file)
        return str(output_file)
    finally:
        resource_stack.close()
