"""Readable staged video-to-video API for StreamDiffusionV2."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Literal

from diffusers.utils import export_to_video as diffusers_export_to_video
import numpy as np
import torch

from models.util import set_seed
from streamv2v.inference import (
    SingleGPUInferencePipeline as StreamBatchInferencePipeline,
    compute_noise_scale_and_step,
)
from streamv2v.inference_common import load_mp4_as_tensor, merge_cli_config, normalize_acceleration_flags
from streamv2v.inference_wo_batch import SingleGPUInferencePipeline as StreamNoBatchInferencePipeline


SingleMode = Literal["single", "single-wo"]


@dataclass
class VideoChunk:
    """One video chunk prepared for the encode -> denoise -> decode loop."""

    frames: torch.Tensor
    start_idx: int
    end_idx: int
    current_start: int
    current_end: int


@dataclass
class EncodedChunk:
    """Encoded latent chunk plus the schedule metadata needed for denoising."""

    noisy_latents: torch.Tensor
    current_start: int
    current_end: int
    noise_scale: float
    current_step: int | None = None


@dataclass
class DenoisedChunk:
    """Denoised latent chunk ready for VAE decoding."""

    denoised_pred: torch.Tensor
    last_frame_only: bool


def _resolve_default_config_path(resource_stack: ExitStack) -> str:
    resource = files("streamv2v.configs").joinpath("wan_causal_dmd_v2v.yaml")
    return str(resource_stack.enter_context(as_file(resource)))


def _resolve_device(device: str | torch.device | None) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if device is None:
        return torch.device("cuda" if cuda_available else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not cuda_available:
        raise RuntimeError("CUDA is not available in the current Python environment")
    if resolved.type == "cuda" and resolved.index is not None:
        torch.cuda.set_device(resolved.index)
    return resolved


def _normalize_video_tensor(
    video: str | Path | torch.Tensor,
    *,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(video, (str, Path)):
        tensor = load_mp4_as_tensor(str(video), resize_hw=(height, width)).unsqueeze(0)
    else:
        tensor = video
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 5:
            raise ValueError("video tensor must have shape [B, C, T, H, W] or [C, T, H, W]")
    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(dtype=torch.bfloat16)
    return tensor.to(device)


def load_video(video_path: str, *, height: int = 480, width: int = 832) -> torch.Tensor:
    """Load a video file as a normalized tensor with shape [C, T, H, W]."""
    return load_mp4_as_tensor(video_path, resize_hw=(height, width))


def export_video(video: np.ndarray, output_path: str, *, fps: int = 16) -> str:
    """Write a `[T, H, W, C]` float video array to an mp4 file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    diffusers_export_to_video(video, str(output_file), fps=fps)
    return str(output_file)


class StreamDiffusionV2Pipeline:
    """Readable staged single-GPU API that mirrors the offline inference flow."""

    def __init__(
        self,
        checkpoint_folder: str,
        *,
        mode: SingleMode = "single",
        config_path: str | None = None,
        device: str | torch.device | None = None,
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
    ) -> None:
        if mode not in {"single", "single-wo"}:
            raise ValueError("StreamDiffusionV2Pipeline only supports 'single' and 'single-wo'")

        self._resource_stack = ExitStack()
        self.mode = mode
        self.device = _resolve_device(device)
        self.checkpoint_folder = checkpoint_folder
        self.noise_scale = float(noise_scale)
        self.height = int(height)
        self.width = int(width)
        self.fps = int(fps)
        self.seed = int(seed)
        self.step = int(step)
        self.profile = bool(profile)
        self.model_type = model_type
        self.prompt: str | None = None

        resolved_config_path = config_path or _resolve_default_config_path(self._resource_stack)
        self.config_path = resolved_config_path
        flags = normalize_acceleration_flags(
            {
                "use_taehv": use_taehv,
                "use_tensorrt": use_tensorrt,
                "fast": fast,
            }
        )
        self.use_taehv = bool(flags["use_taehv"])
        self.use_tensorrt = bool(flags["use_tensorrt"])
        self.fast = bool(flags["fast"])
        config_args = {
            "config_path": resolved_config_path,
            "checkpoint_folder": checkpoint_folder,
            "noise_scale": noise_scale,
            "height": height,
            "width": width,
            "fps": fps,
            "step": step,
            "seed": seed,
            "model_type": model_type,
            "profile": profile,
            "use_taehv": self.use_taehv,
            "use_tensorrt": self.use_tensorrt,
            "fast": self.fast,
            "t2v": False,
            "target_fps": None,
            "fixed_noise_scale": False,
            "num_frames": 81,
        }
        self.config = merge_cli_config(resolved_config_path, config_args)

        manager_cls = (
            StreamBatchInferencePipeline if mode == "single" else StreamNoBatchInferencePipeline
        )
        torch.set_grad_enabled(False)
        set_seed(self.seed)
        self.pipeline_manager = manager_cls(self.config, self.device)
        self.pipeline_manager.load_model(checkpoint_folder)
        self.chunk_size = 4 * self.config.num_frame_per_block
        self.num_steps = len(self.pipeline_manager.pipeline.denoising_step_list)
        self._next_chunk_index = 0

    def close(self) -> None:
        self._resource_stack.close()

    def __enter__(self) -> "StreamDiffusionV2Pipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def enable_acceleration(
        self,
        *,
        use_taehv: bool = False,
        use_tensorrt: bool = False,
        fast: bool = False,
    ) -> "StreamDiffusionV2Pipeline":
        """Rebuild the pipeline with the requested acceleration flags."""
        replacement = StreamDiffusionV2Pipeline(
            checkpoint_folder=self.checkpoint_folder,
            mode=self.mode,
            config_path=self.config_path,
            device=self.device,
            noise_scale=self.noise_scale,
            height=self.height,
            width=self.width,
            fps=self.fps,
            step=self.step,
            seed=self.seed,
            model_type=self.model_type,
            use_taehv=use_taehv,
            use_tensorrt=use_tensorrt,
            fast=fast,
            profile=self.profile,
        )
        self.close()
        self.__dict__.update(replacement.__dict__)
        return self

    def prepare(self, prompt: str) -> None:
        """Reset the stream state and store the prompt for the next denoising pass."""
        self.prompt = prompt
        self.pipeline_manager.reset_stream_state(reset_vae_flags=True)
        self.pipeline_manager.processed = 0
        self._next_chunk_index = 0

    def chunk_video(self, video: str | Path | torch.Tensor) -> list[VideoChunk]:
        """Split a full input video into the same chunks used by the offline inference loop."""
        input_video = _normalize_video_tensor(
            video,
            height=self.height,
            width=self.width,
            device=self.device,
        )
        _, _, total_frames, _, _ = input_video.shape
        if total_frames < 1 + self.chunk_size:
            raise ValueError(f"video must contain at least {1 + self.chunk_size} frames")

        chunks: list[VideoChunk] = []
        start_idx = 0
        end_idx = 1 + self.chunk_size
        current_start = 0
        current_end = self.pipeline_manager.pipeline.frame_seq_length * (1 + self.chunk_size // 4)

        chunks.append(
            VideoChunk(
                frames=input_video[:, :, start_idx:end_idx],
                start_idx=start_idx,
                end_idx=end_idx,
                current_start=current_start,
                current_end=current_end,
            )
        )

        while True:
            start_idx = end_idx
            end_idx = end_idx + self.chunk_size
            if end_idx > total_frames:
                break
            current_start = current_end
            current_end = current_end + (self.chunk_size // 4) * self.pipeline_manager.pipeline.frame_seq_length
            chunks.append(
                VideoChunk(
                    frames=input_video[:, :, start_idx:end_idx],
                    start_idx=start_idx,
                    end_idx=end_idx,
                    current_start=current_start,
                    current_end=current_end,
                )
            )
        return chunks

    @torch.inference_mode()
    def encode_chunk(
        self,
        input_video: str | Path | torch.Tensor,
        chunk: VideoChunk,
        *,
        previous_noise_scale: float | None = None,
        initial_noise_scale: float | None = None,
    ) -> EncodedChunk:
        """Encode one chunk in the same style as the offline inference loop."""
        full_video = _normalize_video_tensor(
            input_video,
            height=self.height,
            width=self.width,
            device=self.device,
        )
        noise_scale = self.noise_scale if previous_noise_scale is None else float(previous_noise_scale)
        init_noise_scale = self.noise_scale if initial_noise_scale is None else float(initial_noise_scale)
        current_step = None

        if chunk.start_idx != 0:
            noise_scale, current_step = compute_noise_scale_and_step(
                full_video,
                chunk.end_idx,
                self.chunk_size,
                noise_scale,
                init_noise_scale,
            )

        latents = self.pipeline_manager._timed_stream_encode(chunk.frames)
        latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
        noise = torch.randn_like(latents)
        return EncodedChunk(
            noisy_latents=noise * noise_scale + latents * (1 - noise_scale),
            current_start=chunk.current_start,
            current_end=chunk.current_end,
            noise_scale=float(noise_scale),
            current_step=current_step,
        )

    @torch.inference_mode()
    def encode_video(self, video: str | Path | torch.Tensor) -> list[EncodedChunk]:
        """Encode a full input video into noisy latent chunks."""
        chunks: list[EncodedChunk] = []
        noise_scale = float(self.noise_scale)
        init_noise_scale = noise_scale
        video_chunks = self.chunk_video(video)
        full_video = _normalize_video_tensor(
            video,
            height=self.height,
            width=self.width,
            device=self.device,
        )
        for chunk in video_chunks:
            encoded_chunk = self.encode_chunk(
                full_video,
                chunk,
                previous_noise_scale=noise_scale,
                initial_noise_scale=init_noise_scale,
            )
            noise_scale = encoded_chunk.noise_scale
            chunks.append(encoded_chunk)
        return chunks

    @torch.inference_mode()
    def denoise_chunks(self, chunks: list[EncodedChunk]) -> list[DenoisedChunk]:
        """Run DiT denoising over the encoded chunks."""
        if not chunks:
            raise ValueError("chunks must not be empty")
        if self.prompt is None:
            raise RuntimeError("Call prepare(prompt) before denoise_chunks(...)")

        self.prepare(self.prompt)
        outputs: list[DenoisedChunk] = []
        for chunk in chunks:
            denoised_chunk = self.denoise_chunk(chunk)
            if denoised_chunk is not None:
                outputs.append(denoised_chunk)
        return outputs

    @torch.inference_mode()
    def denoise_chunk(self, chunk: EncodedChunk) -> DenoisedChunk | None:
        """Run DiT on one encoded chunk and return a decodable latent when available."""
        if self.prompt is None:
            raise RuntimeError("Call prepare(prompt) before denoise_chunk(...)")

        if self._next_chunk_index == 0:
            if self.mode == "single":
                denoised_pred = self.pipeline_manager.prepare_pipeline(
                    text_prompts=[self.prompt],
                    noise=chunk.noisy_latents,
                    current_start=chunk.current_start,
                    current_end=chunk.current_end,
                )
            else:
                denoised_pred = self.pipeline_manager.prepare_pipeline(
                    text_prompts=[self.prompt],
                    noise=chunk.noisy_latents,
                    current_start=chunk.current_start,
                    current_end=chunk.current_end,
                    batch_denoise=False,
                )
            self._next_chunk_index += 1
            return DenoisedChunk(denoised_pred=denoised_pred, last_frame_only=False)

        current_start = chunk.current_start
        current_end = chunk.current_end

        if current_start // self.pipeline_manager.pipeline.frame_seq_length >= self.pipeline_manager.t_refresh:
            current_start = self.pipeline_manager.pipeline.kv_cache_length - self.pipeline_manager.pipeline.frame_seq_length
            current_end = current_start + (self.chunk_size // 4) * self.pipeline_manager.pipeline.frame_seq_length

        if self.mode == "single":
            denoised_pred = self.pipeline_manager.pipeline.inference_stream(
                noise=chunk.noisy_latents,
                current_start=current_start,
                current_end=current_end,
                current_step=chunk.current_step,
            )
            self.pipeline_manager.processed += 1
            self._next_chunk_index += 1
            if self.pipeline_manager.processed < self.num_steps:
                return None
            return DenoisedChunk(denoised_pred=denoised_pred, last_frame_only=True)

        denoised_pred = self.pipeline_manager.pipeline.inference_wo_batch(
            noise=chunk.noisy_latents,
            current_start=current_start,
            current_end=current_end,
            current_step=chunk.current_step,
        )
        self.pipeline_manager.processed += 1
        self._next_chunk_index += 1
        return DenoisedChunk(denoised_pred=denoised_pred, last_frame_only=True)

    @torch.inference_mode()
    def decode_chunks(self, chunks: list[DenoisedChunk]) -> np.ndarray:
        """Decode denoised latent chunks into a `[T, H, W, C]` video array."""
        if not chunks:
            raise ValueError("chunks must not be empty")
        decoded = [self.decode_chunk(chunk) for chunk in chunks]
        return np.concatenate(decoded, axis=0)

    @torch.inference_mode()
    def decode_chunk(self, chunk: DenoisedChunk) -> np.ndarray:
        """Decode one denoised latent chunk into `[T, H, W, C]` frames."""
        return self.pipeline_manager._decode_video_array(
            chunk.denoised_pred,
            last_frame_only=chunk.last_frame_only,
        )

    @torch.inference_mode()
    def __call__(self, video: str | Path | torch.Tensor) -> np.ndarray:
        """Run the full staged pipeline after `prepare(prompt)` has been called."""
        encoded = self.encode_video(video)
        denoised = self.denoise_chunks(encoded)
        return self.decode_chunks(denoised)
